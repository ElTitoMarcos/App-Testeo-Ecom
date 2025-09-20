from __future__ import annotations

import csv
import io
import hashlib
import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

from product_research_app.db import get_db, get_last_performance_config, init_db_performance
from product_research_app.database import (
    append_import_job_metrics,
    clear_staging_for_job,
    create_import_job,
    json_dump,
    merge_staging_into_products,
    transition_job_items,
    update_import_job_progress,
)

logger = logging.getLogger(__name__)


class ImportCancelledError(RuntimeError):
    """Raised when an import operation is cancelled by the caller."""


StatusCallback = Callable[..., None]
DEFAULT_BATCH_SIZE = 2000


def _sanitize(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


FIELD_ALIASES: dict[str, Sequence[str]] = {
    "id": ["id"],
    "name": ["name", "nombre", "productname", "product", "title"],
    "description": ["description", "descripcion", "desc"],
    "category": ["category", "categoria", "niche", "segment"],
    "category_path": ["category_path", "categorypath", "path"],
    "price": ["price", "precio", "cost", "unitprice"],
    "currency": ["currency", "moneda"],
    "image_url": [
        "image_url",
        "image",
        "imagen",
        "img",
        "imgurl",
        "picture",
        "imageurl",
        "imagelink",
        "mainimage",
        "mainimageurl",
    ],
    "desire": ["desire", "deseo"],
    "desire_magnitude": ["desire_magnitude", "desiremag", "magnituddeseo"],
    "awareness_level": ["awareness_level", "awareness", "nivelconsciencia"],
    "competition_level": ["competition_level", "competition", "saturacionmercado"],
    "date_range": ["date_range", "daterange", "rangofechas", "fecharango"],
    "launch_date": ["launch_date", "launchdate", "fechalanzamiento"],
    "rating": ["rating", "valoracion", "stars", "productrating"],
    "units_sold": ["units_sold", "unitssold", "units", "itemsold", "items_sold", "sold"],
    "revenue": ["revenue", "sales", "ingresos"],
    "conversion_rate": ["conversion_rate", "conversion", "tasaconversion", "cr", "conversionrate"],
    "winner_score": ["winner_score", "winnerscore"],
    "source": ["source", "fuente"],
    "brand": ["brand", "marca", "seller"],
    "asin": ["asin", "productasin", "asin13"],
    "url": [
        "url",
        "producturl",
        "product_url",
        "link",
        "productlink",
        "landingpage",
        "landing_page",
        "landingpageurl",
        "landing_page_url",
    ],
}

ALIASES_SANITIZED = {
    field: [_sanitize(alias) for alias in aliases]
    for field, aliases in FIELD_ALIASES.items()
}

_SIG_NORMALIZE_RE = re.compile(r"\s+")


def _normalize_sig_part(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    return _SIG_NORMALIZE_RE.sub(" ", text)


def _compute_sig_hash(name: str, brand: Optional[str], asin: Optional[str], url: Optional[str]) -> str:
    payload = "|".join(
        _normalize_sig_part(part) for part in (name, brand, asin, url)
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _parse_optional_number(value: Any, *, as_int: bool = False) -> Optional[float]:
    if value in (None, ""):
        return None
    s = str(value).strip()
    if not s:
        return None
    multiplier = 1.0
    if s.lower().endswith("m"):
        multiplier = 1_000_000.0
        s = s[:-1]
    elif s.lower().endswith("k"):
        multiplier = 1_000.0
        s = s[:-1]
    s = (
        s.replace("€", "")
        .replace("$", "")
        .replace("%", "")
        .replace(".", "")
        .replace(",", ".")
    )
    try:
        num = float(s) * multiplier
    except Exception:
        return None
    if as_int:
        try:
            return float(int(round(num)))
        except Exception:
            return None
    return num


def _pick(
    row: Mapping[str, Any],
    sanitized: Mapping[str, str],
    field: str,
    recognised: set[str],
) -> tuple[Optional[str], Optional[Any]]:
    for alias in ALIASES_SANITIZED.get(field, ()):  # type: ignore[arg-type]
        original = sanitized.get(alias)
        if original is None:
            continue
        value = row.get(original)
        if isinstance(value, str):
            value = value.strip()
        if value in (None, ""):
            continue
        recognised.add(original)
        return original, value
    return None, None


def _resolve_db_path(conn: sqlite3.Connection) -> str:
    cur = conn.execute("PRAGMA database_list;")
    rows = cur.fetchall()
    for _, name, path in rows:
        if name == "main" and path:
            return str(path)
    return "product_research_app/data.sqlite3"


def _iter_csv_bytes(payload: bytes) -> Iterator[Mapping[str, Any]]:
    buffer = io.BytesIO(payload)
    with io.TextIOWrapper(buffer, encoding="utf-8", errors="ignore", newline="") as text_stream:
        reader = csv.DictReader(text_stream)
        for row in reader:
            yield {k: v for k, v in row.items()}


@dataclass
class ImportSummary:
    job_id: int
    total_rows: int
    unique_rows: int
    batches: int
    total_ms: float
    throughput_rps: float


class BulkImporter:
    def __init__(
        self,
        db_path: str,
        job_id: int,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        source: Optional[str] = None,
        status_cb: Optional[StatusCallback] = None,
        should_abort: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.db_path = db_path
        self.job_id = job_id
        self.batch_size = max(1000, min(batch_size, 5000))
        self.source = source or "upload"
        self.status_cb: StatusCallback = status_cb or (lambda **_: None)
        self.write_conn = self._open_connection()
        self.status_conn = self._open_connection()
        self.pending: list[dict[str, Any]] = []
        self.processed = 0
        self.batches = 0
        self._unique_hashes: set[str] = set()
        self._summary: Optional[ImportSummary] = None
        self._start = 0.0
        self._should_abort = should_abort or (lambda: False)

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        init_db_performance(self.db_path, connection=conn)
        return conn

    def close(self) -> None:
        for conn in (self.write_conn, self.status_conn):
            try:
                conn.close()
            except Exception:
                pass

    def _raise_if_cancelled(self) -> None:
        try:
            if bool(self._should_abort()):
                raise ImportCancelledError("Import cancelled")
        except ImportCancelledError:
            raise
        except Exception:
            # If the callback itself fails we ignore the error and continue.
            return

    @property
    def summary(self) -> ImportSummary:
        if self._summary is None:
            return ImportSummary(self.job_id, 0, 0, 0, 0.0, 0.0)
        return self._summary

    def run(self, rows: Iterator[Mapping[str, Any]]) -> ImportSummary:
        self._start = time.perf_counter()
        self._update_status(phase="parse", status="running", processed=0, total=0)
        self.status_cb(stage="prepare", done=0, total=0)
        self.write_conn.execute("BEGIN IMMEDIATE;")
        try:
            self._raise_if_cancelled()
            for record in rows:
                self._raise_if_cancelled()
                prepared = self._prepare_record(record)
                if not prepared:
                    continue
                self.pending.append(prepared)
                self.processed += 1
                if len(self.pending) >= self.batch_size:
                    self._flush_pending()
            if self.pending:
                self._flush_pending()
            self._raise_if_cancelled()
            transition_job_items(self.write_conn, self.job_id, "raw", "pending_enrich")
            self._raise_if_cancelled()
            merge_staging_into_products(self.write_conn, self.job_id)
            unique_rows = len(self._unique_hashes)
            clear_staging_for_job(self.write_conn, self.job_id)
            self.write_conn.execute("COMMIT;")
        except Exception:
            self.write_conn.execute("ROLLBACK;")
            raise
        total_ms = (time.perf_counter() - self._start) * 1000 if self._start else 0.0
        throughput = (self.processed / (total_ms / 1000.0)) if total_ms else 0.0
        self.status_cb(stage="commit", done=self.processed, total=self.processed)
        self._summary = ImportSummary(
            self.job_id,
            self.processed,
            len(self._unique_hashes),
            self.batches,
            total_ms,
            throughput,
        )
        logger.info(
            "Import finished job=%s rows=%d unique=%d ms=%.2f batches=%d throughput=%.2f",
            self.job_id,
            self.processed,
            len(self._unique_hashes),
            total_ms,
            self.batches,
            throughput,
        )
        return self.summary

    def _prepare_record(self, record: Mapping[str, Any]) -> Optional[dict[str, Any]]:
        if not isinstance(record, Mapping):
            return None
        row = dict(record)
        sanitized: dict[str, str] = {}
        for key in list(row.keys()):
            if key is None:
                continue
            norm = _sanitize(str(key))
            if not norm:
                continue
            sanitized.setdefault(norm, str(key))
        recognised: set[str] = set()
        _, raw_name = _pick(row, sanitized, "name", recognised)
        if raw_name is None:
            return None
        name = str(raw_name).strip()
        if not name:
            return None
        _, raw_description = _pick(row, sanitized, "description", recognised)
        description = str(raw_description).strip() if raw_description not in (None, "") else None
        _, raw_category_path = _pick(row, sanitized, "category_path", recognised)
        category_path = str(raw_category_path).strip() if raw_category_path not in (None, "") else None
        _, raw_category = _pick(row, sanitized, "category", recognised)
        category_val = raw_category if raw_category not in (None, "") else category_path
        category = str(category_val).strip() if category_val not in (None, "") else None
        _, raw_price = _pick(row, sanitized, "price", recognised)
        price = _parse_optional_number(raw_price)
        _, raw_currency = _pick(row, sanitized, "currency", recognised)
        currency = str(raw_currency).strip() if raw_currency not in (None, "") else None
        _, raw_image = _pick(row, sanitized, "image_url", recognised)
        image_url = str(raw_image).strip() if raw_image not in (None, "") else None
        _, raw_brand = _pick(row, sanitized, "brand", recognised)
        brand = str(raw_brand).strip() if raw_brand not in (None, "") else None
        _, raw_asin = _pick(row, sanitized, "asin", recognised)
        asin = str(raw_asin).strip() if raw_asin not in (None, "") else None
        _, raw_url = _pick(row, sanitized, "url", recognised)
        product_url = str(raw_url).strip() if raw_url not in (None, "") else None
        _, raw_desire = _pick(row, sanitized, "desire", recognised)
        desire = str(raw_desire).strip() if raw_desire not in (None, "") else None
        _, raw_desire_mag = _pick(row, sanitized, "desire_magnitude", recognised)
        desire_mag = str(raw_desire_mag).strip() if raw_desire_mag not in (None, "") else None
        _, raw_awareness = _pick(row, sanitized, "awareness_level", recognised)
        awareness = str(raw_awareness).strip() if raw_awareness not in (None, "") else None
        _, raw_competition = _pick(row, sanitized, "competition_level", recognised)
        competition = str(raw_competition).strip() if raw_competition not in (None, "") else None
        _, raw_range = _pick(row, sanitized, "date_range", recognised)
        date_range = str(raw_range).strip() if raw_range not in (None, "") else None
        _, raw_launch = _pick(row, sanitized, "launch_date", recognised)
        launch_date = str(raw_launch).strip() if raw_launch not in (None, "") else None
        if launch_date:
            launch_date = launch_date[:10]
        _, raw_rating = _pick(row, sanitized, "rating", recognised)
        rating = _parse_optional_number(raw_rating)
        _, raw_units = _pick(row, sanitized, "units_sold", recognised)
        units_sold = _parse_optional_number(raw_units, as_int=True)
        _, raw_revenue = _pick(row, sanitized, "revenue", recognised)
        revenue = _parse_optional_number(raw_revenue)
        _, raw_conversion = _pick(row, sanitized, "conversion_rate", recognised)
        conversion_rate = _parse_optional_number(raw_conversion)
        _, raw_winner = _pick(row, sanitized, "winner_score", recognised)
        winner_score = None
        if raw_winner not in (None, ""):
            winner_score = _parse_optional_number(raw_winner, as_int=True)
            if winner_score is not None:
                winner_score = int(winner_score)
        _, raw_source = _pick(row, sanitized, "source", recognised)
        source_val = str(raw_source).strip() if raw_source not in (None, "") else None
        if not source_val:
            source_val = self.source
        extras: dict[str, Any] = {}
        if rating is not None:
            extras["rating"] = rating
        if units_sold is not None:
            extras["units_sold"] = int(units_sold)
        if revenue is not None:
            extras["revenue"] = revenue
        if conversion_rate is not None:
            extras["conversion_rate"] = conversion_rate
        if launch_date:
            extras["launch_date"] = launch_date
        if category_path and (not category or category_path != category):
            extras["category_path"] = category_path
        if brand:
            extras["brand"] = brand
        if asin:
            extras["asin"] = asin
        if product_url:
            extras["product_url"] = product_url
        for key, value in row.items():
            if key in recognised or key is None:
                continue
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    continue
            extras[key] = value
        sig_hash = _compute_sig_hash(name, brand, asin, product_url)
        if not sig_hash:
            return None
        return {
            "sig_hash": sig_hash,
            "name": name,
            "description": description,
            "category": category,
            "price": float(price) if price is not None else None,
            "currency": currency,
            "image_url": image_url,
            "brand": brand,
            "asin": asin,
            "product_url": product_url,
            "source": source_val,
            "import_date": datetime.utcnow().isoformat(),
            "desire": desire,
            "desire_magnitude": desire_mag,
            "awareness_level": awareness,
            "competition_level": competition,
            "date_range": date_range,
            "winner_score": winner_score,
            "extra": extras,
            "raw": row,
        }

    def _flush_pending(self) -> None:
        if not self.pending:
            return
        self._raise_if_cancelled()
        batch_start = time.perf_counter()
        now = datetime.utcnow().isoformat()
        batch = list(self.pending)
        self.pending.clear()
        cur = self.write_conn.cursor()
        items_payload = [
            (self.job_id, row["sig_hash"], json_dump(row["raw"]), "raw", now)
            for row in batch
        ]
        staging_payload = [
            (
                self.job_id,
                row["sig_hash"],
                row["name"],
                row.get("description"),
                row.get("category"),
                row.get("price"),
                row.get("currency"),
                row.get("image_url"),
                row.get("brand"),
                row.get("asin"),
                row.get("product_url"),
                row["source"],
                row["import_date"],
                row.get("desire"),
                row.get("desire_magnitude"),
                row.get("awareness_level"),
                row.get("competition_level"),
                row.get("date_range"),
                row.get("winner_score"),
                json_dump(row["extra"]),
            )
            for row in batch
        ]
        cur.executemany(
            """
            INSERT INTO items (job_id, sig_hash, raw, state, updated_at)
            VALUES (?, ?, json(?), ?, ?)
            """,
            items_payload,
        )
        cur.executemany(
            """
            INSERT OR REPLACE INTO products_staging (
                job_id, sig_hash, name, description, category, price, currency,
                image_url, brand, asin, product_url, source, import_date,
                desire, desire_magnitude, awareness_level, competition_level,
                date_range, winner_score, extra
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, json(?))
            """,
            staging_payload,
        )
        batch_size = len(batch)
        for row in batch:
            self._unique_hashes.add(row["sig_hash"])
        self.batches += 1
        elapsed_ms = (time.perf_counter() - batch_start) * 1000
        batch_throughput = batch_size / ((elapsed_ms / 1000.0) or 1.0)
        append_import_job_metrics(
            self.write_conn,
            self.job_id,
            self.batches,
            batch_size,
            elapsed_ms,
            batch_throughput,
            commit=False,
        )
        logger.info(
            "Import batch job=%s batch=%d rows=%d unique=%d ms=%.2f throughput=%.2f",
            self.job_id,
            self.batches,
            batch_size,
            len(self._unique_hashes),
            elapsed_ms,
            batch_throughput,
        )
        self.status_cb(
            stage="insert",
            done=self.processed,
            total=self.processed,
            batch=batch_size,
        )
        self._update_status(
            phase="insert",
            processed=self.processed,
            total=self.processed,
            rows_imported=len(self._unique_hashes),
        )

    def _update_status(self, **kwargs: Any) -> None:
        update_import_job_progress(self.status_conn, self.job_id, **kwargs)


def _prepare_rows(records: Iterable[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
    for record in records:
        if isinstance(record, Mapping):
            yield record


def fast_import(
    csv_bytes: bytes,
    *,
    status_cb: StatusCallback = lambda **_: None,
    source: Optional[str] = None,
    job_id: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    db_path: Optional[str] = None,
    should_abort: Optional[Callable[[], bool]] = None,
) -> int:
    base_conn = get_db()
    resolved_path = db_path or _resolve_db_path(base_conn)
    pragmas = get_last_performance_config()
    config = {"batch_size": batch_size, "pragmas": pragmas, "source": source or "upload"}
    created_here = False
    if job_id is None:
        job_id = create_import_job(
            base_conn,
            status="running",
            phase="parse",
            total=0,
            processed=0,
            config=config,
        )
        created_here = True
    else:
        update_import_job_progress(base_conn, job_id, status="running", phase="parse", processed=0, total=0, config=config)
    importer = BulkImporter(
        resolved_path,
        job_id,
        batch_size=batch_size,
        source=source,
        status_cb=status_cb,
        should_abort=should_abort,
    )
    try:
        summary = importer.run(_iter_csv_bytes(csv_bytes))
        update_import_job_progress(
            base_conn,
            job_id,
            phase="done",
            status="done",
            processed=summary.total_rows,
            total=summary.total_rows,
            rows_imported=summary.unique_rows,
            metrics={
                "total_rows": summary.total_rows,
                "unique_rows": summary.unique_rows,
                "batches": summary.batches,
                "total_ms": summary.total_ms,
                "throughput_rps": summary.throughput_rps,
                "batch_size": batch_size,
            },
        )
        return summary.unique_rows
    except ImportCancelledError:
        logger.info("Fast import cancelled job=%s", job_id)
        update_import_job_progress(
            base_conn,
            job_id,
            status="cancelled",
            phase="cancelled",
            processed=importer.processed,
            total=importer.processed,
            rows_imported=0,
            error=None,
        )
        raise
    except Exception as exc:
        logger.exception("Fast import failed job=%s", job_id)
        update_import_job_progress(
            base_conn,
            job_id,
            status="error",
            phase="done",
            error=str(exc),
        )
        raise
    finally:
        importer.close()
        if created_here:
            base_conn.commit()


def fast_import_records(
    records: Iterable[Mapping[str, Any]],
    *,
    status_cb: StatusCallback = lambda **_: None,
    source: Optional[str] = None,
    job_id: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    db_path: Optional[str] = None,
    should_abort: Optional[Callable[[], bool]] = None,
) -> int:
    base_conn = get_db()
    resolved_path = db_path or _resolve_db_path(base_conn)
    pragmas = get_last_performance_config()
    config = {"batch_size": batch_size, "pragmas": pragmas, "source": source or "upload"}
    created_here = False
    if job_id is None:
        job_id = create_import_job(
            base_conn,
            status="running",
            phase="parse",
            total=0,
            processed=0,
            config=config,
        )
        created_here = True
    else:
        update_import_job_progress(base_conn, job_id, status="running", phase="parse", processed=0, total=0, config=config)
    importer = BulkImporter(
        resolved_path,
        job_id,
        batch_size=batch_size,
        source=source,
        status_cb=status_cb,
        should_abort=should_abort,
    )
    try:
        summary = importer.run(_prepare_rows(records))
        update_import_job_progress(
            base_conn,
            job_id,
            phase="done",
            status="done",
            processed=summary.total_rows,
            total=summary.total_rows,
            rows_imported=summary.unique_rows,
            metrics={
                "total_rows": summary.total_rows,
                "unique_rows": summary.unique_rows,
                "batches": summary.batches,
                "total_ms": summary.total_ms,
                "throughput_rps": summary.throughput_rps,
                "batch_size": batch_size,
            },
        )
        return summary.unique_rows
    except ImportCancelledError:
        logger.info("Fast record import cancelled job=%s", job_id)
        update_import_job_progress(
            base_conn,
            job_id,
            status="cancelled",
            phase="cancelled",
            processed=importer.processed,
            total=importer.processed,
            rows_imported=0,
            error=None,
        )
        raise
    except Exception as exc:
        logger.exception("Fast record import failed job=%s", job_id)
        update_import_job_progress(
            base_conn,
            job_id,
            status="error",
            phase="done",
            error=str(exc),
        )
        raise
    finally:
        importer.close()
        if created_here:
            base_conn.commit()


def benchmark_bulk_import(
    row_count: int = 10_000,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    db_path: Optional[str] = None,
) -> ImportSummary:
    logger.info(
        "Starting benchmark import rows=%d batch_size=%d", row_count, batch_size
    )

    def _records() -> Iterator[Mapping[str, Any]]:
        for idx in range(row_count):
            yield {
                "title": f"Synthetic Product {idx}",
                "price": 19.99,
                "brand": f"Brand {idx % 50}",
                "asin": f"B00{idx:06d}",
                "url": f"https://example.com/product/{idx}",
                "category": "synthetic",
            }

    start = time.perf_counter()
    unique_rows = fast_import_records(
        _records(),
        source="benchmark",
        batch_size=batch_size,
        db_path=db_path,
        status_cb=lambda **_: None,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    throughput = row_count / ((elapsed_ms / 1000) or 1.0)
    logger.info(
        "Benchmark completed rows=%d unique=%d ms=%.2f throughput=%.2f",
        row_count,
        unique_rows,
        elapsed_ms,
        throughput,
    )
    return ImportSummary(0, row_count, unique_rows, 0, elapsed_ms, throughput)
