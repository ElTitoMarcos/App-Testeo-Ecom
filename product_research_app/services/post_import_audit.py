from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .. import database
from . import ai_columns, audit_config
from .desire_utils import cleanse, looks_like_product_desc

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = APP_DIR / "data.sqlite3"


def _coerce_conn(db: Any) -> Tuple[sqlite3.Connection, bool]:
    if db is None:
        return database.get_connection(DEFAULT_DB_PATH), True
    if isinstance(db, (str, Path)):
        return database.get_connection(Path(db)), True
    if hasattr(db, "cursor"):
        return db, False  # type: ignore[return-value]
    candidate = getattr(db, "connection", None)
    if candidate is not None and hasattr(candidate, "cursor"):
        return candidate, False  # type: ignore[return-value]
    raise TypeError("db must be a sqlite3.Connection, path or wrapper with .connection")


def _get_product_columns(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute("PRAGMA table_info(products)")
    return {str(row[1]) for row in cur.fetchall()}


def _count_products(conn: sqlite3.Connection, ids: Optional[Sequence[int]] = None) -> int:
    cur = conn.cursor()
    if ids:
        uniq = sorted({int(pid) for pid in ids})
        if not uniq:
            return 0
        placeholders = ",".join(["?"] * len(uniq))
        cur.execute(f"SELECT COUNT(*) FROM products WHERE id IN ({placeholders})", tuple(uniq))
    else:
        cur.execute("SELECT COUNT(*) FROM products")
    row = cur.fetchone()
    return int(row[0]) if row else 0


def scan_missing(db: Any, ids: Optional[Sequence[int]] = None) -> Dict[int, List[str]]:
    conn, should_close = _coerce_conn(db)
    missing: Dict[int, List[str]] = {}
    try:
        for row in database.iter_products(conn, ids=ids):
            data = {k: row[k] for k in row.keys()}  # type: ignore[attr-defined]
            try:
                pid = int(data.get("id"))
            except Exception:
                continue
            to_fill: List[str] = []
            for field in audit_config.REQUIRED_FIELDS.keys():
                if audit_config.should_fill(field, data):
                    to_fill.append(field)
            if to_fill:
                missing[pid] = to_fill
    finally:
        if should_close:
            conn.close()
    return missing


def _prepare_desire_payload(
    pid: int,
    payload: Mapping[str, Any],
    product_row: Mapping[str, Any],
    columns: set[str],
) -> Optional[Dict[str, Any]]:
    title = str(product_row.get("name") or product_row.get("title") or "")
    desire_txt = str(payload.get("desire") or "").strip()
    if desire_txt:
        cleaned = cleanse(desire_txt).strip()
        if cleaned:
            desire_txt = cleaned
    if not desire_txt:
        return None
    if len(desire_txt) < 280 or len(desire_txt) > 420:
        logger.info("audit: fail product=%s reason=desire_length", pid)
        return None
    if looks_like_product_desc(desire_txt, title):
        logger.info("audit: fail product=%s reason=looks_like_product", pid)
        return None

    update: Dict[str, Any] = {}
    if "desire" in columns:
        update["desire"] = desire_txt

    desire_primary = payload.get("desire_primary")
    if desire_primary in ("", None):
        desire_primary = None
    if desire_primary and "desire_primary" in columns:
        update["desire_primary"] = desire_primary

    raw_magnitude = payload.get("desire_magnitude")
    magnitude_obj: Optional[Dict[str, Any]] = None
    if isinstance(raw_magnitude, Mapping):
        magnitude_obj = {
            "scope": raw_magnitude.get("scope"),
            "urgency": raw_magnitude.get("urgency"),
            "staying_power": raw_magnitude.get("staying_power"),
            "overall": raw_magnitude.get("overall"),
        }
    elif isinstance(raw_magnitude, str):
        try:
            parsed = json.loads(raw_magnitude)
            if isinstance(parsed, Mapping):
                magnitude_obj = {
                    "scope": parsed.get("scope"),
                    "urgency": parsed.get("urgency"),
                    "staying_power": parsed.get("staying_power"),
                    "overall": parsed.get("overall"),
                }
        except json.JSONDecodeError:
            magnitude_obj = None
    if magnitude_obj and "desire_magnitude" in columns:
        update["desire_magnitude"] = json.dumps(magnitude_obj)

    awareness = payload.get("awareness_level")
    if awareness and "awareness_level" in columns:
        update["awareness_level"] = awareness

    competition = payload.get("competition_level")
    if competition and "competition_level" in columns:
        update["competition_level"] = competition

    ai_label = payload.get("ai_desire_label") or None
    if not ai_label and desire_primary:
        ai_label = str(desire_primary)
    if not ai_label:
        ai_label = " ".join(desire_txt.split()[:8]).strip()
    if ai_label and "ai_desire_label" in columns:
        update["ai_desire_label"] = ai_label

    return update if update else None


def fill_batch_desire(
    db: Any,
    ids: List[int],
    *,
    batch_size: int = 32,
    parallel: int = 3,
) -> None:
    if not ids:
        return
    conn, should_close = _coerce_conn(db)
    try:
        columns = _get_product_columns(conn)
        unique_ids = []
        seen: set[int] = set()
        for raw in ids:
            try:
                pid = int(raw)
            except Exception:
                continue
            if pid in seen:
                continue
            unique_ids.append(pid)
            seen.add(pid)
        total = len(unique_ids)
        for start in range(0, total, batch_size):
            chunk = unique_ids[start : start + batch_size]
            if not chunk:
                continue
            logger.info(
                "audit: batch start %d..%d / total %d",
                start + 1,
                start + len(chunk),
                total,
            )
            result = ai_columns.run_ai_fill_job(
                0,
                chunk,
                microbatch=batch_size,
                parallelism=parallel,
                status_cb=None,
            )
            ok_map: Dict[str, Mapping[str, Any]] = result.get("ok", {}) or {}
            ko_map: Dict[str, Any] = result.get("ko", {}) or {}
            if ko_map:
                for pid_str, reason in ko_map.items():
                    try:
                        pid = int(pid_str)
                    except Exception:
                        continue
                    logger.info("audit: fail product=%s reason=%s", pid, reason or "error")
            if not ok_map:
                continue
            product_rows = {
                int(row["id"]): {k: row[k] for k in row.keys()}  # type: ignore[attr-defined]
                for row in database.get_products_by_ids(conn, [int(pid) for pid in ok_map.keys()])
            }
            assignments: Dict[int, Dict[str, Any]] = {}
            for pid_str, payload in ok_map.items():
                try:
                    pid = int(pid_str)
                except Exception:
                    continue
                row = product_rows.get(pid)
                if not row:
                    continue
                update_payload = _prepare_desire_payload(pid, payload, row, columns)
                if update_payload:
                    assignments[pid] = update_payload
            if assignments:
                cur = conn.cursor()
                for pid, data in assignments.items():
                    sets = ",".join(f"{key}=?" for key in data.keys())
                    params = list(data.values())
                    params.append(pid)
                    cur.execute(f"UPDATE products SET {sets} WHERE id=?", params)
                conn.commit()
    finally:
        if should_close:
            conn.close()


def derive_missing_locally(row: Dict[str, Any]) -> None:
    current = str(row.get("ai_desire_label") or "").strip()
    if current:
        return
    primary = row.get("desire_primary")
    if not primary:
        return
    row["ai_desire_label"] = str(primary)


def run_audit(
    db: Any,
    ids: Optional[Sequence[int]] = None,
    batch_size: int = 32,
    parallel: int = 3,
    max_passes: int = 2,
    logger_obj: Optional[Any] = None,
) -> Dict[str, int]:
    conn, should_close = _coerce_conn(db)
    summary = {"checked": 0, "updated": 0, "skipped": 0, "failed": 0}
    try:
        total = _count_products(conn, ids)
        summary["checked"] = total
        initial_missing = scan_missing(conn, ids=ids)
        summary["skipped"] = max(0, total - len(initial_missing))
        current_missing = initial_missing
        for attempt in range(max_passes):
            if not current_missing:
                break
            desire_ids: List[int] = []
            derived_ids: List[int] = []
            for pid, fields in current_missing.items():
                via_types = {audit_config.REQUIRED_FIELDS[f]["via"] for f in fields if f in audit_config.REQUIRED_FIELDS}
                if "DESIRE" in via_types:
                    desire_ids.append(pid)
                elif via_types == {"DERIVED"}:
                    derived_ids.append(pid)
            if desire_ids:
                for start in range(0, len(desire_ids), batch_size):
                    chunk = desire_ids[start : start + batch_size]
                    fill_batch_desire(
                        conn,
                        chunk,
                        batch_size=min(batch_size, 32),
                        parallel=parallel,
                    )
            if derived_ids:
                rows = database.get_products_by_ids(conn, derived_ids)
                cur = conn.cursor()
                for row in rows:
                    data = {k: row[k] for k in row.keys()}  # type: ignore[attr-defined]
                    before = data.get("ai_desire_label")
                    derive_missing_locally(data)
                    after = data.get("ai_desire_label")
                    if after and after != before:
                        cur.execute(
                            "UPDATE products SET ai_desire_label=? WHERE id=?",
                            (after, int(data.get("id"))),
                        )
                conn.commit()
            current_missing = scan_missing(conn, ids=ids)
        final_missing = current_missing
        summary["failed"] = len(final_missing)
        summary["updated"] = max(0, len(initial_missing) - len(final_missing))
        target_logger = logger_obj if logger_obj is not None else logger
        try:
            target_logger.info(
                "audit: done checked=%s updated=%s failed=%s",
                summary["checked"],
                summary["updated"],
                summary["failed"],
            )
        except Exception:
            try:
                globals()["logger"].info(
                    "audit: done checked=%s updated=%s failed=%s",
                    summary["checked"],
                    summary["updated"],
                    summary["failed"],
                )
            except Exception:
                pass
        return summary
    finally:
        if should_close:
            conn.close()
