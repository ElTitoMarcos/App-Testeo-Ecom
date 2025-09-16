from __future__ import annotations

import csv
import io
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from product_research_app.db import get_db
from product_research_app.utils.timing import phase


PRODUCTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    title TEXT,
    price REAL,
    units_sold REAL,
    revenue REAL,
    rating REAL,
    desire TEXT,
    competition TEXT,
    oldness TEXT,
    awareness TEXT,
    category TEXT,
    store TEXT,
    description TEXT,
    dateAdded TEXT
);
"""

PRODUCT_COLUMN_TYPES: dict[str, str] = {
    "title": "TEXT",
    "price": "REAL",
    "units_sold": "REAL",
    "revenue": "REAL",
    "rating": "REAL",
    "desire": "TEXT",
    "competition": "TEXT",
    "oldness": "TEXT",
    "awareness": "TEXT",
    "category": "TEXT",
    "store": "TEXT",
    "description": "TEXT",
    "dateAdded": "TEXT",
}

UPSERT_SQL = """
INSERT INTO products (
    id, title, price, units_sold, revenue, rating, desire, competition,
    oldness, awareness, category, store, description, dateAdded
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    title=excluded.title,
    price=excluded.price,
    units_sold=excluded.units_sold,
    revenue=excluded.revenue,
    rating=excluded.rating,
    desire=excluded.desire,
    competition=excluded.competition,
    oldness=excluded.oldness,
    awareness=excluded.awareness,
    category=excluded.category,
    store=excluded.store,
    description=excluded.description,
    dateAdded=excluded.dateAdded;
"""

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "ID"),
    "title": ("title", "name", "product", "producto"),
    "price": ("price", "precio", "cost"),
    "units_sold": (
        "units_sold",
        "unitsSold",
        "units",
        "unitssold",
        "ventas",
        "sold",
        "orders",
    ),
    "revenue": ("revenue", "sales", "ingresos", "income"),
    "rating": ("rating", "valoracion", "stars"),
    "desire": ("desire", "desire_score", "deseo"),
    "competition": ("competition", "competition_level", "saturacion"),
    "oldness": ("oldness", "age", "edad", "antiguedad"),
    "awareness": ("awareness", "awareness_level", "consciencia"),
    "category": ("category", "categoria", "niche"),
    "store": ("store", "tienda", "shop", "seller"),
    "description": ("description", "descripcion", "desc"),
    "dateAdded": ("dateAdded", "date_added", "added", "fecha"),
}

SECONDARY_INDEXES: dict[str, str] = {
    "idx_products_category": "CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);",
    "idx_products_store": "CREATE INDEX IF NOT EXISTS idx_products_store ON products(store);",
    "idx_products_date": "CREATE INDEX IF NOT EXISTS idx_products_date ON products(dateAdded);",
}

def _normalize_key(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _lookup(row: Mapping[str, object], sanitized: dict[str, str], field: str) -> object | None:
    aliases = FIELD_ALIASES.get(field, ())
    for alias in aliases:
        key = sanitized.get(_normalize_key(alias))
        if key is None:
            continue
        value = row.get(key)
        if isinstance(value, str):
            value = value.strip()
        if value in (None, ""):
            continue
        return value
    return None


def _as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    multiplier = 1.0
    last = text[-1].lower()
    if last == "m":
        multiplier = 1_000_000.0
        text = text[:-1]
    elif last == "k":
        multiplier = 1_000.0
        text = text[:-1]

    text = (
        text.replace("€", "")
        .replace("$", "")
        .replace("%", "")
        .replace("\u00a0", "")
        .replace(" ", "")
    )
    if not text:
        return None

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            decimal_sep, thousands_sep = ",", "."
        else:
            decimal_sep, thousands_sep = ".", ","
        text = text.replace(thousands_sep, "")
        text = text.replace(decimal_sep, ".")
    elif text.count(",") > 1:
        text = text.replace(",", "")
    elif text.count(".") > 1:
        text = text.replace(".", "")
    else:
        text = text.replace(",", ".")

    try:
        return float(text) * multiplier
    except ValueError:
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        if match:
            try:
                return float(match.group(0)) * multiplier
            except ValueError:
                return None
    return None


def _as_int(value: object) -> int | None:
    num = _as_float(value)
    if num is None:
        return None
    try:
        return int(round(num))
    except Exception:
        return None

def _resolve_numeric_columns(fieldnames: Iterable[str | None]) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    for name in fieldnames:
        if name is None:
            continue
        norm = _normalize_key(str(name))
        if not norm:
            continue
        sanitized[norm] = str(name)

    resolved: dict[str, str] = {}
    numeric_fields = ("price", "units_sold", "revenue", "rating", "oldness", "awareness")
    for field in numeric_fields:
        for alias in FIELD_ALIASES.get(field, ()):  # pragma: no branch - tiny tuple
            key = _normalize_key(alias)
            actual = sanitized.get(key)
            if actual is not None:
                resolved[field] = actual
                break
    return resolved


def _vectorized_type_cast(
    records: list[dict[str, Any]],
    resolved: Mapping[str, str],
) -> None:
    if not records or not resolved:
        return

    casters: dict[str, Callable[[object], Any]] = {
        "price": _as_float,
        "units_sold": _as_int,
        "revenue": _as_float,
        "rating": _as_float,
        "oldness": _as_int,
        "awareness": _as_float,
    }

    for field, column in resolved.items():
        caster = casters.get(field)
        if caster is None:
            continue
        converted = [caster(record.get(column)) for record in records]
        for record, value in zip(records, converted):
            record[column] = value


def _coerce_text(value: object | None) -> str | None:
    if value in (None, ""):
        return None
    return str(value).strip()


def _normalize_for_dedup(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.lower().split())


def _row_from_mapping(
    row: Mapping[str, object],
    seen_ids: set[int],
    seen_hashes: set[tuple[str, str]],
) -> tuple[int, str, float | None, float | None, float | None, float | None, object | None,
           object | None, object | None, object | None, str | None, str | None, str | None, str | None] | None:
    sanitized = {}
    for key in row.keys():
        if key is None:
            continue
        norm = _normalize_key(str(key))
        if not norm:
            continue
        sanitized.setdefault(norm, str(key))

    product_id = _as_int(_lookup(row, sanitized, "id"))
    if product_id is None or product_id <= 0:
        return None
    if product_id in seen_ids:
        return None
    seen_ids.add(product_id)

    title_val = _coerce_text(_lookup(row, sanitized, "title")) or ""
    store_val = _coerce_text(_lookup(row, sanitized, "store")) or ""
    hash_key = (_normalize_for_dedup(title_val), _normalize_for_dedup(store_val))
    if hash_key != ("", "") and hash_key in seen_hashes:
        return None
    seen_hashes.add(hash_key)

    price_val = _as_float(_lookup(row, sanitized, "price"))
    units_val = _as_int(_lookup(row, sanitized, "units_sold"))
    revenue_val = _as_float(_lookup(row, sanitized, "revenue"))
    rating_val = _as_float(_lookup(row, sanitized, "rating"))

    desire_val = _lookup(row, sanitized, "desire")
    competition_val = _lookup(row, sanitized, "competition")
    oldness_val = _as_int(_lookup(row, sanitized, "oldness"))
    awareness_val = _as_float(_lookup(row, sanitized, "awareness"))

    category_val = _coerce_text(_lookup(row, sanitized, "category"))
    description_val = _coerce_text(_lookup(row, sanitized, "description"))
    date_added_val = _coerce_text(_lookup(row, sanitized, "dateAdded"))

    return (
        product_id,
        title_val,
        price_val,
        units_val,
        revenue_val,
        rating_val,
        desire_val,
        competition_val,
        oldness_val,
        awareness_val,
        category_val,
        store_val or None,
        description_val,
        date_added_val,
    )


def _prepare_rows(records: Iterable[Mapping[str, object]]) -> list[tuple]:
    rows: list[tuple] = []
    seen_ids: set[int] = set()
    seen_hashes: set[tuple[str, str]] = set()
    for record in records:
        if not isinstance(record, Mapping):
            continue
        prepared = _row_from_mapping(record, seen_ids, seen_hashes)
        if prepared is None:
            continue
        rows.append(prepared)
    return rows


def _rows_from_csv(csv_bytes: bytes) -> list[tuple]:
    text = csv_bytes.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    records: list[dict[str, Any]] = list(reader)
    if not records:
        return []

    resolved = _resolve_numeric_columns(reader.fieldnames or [])
    if resolved:
        _vectorized_type_cast(records, resolved)

    return _prepare_rows(records)

def _ensure_products_schema(conn) -> None:
    conn.execute(PRODUCTS_TABLE_SQL)
    existing = {
        row[1]
        for row in conn.execute("PRAGMA table_info(products)")
    }
    for column, col_type in PRODUCT_COLUMN_TYPES.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE products ADD COLUMN {column} {col_type};")


def _ensure_unique_index(conn) -> None:
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_products_id ON products(id);")

def _drop_secondary_indexes(conn) -> None:
    for name in SECONDARY_INDEXES:
        conn.execute(f"DROP INDEX IF EXISTS {name};")


def _recreate_secondary_indexes(conn) -> None:
    for sql in SECONDARY_INDEXES.values():
        conn.execute(sql)

def _apply_pragmas(conn) -> dict[str, object]:
    original: dict[str, object] = {}
    try:
        original["journal_mode"] = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    except Exception:
        original["journal_mode"] = None
    try:
        original["synchronous"] = conn.execute("PRAGMA synchronous;").fetchone()[0]
    except Exception:
        original["synchronous"] = None
    try:
        original["temp_store"] = conn.execute("PRAGMA temp_store;").fetchone()[0]
    except Exception:
        original["temp_store"] = None
    try:
        original["foreign_keys"] = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
    except Exception:
        original["foreign_keys"] = None

    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA foreign_keys=OFF;")
    return original


def _restore_pragmas(conn, original: dict[str, object]) -> None:
    jm = original.get("journal_mode")
    if jm and str(jm).upper() != "WAL":
        conn.execute(f"PRAGMA journal_mode={jm};")
    else:
        conn.execute("PRAGMA journal_mode=WAL;")

    sync = original.get("synchronous")
    if sync is not None:
        conn.execute(f"PRAGMA synchronous={sync};")
    temp_store = original.get("temp_store")
    if temp_store is not None:
        conn.execute(f"PRAGMA temp_store={temp_store};")

    fk = original.get("foreign_keys")
    if fk in (0, 1):
      
        conn.execute(f"PRAGMA foreign_keys={'ON' if fk else 'OFF'};")
    else:
        conn.execute("PRAGMA foreign_keys=ON;")

    conn = get_db()
    _ensure_products_schema(conn)
    _ensure_unique_index(conn)

def _import_rows(
    rows: Sequence[tuple],
    status_cb,
    phase_recorder: Callable[[Mapping[str, int]], None] | None = None,
) -> int:
    total = len(rows)
    status_cb(total=total)

    conn = get_db()
    _ensure_products_schema(conn)
    _ensure_unique_index(conn)

    if rows:
        with phase("drop_product_indexes") as info:
            _drop_secondary_indexes(conn)
        if phase_recorder is not None:
            phase_recorder(info)
    original_pragmas = _apply_pragmas(conn)
    try:
        rows_imported = 0
        with conn:
            cursor = conn.cursor()
            try:
                if rows:
                    status_cb(stage="db_bulk_insert", done=0, total=total)
                    cursor.executemany(UPSERT_SQL, rows)
                    rows_imported = total
                status_cb(
                    stage="db_bulk_insert",
                    done=rows_imported,
                    total=total,
                    imported=rows_imported,
                )
            finally:
                cursor.close()
    finally:
        _restore_pragmas(conn, original_pragmas)

    if rows:
        with phase("rebuild_product_indexes") as info:
            _recreate_secondary_indexes(conn)
        if phase_recorder is not None:
            phase_recorder(info)

    return total

def fast_import_adaptive(
    csv_bytes: bytes,
    status_cb=lambda **_: None,
    phase_recorder: Callable[[Mapping[str, int]], None] | None = None,
):
    if not isinstance(csv_bytes, (bytes, bytearray)):
        csv_bytes = bytes(csv_bytes)

    rows = _rows_from_csv(csv_bytes)
    product_ids = [
        int(row[0])
        for row in rows
        if isinstance(row, tuple) and row and row[0] is not None
    ]

    rows_imported = _import_rows(rows, status_cb, phase_recorder=phase_recorder)

    def _optimize():
        return None

    _optimize.rows_imported = rows_imported  # type: ignore[attr-defined]
    _optimize.product_ids = product_ids  # type: ignore[attr-defined]
    return _optimize


def fast_import(
    csv_bytes,
    status_cb=lambda **_: None,
    source=None,
    phase_recorder: Callable[[Mapping[str, int]], None] | None = None,
):
    optimize = fast_import_adaptive(
        csv_bytes,
        status_cb=status_cb,
        phase_recorder=phase_recorder,
    )
    try:
        optimize()
    except Exception:
        raise
    return int(getattr(optimize, "rows_imported", 0) or 0)

def fast_import_records(
    records: Iterable[Mapping[str, object]],
    status_cb=lambda **_: None,
    source=None,
    phase_recorder: Callable[[Mapping[str, int]], None] | None = None,
):
    if not isinstance(records, Sequence):
        records = list(records)
    rows = _prepare_rows(records)
    return _import_rows(rows, status_cb, phase_recorder=phase_recorder)
