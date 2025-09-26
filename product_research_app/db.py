import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union


logger = logging.getLogger(__name__)

_DB: Optional[sqlite3.Connection] = None
_DB_PATH: Optional[str] = None
_DB_LOCK = threading.Lock()
_PERF_APPLIED: dict[str, bool] = {}
_PERF_CONFIG: dict[str, Union[str, int]] = {
    "journal_mode": "WAL",
    "synchronous": "NORMAL",
    "temp_store": "MEMORY",
    "mmap_size": 268_435_456,
}


def _is_sqlite_url(target: Union[str, Path]) -> bool:
    target_str = str(target)
    if target_str.startswith("sqlite://"):
        return True
    if ":memory:" in target_str:
        return True
    return not any(target_str.startswith(prefix) for prefix in ("postgresql://", "mysql://", "mariadb://", "oracle://"))


def init_db_performance(db_url_or_path: Union[str, Path], connection: Optional[sqlite3.Connection] = None) -> None:
    """Apply high performance PRAGMA settings for SQLite databases.

    The function is a no-op for non-SQLite URLs.  When ``connection`` is not
    provided a temporary connection is opened and closed immediately after the
    PRAGMAs are set.  The call is idempotent and the settings are only logged
    once per database path.
    """

    target = str(db_url_or_path)
    if not _is_sqlite_url(target):
        return

    if _PERF_APPLIED.get(target):
        return

    close_after = False
    conn = connection
    if conn is None:
        conn = sqlite3.connect(target, check_same_thread=False)
        close_after = True
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA temp_store=MEMORY;")
        cur.execute("PRAGMA mmap_size=268435456;")
        conn.commit()
        _PERF_APPLIED[target] = True
        logger.info("PRAGMA set: WAL,NORMAL,MEMORY,mmap=256MB")
    finally:
        if close_after and conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def get_last_performance_config() -> dict[str, Union[str, int]]:
    """Return the last applied PRAGMA configuration."""

    return dict(_PERF_CONFIG)


def get_db(path: str = "product_research_app/data.sqlite3", write: bool = False) -> sqlite3.Connection:
    """Return a cached SQLite connection.

    The connection is shared across the process to avoid re‑initializing the
    database on every request.  When ``path`` changes the previous connection is
    closed and a new one is opened lazily.  ``write`` is accepted for
    compatibility with existing call sites but currently unused.
    """

    global _DB, _DB_PATH

    target_path = path or _DB_PATH or "product_research_app/data.sqlite3"
    if _DB is None or _DB_PATH != target_path:
        with _DB_LOCK:
            if _DB is not None and _DB_PATH != target_path:
                try:
                    _DB.close()
                except Exception:
                    pass
                _DB = None
            if _DB is None:
                conn = sqlite3.connect(target_path, check_same_thread=False, isolation_level=None)
                conn.execute("PRAGMA foreign_keys=ON;")
                init_db_performance(target_path, connection=conn)
                conn.row_factory = sqlite3.Row
                _DB = conn
                _DB_PATH = target_path
    return _DB


def close_db():
    """Close the cached connection.

    Useful for tests that need to reset the database path between runs."""

    global _DB, _DB_PATH
    with _DB_LOCK:
        if _DB is not None:
            try:
                _DB.close()
            except Exception:
                pass
        _DB = None
        _DB_PATH = None




def _clean_desire_magnitude(value: Any) -> tuple[Any, bool]:
    """Return a sanitized desire_magnitude and whether the input carried data."""

    if value in (None, ""):
        return 0.0, False
    try:
        mag = float(value)
    except (TypeError, ValueError):
        text_val = str(value).strip()
        return text_val, bool(text_val)
    if mag > 1.0:
        mag = max(0.0, min(100.0, mag))
    else:
        mag = max(0.0, min(1.0, mag))
    return mag, True


def upsert_ai_columns(conn: sqlite3.Connection, rows: Iterable[Dict[str, Any]]) -> int:
    """Update AI columns for existing products only."""

    rows_list = list(rows or [])
    if not rows_list:
        return 0

    try:
        col_rows = conn.execute("PRAGMA table_info(products)").fetchall()
        available_cols = {str(row[1]) for row in col_rows}
    except Exception:
        available_cols = set()

    label_column: Optional[str] = None
    if "ai_desire_label" in available_cols:
        label_column = "ai_desire_label"
    elif "desire" in available_cols:
        label_column = "desire"

    update_columns: list[str] = []
    if label_column:
        update_columns.append(label_column)
    for name in ("desire_magnitude", "awareness_level", "competition_level"):
        if name in available_cols:
            update_columns.append(name)

    if not update_columns:
        logger.info("upsert_ai_columns skip=no_target_columns")
        return 0

    sql = "UPDATE products SET " + ", ".join(f"{col} = ?" for col in update_columns) + " WHERE id = ?"

    data: list[tuple[Any, ...]] = []
    for row in rows_list:
        if not isinstance(row, dict):
            logger.info("upsert_ai_columns skip=invalid_row row=%s", row)
            continue
        try:
            product_id = int(row["product_id"])
        except Exception:
            logger.info("upsert_ai_columns skip=invalid_id row=%s", row)
            continue

        values: list[Any] = []
        has_payload = False
        for col in update_columns:
            if col == label_column:
                source_keys = ["ai_desire_label", "desire"] if label_column == "ai_desire_label" else [label_column, "ai_desire_label"]
                label_val: Any = ""
                for key in source_keys:
                    if key in row and row.get(key) is not None:
                        label_val = row.get(key)
                        break
                label_clean = str(label_val or "").strip()
                if label_clean:
                    has_payload = True
                values.append(label_clean)
            elif col == "desire_magnitude":
                cleaned_mag, mag_present = _clean_desire_magnitude(row.get("desire_magnitude"))
                if mag_present:
                    has_payload = True
                values.append(cleaned_mag)
            else:
                raw_val = row.get(col)
                text_val = str(raw_val or "").strip()
                if text_val:
                    has_payload = True
                values.append(text_val)

        if not has_payload:
            logger.info("upsert_ai_columns skip=missing_values product_id=%s", product_id)
            continue

        values.append(product_id)
        data.append(tuple(values))

    if not data:
        return 0

    cur = conn.cursor()
    ok = 0
    try:
        cur.executemany(sql, data)
        ok = cur.rowcount if cur.rowcount not in (None, -1) else len(data)
        conn.commit()
    except Exception as exc:
        logger.exception("upsert_ai_columns failed: %s", exc)
        conn.rollback()
        ok = 0
    finally:
        cur.close()
    return ok
