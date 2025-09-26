import logging
import sqlite3
import threading
from datetime import datetime
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


def upsert_ai_columns(conn: sqlite3.Connection, rows: Iterable[Dict[str, Any]]) -> int:
    """Insert or update AI columns for provided product rows.

    Args:
        conn: SQLite connection.
        rows: Iterable of dictionaries with at least ``product_id`` and
            the AI-derived fields.

    Returns:
        Number of rows persisted.
    """

    rows = list(rows or [])
    if not rows:
        return 0

    try:
        col_rows = conn.execute("PRAGMA table_info(products)").fetchall()
        available_cols = {str(row[1]) for row in col_rows}
    except Exception:
        available_cols = {"id"}

    label_targets = []
    if "ai_desire_label" in available_cols:
        label_targets.append("ai_desire_label")
    if "desire" in available_cols:
        label_targets.append("desire")

    supported_fields = [
        ("desire_magnitude", "desire_magnitude"),
        ("awareness_level", "awareness_level"),
        ("competition_level", "competition_level"),
    ]

    now_iso = datetime.utcnow().isoformat()
    processed = 0
    cur = conn.cursor()
    began_tx = False
    try:
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
            began_tx = True
        for row in rows:
            if not isinstance(row, dict):
                logger.info("upsert_ai_columns skip=invalid_row row=%s", row)
                continue
            try:
                product_id = int(row.get("product_id"))
            except Exception:
                logger.info("upsert_ai_columns skip=invalid_id row=%s", row)
                continue

            payload: Dict[str, Any] = {}
            label_val = row.get("ai_desire_label")
            if label_val in (None, ""):
                label_val = row.get("desire")
            if label_val not in (None, "") and label_targets:
                for target in label_targets:
                    payload[target] = label_val

            for src_key, db_col in supported_fields:
                if db_col not in available_cols:
                    continue
                value = row.get(src_key)
                if value is not None:
                    payload[db_col] = value

            if not payload:
                logger.info(
                    "upsert_ai_columns skip=missing_values product_id=%s", product_id
                )
                continue

            if "ai_columns_completed_at" in available_cols:
                payload["ai_columns_completed_at"] = now_iso

            columns = ["id", *payload.keys()]
            placeholders = ",".join(["?"] * len(columns))
            assignments = ", ".join(f"{col}=excluded.{col}" for col in payload.keys())
            values = [product_id, *[payload[col] for col in payload.keys()]]
            cur.execute(
                f"INSERT INTO products ({', '.join(columns)}) VALUES ({placeholders}) "
                f"ON CONFLICT(id) DO UPDATE SET {assignments}",
                values,
            )
            processed += 1
            logger.info(
                "upsert_ai_columns saved product_id=%s fields=%s",
                product_id,
                sorted(payload.keys()),
            )

        if began_tx:
            conn.commit()
    except Exception:
        if began_tx and conn.in_transaction:
            conn.rollback()
        raise

    return processed
