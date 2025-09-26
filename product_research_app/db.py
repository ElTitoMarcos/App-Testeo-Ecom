import logging
import sqlite3
import threading
from pathlib import Path
from typing import Optional, Union


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


def upsert_ai_columns(conn: sqlite3.Connection, rows: list[dict[str, object]]) -> int:
    """Update AI-related columns for existing products.

    Args:
        conn: Active SQLite connection.
        rows: Sequence of dicts describing the desired updates.

    Returns:
        Number of updated rows.
    """

    if not rows:
        return 0

    payload = []
    for r in rows:
        pid = int(r.get("id"))
        desire = r.get("desire")
        ai_label = r.get("ai_desire_label") or r.get("desire")
        mag = r.get("desire_magnitude")
        aware = r.get("awareness_level")
        comp = r.get("competition_level")
        payload.append((desire, ai_label, mag, aware, comp, pid))

    sql = """
    UPDATE products
       SET
         desire              = COALESCE(?, desire),
         ai_desire_label     = COALESCE(?, ai_desire_label),
         desire_magnitude    = COALESCE(?, desire_magnitude),
         awareness_level     = COALESCE(?, awareness_level),
         competition_level   = COALESCE(?, competition_level)
     WHERE id = ?
    """
    cur = conn.cursor()
    cur.executemany(sql, payload)
    conn.commit()
    rowcount = cur.rowcount
    cur.close()
    return rowcount


def filter_missing_ai_columns(conn: sqlite3.Connection, ids: list[int]) -> list[int]:
    """Return IDs from *ids* that still lack AI column values."""

    if not ids:
        return []

    placeholders = ",".join(["?"] * len(ids))
    sql = f"""
      SELECT id FROM products
       WHERE id IN ({placeholders})
         AND (ai_desire_label IS NULL
          OR desire_magnitude IS NULL
          OR awareness_level IS NULL
          OR competition_level IS NULL)
    """
    cur = conn.execute(sql, ids)
    rows = cur.fetchall()
    return [int(row[0]) for row in rows]
