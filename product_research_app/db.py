import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Optional, Union


logger = logging.getLogger(__name__)

_DB: Optional[sqlite3.Connection] = None
_DB_PATH: Optional[str] = None
_DB_LOCK = threading.Lock()
_TLS = threading.local()
_PERF_APPLIED: dict[str, bool] = {}
_PERF_CONFIG: dict[str, Union[str, int]] = {
    "journal_mode": "WAL",
    "synchronous": "NORMAL",
    "temp_store": "MEMORY",
    "mmap_size": 268_435_456,
    "busy_timeout": 5_000,
    "foreign_keys": "ON",
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


def _resolve_db_path() -> str:
    env_path = os.environ.get("PRAPP_DB_PATH")
    if env_path:
        return str(Path(env_path).expanduser().resolve())
    default_path = Path(__file__).resolve().parent / "data.sqlite3"
    return str(default_path)


def _make_conn() -> sqlite3.Connection:
    path = _resolve_db_path()
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=5000;")
    init_db_performance(path, connection=conn)
    global _DB_PATH
    _DB_PATH = path
    return conn


def get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection."""

    conn: Optional[sqlite3.Connection] = getattr(_TLS, "conn", None)
    try:
        cursor = conn.cursor() if conn is not None else None
    except sqlite3.ProgrammingError:
        cursor = None
    if conn is None or cursor is None:
        conn = _make_conn()
        _TLS.conn = conn
    elif cursor is not None:
        cursor.close()
    return conn


def get_db(path: str = "product_research_app/data.sqlite3", write: bool = False) -> sqlite3.Connection:
    """Backwards compatibility wrapper for existing call sites."""

    current_path = _resolve_db_path()
    if path and path != current_path:
        new_path = str(Path(path).expanduser().resolve())
        os.environ["PRAPP_DB_PATH"] = new_path
        close_db()
    return get_conn()


def close_db():
    """Close the cached connection.

    Useful for tests that need to reset the database path between runs."""

    conn = getattr(_TLS, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
    if hasattr(_TLS, "conn"):
        delattr(_TLS, "conn")

    global _DB, _DB_PATH
    with _DB_LOCK:
        if _DB is not None:
            try:
                _DB.close()
            except Exception:
                pass
        _DB = None
        _DB_PATH = None
