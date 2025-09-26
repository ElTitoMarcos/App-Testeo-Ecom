import logging
import sqlite3
import threading
from pathlib import Path
from typing import Optional, Union


def _table_has_column(conn, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


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


def upsert_ai_columns(conn, rows):
    """
    rows: [{"product_id": int,
            "ai_desire_label": Optional[str],   # puede no venir
            "desire": Optional[str],            # texto corto/etiqueta fallback
            "desire_magnitude": Optional[float|int],
            "awareness_level": Optional[float|int],
            "competition_level": Optional[float|int]}]
    Actualiza ÚNICAMENTE columnas IA del producto existente. Tolerante si
    'ai_desire_label' no existe en el esquema: usará 'desire' en su lugar.
    """
    if not rows:
        return 0

    has_ai_label = _table_has_column(conn, "products", "ai_desire_label")

    # columnas numéricas siempre presentes en este fix
    set_cols = []
    if has_ai_label:
        set_cols.append("ai_desire_label")
    else:
        set_cols.append("desire")
    set_cols += ["desire_magnitude", "awareness_level", "competition_level"]

    set_sql = ", ".join([f"{c} = COALESCE(?, {c})" for c in set_cols])
    sql = f"UPDATE products SET {set_sql} WHERE id = ?"

    payload = []
    for r in rows:
        pid = int(r["product_id"])
        # label: preferimos 'ai_desire_label', luego 'desire'
        label = (r.get("ai_desire_label") or r.get("desire") or "").strip()
        mag = r.get("desire_magnitude")
        aware = r.get("awareness_level")
        comp = r.get("competition_level")
        vals = [label, mag, aware, comp]
        vals.append(pid)
        payload.append(vals)

    cur = conn.cursor()
    try:
        cur.executemany(sql, payload)
        conn.commit()
        updated = cur.rowcount if cur.rowcount is not None else 0
        logging.info("upsert_ai_columns: updated=%s has_ai_label=%s", updated, has_ai_label)
        return updated
    except Exception as e:
        logging.exception("upsert_ai_columns failed: %s", e)
        conn.rollback()
        return 0
    finally:
        cur.close()


def filter_missing_ai_columns(conn, ids):
    if not ids:
        return []
    has_ai_label = _table_has_column(conn, "products", "ai_desire_label")
    qmarks = ",".join(["?"] * len(ids))

    # Columna de texto a verificar: ai_desire_label (si existe) o desire
    txt_col = "ai_desire_label" if has_ai_label else "desire"

    sql = f"""
      SELECT id FROM products
       WHERE id IN ({qmarks})
         AND (
              ({txt_col} IS NULL OR TRIM({txt_col}) = '')
           OR desire_magnitude IS NULL
           OR awareness_level IS NULL
           OR competition_level IS NULL
         )
    """
    cur = conn.execute(sql, ids)
    out = [r[0] for r in cur.fetchall()]
    logging.info("filter_missing_ai_columns: need=%d using=%s", len(out), txt_col)
    return out
