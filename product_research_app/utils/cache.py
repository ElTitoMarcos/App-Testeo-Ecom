from __future__ import annotations

import json
import sqlite3
import hashlib
import time
import os
from typing import Optional, Any

try:
    from product_research_app import db as _db  # type: ignore
    _get_conn = getattr(_db, "get_conn", None)
except Exception:
    _get_conn = None

from .paths import get_database_path


def _conn() -> sqlite3.Connection:
    if _get_conn:
        return _get_conn()
    path = os.getenv("PRAPP_DB_PATH") or str(get_database_path())
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_cache():
    c = _conn()
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS ai_cache (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL,
      created_at INTEGER NOT NULL
    )
    """
    )
    c.commit()


def make_key(payload: Any, version: str = "v1") -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256((version + "::" + blob).encode("utf-8")).hexdigest()
    return h


def get(key: str) -> Optional[dict]:
    c = _conn()
    r = c.execute("SELECT value FROM ai_cache WHERE key=?", (key,)).fetchone()
    if not r:
        return None
    try:
        return json.loads(r["value"])
    except Exception:
        return None


def set_(key: str, value: dict) -> None:
    c = _conn()
    c.execute(
        "INSERT OR REPLACE INTO ai_cache(key, value, created_at) VALUES (?, ?, ?)",
        (key, json.dumps(value, ensure_ascii=False), int(time.time())),
    )
    c.commit()
