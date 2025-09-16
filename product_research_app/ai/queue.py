from __future__ import annotations

import logging
from typing import Iterable, List

from product_research_app.db import get_db


logger = logging.getLogger(__name__)


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ai_task_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type TEXT NOT NULL,
    product_id INTEGER NOT NULL,
    enqueued_at REAL NOT NULL DEFAULT (strftime('%s','now')),
    UNIQUE(task_type, product_id)
);
"""

_CREATE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_ai_task_queue_task_id "
    "ON ai_task_queue(task_type, id);"
)

_DELETE_SQL = "DELETE FROM ai_task_queue WHERE id = ?;"

_SELECT_SQL = (
    "SELECT id, product_id FROM ai_task_queue "
    "WHERE task_type = ? ORDER BY id ASC LIMIT ?;"
)

_INSERT_SQL = "INSERT OR IGNORE INTO ai_task_queue (task_type, product_id) VALUES (?, ?);"


def _ensure_schema(conn) -> None:
    conn.execute(_CREATE_TABLE_SQL)
    conn.execute(_CREATE_INDEX_SQL)


def _coerce_ids(product_ids: Iterable[int]) -> List[int]:
    cleaned: List[int] = []
    for value in product_ids:
        try:
            num = int(value)
        except Exception:
            continue
        if num <= 0:
            continue
        cleaned.append(num)
    return cleaned


def enqueue_post_import(task_type: str, product_ids: Iterable[int]) -> int:
    """Persist a batch of post-import tasks for later AI processing."""

    if not task_type:
        return 0
    ids = _coerce_ids(product_ids)
    if not ids:
        return 0

    conn = get_db()
    _ensure_schema(conn)

    before = conn.total_changes
    params = [(task_type, pid) for pid in ids]
    with conn:
        conn.executemany(_INSERT_SQL, params)
    inserted = conn.total_changes - before
    if inserted:
        logger.debug(
            "ai_queue enqueue task_type=%s inserted=%s total_ids=%s",
            task_type,
            inserted,
            len(ids),
        )
    return max(inserted, 0)


def dequeue_batch(task_type: str, limit: int) -> List[int]:
    """Return and remove up to ``limit`` product IDs for ``task_type``."""

    try:
        limit_int = int(limit)
    except Exception:
        limit_int = 0
    if not task_type or limit_int <= 0:
        return []

    conn = get_db()
    _ensure_schema(conn)

    rows = conn.execute(_SELECT_SQL, (task_type, limit_int)).fetchall()
    if not rows:
        return []

    queue_ids: List[int] = []
    product_ids: List[int] = []
    for row in rows:
        try:
            queue_ids.append(int(row["id"]))
            product_ids.append(int(row["product_id"]))
        except Exception:
            queue_ids.append(int(row[0]))
            product_ids.append(int(row[1]))

    with conn:
        conn.executemany(_DELETE_SQL, [(qid,) for qid in queue_ids])

    logger.debug(
        "ai_queue dequeue task_type=%s count=%s", task_type, len(product_ids)
    )
    return product_ids
