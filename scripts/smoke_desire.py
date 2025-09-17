from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Iterable, List

from product_research_app.ai.ai_status import get_status
from product_research_app.ai.runner import BatchResult, run_desire_only

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "product_research_app" / "data.sqlite3"
DEFAULT_LIMIT = 100


def _parse_ids(raw: str) -> List[int]:
    values: List[int] = []
    for chunk in raw.replace(",", " ").split():
        try:
            num = int(chunk.strip())
        except Exception:
            continue
        if num > 0:
            values.append(num)
    return values


def _fetch_latest_job(conn: sqlite3.Connection) -> sqlite3.Row | None:
    cur = conn.execute(
        "SELECT id, created_at, rows_imported, ai_pending FROM import_jobs ORDER BY created_at DESC LIMIT 1"
    )
    return cur.fetchone()


def _fetch_job(conn: sqlite3.Connection, task_id: int) -> sqlite3.Row | None:
    cur = conn.execute(
        "SELECT id, created_at, rows_imported, ai_pending FROM import_jobs WHERE id=?",
        (task_id,),
    )
    return cur.fetchone()


def _determine_product_ids(
    conn: sqlite3.Connection,
    job_row: sqlite3.Row | None,
    args_ids: str | None,
    limit: int,
) -> List[int]:
    if args_ids:
        return _parse_ids(args_ids)

    if job_row is None:
        raise RuntimeError("No import job information available; provide --ids")

    pending_ids_raw = job_row["ai_pending"] if "ai_pending" in job_row.keys() else None
    if pending_ids_raw:
        try:
            pending = json.loads(pending_ids_raw)
        except Exception:
            pending = []
        else:
            if isinstance(pending, list):
                parsed = [int(pid) for pid in pending if isinstance(pid, (int, float, str))]
                parsed = [pid for pid in parsed if isinstance(pid, int) and pid > 0]
                if parsed:
                    return parsed[:limit]

    rows_imported = int(job_row["rows_imported"] or 0)
    created_at = job_row["created_at"]
    if rows_imported > 0:
        base_target = rows_imported
    else:
        base_target = DEFAULT_LIMIT
    if limit and limit > 0:
        target = min(limit, base_target) if rows_imported > 0 else limit
    else:
        target = base_target
    target = max(1, target)

    ids: List[int] = []
    if created_at:
        cur = conn.execute(
            "SELECT id FROM products WHERE import_date >= ? ORDER BY id ASC LIMIT ?",
            (created_at, target),
        )
        ids = [int(row[0]) for row in cur.fetchall()]
    if not ids:
        cur = conn.execute(
            "SELECT id FROM products ORDER BY id DESC LIMIT ?",
            (target,),
        )
        ids = [int(row[0]) for row in cur.fetchall()][::-1]
    return ids


def _print_samples(conn: sqlite3.Connection, ids: Iterable[int]) -> None:
    id_list = list(dict.fromkeys(int(pid) for pid in ids if pid is not None))
    if not id_list:
        print("No product ids to sample.")
        return
    placeholders = ",".join("?" for _ in id_list)
    cur = conn.execute(
        f"SELECT COUNT(*) FROM products WHERE id IN ({placeholders}) "
        "AND ai_desire IS NOT NULL AND desire_magnitude IS NOT NULL",
        tuple(id_list),
    )
    total = cur.fetchone()[0]
    print(f"Products with desire data: {total}/{len(id_list)}")
    cur = conn.execute(
        f"SELECT id, ai_desire, ai_desire_label, desire_magnitude "
        f"FROM products WHERE id IN ({placeholders}) AND ai_desire IS NOT NULL "
        "ORDER BY id ASC LIMIT 5",
        tuple(id_list),
    )
    rows = cur.fetchall()
    if not rows:
        print("No samples available.")
        return
    for row in rows:
        desire_text = str(row["ai_desire"] or "").replace("\n", " ")
        snippet = desire_text[:120]
        print(
            f"- #{row['id']} label={row['ai_desire_label']} magnitude={row['desire_magnitude']} text={snippet}"
        )


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Smoke test for the desire pipeline")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="SQLite database path")
    parser.add_argument("--task-id", help="Import task id to reuse", default=None)
    parser.add_argument("--ids", help="Comma/space separated product ids", default=None)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Maximum number of ids to inspect")
    args = parser.parse_args(argv)

    db_path = args.db_path
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        job_row: sqlite3.Row | None
        task_id_value = args.task_id
        job_row = None
        if task_id_value:
            try:
                job_id = int(task_id_value)
            except ValueError:
                job_id = None
            if job_id is not None:
                job_row = _fetch_job(conn, job_id)
            task_id = str(task_id_value)
        else:
            job_row = _fetch_latest_job(conn)
            if not job_row:
                raise RuntimeError("No import jobs found; provide --ids explicitly")
            task_id = str(job_row["id"])

        product_ids = _determine_product_ids(conn, job_row, args.ids, args.limit)
        product_ids = [pid for pid in product_ids if isinstance(pid, int) and pid > 0]
        product_ids = list(dict.fromkeys(product_ids))
        if not product_ids:
            raise RuntimeError("No product ids resolved; use --ids to provide them")
    finally:
        conn.close()

    print(f"Running desire pipeline for task_id={task_id} ids={product_ids[:10]}...", flush=True)
    result: BatchResult = run_desire_only(task_id, product_ids)
    print(
        f"Runner result: processed={result.processed} failed={result.failed} calls={result.calls}",
        flush=True,
    )

    status_snapshot = get_status(task_id)
    if status_snapshot:
        desire_status = status_snapshot.get("desire") or {}
        print(f"Status desire snapshot: {desire_status}")
        if status_snapshot.get("notes"):
            print(f"Status notes: {status_snapshot['notes']}")
        if status_snapshot.get("last_error"):
            print(f"Last error: {status_snapshot['last_error']}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        _print_samples(conn, product_ids)
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
