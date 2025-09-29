from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from . import config, database
from .services import ai_columns


@dataclass
class Job:
    id: str
    product_ids: List[int]
    total: int
    processed: int = 0
    done: bool = False
    cancelled: bool = False
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    est_ms: int = 0
    parallelism: int = 1
    cancel_event: threading.Event = field(default_factory=threading.Event)
    counts: Dict[str, int] = field(default_factory=dict)
    message: Optional[str] = None
    finished_at: Optional[float] = None
    db_path: Path | None = None

    def remaining(self) -> int:
        return max(0, self.total - self.processed)


JOBS: Dict[str, Job] = {}
LOCK = threading.Lock()
AVG_MS_PER_ITEM = 2500
MIN_ESTIMATE_MS = 4000
EMA_ALPHA = 0.3


def _sanitize_ids(raw_ids: Iterable[int | str]) -> List[int]:
    seen: set[int] = set()
    clean: List[int] = []
    for value in raw_ids:
        try:
            num = int(value)
        except Exception:
            continue
        if num in seen:
            continue
        seen.add(num)
        clean.append(num)
    return clean


def _fetch_all_product_ids(db_path: Path) -> List[int]:
    conn = database.get_connection(db_path)
    try:
        database.initialize_database(conn)
        cur = conn.cursor()
        cur.execute("SELECT id FROM products ORDER BY id ASC")
        rows = cur.fetchall()
        return [int(row[0]) for row in rows]
    finally:
        conn.close()


def estimate_ms(total: int, parallelism: int) -> int:
    if total <= 0:
        return MIN_ESTIMATE_MS
    effective_parallelism = max(1, parallelism)
    return int(max(MIN_ESTIMATE_MS, (total * AVG_MS_PER_ITEM) / effective_parallelism))


def _resolve_parallelism() -> int:
    runtime_cfg = config.get_ai_runtime_config()
    try:
        return max(1, int(runtime_cfg.get("parallelism", 4) or 4))
    except Exception:
        return 4


def start_job(
    product_ids: Optional[Sequence[int | str]] = None,
    *,
    db_path: Path | str | None = None,
) -> Job:
    resolved_db_path = Path(db_path or ai_columns.DB_PATH)
    if product_ids is None:
        ids = _fetch_all_product_ids(resolved_db_path)
    else:
        ids = _sanitize_ids(product_ids)
    job = Job(
        id=str(uuid.uuid4()),
        product_ids=ids,
        total=len(ids),
        parallelism=_resolve_parallelism(),
        db_path=resolved_db_path,
    )
    job.est_ms = estimate_ms(job.total, job.parallelism)
    with LOCK:
        JOBS[job.id] = job
    worker = threading.Thread(target=_worker, args=(job,), daemon=True)
    worker.start()
    return job


def _update_average(processed: int, elapsed_ms: float) -> None:
    global AVG_MS_PER_ITEM
    if processed <= 0 or elapsed_ms <= 0:
        return
    per_item = elapsed_ms / processed
    AVG_MS_PER_ITEM = int((1 - EMA_ALPHA) * AVG_MS_PER_ITEM + EMA_ALPHA * per_item)


def _status_callback(job: Job):
    def _cb(**payload: Dict[str, object]) -> None:
        counts = payload.get("counts")
        if counts is None:
            counts = payload.get("ai_counts")
        done_val = payload.get("done")
        processed = 0
        if isinstance(done_val, int):
            processed = done_val
        elif counts and isinstance(counts, dict):
            ok = int((counts or {}).get("ok", 0) or 0)
            cached = int((counts or {}).get("cached", 0) or 0)
            processed = ok + cached
        msg = payload.get("message")
        with LOCK:
            if isinstance(counts, dict):
                job.counts = {k: v for k, v in counts.items() if isinstance(v, (int, float))}
            job.processed = min(job.total, max(job.processed, int(processed)))
            if isinstance(msg, str):
                job.message = msg
    return _cb


def _worker(job: Job) -> None:
    start_ts = time.time()
    status_cb = _status_callback(job)
    previous_path = ai_columns.DB_PATH
    if job.db_path and job.db_path != previous_path:
        ai_columns.DB_PATH = job.db_path
    try:
        if not job.product_ids:
            with LOCK:
                job.done = True
                job.finished_at = time.time()
            return
        try:
            result = ai_columns.run_ai_fill_job(
                0,
                job.product_ids,
                parallelism=job.parallelism,
                status_cb=status_cb,
                cancel_event=job.cancel_event,
            )
            cancelled = bool(result.get("cancelled"))
            error = result.get("error")
            counts = result.get("counts") or {}
            processed = int(counts.get("ok", 0) or 0) + int(counts.get("cached", 0) or 0)
            with LOCK:
                job.processed = min(job.total, max(job.processed, processed))
                job.counts = {k: v for k, v in counts.items() if isinstance(v, (int, float))}
                job.done = not cancelled and error in (None, "")
                job.cancelled = cancelled or job.cancel_event.is_set()
                job.error = error if error and not job.cancelled else None
        except Exception as exc:  # pragma: no cover - defensive
            with LOCK:
                job.error = str(exc)
                job.done = False
        finally:
            with LOCK:
                job.finished_at = time.time()
    finally:
        if job.db_path is not None:
            ai_columns.DB_PATH = previous_path
        elapsed_ms = (time.time() - start_ts) * 1000.0
        with LOCK:
            processed = job.processed
        _update_average(processed, elapsed_ms)


def get_job(job_id: str) -> Optional[Job]:
    with LOCK:
        return JOBS.get(job_id)


def cancel_job(job_id: str) -> bool:
    job = get_job(job_id)
    if not job:
        return False
    job.cancel_event.set()
    with LOCK:
        job.cancelled = True
    return True


def job_to_dict(job: Job) -> Dict[str, object]:
    with LOCK:
        elapsed = (time.time() - job.started_at) * 1000.0
        counts = dict(job.counts)
        processed = job.processed
        total = max(0, job.total)
        pct = 0.0 if total == 0 else round((processed / total) * 100, 2)
        if job.done:
            eta = 0
        elif job.cancelled and job.finished_at:
            eta = 0
        else:
            remaining = max(0, total - processed)
            effective_parallelism = max(1, job.parallelism)
            eta = int(max(0, remaining * AVG_MS_PER_ITEM / effective_parallelism))
        return {
            "job_id": job.id,
            "done": job.done,
            "cancelled": job.cancelled,
            "processed": processed,
            "total": total,
            "pct": pct,
            "eta_ms_left": eta,
            "error": job.error,
            "message": job.message,
            "counts": counts,
            "estimated_ms": job.est_ms,
            "elapsed_ms": int(max(0, elapsed)),
            "parallelism": job.parallelism,
        }


def serialize_job(job: Job) -> Dict[str, object]:
    data = job_to_dict(job)
    data.update({
        "parallelism": job.parallelism,
        "started_at": job.started_at,
    })
    return data
