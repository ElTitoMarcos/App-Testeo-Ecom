from __future__ import annotations

"""In-memory job tracking and progress publication for AI fills."""

import itertools
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class AiJobState:
    job_id: int = 0
    status: str = "idle"  # queued | running | done | error | idle
    total: int = 0
    processed: int = 0
    avg_sec_per_item: float = 0.0
    started_at: float = 0.0
    duration_accum: float = 0.0


_lock = threading.Lock()
_state = AiJobState()
_id_counter = itertools.count(1)
_subscribers: set[queue.Queue[dict[str, Any]]] = set()
_last_progress: Optional[dict[str, Any]] = None


def _eta_ms_locked() -> int:
    remaining = max(0, _state.total - _state.processed)
    if remaining <= 0:
        return 0
    avg = max(0.0, float(_state.avg_sec_per_item))
    eta = max(0.0, (avg - 0.2)) * remaining * 1000.0
    return int(round(eta))


def _progress_payload_locked() -> dict[str, Any]:
    progress = 0.0
    if _state.total > 0:
        progress = min(1.0, _state.processed / float(_state.total))
    elif _state.processed > 0:
        progress = 1.0
    return {
        "job_id": _state.job_id,
        "status": _state.status,
        "processed": _state.processed,
        "total": _state.total,
        "eta_ms": _eta_ms_locked(),
        "progress": progress,
    }


def _publish_locked(event_type: str, data: Dict[str, Any]) -> None:
    global _last_progress
    payload = {"type": event_type, "data": data}
    dead: list[queue.Queue[dict[str, Any]]] = []
    for q in list(_subscribers):
        try:
            q.put_nowait(payload)
        except queue.Full:
            dead.append(q)
    for q in dead:
        _subscribers.discard(q)
    if event_type == "progress":
        _last_progress = dict(data)
    elif event_type in {"done", "error"}:
        progress_snapshot = _progress_payload_locked()
        progress_snapshot["status"] = data.get("status", _state.status)
        _last_progress = progress_snapshot


def subscribe_progress_queue() -> queue.Queue[dict[str, Any]]:
    q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=200)
    with _lock:
        _subscribers.add(q)
        if _last_progress:
            try:
                q.put_nowait({"type": "progress", "data": dict(_last_progress)})
            except queue.Full:
                pass
    return q


def unsubscribe_progress_queue(q: queue.Queue[dict[str, Any]]) -> None:
    with _lock:
        _subscribers.discard(q)


def start_job(total: int) -> int:
    with _lock:
        job_id = next(_id_counter)
        _state.job_id = job_id
        _state.status = "running"
        _state.total = max(0, int(total or 0))
        _state.processed = 0
        _state.avg_sec_per_item = 0.0
        _state.duration_accum = 0.0
        _state.started_at = time.time()
        payload = _progress_payload_locked()
        _publish_locked("progress", payload)
        return job_id


def record_batch(batch_size: int, duration_sec: float) -> dict[str, Any]:
    with _lock:
        processed_inc = max(0, int(batch_size or 0))
        _state.processed = min(_state.total, _state.processed + processed_inc)
        _state.duration_accum += max(0.0, float(duration_sec or 0.0))
        done = max(1, _state.processed)
        _state.avg_sec_per_item = (
            _state.duration_accum / done if done > 0 else 0.0
        )
        _state.status = "running"
        payload = _progress_payload_locked()
        _publish_locked("progress", payload)
        return payload


def finish_job(status: str = "done") -> int:
    event_type = "done" if status == "done" else "error"
    with _lock:
        if status == "done":
            _state.processed = max(_state.total, _state.processed)
        _state.status = status
        payload = {
            "processed": _state.processed,
            "total": _state.total,
            "status": status,
        }
        _publish_locked(event_type, payload)
        return _state.job_id


def mark_error() -> int:
    return finish_job(status="error")


def get_progress_payload() -> dict[str, Any]:
    with _lock:
        if _state.job_id == 0:
            return {
                "job_id": 0,
                "status": "idle",
                "processed": 0,
                "total": 0,
                "eta_ms": 0,
                "progress": 0.0,
            }
        return _progress_payload_locked()
