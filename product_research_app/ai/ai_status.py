"""In-memory tracking for background AI post-processing tasks."""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from typing import Any, Dict, MutableMapping, Optional

DEFAULT_STATUS: Dict[str, Any] = {
    "state": "IDLE",
    "desire": {"requested": 0, "processed": 0, "failed": 0},
    "imputacion": {"requested": 0, "processed": 0, "failed": 0},
    "winner_score": {"requested": 0, "processed": 0, "failed": 0},
    "notes": [],
    "poll_interval_ms": 2500,
    "last_error": None,
    "trace": None,
    "started_at": None,
    "finished_at": None,
}

AI_STATUS: Dict[str, Dict[str, Any]] = {}
_STATUS_LOCK = threading.Lock()


def _blank_status() -> Dict[str, Any]:
    status = deepcopy(DEFAULT_STATUS)
    status["updated_at"] = time.time()
    return status


def init_status(task_id: str) -> Dict[str, Any]:
    """Initialise the status entry for a task, resetting any previous data."""

    tid = str(task_id)
    with _STATUS_LOCK:
        status = _blank_status()
        status["task_id"] = tid
        status["started_at"] = time.time()
        AI_STATUS[tid] = status
        return deepcopy(status)


def _merge_dict(target: MutableMapping[str, Any], updates: MutableMapping[str, Any]) -> None:
    for key, value in updates.items():
        if key == "notes":
            notes = target.setdefault("notes", [])
            if isinstance(value, (list, tuple, set)):
                for note in value:
                    if note is None:
                        continue
                    notes.append(str(note))
            elif value is not None:
                notes.append(str(value))
            continue
        existing = target.get(key)
        if isinstance(existing, MutableMapping) and isinstance(value, MutableMapping):
            _merge_dict(existing, value)
        else:
            target[key] = value


def update_status(task_id: str, **partial: Any) -> Dict[str, Any]:
    """Update an existing task status with partial data."""

    tid = str(task_id)
    with _STATUS_LOCK:
        status = AI_STATUS.setdefault(tid, _blank_status())
        if partial:
            _merge_dict(status, partial)
        status["updated_at"] = time.time()
        if status.get("state") == "DONE" and status.get("finished_at") is None:
            status["finished_at"] = status.get("updated_at")
        return deepcopy(status)


def get_status(task_id: str) -> Dict[str, Any] | None:
    """Return a copy of the status for the given task id if present."""

    tid = str(task_id)
    with _STATUS_LOCK:
        status = AI_STATUS.get(tid)
        if status is None:
            return None
        return deepcopy(status)


def set_error(task_id: str, message: str, trace_tail: Optional[str] = None) -> Dict[str, Any]:
    """Register an error for the task keeping the latest trace snippet."""

    payload: Dict[str, Any] = {
        "state": "ERROR",
        "last_error": message,
        "trace": trace_tail,
    }
    notes = []
    if message:
        notes.append(message)
    if trace_tail:
        notes.append("trace_available")
    if notes:
        payload["notes"] = notes
    return update_status(task_id, **payload)
