"""In-memory tracking for background AI post-processing tasks."""

from __future__ import annotations

from copy import deepcopy
import threading
from typing import Any, Dict, MutableMapping

DEFAULT_STATUS: Dict[str, Any] = {
    "state": "IDLE",
    "desire": {"requested": 0, "processed": 0, "failed": 0},
    "imputacion": {"requested": 0, "processed": 0, "failed": 0},
    "winner_score": {"requested": 0, "processed": 0, "failed": 0},
    "notes": [],
}

AI_STATUS: Dict[str, Dict[str, Any]] = {}
_STATUS_LOCK = threading.Lock()


def _blank_status() -> Dict[str, Any]:
    return deepcopy(DEFAULT_STATUS)


def init_status(task_id: str) -> Dict[str, Any]:
    """Initialise the status entry for a task, resetting any previous data."""

    tid = str(task_id)
    with _STATUS_LOCK:
        status = _blank_status()
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
        return deepcopy(status)


def get_status(task_id: str) -> Dict[str, Any] | None:
    """Return a copy of the status for the given task id if present."""

    tid = str(task_id)
    with _STATUS_LOCK:
        status = AI_STATUS.get(tid)
        if status is None:
            return None
        return deepcopy(status)
