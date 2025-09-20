"""In-memory manager to coordinate import cancellation state."""
from __future__ import annotations

from threading import RLock
from typing import Any, Dict, Optional


class CancelledImport(Exception):
    """Raised when an import job is cancelled by the user."""


class ImportManager:
    def __init__(self) -> None:
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = RLock()

    def start(self, task_id: str, *, status: str = "queued", message: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            state = self._tasks.setdefault(task_id, {})
            state["cancelled"] = False
            state["status"] = status
            if message is not None:
                state["message"] = message
            state.setdefault("progress", 0)
            return dict(state)

    def update(
        self,
        task_id: str,
        *,
        status: Optional[str] = None,
        message: Optional[str] = None,
        progress: Optional[int] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        with self._lock:
            state = self._tasks.setdefault(task_id, {"cancelled": False})
            if status is not None:
                state["status"] = status
                if status == "cancelled":
                    state["cancelled"] = True
            if message is not None:
                state["message"] = message
            if progress is not None:
                state["progress"] = progress
            for key in ("done", "total", "error", "stage"):
                if key in extra and extra[key] is not None:
                    state[key] = extra[key]
            return dict(state)

    def set_status(self, task_id: str, status: str, message: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
        return self.update(task_id, status=status, message=message, **extra)

    def cancel(self, task_id: str, message: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            state = self._tasks.setdefault(task_id, {})
            state["cancelled"] = True
            state["status"] = "cancelled"
            if message is not None:
                state["message"] = message
            else:
                state.setdefault("message", "Cancelado")
            return dict(state)

    def is_cancelled(self, task_id: str) -> bool:
        with self._lock:
            return bool(self._tasks.get(task_id, {}).get("cancelled"))

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            state = self._tasks.get(task_id)
            if not state:
                return None
            return dict(state)

    def clear(self, task_id: str) -> None:
        with self._lock:
            self._tasks.pop(task_id, None)


manager = ImportManager()

__all__ = ["CancelledImport", "ImportManager", "manager"]
