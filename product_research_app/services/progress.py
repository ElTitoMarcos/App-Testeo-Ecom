from __future__ import annotations

"""Thread-safe progress tracking for AI fill jobs."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from time import time
from typing import Dict, Iterator, Optional

import contextvars
from contextvars import Token


@dataclass
class JobProgress:
    total_items: int
    triage_planned: int = 0
    triage_done: int = 0
    api_planned: int = 0
    api_done: int = 0
    post_done: bool = False
    started_at: float = field(default_factory=time)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def plan_triage(self, n: int) -> None:
        if n <= 0:
            return
        with self._lock:
            self.triage_planned += int(n)

    def inc_triage(self, n: int = 1) -> None:
        if n <= 0:
            return
        with self._lock:
            self.triage_done += int(n)

    def plan_api(self, n: int) -> None:
        if n <= 0:
            return
        with self._lock:
            self.api_planned += int(n)

    def inc_api(self, n: int = 1) -> None:
        if n <= 0:
            return
        with self._lock:
            self.api_done += int(n)

    def mark_post(self) -> None:
        with self._lock:
            self.post_done = True

    def snapshot(self) -> float:
        with self._lock:
            w_triage = 0.20
            w_api = 0.78
            w_post = 0.02

            if self.post_done:
                return 1.0

            triage_part = 1.0 if self.triage_planned == 0 else min(
                1.0, self.triage_done / max(self.triage_planned, 1)
            )
            api_part = 1.0 if self.api_planned == 0 else min(
                1.0, self.api_done / max(self.api_planned, 1)
            )

            pct = (w_triage * triage_part) + (w_api * api_part)
            pct = max(0.0, min(1.0, pct))
            return pct


_trackers: Dict[str, JobProgress] = {}
_trackers_lock = Lock()
_current_job_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "prapp_current_job_progress_id", default=None
)


def ensure(job_id: str, total_items: int) -> JobProgress:
    key = job_id.strip()
    if not key:
        raise ValueError("job_id must be a non-empty string")
    with _trackers_lock:
        tracker = _trackers.get(key)
        if tracker is None:
            tracker = JobProgress(total_items=total_items)
            _trackers[key] = tracker
        else:
            if total_items >= 0:
                tracker.total_items = total_items
        return tracker


def get(job_id: str) -> Optional[JobProgress]:
    key = job_id.strip()
    if not key:
        return None
    with _trackers_lock:
        return _trackers.get(key)


def snapshot(job_id: str) -> Optional[float]:
    tracker = get(job_id)
    if tracker is None:
        return None
    return tracker.snapshot()


@contextmanager
def track_job(job_id: str) -> Iterator[None]:
    token = _current_job_id.set(job_id)
    try:
        yield
    finally:
        _current_job_id.reset(token)


def notify_api_success(job_id: Optional[str] = None) -> None:
    key = job_id.strip() if isinstance(job_id, str) else None
    if not key:
        key = _current_job_id.get()
    if not key:
        return
    tracker = get(key)
    if tracker is None:
        return
    tracker.inc_api(1)


def bind(job_id: str) -> Token:
    return _current_job_id.set(job_id)


def reset(token: Optional[Token]) -> None:
    if token is None:
        return
    _current_job_id.reset(token)

