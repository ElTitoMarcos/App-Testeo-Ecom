"""Global GPT orchestration helpers with rate limiting and metrics."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

from .. import settings

logger = logging.getLogger(__name__)


class _SimpleRateLimiter:
    """Minimal rate limiter using a monotonic clock."""

    def __init__(self, max_rps: float) -> None:
        self._interval = 1.0 / max_rps if max_rps > 0 else 0.0
        self._lock = threading.Lock()
        self._next_time = 0.0

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                wait = self._next_time - now
                if wait <= 0:
                    if self._interval > 0:
                        target = max(self._next_time, now) + self._interval
                    else:
                        target = max(self._next_time, now)
                    self._next_time = target
                    return
            if wait <= 0:
                if self._interval <= 0:
                    return
                wait = self._interval
            time.sleep(min(wait, 1.0))

    def defer(self, delay: float) -> None:
        delay = max(0.0, float(delay or 0.0))
        if delay <= 0:
            return
        with self._lock:
            now = time.monotonic()
            target = now + delay
            if target > self._next_time:
                self._next_time = target


class _GlobalLimiter:
    """Coordinate global GPT concurrency with retry back-off."""

    def __init__(self) -> None:
        self._default_limit = max(1, int(settings.GPT_MAX_PARALLEL or 1))
        self._current_limit = self._default_limit
        self._restore_at = 0.0
        self._inflight = 0
        self._cond = threading.Condition()
        self._rate = _SimpleRateLimiter(settings.GPT_MAX_RPS)

    def acquire(self) -> None:
        self._rate.acquire()
        with self._cond:
            while True:
                self._maybe_restore_locked()
                if self._inflight < self._current_limit:
                    self._inflight += 1
                    return
                timeout = None
                if self._current_limit < self._default_limit:
                    timeout = max(0.0, self._restore_at - time.monotonic())
                    if timeout <= 0:
                        self._maybe_restore_locked()
                        continue
                self._cond.wait(timeout)

    def release(self) -> None:
        with self._cond:
            self._inflight = max(0, self._inflight - 1)
            self._maybe_restore_locked()
            self._cond.notify_all()

    def handle_retry_after(self, delay: float) -> None:
        self._rate.defer(delay)
        with self._cond:
            if self._current_limit != 1:
                logger.info(
                    "gpt_orchestrator reducing parallelism due to rate limit: %s -> 1",
                    self._current_limit,
                )
            self._current_limit = 1
            self._restore_at = max(self._restore_at, time.monotonic() + 60.0)
            self._cond.notify_all()

    def _maybe_restore_locked(self) -> None:
        if self._current_limit < self._default_limit and time.monotonic() >= self._restore_at:
            self._current_limit = self._default_limit
            self._restore_at = 0.0
            self._cond.notify_all()


_GLOBAL_LIMITER = _GlobalLimiter()


@dataclass
class _ImportMetrics:
    calls_total: int = 0
    cache_saved: int = 0
    batch_count: int = 0
    batch_total: int = 0
    budget_warning_emitted: bool = False

    def add_call(self, batch_size: int) -> None:
        self.calls_total += 1
        self.batch_count += 1
        if batch_size > 0:
            self.batch_total += batch_size

    def add_cache_saved(self, count: int) -> None:
        if count > 0:
            self.cache_saved += count


_IMPORT_METRICS: Dict[str, _ImportMetrics] = {}
_METRICS_LOCK = threading.Lock()


def _normalize_import_id(import_id: Optional[object]) -> Optional[str]:
    if import_id is None:
        return None
    text = str(import_id).strip()
    return text or None


def start_import(import_id: Optional[object]) -> None:
    key = _normalize_import_id(import_id)
    if key is None:
        return
    with _METRICS_LOCK:
        _IMPORT_METRICS[key] = _ImportMetrics()


def record_cache_saved(import_id: Optional[object], count: int) -> None:
    key = _normalize_import_id(import_id)
    if key is None:
        return
    saved = max(0, int(count or 0))
    if saved <= 0:
        return
    with _METRICS_LOCK:
        metrics = _IMPORT_METRICS.setdefault(key, _ImportMetrics())
        metrics.add_cache_saved(saved)


def _record_call_metrics(key: str, batch_size: int) -> None:
    batch = max(0, int(batch_size or 0))
    with _METRICS_LOCK:
        metrics = _IMPORT_METRICS.setdefault(key, _ImportMetrics())
        metrics.add_call(batch)
        budget = settings.GPT_BUDGET_CALLS_PER_IMPORT
        if budget > 0 and metrics.calls_total > budget and not metrics.budget_warning_emitted:
            metrics.budget_warning_emitted = True
            logger.warning(
                "gpt_orchestrator import=%s exceeded GPT budget: used=%s budget=%s",
                key,
                metrics.calls_total,
                budget,
            )


def record_call(import_id: Optional[object], batch_size: int) -> None:
    key = _normalize_import_id(import_id)
    if key is None:
        return
    _record_call_metrics(key, batch_size)


def flush_import_metrics(import_id: Optional[object]) -> None:
    key = _normalize_import_id(import_id)
    if key is None:
        return
    with _METRICS_LOCK:
        metrics = _IMPORT_METRICS.pop(key, _ImportMetrics())
    avg = (metrics.batch_total / metrics.batch_count) if metrics.batch_count else 0.0
    logger.info(
        "gpt_orchestrator import=%s calls_total=%s calls_saved_by_cache=%s avg_batch_size=%.2f",
        key,
        metrics.calls_total,
        metrics.cache_saved,
        avg,
    )


@contextlib.contextmanager
def acquire_call(import_id: Optional[object], task_type: str, batch_size: int) -> Iterator[None]:
    """Acquire global rate/concurrency slots for a GPT call."""

    _GLOBAL_LIMITER.acquire()
    try:
        record_call(import_id, batch_size)
        yield
    finally:
        _GLOBAL_LIMITER.release()


def handle_retry_after(delay: float) -> None:
    """Notify the limiter about a Retry-After header."""

    _GLOBAL_LIMITER.handle_retry_after(delay)


def get_timeout() -> int:
    """Return the configured GPT timeout in seconds."""

    return max(1, int(settings.GPT_TIMEOUT or 20))
