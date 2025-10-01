"""Async rate limiter shared across OpenAI calls."""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Deque, Optional, Tuple

from product_research_app.config import (
    OPENAI_HEADROOM,
    OPENAI_MAX_CONCURRENCY,
    OPENAI_RPM,
    OPENAI_TPM,
)

logger = logging.getLogger(__name__)


def _apply_headroom(value: int, headroom: float) -> int:
    return max(1, int(value * headroom))


def _parse_retry_after_header(headers: Optional[dict]) -> Optional[float]:
    if not headers:
        return None
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        try:
            parsed = parsedate_to_datetime(str(raw))
        except Exception:
            return None
        if not parsed:
            return None
        now = datetime.utcnow()
        if parsed.tzinfo is not None:
            now = now.replace(tzinfo=parsed.tzinfo)
        return max(0.0, (parsed - now).total_seconds())


_MESSAGE_RETRY = re.compile(
    r"(try again in|in)\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|milliseconds|s|sec|seconds)?",
    re.I,
)


def _parse_retry_from_message(message: str) -> Optional[float]:
    if not message:
        return None
    match = _MESSAGE_RETRY.search(message)
    if not match:
        return None
    value = match.group(2)
    unit = (match.group(3) or "s").lower()
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if unit.startswith("ms"):
        numeric /= 1000.0
    return max(0.0, numeric)


def decorrelated_jitter_sleep(prev: float, cap: float) -> float:
    base = 0.3
    prev = max(0.0, float(prev or 0.0))
    cap = max(base, float(cap or base))
    span = prev * 3 if prev > 0 else 1.0
    sleep_for = min(cap, random.uniform(base, max(base, span)))
    time.sleep(sleep_for)
    return sleep_for


async def async_decorrelated_jitter_sleep(prev: float, cap: float) -> float:
    base = 0.3
    prev = max(0.0, float(prev or 0.0))
    cap = max(base, float(cap or base))
    span = prev * 3 if prev > 0 else 1.0
    sleep_for = min(cap, random.uniform(base, max(base, span)))
    await asyncio.sleep(sleep_for)
    return sleep_for


class GlobalRateLimiter:
    """Simple sliding-window limiter supporting TPM + RPM budgets."""

    def __init__(
        self,
        tpm: int,
        rpm: int,
        *,
        headroom: float = 0.85,
        max_concurrency: int = 4,
    ) -> None:
        self.tpm_limit = _apply_headroom(tpm, headroom)
        self.rpm_limit = _apply_headroom(rpm, headroom)
        self._token_events: Deque[Tuple[float, int]] = deque()
        self._request_events: Deque[Tuple[float, int]] = deque()
        self._token_total = 0
        self._request_total = 0
        self._lock = asyncio.Lock()
        self._sem = asyncio.Semaphore(max(1, int(max_concurrency)))
        self._backoff_prev = 0.0

    def _purge(self, now: float) -> None:
        window_start = now - 60.0
        while self._token_events and self._token_events[0][0] <= window_start:
            _, amount = self._token_events.popleft()
            self._token_total = max(0, self._token_total - amount)
        while self._request_events and self._request_events[0][0] <= window_start:
            _, amount = self._request_events.popleft()
            self._request_total = max(0, self._request_total - amount)

    async def _acquire(self, tokens: int, requests: int) -> None:
        tokens = max(0, int(tokens))
        requests = max(1, int(requests))
        async with self._lock:
            while True:
                now = time.monotonic()
                self._purge(now)
                token_ok = (self._token_total + tokens) <= self.tpm_limit
                request_ok = (self._request_total + requests) <= self.rpm_limit
                if token_ok and request_ok:
                    if tokens:
                        self._token_events.append((now, tokens))
                        self._token_total += tokens
                    if requests:
                        self._request_events.append((now, requests))
                        self._request_total += requests
                    return
                waits: list[float] = []
                if not token_ok and self._token_events:
                    earliest = self._token_events[0][0]
                    waits.append(max(0.05, (earliest + 60.0) - now))
                if not request_ok and self._request_events:
                    earliest = self._request_events[0][0]
                    waits.append(max(0.05, (earliest + 60.0) - now))
                sleep_for = min(waits) if waits else 0.05
                await asyncio.sleep(min(sleep_for, 1.0))

    async def _backoff(self, exc: Exception) -> None:
        status = getattr(exc, "status_code", None)
        if status is None:
            response = getattr(exc, "response", None)
            if response is not None:
                status = getattr(response, "status_code", None)
        message = str(exc) if exc else ""
        is_rate_limited = False
        try:
            if status is not None and int(status) == 429:
                is_rate_limited = True
        except Exception:
            pass
        lowered = message.lower()
        if not is_rate_limited and any(marker in lowered for marker in ("429", "rate limit", "too many requests")):
            is_rate_limited = True
        if not is_rate_limited:
            self._backoff_prev = 0.0
            return

        retry_after = getattr(exc, "retry_after", None)
        if retry_after is None:
            headers = getattr(getattr(exc, "response", None), "headers", None)
            retry_after = _parse_retry_after_header(headers)
        if retry_after is None:
            retry_after = _parse_retry_from_message(message)

        if retry_after is not None:
            delay = max(0.0, float(retry_after))
        else:
            prev = self._backoff_prev if self._backoff_prev > 0 else 0.5
            delay = min(60.0, prev * 2.0)
            delay *= random.uniform(0.5, 1.5)
            delay = max(0.1, delay)
        self._backoff_prev = delay
        logger.warning("rate_limiter.backoff delay=%.2fs", delay)
        await asyncio.sleep(delay)

    def _reset_backoff(self) -> None:
        self._backoff_prev = 0.0

    @asynccontextmanager
    async def reserve(self, *, tokens: int, requests: int = 1):
        async with self._sem:
            await self._acquire(tokens, requests)
            error: Optional[Exception] = None
            try:
                yield
            except Exception as exc:  # pragma: no cover - propagated to caller
                error = exc
                await self._backoff(exc)
                raise
            finally:
                if error is None:
                    self._reset_backoff()


limiter = GlobalRateLimiter(
    OPENAI_TPM,
    OPENAI_RPM,
    headroom=OPENAI_HEADROOM,
    max_concurrency=OPENAI_MAX_CONCURRENCY,
)

__all__ = ["limiter", "GlobalRateLimiter", "async_decorrelated_jitter_sleep", "decorrelated_jitter_sleep"]
