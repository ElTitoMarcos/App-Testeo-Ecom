"""Adaptive rate limiting helpers for GPT usage.

This module tracks token consumption and recent request cadence to offer
throughput hints and apply soft throttling before issuing requests.  It keeps
an exponential moving average (EMA) of prompt and completion tokens so callers
can gauge the expected footprint of the next call.  A cooperative
``TokenRateLimiter`` instance exposes helpers to reserve capacity, record
actual usage and derive concurrency suggestions for schedulers.

Usage summary
-------------

* Await :meth:`TokenRateLimiter.acquire` before performing an HTTP request.
  The method blocks (with sleeps) until sending another request would stay
  within the configured RPM/TPM soft limits.  It returns a release callback
  that **must** be invoked once the request finishes (successfully or not).
* After obtaining a successful response, call
  :meth:`TokenRateLimiter.record_tokens` with the real prompt and completion
  usage reported by the API.  This updates EMAs and the rolling minute window
  used by ``acquire``.
* Use :meth:`TokenRateLimiter.suggest_concurrency` to dynamically adjust the
  worker pool.  This function relies on the EMAs and the detected limits (when
  available from response headers) to stay comfortably below the hard API
  ceilings.
* When handling HTTP 429 responses, feed the attempt counter into
  :meth:`TokenRateLimiter.backoff_on_429` to obtain an exponential backoff with
  jitter that honours ``Retry-After`` hints.

The module instantiates a shared ``rate_limiter`` that reads configuration from
environment variables (with sensible defaults) so every caller within the
process cooperates.
"""

from __future__ import annotations

import asyncio
import math
import os
import random
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, Mapping, Optional

WINDOW_SECONDS = 60.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, default)))
    except Exception:
        return default


class ExponentialMovingAverage:
    """Simple EMA helper used to track prompt/completion usage."""

    def __init__(self, alpha: float, initial: Optional[float] = None) -> None:
        self.alpha = max(0.0, min(1.0, alpha)) or 0.3
        self.value: float = float(initial) if initial is not None else 0.0
        self.initialized = initial is not None
        self.samples = 0

    def update(self, sample: float) -> float:
        sample = max(0.0, float(sample))
        if not self.initialized:
            self.value = sample
            self.initialized = True
        else:
            self.value = self.alpha * sample + (1.0 - self.alpha) * self.value
        self.samples += 1
        return self.value

    def get(self, default: float) -> float:
        if self.initialized:
            return self.value
        return default


class TokenRateLimiter:
    """Cooperative limiter based on tokens-per-minute and requests-per-minute."""

    def __init__(self) -> None:
        target_prompt = _env_int("AI_TARGET_PROMPT_TOKENS", 4000)
        self.target_prompt_tokens = max(1, target_prompt)
        self.prompt_ema = ExponentialMovingAverage(0.25, self.target_prompt_tokens)
        self.completion_ema = ExponentialMovingAverage(
            0.25, max(256, self.target_prompt_tokens // 4)
        )

        self.detected_tpm: float = float(_env_int("AI_DETECTED_TPM", 500_000))
        self.detected_rpm: float = float(_env_int("AI_DETECTED_RPM", 500))
        self.tpm_soft_limit = max(0.1, min(1.0, _env_float("AI_TPM_SOFT_LIMIT", 0.80)))
        self.rpm_soft_limit = max(0.1, min(1.0, _env_float("AI_RPM_SOFT_LIMIT", 0.80)))

        self.min_concurrency = max(1, _env_int("AI_MIN_CONCURRENCY", 2))
        self.max_concurrency = max(self.min_concurrency, _env_int("AI_MAX_CONCURRENCY", 16))
        self.backoff_base_ms = max(50, _env_int("AI_BACKOFF_BASE_MS", 250))

        self._lock = threading.Lock()
        self._token_events: Deque[tuple[float, int]] = deque()
        self._request_events: Deque[float] = deque()
        self._token_total: float = 0.0
        self._request_total: int = 0
        self._inflight_tokens: float = 0.0
        self._inflight_requests: int = 0

    # ------------------------------------------------------------------ utils
    def _prune_locked(self, now: float) -> None:
        cutoff = now - WINDOW_SECONDS
        while self._token_events and self._token_events[0][0] <= cutoff:
            ts, amount = self._token_events.popleft()
            self._token_total = max(0.0, self._token_total - float(amount))
        while self._request_events and self._request_events[0] <= cutoff:
            self._request_events.popleft()
            self._request_total = max(0, self._request_total - 1)

    def _smoothed_update(self, attr: str, new_value: Optional[float]) -> None:
        if new_value is None or new_value <= 0:
            return
        with self._lock:
            current = getattr(self, attr)
            if current <= 0:
                setattr(self, attr, float(new_value))
            else:
                blended = 0.7 * float(current) + 0.3 * float(new_value)
                setattr(self, attr, max(1.0, blended))

    # ----------------------------------------------------------------- public
    async def acquire(self, estimated_tokens: int) -> Callable[[bool], None]:
        """Reserve capacity for a request and return a release callback."""

        estimate = max(1, int(estimated_tokens or 0))
        fallback = self.tokens_per_req()
        if estimate < fallback:
            estimate = fallback

        while True:
            wait_time = 0.0
            with self._lock:
                now = time.monotonic()
                self._prune_locked(now)

                tpm_cap = self.detected_tpm * self.tpm_soft_limit if self.detected_tpm else None
                rpm_cap = self.detected_rpm * self.rpm_soft_limit if self.detected_rpm else None

                projected_tokens = self._token_total + self._inflight_tokens + float(estimate)
                projected_requests = self._request_total + self._inflight_requests + 1

                ok_tpm = tpm_cap is None or projected_tokens <= tpm_cap
                ok_rpm = rpm_cap is None or projected_requests <= rpm_cap

                if ok_tpm and ok_rpm:
                    self._inflight_tokens += float(estimate)
                    self._inflight_requests += 1
                    break

                if not ok_tpm and self._token_events:
                    wait_time = max(wait_time, WINDOW_SECONDS - (now - self._token_events[0][0]))
                if not ok_rpm and self._request_events:
                    wait_time = max(wait_time, WINDOW_SECONDS - (now - self._request_events[0]))
                if wait_time <= 0:
                    wait_time = 0.05

            await asyncio.sleep(wait_time)

        released = False

        def _release(success: bool = True) -> None:
            nonlocal released
            if released:
                return
            with self._lock:
                self._inflight_tokens = max(0.0, self._inflight_tokens - float(estimate))
                self._inflight_requests = max(0, self._inflight_requests - 1)
            released = True

        return _release

    def record_tokens(self, prompt_toks: int, completion_toks: int) -> None:
        """Record real token usage after a successful response."""

        prompt = max(0, int(prompt_toks or 0))
        completion = max(0, int(completion_toks or 0))
        total = prompt + completion
        now = time.monotonic()

        with self._lock:
            self._prune_locked(now)
            self.prompt_ema.update(prompt)
            self.completion_ema.update(completion)
            self._token_events.append((now, total))
            self._token_total += float(total)
            self._request_events.append(now)
            self._request_total += 1

    def tokens_per_req(self) -> int:
        prompt_est = self.prompt_ema.get(float(self.target_prompt_tokens))
        completion_est = self.completion_ema.get(max(256.0, self.target_prompt_tokens / 4))
        total = max(prompt_est + completion_est, float(self.target_prompt_tokens))
        return max(1, int(total))

    def suggest_concurrency(self) -> int:
        tokens = max(1, self.tokens_per_req())
        tpm_cap = self.detected_tpm * self.tpm_soft_limit if self.detected_tpm else None
        rpm_cap = self.detected_rpm * self.rpm_soft_limit if self.detected_rpm else None

        conc_from_tpm = math.floor(tpm_cap / tokens) if tpm_cap else self.max_concurrency
        conc_from_rpm = math.floor(rpm_cap) if rpm_cap else self.max_concurrency
        candidate = min(conc_from_tpm, conc_from_rpm, self.max_concurrency)
        return max(self.min_concurrency, candidate if candidate > 0 else self.min_concurrency)

    def backoff_on_429(self, attempt: int, reset_seconds: Optional[float] = None) -> float:
        attempt = max(1, int(attempt))
        base_ms = self.backoff_base_ms * (2 ** (attempt - 1))
        if reset_seconds is not None and reset_seconds > 0:
            base_ms = max(base_ms, reset_seconds * 1000.0)
        jitter_ms = random.uniform(0, base_ms * 0.25)
        sleep_ms = min(base_ms + jitter_ms, 60_000.0)
        return sleep_ms / 1000.0

    # ------------------------------------------------------------- observations
    def update_from_headers(self, headers: Optional[Mapping[str, str]]) -> None:
        if not headers:
            return
        lower_map: Dict[str, str] = {k.lower(): v for k, v in headers.items()}

        def _to_int(name: str) -> Optional[int]:
            raw = lower_map.get(name)
            if raw is None:
                return None
            try:
                return int(float(raw))
            except Exception:
                return None

        def _to_float(name: str) -> Optional[float]:
            raw = lower_map.get(name)
            if raw is None:
                return None
            try:
                return float(raw)
            except Exception:
                return None

        limit_tokens = _to_int("x-ratelimit-limit-tokens")
        limit_requests = _to_int("x-ratelimit-limit-requests")
        remaining_tokens = _to_int("x-ratelimit-remaining-tokens")
        remaining_requests = _to_int("x-ratelimit-remaining-requests")
        reset_tokens = _to_float("x-ratelimit-reset-tokens")
        reset_requests = _to_float("x-ratelimit-reset-requests")

        if limit_tokens:
            self._smoothed_update("detected_tpm", float(limit_tokens))
        elif remaining_tokens is not None and reset_tokens:
            est_tpm = (remaining_tokens / max(reset_tokens, 0.1)) * WINDOW_SECONDS
            self._smoothed_update("detected_tpm", est_tpm)

        if limit_requests:
            self._smoothed_update("detected_rpm", float(limit_requests))
        elif remaining_requests is not None and reset_requests:
            est_rpm = (remaining_requests / max(reset_requests, 0.1)) * WINDOW_SECONDS
            self._smoothed_update("detected_rpm", est_rpm)

    def update_detected_limits(
        self, *, tokens_per_minute: Optional[float] = None, requests_per_minute: Optional[float] = None
    ) -> None:
        self._smoothed_update("detected_tpm", tokens_per_minute)
        self._smoothed_update("detected_rpm", requests_per_minute)


rate_limiter = TokenRateLimiter()

