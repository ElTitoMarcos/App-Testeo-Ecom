import asyncio
import logging
import os
import random
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Deque, Tuple


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


def _resolve_headroom() -> float:
    raw = os.getenv("AI_RL_HEADROOM")
    if raw is None:
        raw = os.getenv("PRAPP_OPENAI_HEADROOM", "0.75")
    try:
        return float(raw)
    except Exception:
        return 0.75


HEADROOM = _resolve_headroom()
RAW_TPM = _env_int("PRAPP_OPENAI_TPM", 30000)
RAW_RPM = _env_int("PRAPP_OPENAI_RPM", 3000)

if RAW_TPM <= 0:
    EFFECTIVE_TPM = 0
else:
    EFFECTIVE_TPM = max(1, int(RAW_TPM * HEADROOM))

if RAW_RPM <= 0:
    EFFECTIVE_RPM = 0
else:
    EFFECTIVE_RPM = max(1, int(RAW_RPM * HEADROOM))

logger = logging.getLogger(__name__)


class _RollingLimiter:
    def __init__(self, eff_rpm: int, eff_tpm: int) -> None:
        self.eff_rpm = max(0, eff_rpm)
        self.eff_tpm = max(0, eff_tpm)
        self.lock = threading.Lock()
        self.request_events: Deque[float] = deque()
        self.token_events: Deque[Tuple[float, int]] = deque()
        self.token_total = 0

    def _cleanup(self, now: float) -> None:
        horizon = now - 60.0
        while self.request_events and self.request_events[0] <= horizon:
            self.request_events.popleft()
        while self.token_events and self.token_events[0][0] <= horizon:
            _, amount = self.token_events.popleft()
            self.token_total -= amount
        if self.token_total < 0:
            self.token_total = 0

    def _record_locked(self, now: float, amount: int) -> None:
        self.request_events.append(now)
        self.token_events.append((now, amount))
        self.token_total += amount

    def preflight(self, amount: int) -> Tuple[float, int, int, int, int, bool]:
        with self.lock:
            now = time.monotonic()
            self._cleanup(now)
            req_count = len(self.request_events)
            token_total = self.token_total

            rpm_wait = 0.0
            if self.eff_rpm > 0 and req_count >= self.eff_rpm:
                earliest = self.request_events[0]
                rpm_wait = max(0.0, (earliest + 60.0) - now)

            tpm_wait = 0.0
            if self.eff_tpm > 0 and token_total + amount > self.eff_tpm:
                excess = token_total + amount - self.eff_tpm
                acc = 0
                wait_until: float | None = None
                for ts, used in self.token_events:
                    acc += used
                    if acc >= excess:
                        wait_until = ts + 60.0
                        break
                if wait_until is not None:
                    tpm_wait = max(0.0, wait_until - now)

            delay = max(rpm_wait, tpm_wait)
            if delay <= 0:
                self._record_locked(now, amount)
                return 0.0, req_count + 1, token_total + amount, self.eff_rpm, self.eff_tpm, True

            return delay, req_count, token_total, self.eff_rpm, self.eff_tpm, False

    def commit(self, amount: int) -> Tuple[int, int]:
        with self.lock:
            now = time.monotonic()
            self._cleanup(now)
            self._record_locked(now, amount)
            return len(self.request_events), self.token_total


_limiter = _RollingLimiter(EFFECTIVE_RPM, EFFECTIVE_TPM)


def _normalize_amount(tokens_in: int, tokens_out: int) -> int:
    try:
        in_val = int(tokens_in or 0)
    except Exception:
        in_val = 0
    try:
        out_val = int(tokens_out or 0)
    except Exception:
        out_val = 0
    total = max(0, in_val) + max(0, out_val)
    return max(1, total)


def snapshot() -> dict[str, int | float | None]:
    return {
        "eff_tpm": EFFECTIVE_TPM,
        "eff_rpm": EFFECTIVE_RPM,
        "headroom": HEADROOM,
        "max_conc": None,
    }


async def acquire(tokens_in: int, tokens_out: int = 0) -> float:
    amount = _normalize_amount(tokens_in, tokens_out)
    delay, req_usage, token_usage, eff_rpm, eff_tpm, committed = _limiter.preflight(amount)
    waited = 0.0
    if delay > 0:
        if delay > 1.0:
            logger.info(
                "rate_limit_wait waited=%.2fs rpm=%d/%s tpm=%d/%s",
                delay,
                req_usage,
                str(eff_rpm or "∞"),
                token_usage,
                str(eff_tpm or "∞"),
            )
        await asyncio.sleep(delay)
        _limiter.commit(amount)
        waited = delay
    elif not committed:
        _limiter.commit(amount)
    return waited


@contextmanager
def reserve(tokens_estimate: int):
    amount = _normalize_amount(tokens_estimate, 0)
    delay, req_usage, token_usage, eff_rpm, eff_tpm, committed = _limiter.preflight(amount)
    try:
        if delay > 0:
            if delay > 1.0:
                logger.info(
                    "rate_limit_wait waited=%.2fs rpm=%d/%s tpm=%d/%s",
                    delay,
                    req_usage,
                    str(eff_rpm or "∞"),
                    token_usage,
                    str(eff_tpm or "∞"),
                )
            time.sleep(delay)
            _limiter.commit(amount)
        elif not committed:
            _limiter.commit(amount)
        yield
    finally:
        # No release step required; limiter tracks rolling windows only.
        pass


def log_throttled(req_id: str | None, retry_after: float | None, waited: float) -> None:
    logger.warning(
        "rate_limit_throttled req_id=%s retry_after=%.2f waited=%.2f",
        req_id or "",
        float(retry_after) if retry_after is not None else -1.0,
        float(waited or 0.0),
    )


def decorrelated_jitter_sleep(prev: float, cap: float) -> float:
    base = 0.3
    next_sleep = min(cap, random.uniform(base, prev * 3 if prev > 0 else 1.0))
    time.sleep(next_sleep)
    return next_sleep

