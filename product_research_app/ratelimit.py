from __future__ import annotations

import os
import random
import threading
import time
from contextlib import contextmanager


def _env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


def _env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


_TPM = _env_int("PRAPP_OPENAI_TPM", 30000)
_RPM = _env_int("PRAPP_OPENAI_RPM", 3000)
_HEADROOM = _env_float("PRAPP_OPENAI_HEADROOM", 0.85)
_MAX_CONC = _env_int("PRAPP_OPENAI_MAX_CONCURRENCY", 2)

_EFF_TPM = max(1, int(_TPM * _HEADROOM))
_EFF_RPM = max(1, int(_RPM * _HEADROOM))


class _TokenBucket:
    def __init__(self, capacity_per_min: int):
        self.capacity = max(1, capacity_per_min)
        self.tokens = self.capacity
        self.lock = threading.Lock()
        self.last = time.monotonic()

    def acquire(self, amount: int):
        with self.lock:
            while True:
                now = time.monotonic()
                refill = (now - self.last) * (self.capacity / 60.0)
                if refill > 0:
                    self.tokens = min(self.capacity, self.tokens + refill)
                    self.last = now
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                deficit = amount - self.tokens
                sleep_s = max(deficit / (self.capacity / 60.0), 0.01)
                self.lock.release()
                try:
                    time.sleep(sleep_s)
                finally:
                    self.lock.acquire()


_tokens_bucket = _TokenBucket(_EFF_TPM)
_requests_bucket = _TokenBucket(_EFF_RPM)

_conc_sem = threading.BoundedSemaphore(_MAX_CONC)


@contextmanager
def reserve(tokens_estimate: int):
    _conc_sem.acquire()
    try:
        _requests_bucket.acquire(1)
        _tokens_bucket.acquire(max(1, tokens_estimate))
        yield
    finally:
        _conc_sem.release()


def decorrelated_jitter_sleep(prev: float, cap: float) -> float:
    base = 0.3
    next_sleep = min(cap, random.uniform(base, prev * 3 if prev > 0 else 1.0))
    time.sleep(next_sleep)
    return next_sleep

