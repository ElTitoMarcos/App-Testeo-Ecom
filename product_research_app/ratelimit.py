from __future__ import annotations

import asyncio
import os, time, threading, random
from contextlib import asynccontextmanager, contextmanager
from typing import Optional


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


_TPM = max(1, _env_int("PRAPP_OPENAI_TPM", 30000))
_RPM = max(1, _env_int("PRAPP_OPENAI_RPM", 3000))
_HEADROOM = max(0.1, min(0.99, _env_float("PRAPP_OPENAI_HEADROOM", 0.90)))
_MAX_CONC = max(1, _env_int("PRAPP_OPENAI_MAX_CONCURRENCY", 8))

_EFF_TPM = 1
_EFF_RPM = 1
_eff_conc = 1

_tokens_bucket: "_TokenBucket"
_requests_bucket: "_TokenBucket"
_conc_sem: threading.BoundedSemaphore
_async_limiter: "AsyncRateLimiter"

_UPDATE_LOCK = threading.Lock()

class _TokenBucket:
    def __init__(self, capacity_per_min: int):
        self.capacity = max(1, capacity_per_min)
        self.tokens = float(self.capacity)
        self.lock = threading.Lock()
        self.last = time.monotonic()
    def acquire(self, amount: int):
        amount = max(1, int(amount))
        with self.lock:
            while True:
                now = time.monotonic()
                # recarga lineal por segundo
                refill = (now - self.last) * (self.capacity / 60.0)
                if refill > 0:
                    self.tokens = min(self.capacity, self.tokens + refill)
                    self.last = now
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                # esperar lo justo para acumular lo que falta
                deficit = amount - self.tokens
                rate_per_s = (self.capacity / 60.0)
                sleep_s = max(deficit / rate_per_s, 0.01)
                # liberar el lock para no bloquear a quienes vengan a recargar/consultar
                self.lock.release()
                try:
                    time.sleep(sleep_s)
                finally:
                    self.lock.acquire()

# buckets globales
_tokens_bucket = _TokenBucket(1)
_requests_bucket = _TokenBucket(1)

# semáforo global para capar concurrencia
_conc_sem = threading.BoundedSemaphore(1)

@contextmanager
def reserve(estimated_tokens: int):
    """
    Embudo global de RPM/TPM + concurrencia.
    Debe envolver TODA llamada real al proveedor (batch y refine).
    """
    _conc_sem.acquire()
    try:
        # 1) limita RPM (una unidad por request)
        _requests_bucket.acquire(1)
        # 2) limita TPM (tokens estimados)
        _tokens_bucket.acquire(max(1, int(estimated_tokens)))
        yield
    finally:
        _conc_sem.release()

def decorrelated_jitter_sleep(prev: float, cap: float) -> float:
    """
    Backoff con "decorrelated jitter" (AWS). Devuelve el sleep usado.
    """
    base = 0.3
    prev = max(0.0, float(prev or 0.0))
    cap = max(base, float(cap or base))
    next_sleep = min(cap, random.uniform(base, prev * 3 if prev > 0 else 1.0))
    time.sleep(next_sleep)
    return next_sleep


class AsyncTokenBucket:
    def __init__(self, capacity_per_min: int):
        self.capacity = max(1, int(capacity_per_min))
        self.tokens = self.capacity
        self.updated = time.time()
        self._lock = asyncio.Lock()

    async def _refill(self):
        now = time.time()
        delta = now - self.updated
        if delta >= 1.0:
            per_sec = self.capacity / 60.0
            add = int(delta * per_sec)
            if add > 0:
                self.tokens = min(self.capacity, self.tokens + add)
                self.updated = now

    async def take(self, n: int = 1):
        n = max(0, int(n))
        async with self._lock:
            while self.tokens < n:
                await self._refill()
                if self.tokens < n:
                    await asyncio.sleep(0.2)
            self.tokens -= n


class AsyncRateLimiter:
    def __init__(self, tpm: int, rpm: int, max_conc: int):
        self.tpm = AsyncTokenBucket(max(1, int(tpm)))
        self.rpm = AsyncTokenBucket(max(1, int(rpm)))
        self.conc = asyncio.Semaphore(max(1, int(max_conc)))

    @asynccontextmanager
    async def async_guard(self, *, tokens: int = 0):
        await self.conc.acquire()
        try:
            await self.rpm.take(1)
            if tokens > 0:
                await self.tpm.take(tokens)
            yield
        finally:
            self.conc.release()


_async_limiter = AsyncRateLimiter(1, 1, 1)


def update_runtime_limits(
    tpm: Optional[int] = None,
    rpm: Optional[int] = None,
    *,
    headroom: Optional[float] = None,
    max_conc: Optional[int] = None,
) -> None:
    global _TPM, _RPM, _HEADROOM, _MAX_CONC, _EFF_TPM, _EFF_RPM, _eff_conc
    global _tokens_bucket, _requests_bucket, _conc_sem, _async_limiter

    with _UPDATE_LOCK:
        if tpm is not None:
            _TPM = max(1, int(tpm))
        if rpm is not None:
            _RPM = max(1, int(rpm))
        if headroom is not None:
            try:
                _HEADROOM = max(0.1, min(0.99, float(headroom)))
            except Exception:
                pass
        if max_conc is not None:
            try:
                _MAX_CONC = max(1, int(max_conc))
            except Exception:
                pass

        _EFF_TPM = max(1, int(_TPM * _HEADROOM))
        _EFF_RPM = max(1, int(_RPM * _HEADROOM))
        _tokens_bucket = _TokenBucket(_EFF_TPM)
        _requests_bucket = _TokenBucket(_EFF_RPM)
        _eff_conc = max(1, min(_MAX_CONC, int(max(1, _MAX_CONC * _HEADROOM))))
        _conc_sem = threading.BoundedSemaphore(_eff_conc)
        _async_limiter = AsyncRateLimiter(_EFF_TPM, _EFF_RPM, _eff_conc)


def get_async_limiter() -> AsyncRateLimiter:
    return _async_limiter


async def async_decorrelated_jitter_sleep(prev: float, cap: float) -> float:
    base = 0.3
    prev = max(0.0, float(prev or 0.0))
    cap = max(base, float(cap or base))
    next_sleep = min(cap, random.uniform(base, prev * 3 if prev > 0 else 1.0))
    await asyncio.sleep(next_sleep)
    return next_sleep


# Inicializa los buckets con los valores actuales del entorno.
update_runtime_limits(_TPM, _RPM, headroom=_HEADROOM, max_conc=_MAX_CONC)
