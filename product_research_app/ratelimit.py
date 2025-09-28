from __future__ import annotations
import os, time, threading, random
from contextlib import contextmanager

def _env_int(name, default):
    try: return int(os.getenv(name, default))
    except: return default
def _env_float(name, default):
    try: return float(os.getenv(name, default))
    except: return default

_TPM = _env_int("PRAPP_OPENAI_TPM", 30000)
_RPM = _env_int("PRAPP_OPENAI_RPM", 3000)
_HEADROOM = _env_float("PRAPP_OPENAI_HEADROOM", 0.85)
_MAX_CONC = _env_int("PRAPP_OPENAI_MAX_CONCURRENCY", 2)

# Efectivo tras aplicar headroom
_EFF_TPM = max(1, int(_TPM * _HEADROOM))
_EFF_RPM = max(1, int(_RPM * _HEADROOM))

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
_tokens_bucket = _TokenBucket(_EFF_TPM)
_requests_bucket = _TokenBucket(_EFF_RPM)

# semáforo global para capar concurrencia
_conc_sem = threading.BoundedSemaphore(_MAX_CONC)

@contextmanager
def reserve(tokens_estimate: int):
    """
    Embudo global de RPM/TPM + concurrencia.
    Debe envolver TODA llamada real al proveedor (batch y refine).
    """
    _conc_sem.acquire()
    try:
        # 1) limita RPM (una unidad por request)
        _requests_bucket.acquire(1)
        # 2) limita TPM (tokens estimados)
        _tokens_bucket.acquire(max(1, int(tokens_estimate)))
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
