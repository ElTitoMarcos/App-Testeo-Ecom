"""High throughput AI enrichment scheduler.

This module implements a token/RPM aware asynchronous scheduler that submits
batches of products to a chat-completions endpoint while respecting provider
limits.  It estimates the token footprint of every request to decide how many
items should be grouped together and adapts concurrency in real time using an
exponential moving average of the observed latency.  Results are persisted
through a DAO interface that exposes ``upsert_ai_fields_bulk``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Límites de la API (ajusta a tu cuenta/modelo)
RPM_LIMIT = int(os.getenv("RPM_LIMIT", "120"))  # requests por minuto
TPM_LIMIT = int(os.getenv("TPM_LIMIT", "400000"))  # tokens por minuto
REQ_TOKEN_HARD_MAX = int(os.getenv("REQ_TOKEN_HARD_MAX", "120000"))  # tope tokens por request

# Planificación
MAX_ITEMS_PER_CALL = int(os.getenv("MAX_ITEMS_PER_CALL", "64"))
TARGET_INPUT_TOKENS_PER_REQ = int(os.getenv("TARGET_INPUT_TOKENS_PER_REQ", "6000"))
CONCURRENCY_HARD_MAX = int(os.getenv("CONCURRENCY_HARD_MAX", "64"))

# Política de coste: si True, asumimos paridad de coste por ítem entre llamadas individuales y agrupadas
COST_PARITY_MODE = os.getenv("COST_PARITY_MODE", "1") in ("1", "true", "True", "yes")

TIMEOUT_REQUEST_SEC = float(os.getenv("TIMEOUT_REQUEST_SEC", "45"))
MAX_RETRIES = 3

GPT_URL = os.getenv("GPT_URL", "https://api.your-model/chat/completions")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4.1-mini")
GPT_API_KEY = (
    os.getenv("GPT_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("ENRICH_API_KEY")
    or os.getenv("AI_API_KEY")
)


def _tokens_est_item(it: dict) -> int:
    """Estimación defensiva tokens/ítem (chars/4) + mínimo por ítem."""

    title = (it.get("title", "")[:120])
    desc = (it.get("description", "")[:240])
    brand = (it.get("brand", "")[:30])
    cat = (it.get("category", "")[:40])
    nums = "".join(
        str(it.get(k, ""))
        for k in ("price", "rating", "units_sold", "revenue", "oldness")
    )
    chars = len(title) + len(desc) + len(brand) + len(cat) + len(nums)
    return max(28, chars // 4)


def _tokens_est_request(items: List[dict]) -> int:
    """Overhead del prompt + suma de ítems."""

    base = 450  # sistema + instrucciones + llaves JSON
    return base + sum(_tokens_est_item(it) for it in items)


def _best_K_for_tokens(items: List[dict], k_cap: int) -> int:
    """Calcula K máximo seguro para tokens/req."""

    if not items:
        return 0
    sample = items[: min(64, len(items))]
    tpi = max(28, sum(_tokens_est_item(it) for it in sample) / len(sample))
    budget = min(TARGET_INPUT_TOKENS_PER_REQ, REQ_TOKEN_HARD_MAX)
    available = max(0, budget - 450)
    if tpi <= 0:
        return max(1, min(k_cap, len(items)))
    k = int(available // tpi)
    return max(1, min(k_cap, k if k > 0 else 1, len(items)))


def _chunks_by_K(items: List[dict], K: int) -> List[List[dict]]:
    if K <= 0:
        return [items]
    return [items[i : i + K] for i in range(0, len(items), K)]


class LatencyEMA:
    def __init__(self, alpha: float = 0.2, init_sec: float = 2.5) -> None:
        self.alpha = alpha
        self.val = init_sec

    def update(self, x: float) -> float:
        self.val = self.alpha * x + (1 - self.alpha) * self.val
        return self.val

    @property
    def sec(self) -> float:
        return max(0.5, self.val)


def _concurrency_cap(t_req_tokens: int, ema: LatencyEMA) -> int:
    """Devuelve la concurrencia máxima respetando RPM/TPM."""

    latency = ema.sec
    c_rpm = max(1, int((RPM_LIMIT * latency) // 60))
    if t_req_tokens <= 0:
        t_req_tokens = 1000
    c_tpm = max(1, int((TPM_LIMIT * latency) // (60 * t_req_tokens)))
    return max(1, min(CONCURRENCY_HARD_MAX, c_rpm, c_tpm))


def _plan_choose_K_and_C(items: List[dict], ema: LatencyEMA) -> tuple[int, int, int]:
    if not items:
        return 1, 1, 1000
    Kmax = _best_K_for_tokens(items, MAX_ITEMS_PER_CALL)
    t_req_A = _tokens_est_request(items[:Kmax])
    C_A = _concurrency_cap(t_req_A, ema)
    ipm_A = C_A * Kmax * 60.0 / ema.sec

    t_req_B = _tokens_est_request(items[:1])
    C_B = _concurrency_cap(t_req_B, ema)
    ipm_B = C_B * 60.0 / ema.sec

    if COST_PARITY_MODE and ipm_B > ipm_A * 1.05:
        return 1, C_B, t_req_B
    return Kmax, C_A, t_req_A


def _build_payload(batch: List[dict], weights: Optional[dict]) -> dict:
    lines: List[dict] = []
    for it in batch:
        lines.append(
            {
                "id": str(it.get("id")),
                "title": (it.get("title", "") or "")[:120],
                "category": (it.get("category", "") or "")[:40],
                "brand": (it.get("brand", "") or "")[:30],
                "price": it.get("price"),
                "rating": it.get("rating"),
                "units_sold": it.get("units_sold"),
                "revenue": it.get("revenue"),
                "oldness": it.get("oldness"),
                "desc": (it.get("description", "") or "")[:240],
            }
        )
    sys_msg = {
        "role": "system",
        "content": "Eres un clasificador estricto. Devuelves SOLO JSON válido.",
    }
    weight_text = f"Weights: {json.dumps(weights)}" if weights else ""
    user_msg = {
        "role": "user",
        "content": (
            "Devuelve SOLO este JSON (sin texto extra): "
            "{\"results\":[{\"id\":\"<id>\",\"desire\":0-100,\"desire_magnitude\":0-100,"
            "\"awareness_level\":0-100,\"competition_level\":0-100,\"winner_score\":0-100}]}\n"
            "Definiciones:\n"
            "- desire: apetencia del comprador; desire_magnitude: intensidad del deseo; "
            "awareness_level: visibilidad en mercado; competition_level: saturación (100=alta); "
            "winner_score: síntesis 0–100; si paso 'weights', pondera según ellos.\n"
            f"Items: {json.dumps(lines, ensure_ascii=False)}\n"
            f"{weight_text}"
        ),
    }
    return {"messages": [sys_msg, user_msg]}


def _clamp01(x: Any) -> int:
    try:
        v = int(round(float(x)))
    except Exception:
        v = 0
    return max(0, min(100, v))


def _parse_results(obj: dict) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    arr = obj.get("results") if isinstance(obj, dict) else None
    if not isinstance(arr, list):
        return out
    for entry in arr:
        pid = str(entry.get("id", "")).strip()
        if not pid:
            continue
        out[pid] = {
            "desire": _clamp01(entry.get("desire")),
            "desire_magnitude": _clamp01(entry.get("desire_magnitude")),
            "awareness_level": _clamp01(entry.get("awareness_level")),
            "competition_level": _clamp01(entry.get("competition_level")),
            "winner_score": _clamp01(entry.get("winner_score")),
        }
    return out


async def _call_gpt(client: httpx.AsyncClient, payload: dict) -> dict:
    body = {"model": GPT_MODEL, **payload}
    response = await client.post(GPT_URL, json=body, timeout=TIMEOUT_REQUEST_SEC)
    response.raise_for_status()
    try:
        return response.json()
    except Exception:
        return json.loads(response.text)


class TokenBucket:
    def __init__(self, capacity: float, refill_per_sec: float) -> None:
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill_per_sec = float(refill_per_sec)
        self.ts = time.monotonic()
        self.lock = asyncio.Lock()

    async def consume(self, amount: float) -> None:
        amount = max(0.0, float(amount))
        async with self.lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.ts
                if elapsed > 0:
                    self.tokens = min(
                        self.capacity, self.tokens + elapsed * self.refill_per_sec
                    )
                    self.ts = now
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                need = (amount - self.tokens) / self.refill_per_sec if self.refill_per_sec else 1.0
                await asyncio.sleep(max(0.01, need))


async def enrich_max_parallel(
    items: List[dict],
    dao: Any,
    weights: Optional[dict] = None,
    *,
    api_key: Optional[str] = None,
) -> int:
    """Procesa ``items`` en paralelo respetando cuotas y actualiza la BD."""

    if not items:
        return 0

    api_key = api_key or GPT_API_KEY
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    ema = LatencyEMA()
    K, C_init, t_req_est = _plan_choose_K_and_C(items, ema)
    batches = _chunks_by_K(items, K)

    bucket_rpm = TokenBucket(RPM_LIMIT, RPM_LIMIT / 60.0)
    bucket_tpm = TokenBucket(TPM_LIMIT, TPM_LIMIT / 60.0)

    sem = asyncio.Semaphore(CONCURRENCY_HARD_MAX)
    written = 0
    buffer: List[dict] = []
    start = time.monotonic()

    async with httpx.AsyncClient(headers=headers) as client:
        async def run_batch(batch: List[dict], attempt: int = 0) -> bool:
            nonlocal written, buffer
            t_est = max(1000, _tokens_est_request(batch))
            await bucket_rpm.consume(1)
            await bucket_tpm.consume(t_est)
            async with sem:
                sent = time.monotonic()
                try:
                    payload = _build_payload(batch, weights)
                    raw = await _call_gpt(client, payload)
                    latency = time.monotonic() - sent
                    ema.update(latency)
                    parsed = _parse_results(raw)
                    rows: List[dict] = []
                    for it in batch:
                        pid = str(it.get("id"))
                        if not pid:
                            continue
                        if pid in parsed:
                            rows.append({"id": it.get("id"), **parsed[pid]})
                    if rows:
                        buffer.extend(rows)
                        if len(buffer) >= 500:
                            dao.upsert_ai_fields_bulk(buffer)
                            written += len(buffer)
                            buffer.clear()
                    logger.info(
                        "IA batch ok size=%d latency=%.2fs", len(batch), latency
                    )
                    return True
                except httpx.HTTPStatusError as exc:
                    latency = time.monotonic() - sent
                    ema.update(latency)
                    retry_after = exc.response.headers.get("Retry-After")
                    wait = None
                    if retry_after:
                        try:
                            wait = float(retry_after)
                        except Exception:
                            wait = None
                    if wait is None:
                        wait = min(30.0, 2.0 ** min(6, attempt))
                    logger.warning(
                        "HTTP %s en batch (intent=%d). Esperando %.2fs", exc.response.status_code, attempt, wait
                    )
                    await asyncio.sleep(wait)
                    if attempt < MAX_RETRIES:
                        return await run_batch(batch, attempt + 1)
                    logger.warning("Batch failed definitively after retries: %s", exc)
                    return False
                except Exception as exc:  # noqa: BLE001
                    latency = time.monotonic() - sent
                    ema.update(latency)
                    backoff = 1.0 * (attempt + 1)
                    logger.warning(
                        "Batch error (%s). Retrying in %.1fs", exc, backoff
                    )
                    if attempt < 2:
                        await asyncio.sleep(backoff)
                        return await run_batch(batch, attempt + 1)
                    logger.exception("Batch error irrecoverable")
                    return False

        tasks: set[asyncio.Task] = set()
        idx = 0
        current_cap = max(1, min(CONCURRENCY_HARD_MAX, C_init))
        while idx < len(batches) and len(tasks) < current_cap:
            tasks.add(asyncio.create_task(run_batch(batches[idx])))
            idx += 1

        while tasks:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            tasks = pending
            for task in done:
                try:
                    task.result()
                except Exception:
                    logger.exception("Task raised unexpectedly")
            # Recalcular concurrencia con latencia observada
            C_target = max(1, min(CONCURRENCY_HARD_MAX, _concurrency_cap(t_req_est, ema)))
            while idx < len(batches) and len(tasks) < C_target:
                tasks.add(asyncio.create_task(run_batch(batches[idx])))
                idx += 1

        if buffer:
            dao.upsert_ai_fields_bulk(buffer)
            written += len(buffer)
            buffer.clear()

    duration = time.monotonic() - start
    ips = (len(items) / duration) if duration > 0 else float("inf")
    logger.info(
        "IA enrich: items=%d K=%d concurrency~%d duration=%.2fs throughput=%.2f items/s",
        len(items),
        K,
        max(1, min(CONCURRENCY_HARD_MAX, C_init)),
        duration,
        ips,
    )
    return written


__all__ = ["enrich_max_parallel"]
