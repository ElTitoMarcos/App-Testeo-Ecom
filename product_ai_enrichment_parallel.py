from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List

import httpx

logger = logging.getLogger(__name__)

RPM_LIMIT = int(os.getenv("RPM_LIMIT", "120"))
TPM_LIMIT = int(os.getenv("TPM_LIMIT", "400000"))
REQ_TOKEN_HARD_MAX = int(os.getenv("REQ_TOKEN_HARD_MAX", "120000"))
TARGET_INPUT_TOKENS_PER_REQ = int(os.getenv("TARGET_INPUT_TOKENS_PER_REQ", "6000"))

MAX_ITEMS_PER_CALL = int(os.getenv("MAX_ITEMS_PER_CALL", "128"))
CONCURRENCY_MAX = int(os.getenv("CONCURRENCY_MAX", "32"))
TIMEOUT_REQUEST_SEC = float(os.getenv("TIMEOUT_REQUEST_SEC", "45"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

GPT_URL = os.getenv("GPT_URL", "https://api.your-model/chat/completions")
GPT_API_KEY = os.getenv("GPT_API_KEY") or os.getenv("OPENAI_API_KEY")
GPT_MODEL = os.getenv("GPT_MODEL")
GPT_HEADERS_RAW = os.getenv("GPT_HEADERS")

_HEADERS: Dict[str, str] = {}
if GPT_API_KEY:
    _HEADERS["Authorization"] = f"Bearer {GPT_API_KEY}"
if GPT_HEADERS_RAW:
    try:
        extra_headers = json.loads(GPT_HEADERS_RAW)
        if isinstance(extra_headers, dict):
            _HEADERS.update({str(k): str(v) for k, v in extra_headers.items()})
    except json.JSONDecodeError:
        logger.warning("Invalid GPT_HEADERS JSON; ignoring")

DB_WRITE_BATCH = int(os.getenv("DB_WRITE_BATCH", "500"))
DB_WRITE_INTERVAL_SEC = float(os.getenv("DB_WRITE_INTERVAL_SEC", "1.0"))

_WRITER_STOP = object()


def _tokens_est_item(it: Dict[str, Any]) -> int:
    title = (it.get("title", "") or "")[:120]
    desc = (it.get("description", "") or "")[:240]
    brand = (it.get("brand", "") or "")[:30]
    cat = (it.get("category", "") or "")[:40]
    nums = f"{it.get('price', '')}{it.get('rating', '')}{it.get('units_sold', '')}{it.get('revenue', '')}{it.get('oldness', '')}"
    chars = len(title) + len(desc) + len(brand) + len(cat) + len(nums)
    return max(28, chars // 4)


def _tokens_est_request(items: List[Dict[str, Any]]) -> int:
    return 450 + sum(_tokens_est_item(x) for x in items)


def _best_K_for_tokens(items: List[Dict[str, Any]], k_cap: int) -> int:
    if not items:
        return 0
    sample = items[: min(64, len(items))]
    tpi = max(28, sum(_tokens_est_item(x) for x in sample) / len(sample))
    budget = min(TARGET_INPUT_TOKENS_PER_REQ, REQ_TOKEN_HARD_MAX)
    available = max(0, budget - 450)
    k = int(available // tpi) if tpi else k_cap
    return max(1, min(k_cap, k))


def _chunk_maximal(items: List[Dict[str, Any]], K: int) -> List[List[Dict[str, Any]]]:
    if K <= 0:
        return []
    chunks: List[List[Dict[str, Any]]] = []
    i = 0
    while i < len(items):
        sub = items[i : i + K]
        while len(sub) > 1 and _tokens_est_request(sub) > min(TARGET_INPUT_TOKENS_PER_REQ, REQ_TOKEN_HARD_MAX):
            mid = len(sub) // 2
            left = sub[:mid]
            right = sub[mid:]
            if _tokens_est_request(left) <= min(TARGET_INPUT_TOKENS_PER_REQ, REQ_TOKEN_HARD_MAX):
                chunks.append(left)
                sub = right
            else:
                sub = left
        chunks.append(sub)
        i += K
    return [c for c in chunks if c]


def _build_payload(batch: List[Dict[str, Any]], weights: Dict[str, Any] | None) -> Dict[str, Any]:
    lines: List[Dict[str, Any]] = []
    for it in batch:
        lines.append(
            {
                "id": str(it["id"]),
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
    sys_msg = {"role": "system", "content": "Eres un clasificador estricto. Devuelves SOLO JSON válido."}
    user_content = (
        "Devuelve SOLO este JSON (sin texto extra): "
        '{"results":[{"id":"<id>","desire":0-100,"desire_magnitude":0-100,'
        '"awareness_level":0-100,"competition_level":0-100,"winner_score":0-100}]}'
        "\nDefiniciones: desire=apetencia; desire_magnitude=intensidad; awareness=visibilidad; "
        "competition=saturación (100=alta); winner_score=0–100 (pondera 'weights' si los paso).\n"
        f"Items: {json.dumps(lines, ensure_ascii=False)}\n"
    )
    if weights:
        user_content += f"Weights: {json.dumps(weights)}"
    user_msg = {"role": "user", "content": user_content}
    payload: Dict[str, Any] = {"messages": [sys_msg, user_msg]}
    if GPT_MODEL and "model" not in payload:
        payload["model"] = GPT_MODEL
    return payload


def _c(x: Any) -> int:
    try:
        v = int(round(float(x)))
    except Exception:
        v = 0
    return max(0, min(100, v))


def _parse_results(obj: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    arr = obj.get("results") if isinstance(obj, dict) else None
    if not isinstance(arr, list):
        return out
    for e in arr:
        pid = str(e.get("id", "")).strip()
        if not pid:
            continue
        out[pid] = {
            "desire": _c(e.get("desire")),
            "desire_magnitude": _c(e.get("desire_magnitude")),
            "awareness_level": _c(e.get("awareness_level")),
            "competition_level": _c(e.get("competition_level")),
            "winner_score": _c(e.get("winner_score")),
        }
    return out


async def _call_gpt(client: httpx.AsyncClient, payload: Dict[str, Any]) -> Dict[str, Any]:
    req_payload = dict(payload)
    if GPT_MODEL and "model" not in req_payload:
        req_payload["model"] = GPT_MODEL
    headers = _HEADERS or None
    response = await client.post(GPT_URL, json=req_payload, headers=headers)
    response.raise_for_status()
    try:
        return response.json()
    except Exception:
        return json.loads(response.text)


class TokenBucket:
    def __init__(self, capacity: float, refill_per_sec: float) -> None:
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill_per_sec = float(refill_per_sec) if refill_per_sec > 0 else float("inf")
        self.ts = time.monotonic()
        self.lock = asyncio.Lock()

    async def consume(self, amount: float) -> None:
        amount = float(max(0.0, amount))
        if amount <= 0:
            return
        while True:
            async with self.lock:
                now = time.monotonic()
                delta = now - self.ts
                if self.refill_per_sec != float("inf"):
                    self.tokens = min(self.capacity, self.tokens + delta * self.refill_per_sec)
                else:
                    self.tokens = self.capacity
                self.ts = now
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                if self.refill_per_sec == float("inf") or self.refill_per_sec <= 0:
                    wait = 0.01
                else:
                    wait = (amount - self.tokens) / self.refill_per_sec + 0.01
            await asyncio.sleep(wait)


async def _writer_loop(queue: asyncio.Queue[Any], dao: Any) -> None:
    buffer: List[Dict[str, Any]] = []
    last_flush = time.monotonic()
    while True:
        try:
            batch = await asyncio.wait_for(queue.get(), timeout=DB_WRITE_INTERVAL_SEC)
        except asyncio.TimeoutError:
            batch = None
        if batch is _WRITER_STOP:
            queue.task_done()
            break
        if batch:
            buffer.extend(batch)
        if batch is not None:
            queue.task_done()
        now = time.monotonic()
        if buffer and (len(buffer) >= DB_WRITE_BATCH or now - last_flush >= DB_WRITE_INTERVAL_SEC):
            try:
                dao.upsert_ai_fields_bulk(list(buffer))
            finally:
                buffer.clear()
            last_flush = now
    if buffer:
        dao.upsert_ai_fields_bulk(list(buffer))
    # drain leftovers if any pending (e.g., sentinel placed multiple times)
    while not queue.empty():
        try:
            pending = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        if pending and pending is not _WRITER_STOP:
            dao.upsert_ai_fields_bulk(list(pending))
        queue.task_done()


async def enrich_min_calls_parallel(
    items: List[Dict[str, Any]],
    dao: Any,
    weights: Dict[str, Any] | None = None,
) -> int:
    if not items:
        return 0

    Kmax = _best_K_for_tokens(items, MAX_ITEMS_PER_CALL)
    batches = _chunk_maximal(items, Kmax)
    if not batches:
        return 0

    bucket_rpm = TokenBucket(RPM_LIMIT, RPM_LIMIT / 60.0 if RPM_LIMIT else float("inf"))
    bucket_tpm = TokenBucket(TPM_LIMIT, TPM_LIMIT / 60.0 if TPM_LIMIT else float("inf"))

    concurrency = max(1, min(CONCURRENCY_MAX, len(batches)))
    sem = asyncio.Semaphore(concurrency)

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits, headers=_HEADERS or None, timeout=TIMEOUT_REQUEST_SEC) as client:
        result_q: asyncio.Queue[Any] = asyncio.Queue()
        writer = asyncio.create_task(_writer_loop(result_q, dao))

        async def run_one(batch: List[Dict[str, Any]], idx: int, attempt: int = 0) -> bool:
            tokens_est = max(1000, _tokens_est_request(batch))
            await bucket_rpm.consume(1)
            await bucket_tpm.consume(tokens_est)
            async with sem:
                t0 = time.monotonic()
                try:
                    payload = _build_payload(batch, weights)
                    raw = await _call_gpt(client, payload)
                    parsed = _parse_results(raw)
                    rows: List[Dict[str, Any]] = []
                    for it in batch:
                        pid = str(it["id"])
                        result = parsed.get(pid)
                        if not result:
                            continue
                        rows.append(
                            {
                                "id": it["id"],
                                "desire": result["desire"],
                                "desire_magnitude": result["desire_magnitude"],
                                "awareness_level": result["awareness_level"],
                                "competition_level": result["competition_level"],
                                "winner_score": result["winner_score"],
                            }
                        )
                    if rows:
                        await result_q.put(rows)
                    duration = time.monotonic() - t0
                    logger.info("IA batch %d ok: items=%d, %.2fs", idx, len(batch), duration)
                    return True
                except httpx.HTTPStatusError as exc:
                    response = exc.response
                    retry_after = response.headers.get("Retry-After") if response else None
                    wait = None
                    if retry_after:
                        try:
                            wait = float(retry_after)
                        except (TypeError, ValueError):
                            wait = None
                    if wait is None:
                        wait = min(30, 2 ** min(6, attempt))
                    logger.warning(
                        "IA batch %d HTTP %s; retry in %.1fs", idx, response.status_code if response else "?", wait
                    )
                    await asyncio.sleep(wait)
                    if attempt < MAX_RETRIES:
                        return await run_one(batch, idx, attempt + 1)
                    logger.error("IA batch %d failed after %d retries", idx, attempt)
                    return False
                except Exception as exc:
                    logger.exception("IA batch %d error: %s", idx, exc)
                    if attempt < min(2, MAX_RETRIES):
                        await asyncio.sleep(1.0 * (attempt + 1))
                        return await run_one(batch, idx, attempt + 1)
                    return False

        tasks = [asyncio.create_task(run_one(batch, i)) for i, batch in enumerate(batches, start=1)]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        await result_q.put(_WRITER_STOP)
        await writer

    ok = sum(1 for r in results if r)
    logger.info(
        "IA paralelo fin: batches=%d, ok=%d, items=%d, Kmax=%d, conc=%d",
        len(batches),
        ok,
        len(items),
        Kmax,
        concurrency,
    )
    return len(items)


def enrich_min_calls(items: List[Dict[str, Any]], dao: Any, weights: Dict[str, Any] | None = None) -> int:
    return asyncio.run(enrich_min_calls_parallel(items, dao, weights))


__all__ = ["enrich_min_calls_parallel", "enrich_min_calls"]
