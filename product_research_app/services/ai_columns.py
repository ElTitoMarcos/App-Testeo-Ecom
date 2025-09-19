import os
import time
import json
import math
import asyncio
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from .. import config, database

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent.parent
DB_PATH = APP_DIR / "data.sqlite3"

RPM_LIMIT = int(os.getenv("AI_RPM_LIMIT", "9999"))
TPM_LIMIT = int(os.getenv("AI_TPM_LIMIT", "99999999"))
REQ_TOKEN_HARD_MAX = int(os.getenv("AI_REQ_TOKEN_HARD_MAX", "240000"))
TARGET_INPUT_TOKENS_PER_REQ = int(os.getenv("AI_TARGET_INPUT_TOKENS_PER_REQ", "120000"))
MAX_ITEMS_PER_CALL = int(os.getenv("AI_MAX_ITEMS_PER_CALL", "512"))
CONCURRENCY_MAX = int(os.getenv("AI_CONCURRENCY_MAX", "64"))
TIMEOUT_REQUEST_SEC = float(os.getenv("AI_TIMEOUT_REQUEST_SEC", "45"))
MAX_RETRIES = int(os.getenv("AI_MAX_RETRIES", "2"))
DB_WRITE_BATCH = int(os.getenv("AI_DB_WRITE_BATCH", "500"))
DB_WRITE_INTERVAL = float(os.getenv("AI_DB_WRITE_INTERVAL", "0.5"))
GPT_URL = os.getenv("AI_GPT_URL") or os.getenv("GPT_URL")

GPT_API_KEY = config.get_api_key() or os.environ.get("OPENAI_API_KEY")
GPT_EXTRA_HEADERS_RAW = os.getenv("AI_GPT_HEADERS")
GPT_MODEL_NAME = os.getenv("AI_GPT_MODEL")

if not GPT_URL:
    GPT_URL = "https://api.openai.com/v1/chat/completions"

_GPT_HEADERS: Dict[str, str] = {}
if GPT_API_KEY:
    _GPT_HEADERS["Authorization"] = f"Bearer {GPT_API_KEY}"
if GPT_EXTRA_HEADERS_RAW:
    try:
        extra_headers = json.loads(GPT_EXTRA_HEADERS_RAW)
        if isinstance(extra_headers, dict):
            _GPT_HEADERS.update({str(k): str(v) for k, v in extra_headers.items()})
    except json.JSONDecodeError:
        logger.warning("Invalid AI_GPT_HEADERS JSON; ignoring")


def _tok_item(it: Dict[str, Any]) -> int:
    title = (it.get("title", "")[:160])
    desc = (it.get("description", "")[:400])
    brand = (it.get("brand", "")[:40])
    cat = (it.get("category", "")[:60])
    nums = f"{it.get('price', '')}{it.get('rating', '')}{it.get('units_sold', '')}{it.get('revenue', '')}{it.get('oldness', '')}"
    chars = len(title) + len(desc) + len(brand) + len(cat) + len(nums)
    return max(28, chars // 4)


def _tok_req(items: List[Dict[str, Any]]) -> int:
    return 600 + sum(_tok_item(x) for x in items)


def _best_K(items: List[Dict[str, Any]], cap: int) -> int:
    if not items:
        return 0
    sample = items[: min(64, len(items))]
    avg = max(28, sum(_tok_item(x) for x in sample) / len(sample))
    budget = min(TARGET_INPUT_TOKENS_PER_REQ, REQ_TOKEN_HARD_MAX)
    avail = max(0, budget - 600)
    k = int(avail // avg) if avg else cap
    return max(1, min(cap, k))


def _chunk_maximal(items: List[Dict[str, Any]], K: int) -> List[List[Dict[str, Any]]]:
    out: List[List[Dict[str, Any]]] = []
    i = 0
    limit = min(TARGET_INPUT_TOKENS_PER_REQ, REQ_TOKEN_HARD_MAX)
    while i < len(items):
        sub = items[i : i + K]
        while len(sub) > 1 and _tok_req(sub) > limit:
            mid = len(sub) // 2
            left = sub[:mid]
            right = sub[mid:]
            if _tok_req(left) <= limit:
                out.append(left)
                sub = right
            else:
                sub = left
        out.append(sub)
        i += K
    return [c for c in out if c]


def _build_payload(batch: List[Dict[str, Any]], weights: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    items = [
        {
            "id": str(it["id"]),
            "title": (it.get("title", "")[:160]),
            "category": (it.get("category", "")[:60]),
            "brand": (it.get("brand", "")[:40]),
            "price": it.get("price"),
            "rating": it.get("rating"),
            "units_sold": it.get("units_sold"),
            "revenue": it.get("revenue"),
            "oldness": it.get("oldness"),
            "desc": (it.get("description", "")[:400]),
        }
        for it in batch
    ]
    sys = {"role": "system", "content": "Eres estricto. Devuelves SOLO JSON válido."}
    usr_content = (
        "Devuelve SOLO JSON: {\"results\":[{\"id\":\"<id>\"," \
        "\"desire\":0-100,\"desire_magnitude\":0-100," \
        "\"awareness_level\":0-100,\"competition_level\":0-100," \
        "\"winner_score\":0-100}]}\n"
        f"Items: {json.dumps(items, ensure_ascii=False)}\n"
    )
    if weights:
        usr_content += f"Weights: {json.dumps(weights)}"
    usr = {"role": "user", "content": usr_content}
    payload = {"messages": [sys, usr]}
    model_name = GPT_MODEL_NAME
    if model_name:
        payload.setdefault("model", model_name)
    return payload


def _clamp(x: Any) -> int:
    try:
        v = int(round(float(x)))
    except Exception:
        v = 0
    return max(0, min(100, v))


def _parse(json_obj: Any) -> Dict[str, Dict[str, int]]:
    res: Dict[str, Dict[str, int]] = {}
    arr = json_obj.get("results") if isinstance(json_obj, dict) else None
    if not isinstance(arr, list):
        return res
    for e in arr:
        pid = str(e.get("id", "")).strip()
        if not pid:
            continue
        res[pid] = {
            "desire": _clamp(e.get("desire")),
            "desire_magnitude": _clamp(e.get("desire_magnitude")),
            "awareness_level": _clamp(e.get("awareness_level")),
            "competition_level": _clamp(e.get("competition_level")),
            "winner_score": _clamp(e.get("winner_score")),
        }
    return res


class _TokenBucket:
    def __init__(self, capacity: int, refill_per_minute: int) -> None:
        if capacity <= 0 or refill_per_minute <= 0:
            self.capacity = float("inf")
            self.refill_per_sec = float("inf")
        else:
            self.capacity = float(capacity)
            self.refill_per_sec = float(refill_per_minute) / 60.0
        self.tokens = float(self.capacity if self.capacity != float("inf") else 0)
        self.ts = time.monotonic()
        self.lock = asyncio.Lock()

    async def consume(self, amount: float) -> None:
        if self.capacity == float("inf"):
            return
        need = max(0.0, float(amount))
        if need <= 0:
            return
        while True:
            async with self.lock:
                now = time.monotonic()
                delta = now - self.ts
                self.ts = now
                self.tokens = min(self.capacity, self.tokens + delta * self.refill_per_sec)
                if self.tokens >= need:
                    self.tokens -= need
                    return
                deficit = need - self.tokens
            wait = deficit / self.refill_per_sec if self.refill_per_sec > 0 else 0.05
            await asyncio.sleep(max(wait, 0.01))


class _AsyncDBWriter:
    def __init__(self, conn) -> None:
        self.conn = conn

    def upsert_ai_fields_bulk(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        cur = self.conn.cursor()
        now = datetime.utcnow().isoformat()
        updates: List[tuple] = []
        for row in rows:
            pid = row.get("id")
            if pid is None:
                continue
            desire = row.get("desire")
            desire_mag = row.get("desire_magnitude")
            awareness = row.get("awareness_level")
            competition = row.get("competition_level")
            updates.append(
                (
                    desire,
                    desire_mag,
                    awareness,
                    competition,
                    now,
                    pid,
                )
            )
        if not updates:
            return
        cur.executemany(
            """
            UPDATE products
            SET desire = COALESCE(?, desire),
                desire_magnitude = COALESCE(?, desire_magnitude),
                awareness_level = COALESCE(?, awareness_level),
                competition_level = COALESCE(?, competition_level),
                ai_columns_completed_at = ?
            WHERE id = ?
            """,
            updates,
        )
        self.conn.commit()


async def _writer_loop(q: asyncio.Queue, dao: Optional[_AsyncDBWriter], logger: logging.Logger) -> None:
    if dao is None:
        # Drain queue without writing if dao missing.
        while True:
            try:
                batch = await asyncio.wait_for(q.get(), timeout=DB_WRITE_INTERVAL)
            except asyncio.TimeoutError:
                continue
            if batch is None:
                break
            q.task_done()
        return

    buf: List[Dict[str, Any]] = []
    last = time.monotonic()
    while True:
        try:
            batch = await asyncio.wait_for(q.get(), timeout=DB_WRITE_INTERVAL)
        except asyncio.TimeoutError:
            batch = None
        if batch:
            buf.extend(batch)
        if batch is not None:
            q.task_done()
        now = time.monotonic()
        if buf and (len(buf) >= DB_WRITE_BATCH or now - last >= DB_WRITE_INTERVAL):
            try:
                dao.upsert_ai_fields_bulk(buf)
            except Exception:
                logger.exception("DB writer flush failed")
            buf.clear()
            last = now


async def _call_gpt(client: httpx.AsyncClient, payload: Dict[str, Any]) -> Any:
    if not GPT_URL:
        raise RuntimeError("AI_GPT_URL not configured")
    response = await client.post(GPT_URL, json=payload, timeout=TIMEOUT_REQUEST_SEC)
    response.raise_for_status()
    try:
        return response.json()
    except Exception:
        return json.loads(response.text)


async def _run_batches_parallel(
    batches: List[List[Dict[str, Any]]],
    dao: Optional[_AsyncDBWriter],
    weights: Optional[Dict[str, Any]],
    logger: logging.Logger,
    results_store: Optional[Dict[str, Dict[str, int]]] = None,
    failure_reasons: Optional[Dict[str, str]] = None,
    retry_counter: Optional[Dict[str, int]] = None,
) -> int:
    if not batches:
        return 0
    conc = max(1, min(CONCURRENCY_MAX, len(batches)))
    rpm_bucket = _TokenBucket(RPM_LIMIT, RPM_LIMIT)
    tpm_bucket = _TokenBucket(TPM_LIMIT, TPM_LIMIT)
    limits = httpx.Limits(max_connections=conc, max_keepalive_connections=conc)
    headers = _GPT_HEADERS or None
    timeout = httpx.Timeout(TIMEOUT_REQUEST_SEC)
    async with httpx.AsyncClient(http2=True, limits=limits, headers=headers, timeout=timeout) as client:
        out_q: asyncio.Queue = asyncio.Queue()
        writer = asyncio.create_task(_writer_loop(out_q, dao, logger))

        async def one(batch: List[Dict[str, Any]], idx: int, attempt: int = 0) -> bool:
            t0 = time.monotonic()
            try:
                tokens_est = max(600, _tok_req(batch))
                await rpm_bucket.consume(1)
                await tpm_bucket.consume(tokens_est)
                payload = _build_payload(batch, weights)
                obj = await _call_gpt(client, payload)
                parsed = _parse(obj)
                rows: List[Dict[str, Any]] = []
                for it in batch:
                    pid = str(it["id"])
                    if pid in parsed:
                        vals = parsed[pid]
                        if results_store is not None:
                            results_store[pid] = vals
                        rows.append(
                            {
                                "id": it["id"],
                                "desire": vals["desire"],
                                "desire_magnitude": vals["desire_magnitude"],
                                "awareness_level": vals["awareness_level"],
                                "competition_level": vals["competition_level"],
                                "winner_score": vals["winner_score"],
                            }
                        )
                    elif failure_reasons is not None:
                        failure_reasons[pid] = "missing_result"
                if rows:
                    await out_q.put(rows)
                logger.info(
                    "IA batch %d OK: items=%d in %.2fs",
                    idx,
                    len(batch),
                    time.monotonic() - t0,
                )
                return True
            except httpx.HTTPStatusError as exc:
                response = exc.response
                retry_after = response.headers.get("Retry-After") if response else None
                try:
                    wait = float(retry_after) if retry_after is not None else None
                except (TypeError, ValueError):
                    wait = None
                if wait is None:
                    wait = min(30, 2 ** min(5, attempt))
                status = response.status_code if response else "?"
                logger.warning("IA batch %d HTTP %s, retry in %.1fs", idx, status, wait)
                if retry_counter is not None and attempt < MAX_RETRIES:
                    retry_counter["count"] = retry_counter.get("count", 0) + 1
                await asyncio.sleep(wait)
                if attempt < MAX_RETRIES:
                    return await one(batch, idx, attempt + 1)
                logger.error("IA batch %d FAILED after %d retries", idx, attempt)
                if failure_reasons is not None:
                    reason = f"http_{status}" if status else "http_error"
                    for it in batch:
                        failure_reasons[str(it["id"])]=reason
                return False
            except Exception as exc:
                logger.exception("IA batch %d error: %s", idx, exc)
                if retry_counter is not None and attempt < MAX_RETRIES:
                    retry_counter["count"] = retry_counter.get("count", 0) + 1
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1.0)
                    return await one(batch, idx, attempt + 1)
                if failure_reasons is not None:
                    for it in batch:
                        failure_reasons[str(it["id"])]="exception"
                return False

        tasks = [asyncio.create_task(one(batch, idx)) for idx, batch in enumerate(batches, start=1)]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        await asyncio.sleep(DB_WRITE_INTERVAL * 1.5)
        writer.cancel()
        try:
            await writer
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("DB writer terminated with error")

    return sum(1 for r in results if r)


def _ensure_conn():
    conn = database.get_connection(DB_PATH)
    database.initialize_database(conn)
    return conn


def _parse_score(val: Any) -> Optional[float]:
    """Parse a score which may be a string label or numeric value.

    Returns a float in [0, 1] or ``None`` if parsing fails."""

    if val is None:
        return None
    if isinstance(val, (int, float)):
        num = float(val)
        if 0 <= num <= 1:
            return num
        if 0 <= num <= 100:
            return num / 100.0
        return None
    if isinstance(val, str):
        txt = val.strip().lower()
        m = re.search(r"\d+(?:\.\d+)?", txt)
        if m:
            try:
                num = float(m.group())
                if num > 1:
                    num /= 100.0
                if 0 <= num <= 1:
                    return num
            except Exception:
                pass
        if txt.startswith("low"):
            return 0.2
        if txt.startswith("med"):
            return 0.5
        if txt.startswith("high"):
            return 0.8
    return None


def _quantile(data: List[float], q: float) -> float:
    """Return the q-th quantile of data using linear interpolation."""

    if not data:
        return 0.0
    s = sorted(data)
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[int(pos)]
    return s[lo] * (hi - pos) + s[hi] * (pos - lo)


def _format_cost_message(cost: float) -> str:
    if cost >= 0.1:
        txt = f"{cost:.2f}"
    else:
        txt = f"{cost:.4f}"
    if "." in txt:
        txt = txt.rstrip("0").rstrip(".")
    return f"importando productos, por favor espere... El coste será de {txt}$"


def _classify_scores(
    pairs: List[tuple[str, float]],
    *,
    winsorize_pct: float,
    min_low_pct: float,
    min_medium_pct: float,
    min_high_pct: float,
) -> tuple[Dict[str, str], Dict[str, int], Dict[str, Any]]:
    labels: Dict[str, str] = {}
    dist = {"Low": 0, "Medium": 0, "High": 0}
    info: Dict[str, Any] = {"q33": None, "q67": None, "fallback": False, "moved_medium": 0, "moved_low": 0, "moved_high": 0}

    if not pairs:
        return labels, dist, info

    values = [s for _, s in pairs]
    n = len(values)
    distinct = len(set(values))

    if n >= 50 and winsorize_pct > 0:
        low_lim = _quantile(values, winsorize_pct)
        high_lim = _quantile(values, 1 - winsorize_pct)
        values = [min(max(s, low_lim), high_lim) for s in values]
        pairs = [(pid, min(max(score, low_lim), high_lim)) for pid, score in pairs]

    q33 = _quantile(values, 1 / 3)
    q67 = _quantile(values, 2 / 3)
    info["q33"] = q33
    info["q67"] = q67

    if n >= 6 and distinct >= 3 and abs(q67 - q33) > 1e-6:
        for pid, score in pairs:
            if score <= q33:
                lab = "Low"
            elif score >= q67:
                lab = "High"
            else:
                lab = "Medium"
            labels[pid] = lab
            dist[lab] += 1
    else:
        info["fallback"] = True
        sorted_pairs = sorted(pairs, key=lambda x: x[1])
        cut1 = round(n / 3)
        cut2 = round(2 * n / 3)
        for idx, (pid, _) in enumerate(sorted_pairs):
            if idx < cut1:
                lab = "Low"
            elif idx < cut2:
                lab = "Medium"
            else:
                lab = "High"
            labels[pid] = lab
            dist[lab] += 1

    min_medium = math.ceil(min_medium_pct * n)
    min_low = math.ceil(min_low_pct * n)
    min_high = math.ceil(min_high_pct * n)

    if dist["Medium"] < min_medium:
        need = min_medium - dist["Medium"]
        candidates = [(abs(score - 0.5), pid) for pid, score in pairs if labels[pid] != "Medium"]
        candidates.sort()
        for _, pid in candidates[:need]:
            prev = labels[pid]
            labels[pid] = "Medium"
            dist["Medium"] += 1
            dist[prev] -= 1
            info["moved_medium"] += 1

    available = max(0, dist["Medium"] - min_medium)
    if dist["Low"] < min_low and available > 0:
        need = min(min_low - dist["Low"], available)
        candidates = [
            (abs(score - q33), pid)
            for pid, score in pairs
            if labels[pid] == "Medium"
        ]
        candidates.sort()
        for _, pid in candidates[:need]:
            labels[pid] = "Low"
            dist["Low"] += 1
            dist["Medium"] -= 1
            info["moved_low"] += 1
        available = max(0, dist["Medium"] - min_medium)

    if dist["High"] < min_high and available > 0:
        need = min(min_high - dist["High"], available)
        candidates = [
            (abs(score - q67), pid)
            for pid, score in pairs
            if labels[pid] == "Medium"
        ]
        candidates.sort()
        for _, pid in candidates[:need]:
            labels[pid] = "High"
            dist["High"] += 1
            dist["Medium"] -= 1
            info["moved_high"] += 1

    return labels, dist, info


def fill_ai_columns(
    product_ids: List[int],
    *,
    model: str | None = None,
    batch_mode: bool | None = None,
    cost_cap_usd: float | None = None,
) -> Dict[str, Any]:
    conn = _ensure_conn()

    cfg_cost = config.get_ai_cost_config()
    api_key = config.get_api_key() or os.environ.get("OPENAI_API_KEY")
    model = model or cfg_cost.get("model")
    batch_mode = batch_mode if batch_mode is not None else (len(product_ids) >= cfg_cost.get("useBatchWhenCountGte", 300))
    cost_cap_usd = cost_cap_usd if cost_cap_usd is not None else cfg_cost.get("costCapUSD")

    global GPT_MODEL_NAME, _GPT_HEADERS
    if model:
        GPT_MODEL_NAME = model

    if api_key:
        _GPT_HEADERS["Authorization"] = f"Bearer {api_key}"

    dao = _AsyncDBWriter(conn)

    total_requested = len(product_ids)
    skipped_existing = 0
    to_process: List[Dict[str, Any]] = []
    selected_ids: List[int] = []
    records: Dict[str, Any] = {}
    now_ts = datetime.utcnow().isoformat()

    for pid in product_ids:
        rec = database.get_product(conn, pid)
        if not rec:
            continue
        if rec["ai_columns_completed_at"]:
            skipped_existing += 1
            continue
        if rec["desire"] or rec["desire_magnitude"] or rec["awareness_level"] or rec["competition_level"]:
            skipped_existing += 1
            database.update_product(conn, pid, ai_columns_completed_at=now_ts)
            continue
        try:
            extra = json.loads(rec["extra"]) if rec["extra"] else {}
        except Exception:
            extra = {}
        selected_ids.append(rec["id"])
        item = {
            "id": rec["id"],
            "name": rec["name"],
            "category": rec["category"],
            "price": rec["price"],
            "rating": extra.get("rating"),
            "units_sold": extra.get("units_sold"),
            "revenue": extra.get("revenue"),
            "conversion_rate": extra.get("conversion_rate"),
            "launch_date": extra.get("launch_date"),
            "date_range": rec["date_range"],
            "image_url": rec["image_url"],
        }
        to_process.append(item)
        records[str(rec["id"])] = rec

    count = len(to_process)
    est_in = count * cfg_cost.get("estTokensPerItemIn", 0)
    est_out = count * cfg_cost.get("estTokensPerItemOut", 0)
    price_map = cfg_cost.get("prices", {}).get(model, {})
    price_in = price_map.get("input", 0.15)
    price_out = price_map.get("output", 0.6)
    cost_estimated = (est_in / 1_000_000) * price_in + (est_out / 1_000_000) * price_out
    truncated = False
    pending_ids: List[int] = []

    if cost_cap_usd is not None and cost_estimated > cost_cap_usd:
        per_item_cost = ((cfg_cost.get("estTokensPerItemIn", 0) / 1_000_000) * price_in + (cfg_cost.get("estTokensPerItemOut", 0) / 1_000_000) * price_out)
        max_items = int(cost_cap_usd // per_item_cost) if per_item_cost > 0 else 0
        to_process = to_process[:max_items]
        records = {str(it["id"]): records[str(it["id"])] for it in to_process}
        pending_ids.extend(selected_ids[max_items:])
        count = len(to_process)
        est_in = count * cfg_cost.get("estTokensPerItemIn", 0)
        est_out = count * cfg_cost.get("estTokensPerItemOut", 0)
        cost_estimated = (est_in / 1_000_000) * price_in + (est_out / 1_000_000) * price_out
        truncated = True

    cost_msg = _format_cost_message(cost_estimated)

    if not api_key or not to_process:
        err_msg = "missing_api_key" if not api_key else None
        logger.info(
            "fill_ai_columns: n_importados=%s n_para_ia=0 n_procesados=0 n_omitidos_por_valor_existente=%s n_reintentados=0 n_error_definitivo=%s truncated=%s cost_estimated_usd=%.4f",
            total_requested,
            skipped_existing,
            len(to_process),
            truncated,
            cost_estimated,
        )
        return {
            "ok": {},
            "ko": {str(it["id"]): err_msg or "skipped" for it in to_process} if err_msg else {},
            "error": err_msg,
            "counts": {
                "n_importados": total_requested,
                "n_para_ia": 0,
                "n_procesados": 0,
                "n_omitidos_por_valor_existente": skipped_existing,
                "n_reintentados": 0,
                "n_error_definitivo": len(to_process) if err_msg else 0,
                "truncated": truncated,
                "cost_estimated_usd": cost_estimated,
            },
            "pending_ids": selected_ids,
            "cost_estimated_usd": cost_estimated,
            "ui_cost_message": cost_msg,
        }

    weights: Optional[Dict[str, Any]] = None
    try:
        weights = config.get_weights()
    except Exception:
        weights = None

    limit_tokens = min(TARGET_INPUT_TOKENS_PER_REQ, REQ_TOKEN_HARD_MAX)
    total_tokens = _tok_req(to_process)
    if total_tokens <= limit_tokens and count <= MAX_ITEMS_PER_CALL:
        batches = [to_process]
        approx_chunk = count
    else:
        approx_chunk = _best_K(to_process, MAX_ITEMS_PER_CALL)
        batches = _chunk_maximal(to_process, approx_chunk)
        if not batches and to_process:
            batches = [to_process]
    if batches:
        first_batch = len(batches[0])
    else:
        first_batch = 0
    logger.info(
        "AI_PARALLEL: n_items=%d total_tokens≈%d batches=%d Kmax=%s conc=%d",
        count,
        total_tokens,
        len(batches),
        str(count) if len(batches) == 1 else str(first_batch),
        min(CONCURRENCY_MAX, len(batches)) if batches else 1,
    )

    ok_raw: Dict[str, Dict[str, Any]] = {}
    ko_all: Dict[str, str] = {}
    retry_counter: Dict[str, int] = {}
    results_store: Dict[str, Dict[str, int]] = {}

    if batches:
        asyncio.run(
            _run_batches_parallel(
                batches,
                dao,
                weights,
                logger,
                results_store=results_store,
                failure_reasons=ko_all,
                retry_counter=retry_counter,
            )
        )

    for pid, vals in results_store.items():
        ok_raw[pid] = {
            "desire": vals.get("desire"),
            "desire_magnitude": vals.get("desire_magnitude"),
            "awareness_level": vals.get("awareness_level"),
            "competition_level": vals.get("competition_level"),
            "winner_score": vals.get("winner_score"),
        }

    for item in to_process:
        pid_str = str(item["id"])
        if pid_str not in ok_raw and pid_str not in ko_all:
            ko_all[pid_str] = "no_result"

    n_retried = retry_counter.get("count", 0)

    cfg_calib = config.get_ai_calibration_config()
    dist_desire = {"Low": 0, "Medium": 0, "High": 0}
    dist_comp = {"Low": 0, "Medium": 0, "High": 0}

    desire_scores: List[tuple[str, float]] = []
    comp_scores: List[tuple[str, float]] = []
    updates_final: Dict[str, Dict[str, Any]] = {}

    for pid, updates in ok_raw.items():
        rec = records.get(pid)
        if not rec:
            ko_all[pid] = "not_found"
            continue
        apply: Dict[str, Any] = {}
        if not rec["desire"] and updates.get("desire"):
            apply["desire"] = updates.get("desire")
        if not rec["awareness_level"] and updates.get("awareness_level"):
            apply["awareness_level"] = updates.get("awareness_level")
        if cfg_calib.get("enabled", True):
            if not rec["desire_magnitude"] and updates.get("desire_magnitude"):
                score = _parse_score(updates.get("desire_magnitude"))
                if score is not None:
                    updates["_desire_score"] = score
                    desire_scores.append((pid, score))
                else:
                    logger.warning("invalid desire_magnitude for %s: %s", pid, updates.get("desire_magnitude"))
            if not rec["competition_level"] and updates.get("competition_level"):
                score = _parse_score(updates.get("competition_level"))
                if score is not None:
                    updates["_competition_score"] = score
                    comp_scores.append((pid, score))
                else:
                    logger.warning("invalid competition_level for %s: %s", pid, updates.get("competition_level"))
        else:
            if not rec["desire_magnitude"] and updates.get("desire_magnitude"):
                apply["desire_magnitude"] = updates.get("desire_magnitude")
            if not rec["competition_level"] and updates.get("competition_level"):
                apply["competition_level"] = updates.get("competition_level")
        updates_final[pid] = apply

    wins_p = cfg_calib.get("winsorize_pct", 0.05)
    min_low_pct = cfg_calib.get("min_low_pct", 0.05)
    min_med_pct = cfg_calib.get("min_medium_pct", 0.05)
    min_high_pct = cfg_calib.get("min_high_pct", 0.05)

    desire_info: Dict[str, Any] = {}
    comp_info: Dict[str, Any] = {}

    if cfg_calib.get("enabled", True) and desire_scores:
        labels, dist_desire, desire_info = _classify_scores(
            desire_scores,
            winsorize_pct=wins_p,
            min_low_pct=min_low_pct,
            min_medium_pct=min_med_pct,
            min_high_pct=min_high_pct,
        )
        for pid, lab in labels.items():
            updates_final[pid]["desire_magnitude"] = lab

    if cfg_calib.get("enabled", True) and comp_scores:
        labels, dist_comp, comp_info = _classify_scores(
            comp_scores,
            winsorize_pct=wins_p,
            min_low_pct=min_low_pct,
            min_medium_pct=min_med_pct,
            min_high_pct=min_high_pct,
        )
        for pid, lab in labels.items():
            updates_final[pid]["competition_level"] = lab

    applied_ok: Dict[str, Dict[str, Any]] = {}
    success = 0
    errors = 0
    for pid, apply in updates_final.items():
        if apply:
            apply["ai_columns_completed_at"] = datetime.utcnow().isoformat()
            database.update_product(conn, int(pid), **apply)
            applied_ok[pid] = {k: v for k, v in apply.items() if k != "ai_columns_completed_at"}
            success += 1
        else:
            database.update_product(conn, int(pid), ai_columns_completed_at=datetime.utcnow().isoformat())
            ko_all[pid] = ko_all.get(pid, "existing")
            errors += 1

    conn.commit()
    logger.info(
        "fill_ai_columns: n_importados=%s n_para_ia=%s n_procesados=%s n_omitidos_por_valor_existente=%s n_reintentados=%s n_error_definitivo=%s truncated=%s cost_estimated_usd=%.4f",
        total_requested,
        len(to_process),
        success,
        skipped_existing,
        n_retried,
        errors,
        truncated,
        cost_estimated,
    )
    logger.info(
        "ai_calibration_desire: dist=%s q33=%.4f q67=%.4f fallback=%s moved_medium=%s moved_low=%s moved_high=%s",
        dist_desire,
        desire_info.get("q33") or 0.0,
        desire_info.get("q67") or 0.0,
        desire_info.get("fallback"),
        desire_info.get("moved_medium"),
        desire_info.get("moved_low"),
        desire_info.get("moved_high"),
    )
    logger.info(
        "ai_calibration_competition: dist=%s q33=%.4f q67=%.4f fallback=%s moved_medium=%s moved_low=%s moved_high=%s",
        dist_comp,
        comp_info.get("q33") or 0.0,
        comp_info.get("q67") or 0.0,
        comp_info.get("fallback"),
        comp_info.get("moved_medium"),
        comp_info.get("moved_low"),
        comp_info.get("moved_high"),
    )
    processed_ids = {int(pid) for pid in applied_ok.keys()} | {int(pid) for pid in ko_all.keys()}
    pending_ids.extend([it["id"] for it in to_process if it["id"] not in processed_ids])
    return {
        "ok": applied_ok,
        "ko": ko_all,
        "counts": {
            "n_importados": total_requested,
            "n_para_ia": len(to_process),
            "n_procesados": success,
            "n_omitidos_por_valor_existente": skipped_existing,
            "n_reintentados": n_retried,
            "n_error_definitivo": errors,
            "truncated": truncated,
            "cost_estimated_usd": cost_estimated,
        },
        "pending_ids": pending_ids,
        "cost_estimated_usd": cost_estimated,
        "ui_cost_message": cost_msg,
    }
