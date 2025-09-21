from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from openai import APIError, APIStatusError, AsyncOpenAI, RateLimitError

from .. import config, database
from ..utils.signature import compute_sig_hash

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent.parent
DB_PATH = APP_DIR / "data.sqlite3"

AI_FIELDS = ("desire", "desire_magnitude", "awareness_level", "competition_level")
StatusCallback = Callable[..., None]


@dataclass
class Candidate:
    id: int
    sig_hash: str
    payload: Dict[str, Any]
    extra: Dict[str, Any]


TRI_LEVELS = {"low": "Low", "medium": "Medium", "high": "High"}
SYSTEM_PROMPT = (
    "Eres un clasificador de productos e-commerce. Devuelve SOLO JSON válido con cuatro campos por producto."
)
LABELS_SCHEMA = {
    "desire": ["Low", "Medium", "High"],
    "desire_magnitude": ["Low", "Medium", "High"],
    "awareness_level": ["Low", "Medium", "High"],
    "competition_level": ["Low", "Medium", "High"],
}
DEFAULT_EST_TOKENS_PER_ITEM = 120
DEFAULT_EST_OUTPUT_TOKENS_PER_ITEM = 32
BRAND_MAX_CHARS = 60
CATEGORY_MAX_CHARS = 80


def _clean_text(value: Optional[str], max_chars: int) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars]


def _get(obj: Dict[str, Any], key: str) -> Any:
    return obj.get(key) or obj.get(key.replace("_level", "")) or obj.get(key.replace("_", ""))


def _normalize_tri(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return TRI_LEVELS.get(text)


class AsyncRateLimiter:
    def __init__(self, tpm_limit: Optional[int], rpm_limit: Optional[int]) -> None:
        self.tpm_limit = max(0, int(tpm_limit or 0))
        self.rpm_limit = max(0, int(rpm_limit or 0))
        self._lock = asyncio.Lock()
        self._window_start = time.monotonic()
        self._tokens_used = 0
        self._requests = 0

    async def acquire(self, tokens: int, requests: int = 1) -> None:
        if self.tpm_limit <= 0 and self.rpm_limit <= 0:
            return
        tokens = max(0, int(tokens))
        requests = max(1, int(requests))
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._window_start
                if elapsed >= 60:
                    self._window_start = now
                    self._tokens_used = 0
                    self._requests = 0
                wait_time = 0.0
                if self.tpm_limit > 0 and tokens > 0 and self._tokens_used + tokens > self.tpm_limit:
                    wait_time = max(wait_time, 60.0 - elapsed)
                if self.rpm_limit > 0 and self._requests + requests > self.rpm_limit:
                    wait_time = max(wait_time, 60.0 - elapsed)
                if wait_time <= 0:
                    self._tokens_used += tokens
                    self._requests += requests
                    return
            await asyncio.sleep(max(wait_time, 0.05))


def _estimate_input_tokens(user_content: str, item_count: int) -> int:
    payload_tokens = max(1, len(user_content) // 4)
    prompt_tokens = max(16, len(SYSTEM_PROMPT) // 4)
    return max(DEFAULT_EST_TOKENS_PER_ITEM * max(1, item_count), payload_tokens + prompt_tokens)


def _estimate_output_tokens(item_count: int) -> int:
    return DEFAULT_EST_OUTPUT_TOKENS_PER_ITEM * max(1, item_count)


def _extract_response_text(response_dict: Dict[str, Any]) -> str:
    outputs = response_dict.get("output") or []
    pieces: List[str] = []
    for block in outputs:
        if not isinstance(block, dict):
            continue
        for part in block.get("content") or []:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    pieces.append(text)
                elif isinstance(text, dict):
                    nested = text.get("text")
                    if isinstance(nested, str):
                        pieces.append(nested)
            elif isinstance(part, str):
                pieces.append(part)
    return "".join(pieces).strip()


def _normalize_desire(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) > 160:
        return text[:160]
    return text


def _parse_batch_response(
    data: Dict[str, Any], item_ids: Sequence[int]
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    if not isinstance(data, dict):
        raise ValueError("invalid_response")
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("missing_results")
    expected = {str(int(pid)) for pid in item_ids}
    successes: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}
    for entry in results:
        if not isinstance(entry, dict):
            continue
        pid = entry.get("id")
        try:
            pid_int = int(pid)
        except Exception:
            continue
        pid_str = str(pid_int)
        if pid_str not in expected or pid_str in successes or pid_str in errors:
            continue
        desire_mag = _normalize_tri(_get(entry, "desire_magnitude"))
        awareness = _normalize_tri(_get(entry, "awareness_level"))
        competition = _normalize_tri(_get(entry, "competition_level"))
        desire = _normalize_desire(entry.get("desire"))
        if desire_mag is None or awareness is None or competition is None:
            errors[pid_str] = "invalid_fields"
            continue
        successes[pid_str] = {
            "desire": desire,
            "desire_magnitude": desire_mag,
            "awareness_level": awareness,
            "competition_level": competition,
        }
    for pid_str in expected:
        if pid_str not in successes and pid_str not in errors:
            errors[pid_str] = "missing"
    return successes, errors


def _build_request_payload(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "task": "classify",
        "labels": LABELS_SCHEMA,
        "items": list(items),
    }


def _resolve_response_format(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"json", "json_object", "json-object"}:
            return {"type": "json_object"}
    return {"type": "json_object"}


async def _perform_single_request(
    client: AsyncOpenAI,
    *,
    payload_items: List[Dict[str, Any]],
    item_ids: List[int],
    model: str,
    temperature: float,
    top_p: float,
    response_format: Dict[str, Any],
    max_retries: int,
    rate_limiter: AsyncRateLimiter,
    stop_event: asyncio.Event,
) -> Dict[str, Any]:
    request_payload = _build_request_payload(payload_items)
    user_content = json.dumps(request_payload, ensure_ascii=False)
    estimated_in = _estimate_input_tokens(user_content, len(payload_items))
    estimated_out = _estimate_output_tokens(len(payload_items))
    last_error = "error"
    status_code = 0
    for attempt in range(max_retries + 1):
        if stop_event.is_set():
            return {
                "ok": {},
                "ko": {str(pid): "cost_cap_reached" for pid in item_ids},
                "usage": {},
                "duration": 0.0,
                "retries": attempt,
                "status_code": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "sent": 0,
                "skipped": True,
            }
        await rate_limiter.acquire(estimated_in, 1)
        try:
            start = time.perf_counter()
            response = await client.responses.create(
                model=model,
                temperature=temperature,
                top_p=top_p,
                response_format=response_format,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            duration = time.perf_counter() - start
            resp_dict = response.model_dump()
            usage = resp_dict.get("usage") or {}
            tokens_in = usage.get("input_tokens") or usage.get("prompt_tokens") or estimated_in
            tokens_out = usage.get("output_tokens") or usage.get("completion_tokens") or estimated_out
            raw_text = _extract_response_text(resp_dict)
            if not raw_text:
                raise ValueError("empty_response")
            parsed = json.loads(raw_text)
            ok_map, ko_map = _parse_batch_response(parsed, item_ids)
            return {
                "ok": ok_map,
                "ko": ko_map,
                "usage": usage,
                "duration": duration,
                "retries": attempt,
                "status_code": 200,
                "tokens_in": int(tokens_in),
                "tokens_out": int(tokens_out),
                "sent": len(payload_items),
                "skipped": False,
            }
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("AI batch parse error: %s", exc)
            return {
                "ok": {},
                "ko": {str(pid): "invalid_json" for pid in item_ids},
                "usage": {},
                "duration": 0.0,
                "retries": attempt,
                "status_code": 200,
                "tokens_in": estimated_in,
                "tokens_out": estimated_out,
                "sent": len(payload_items),
                "skipped": False,
            }
        except RateLimitError as exc:
            if attempt < max_retries:
                delay = min(10.0, 0.6 * (2**attempt)) + random.random() * 0.3
                await asyncio.sleep(delay)
                continue
            last_error = str(exc)
            status_code = getattr(exc, "status_code", 429) or 429
        except APIStatusError as exc:
            status_code = getattr(exc, "status_code", 0) or 0
            if status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                delay = min(10.0, 0.6 * (2**attempt)) + random.random() * 0.3
                await asyncio.sleep(delay)
                continue
            last_error = str(exc)
        except APIError as exc:
            last_error = str(exc)
            status_code = getattr(exc, "status_code", 0) or 0
        except Exception as exc:
            logger.exception("AI batch unexpected error", exc_info=True)
            last_error = str(exc)
            status_code = 0
        break
    else:
        last_error = "retry_limit"
        status_code = 0

    return {
        "ok": {},
        "ko": {str(pid): last_error or "error" for pid in item_ids},
        "usage": {},
        "duration": 0.0,
        "retries": max_retries,
        "status_code": status_code,
        "tokens_in": estimated_in,
        "tokens_out": estimated_out,
        "sent": len(payload_items),
        "skipped": False,
    }


async def _run_batch_task(
    batch_no: int,
    candidates: Sequence[Candidate],
    *,
    client: AsyncOpenAI,
    model: str,
    temperature: float,
    top_p: float,
    response_format: Dict[str, Any],
    max_retries: int,
    rate_limiter: AsyncRateLimiter,
    stop_event: asyncio.Event,
    semaphore: asyncio.Semaphore,
    results_queue: "asyncio.Queue[Dict[str, Any]]",
) -> None:
    item_ids = [cand.id for cand in candidates]
    payload_items = [cand.payload for cand in candidates]
    async with semaphore:
        if stop_event.is_set():
            await results_queue.put(
                {
                    "batch_no": batch_no,
                    "item_ids": item_ids,
                    "ok": {},
                    "ko": {str(pid): "cost_cap_reached" for pid in item_ids},
                    "usage": {},
                    "duration": 0.0,
                    "retries": 0,
                    "status_code": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "sent": 0,
                    "skipped": True,
                }
            )
            return
        try:
            result = await _perform_single_request(
                client,
                payload_items=payload_items,
                item_ids=item_ids,
                model=model,
                temperature=temperature,
                top_p=top_p,
                response_format=response_format,
                max_retries=max_retries,
                rate_limiter=rate_limiter,
                stop_event=stop_event,
            )
        except Exception:
            logger.exception("AI batch task failed", exc_info=True)
            result = {
                "ok": {},
                "ko": {str(pid): "error" for pid in item_ids},
                "usage": {},
                "duration": 0.0,
                "retries": 0,
                "status_code": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "sent": 0,
                "skipped": False,
            }
        result["batch_no"] = batch_no
        result["item_ids"] = item_ids
        await results_queue.put(result)


def _run_async(coro: "asyncio.Future[Any]") -> Any:
    try:
        return asyncio.run(coro)
    except RuntimeError as exc:
        if "running event loop" in str(exc):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        raise


def _ensure_conn():
    conn = database.get_connection(DB_PATH)
    database.initialize_database(conn)
    return conn


def _parse_score(val: Any) -> Optional[float]:
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
        m = math.nan
        try:
            for part in txt.replace("%", " ").split():
                if not part:
                    continue
                num = float(part)
                m = num
                break
        except Exception:
            m = math.nan
        if not math.isnan(m):
            if m > 1:
                m /= 100.0
            if 0 <= m <= 1:
                return m
        if txt.startswith("low"):
            return 0.2
        if txt.startswith("med"):
            return 0.5
        if txt.startswith("high"):
            return 0.8
    return None


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
    info: Dict[str, Any] = {
        "q33": None,
        "q67": None,
        "fallback": False,
        "moved_medium": 0,
        "moved_low": 0,
        "moved_high": 0,
        "skipped": False,
    }

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

    if n < 6 or distinct < 3 or abs(q67 - q33) <= 1e-6:
        info["skipped"] = True
        for _, score in pairs:
            if score <= q33:
                dist["Low"] += 1
            elif score >= q67:
                dist["High"] += 1
            else:
                dist["Medium"] += 1
        return labels, dist, info

    for pid, score in pairs:
        if score <= q33:
            lab = "Low"
        elif score >= q67:
            lab = "High"
        else:
            lab = "Medium"
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


def _quantile(data: List[float], q: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[int(pos)]
    return s[lo] * (hi - pos) + s[hi] * (pos - lo)


def _calculate_cost(usage: Dict[str, Any], price_in: float, price_out: float) -> float:
    prompt = usage.get("prompt_tokens")
    if prompt is None:
        prompt = usage.get("input_tokens") or usage.get("tokens_in")
    if prompt is None:
        prompt = usage.get("total_tokens")
    completion = usage.get("completion_tokens")
    if completion is None:
        completion = usage.get("output_tokens") or usage.get("tokens_out")
    if completion is None and prompt is not None:
        try:
            total = float(usage.get("total_tokens", 0))
            completion = max(0.0, total - float(prompt))
        except Exception:
            completion = 0.0
    prompt_val = float(prompt or 0.0)
    completion_val = float(completion or 0.0)
    return (prompt_val / 1_000_000.0) * price_in + (completion_val / 1_000_000.0) * price_out


def _apply_ai_updates(conn, updates: Dict[int, Dict[str, Any]]) -> None:
    if not updates:
        return
    now_iso = datetime.utcnow().isoformat()
    cur = conn.cursor()
    for product_id, payload in updates.items():
        assignments: List[str] = []
        params: List[Any] = []
        for field in AI_FIELDS:
            if field in payload and payload[field] is not None:
                assignments.append(f"{field}=?")
                params.append(payload[field])
        assignments.append("ai_columns_completed_at=?")
        params.append(now_iso)
        params.append(int(product_id))
        cur.execute(
            f"UPDATE products SET {', '.join(assignments)} WHERE id=?",
            params,
        )
    conn.commit()


def _build_payload(row: Any, extra: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    max_title_default = int(config.AI_CFG_DEFAULTS.get("max_title_chars", 120))
    max_desc_default = int(config.AI_CFG_DEFAULTS.get("max_desc_chars", 220))
    max_title = int(cfg.get("max_title_chars") or max_title_default)
    max_desc = int(cfg.get("max_desc_chars") or max_desc_default)
    title_source = (
        extra.get("title")
        or extra.get("name")
        or row.get("title")
        or row.get("name")
    )
    desc_source = (
        row.get("description")
        or extra.get("description")
        or extra.get("descripcion")
        or extra.get("short_description")
        or extra.get("product_description")
        or extra.get("desc")
    )
    brand_source = extra.get("brand") or extra.get("marca") or extra.get("seller")
    category_source = (
        row.get("category")
        or extra.get("category")
        or extra.get("categoria")
        or extra.get("cat")
    )
    return {
        "id": row["id"],
        "title": _clean_text(title_source or row.get("name"), max_title),
        "desc": _clean_text(desc_source, max_desc),
        "brand": _clean_text(brand_source, BRAND_MAX_CHARS),
        "category": _clean_text(category_source, CATEGORY_MAX_CHARS),
    }


def _emit_status(
    callback: Optional[StatusCallback],
    *,
    phase: str,
    counts: Dict[str, Any],
    total: int,
    done: int,
    message: Optional[str] = None,
) -> None:
    if callback is None:
        return
    payload = {
        "phase": phase,
        "ai_counts": counts,
        "ai_total": total,
        "ai_done": done,
        "pct_ai": int(round((done / max(total, 1)) * 100)) if total else 100,
        "state": "done",
    }
    if message:
        payload["message"] = message
    try:
        callback(**payload)
    except Exception:
        logger.debug("status callback failed", exc_info=True)



def run_ai_fill_job(
    job_id: int,
    product_ids: Sequence[int],
    *,
    microbatch: int = 32,
    parallelism: Optional[int] = None,
    status_cb: Optional[StatusCallback] = None,
) -> Dict[str, Any]:
    start_ts = time.perf_counter()
    conn = _ensure_conn()
    runtime_cfg = config.get_ai_runtime_config()

    if parallelism is None:
        parallelism = runtime_cfg.get("parallelism", 8)
    try:
        parallelism = int(parallelism)
    except Exception:
        parallelism = int(runtime_cfg.get("parallelism", 1) or 1)
    parallelism = max(1, parallelism)

    micro_default = runtime_cfg.get("microbatch", microbatch)
    try:
        microbatch_size = int(microbatch or micro_default or 1)
    except Exception:
        microbatch_size = int(micro_default or 1)
    microbatch_size = max(1, microbatch_size)

    cache_enabled = bool(runtime_cfg.get("cache_enabled", True))
    cache_version = int(runtime_cfg.get("version", 1) or 1)

    tpm_limit_raw = runtime_cfg.get("tpm_limit")
    try:
        tpm_limit = None if tpm_limit_raw in (None, "") else int(tpm_limit_raw)
    except Exception:
        tpm_limit = None

    rpm_limit_raw = runtime_cfg.get("rpm_limit")
    try:
        rpm_limit = None if rpm_limit_raw in (None, "") else int(rpm_limit_raw)
    except Exception:
        rpm_limit = None

    batch_cfg = config.get_ai_batch_config()
    max_retries = int(batch_cfg.get("MAX_RETRIES", 3) or 3)

    cost_cfg = config.get_ai_cost_config()
    model = runtime_cfg.get("model") or cost_cfg.get("model") or config.get_model()
    try:
        temperature = float(runtime_cfg.get("temperature", 0) or 0.0)
    except Exception:
        temperature = 0.0
    try:
        top_p = float(runtime_cfg.get("top_p", 0) or 0.0)
    except Exception:
        top_p = 0.0
    response_format = _resolve_response_format(runtime_cfg.get("response_format"))

    cost_cap_val = runtime_cfg.get("costCapUSD")
    if cost_cap_val is None:
        cost_cap_val = cost_cfg.get("costCapUSD")
    if cost_cap_val is not None:
        try:
            cost_cap = float(cost_cap_val)
        except Exception:
            cost_cap = None
    else:
        cost_cap = None

    price_map = cost_cfg.get("prices", {}).get(model, {})
    price_in = float(price_map.get("input", 0.0))
    price_out = float(price_map.get("output", 0.0))
    est_tokens_in = int(cost_cfg.get("estTokensPerItemIn", DEFAULT_EST_TOKENS_PER_ITEM) or DEFAULT_EST_TOKENS_PER_ITEM)
    est_tokens_out = int(cost_cfg.get("estTokensPerItemOut", DEFAULT_EST_OUTPUT_TOKENS_PER_ITEM) or DEFAULT_EST_OUTPUT_TOKENS_PER_ITEM)

    api_key = config.get_api_key() or os.environ.get("OPENAI_API_KEY")

    job_updates_enabled = job_id is not None and int(job_id) > 0

    requested_ids: List[int] = []
    seen_ids: set[int] = set()
    for pid in product_ids:
        try:
            num = int(pid)
        except Exception:
            continue
        if num in seen_ids:
            continue
        seen_ids.add(num)
        requested_ids.append(num)

    rows = database.get_products_by_ids(conn, requested_ids)
    row_map = {int(row["id"]): dict(row) for row in rows}

    candidates: List[Candidate] = []
    sig_updates: List[tuple[str, int]] = []
    skipped_existing = 0

    for pid in requested_ids:
        row = row_map.get(pid)
        if row is None:
            skipped_existing += 1
            continue
        extra: Dict[str, Any] = {}
        if row["extra"]:
            try:
                extra = json.loads(row["extra"])
            except Exception:
                extra = {}
        already_done = row.get("ai_columns_completed_at")
        existing = {field: row.get(field) for field in AI_FIELDS}
        if already_done and all(existing.get(field) for field in AI_FIELDS):
            skipped_existing += 1
            continue
        name = row.get("name")
        if not name:
            skipped_existing += 1
            continue
        brand = extra.get("brand")
        asin = extra.get("asin")
        product_url = extra.get("product_url")
        sig_hash = row.get("sig_hash") or compute_sig_hash(name, brand, asin, product_url)
        if sig_hash and not row.get("sig_hash"):
            sig_updates.append((sig_hash, pid))
        payload = _build_payload(row, extra, runtime_cfg)
        candidates.append(Candidate(id=pid, sig_hash=sig_hash, payload=payload, extra=extra))

    if sig_updates:
        cur = conn.cursor()
        for sig_hash, pid in sig_updates:
            cur.execute("UPDATE products SET sig_hash=? WHERE id=?", (sig_hash, pid))
        conn.commit()

    total_items = len(candidates)

    if tpm_limit and tpm_limit > 0:
        est_per_item = max(DEFAULT_EST_TOKENS_PER_ITEM, est_tokens_in or DEFAULT_EST_TOKENS_PER_ITEM)
        per_request = est_per_item * microbatch_size
        if per_request > tpm_limit:
            microbatch_size = max(1, tpm_limit // max(est_per_item, 1))
        if microbatch_size <= 0:
            microbatch_size = 1
        per_request = est_per_item * microbatch_size
        if per_request > 0:
            max_req_per_min = max(1, tpm_limit // per_request)
            parallelism = max(1, min(parallelism, max_req_per_min))
    if rpm_limit and rpm_limit > 0:
        parallelism = max(1, min(parallelism, rpm_limit))

    if job_updates_enabled:
        database.start_import_job_ai(conn, int(job_id), total_items)

    counts: Dict[str, int] = {
        "queued": total_items,
        "sent": 0,
        "ok": 0,
        "ko": 0,
        "cached": 0,
        "retried": 0,
    }
    cost_spent = 0.0
    pending_set: set[int] = {cand.id for cand in candidates}
    counts_with_cost: Dict[str, Any] = {**counts, "cost_spent_usd": cost_spent}

    if job_updates_enabled:
        database.set_import_job_ai_counts(conn, int(job_id), counts_with_cost, sorted(pending_set))
        database.update_import_job_ai_progress(conn, int(job_id), 0)
    _emit_status(status_cb, phase="enrich", counts=counts_with_cost, total=total_items, done=0)

    applied_outputs: Dict[int, Dict[str, Any]] = {}
    fail_reasons: Dict[int, str] = {}

    if total_items == 0:
        if job_updates_enabled and skipped_existing:
            database.set_import_job_ai_counts(conn, int(job_id), counts_with_cost, [])
        _emit_status(status_cb, phase="enrich", counts=counts_with_cost, total=total_items, done=0)
        conn.close()
        return {
            "counts": counts_with_cost,
            "pending_ids": [],
            "error": None,
            "ok": applied_outputs,
            "ko": fail_reasons,
            "skipped_existing": skipped_existing,
            "total_requested": len(requested_ids),
        }

    candidate_map = {cand.id: cand for cand in candidates}

    cache_rows: Dict[str, Any] = {}
    if cache_enabled:
        sig_hashes = [cand.sig_hash for cand in candidates if cand.sig_hash]
        if sig_hashes:
            cache_rows = database.get_ai_cache_entries(
                conn,
                sig_hashes,
                model=model,
                version=cache_version,
            )

    remaining: List[Candidate] = []
    if cache_rows:
        cached_updates: Dict[int, Dict[str, Any]] = {}
        for cand in candidates:
            cache_row = cache_rows.get(cand.sig_hash)
            if not cache_row:
                remaining.append(cand)
                continue
            update_payload = {
                "desire": cache_row["desire"],
                "desire_magnitude": cache_row["desire_magnitude"],
                "awareness_level": cache_row["awareness_level"],
                "competition_level": cache_row["competition_level"],
            }
            cached_updates[cand.id] = update_payload
            applied_outputs[cand.id] = {k: v for k, v in update_payload.items() if v is not None}
            if cand.sig_hash:
                database.upsert_ai_cache_entry(
                    conn,
                    cand.sig_hash,
                    model=model,
                    version=cache_version,
                    desire=update_payload.get("desire"),
                    desire_magnitude=update_payload.get("desire_magnitude"),
                    awareness_level=update_payload.get("awareness_level"),
                    competition_level=update_payload.get("competition_level"),
                )
            pending_set.discard(cand.id)
            counts["cached"] += 1
        if cached_updates:
            _apply_ai_updates(conn, cached_updates)
        counts_with_cost = {**counts, "cost_spent_usd": cost_spent}
        done_val = counts["ok"] + counts["cached"]
        if job_updates_enabled:
            database.update_import_job_ai_progress(conn, int(job_id), done_val)
            database.set_import_job_ai_counts(conn, int(job_id), counts_with_cost, sorted(pending_set))
        _emit_status(
            status_cb,
            phase="enrich",
            counts=counts_with_cost,
            total=total_items,
            done=done_val,
            message=f"IA columnas {done_val}/{total_items}",
        )
    else:
        remaining = candidates

    if not api_key:
        result_error = "missing_api_key"
        for pid in list(pending_set):
            fail_reasons[pid] = result_error
        counts_with_cost = {**counts, "cost_spent_usd": cost_spent}
        if job_updates_enabled:
            database.set_import_job_ai_counts(conn, int(job_id), counts_with_cost, sorted(pending_set))
            database.set_import_job_ai_error(conn, int(job_id), result_error)
        _emit_status(
            status_cb,
            phase="enrich",
            counts=counts_with_cost,
            total=total_items,
            done=counts["cached"],
            message="IA pendiente",
        )
        conn.close()
        return {
            "counts": counts_with_cost,
            "pending_ids": sorted(pending_set),
            "error": result_error,
            "ok": applied_outputs,
            "ko": fail_reasons,
            "skipped_existing": skipped_existing,
            "total_requested": len(requested_ids),
        }

    success_records: Dict[int, Dict[str, Any]] = {}
    desire_scores: List[Tuple[str, float]] = []
    comp_scores: List[Tuple[str, float]] = []

    batches: List[List[Candidate]] = []
    for idx in range(0, len(remaining), microbatch_size):
        chunk = remaining[idx : idx + microbatch_size]
        if chunk:
            batches.append(chunk)

    stop_event = asyncio.Event()

    async def _process_batches() -> Tuple[float, bool]:
        if not batches:
            return 0.0, False
        rate_limiter = AsyncRateLimiter(tpm_limit, rpm_limit)
        semaphore = asyncio.Semaphore(parallelism)
        results_queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
        local_cost = 0.0
        async with AsyncOpenAI(api_key=api_key) as client:
            tasks = [
                asyncio.create_task(
                    _run_batch_task(
                        batch_no,
                        batch_candidates,
                        client=client,
                        model=model,
                        temperature=temperature,
                        top_p=top_p,
                        response_format=response_format,
                        max_retries=max_retries,
                        rate_limiter=rate_limiter,
                        stop_event=stop_event,
                        semaphore=semaphore,
                        results_queue=results_queue,
                    )
                )
                for batch_no, batch_candidates in enumerate(batches, start=1)
            ]
            processed = 0
            total_batches = len(tasks)
            while processed < total_batches:
                result = await results_queue.get()
                processed += 1
                item_ids = result.get("item_ids", [])
                sent = int(result.get("sent", len(item_ids)))
                if not result.get("skipped"):
                    counts["sent"] += sent
                counts["retried"] += int(result.get("retries", 0) or 0)
                ok_map: Dict[str, Dict[str, Any]] = result.get("ok", {}) or {}
                ko_map: Dict[str, str] = result.get("ko", {}) or {}
                duration = float(result.get("duration", 0.0) or 0.0)
                usage = result.get("usage", {}) or {}
                tokens_in = int(result.get("tokens_in", 0) or 0)
                tokens_out = int(result.get("tokens_out", 0) or 0)
                status_code = result.get("status_code", 0)
                if usage:
                    local_cost += _calculate_cost(usage, price_in, price_out)
                success_updates: Dict[int, Dict[str, Any]] = {}
                for pid_str, payload in ok_map.items():
                    try:
                        pid = int(pid_str)
                    except Exception:
                        continue
                    success_updates[pid] = payload
                    pending_set.discard(pid)
                    fail_reasons.pop(pid, None)
                for pid_str, reason in ko_map.items():
                    try:
                        pid = int(pid_str)
                    except Exception:
                        continue
                    pending_set.add(pid)
                    fail_reasons[pid] = reason or "error"
                if success_updates:
                    _apply_ai_updates(conn, success_updates)
                    for pid, payload in success_updates.items():
                        candidate = candidate_map.get(pid)
                        sig_hash = candidate.sig_hash if candidate else ""
                        record = {"sig_hash": sig_hash, "updates": payload.copy()}
                        parsed_desire = _parse_score(payload.get("desire_magnitude"))
                        parsed_comp = _parse_score(payload.get("competition_level"))
                        if parsed_desire is not None:
                            desire_scores.append((str(pid), parsed_desire))
                            record["_desire_score"] = parsed_desire
                        if parsed_comp is not None:
                            comp_scores.append((str(pid), parsed_comp))
                            record["_competition_score"] = parsed_comp
                        success_records[pid] = record
                        applied_outputs[pid] = {k: v for k, v in payload.items() if v is not None}
                counts["ok"] += len(success_updates)
                counts["ko"] += len(ko_map)
                if job_updates_enabled and not result.get("skipped"):
                    throughput = (len(item_ids) / duration) if duration > 0 else 0.0
                    database.append_ai_job_metric(
                        conn,
                        int(job_id),
                        result.get("batch_no", processed),
                        len(item_ids),
                        duration * 1000.0,
                        throughput,
                        cached_hits=0,
                    )
                done_val = counts["ok"] + counts["cached"]
                counts_with_cost_local = {**counts, "cost_spent_usd": local_cost}
                if job_updates_enabled:
                    database.update_import_job_ai_progress(conn, int(job_id), done_val)
                    database.set_import_job_ai_counts(conn, int(job_id), counts_with_cost_local, sorted(pending_set))
                _emit_status(
                    status_cb,
                    phase="enrich",
                    counts=counts_with_cost_local,
                    total=total_items,
                    done=done_val,
                    message=f"IA columnas {done_val}/{total_items}",
                )
                logger.info(
                    "AI_ENRICH batch k=%s dur_ms=%.0f rps=%.2f tokens_in=%d tokens_out=%d rc=%s",
                    result.get("batch_no", processed),
                    duration * 1000.0,
                    (len(item_ids) / duration) if duration > 0 else 0.0,
                    tokens_in,
                    tokens_out,
                    status_code,
                )
                if cost_cap is not None and local_cost >= float(cost_cap) and not stop_event.is_set():
                    stop_event.set()
            await asyncio.gather(*tasks)
        return local_cost, stop_event.is_set()

    cost_limit_hit = False
    if batches:
        cost_spent, cost_limit_hit = _run_async(_process_batches())
    counts_with_cost = {**counts, "cost_spent_usd": cost_spent}

    result_error: Optional[str] = None
    if cost_cap is not None and (cost_limit_hit or cost_spent >= float(cost_cap)):
        result_error = "cost_cap_reached"

    cfg_calib = config.get_ai_calibration_config()
    if cfg_calib.get("enabled", True) and success_records:
        wins = float(cfg_calib.get("winsorize_pct", 0.05) or 0.0)
        min_low = float(cfg_calib.get("min_low_pct", 0.05) or 0.0)
        min_med = float(cfg_calib.get("min_medium_pct", 0.05) or 0.0)
        min_high = float(cfg_calib.get("min_high_pct", 0.05) or 0.0)
        desire_info: Dict[str, Any] = {}
        comp_info: Dict[str, Any] = {}
        dist_desire: Dict[str, int] = {}
        dist_comp: Dict[str, int] = {}
        if desire_scores:
            labels, dist_desire, desire_info = _classify_scores(
                desire_scores,
                winsorize_pct=wins,
                min_low_pct=min_low,
                min_medium_pct=min_med,
                min_high_pct=min_high,
            )
            for pid_str, label in labels.items():
                pid = int(pid_str)
                rec = success_records.get(pid)
                if rec and rec["updates"].get("desire_magnitude") != label:
                    conn.execute("UPDATE products SET desire_magnitude=? WHERE id=?", (label, pid))
                    rec["updates"]["desire_magnitude"] = label
                    if pid in applied_outputs:
                        applied_outputs[pid]["desire_magnitude"] = label
        if comp_scores:
            labels, dist_comp, comp_info = _classify_scores(
                comp_scores,
                winsorize_pct=wins,
                min_low_pct=min_low,
                min_medium_pct=min_med,
                min_high_pct=min_high,
            )
            for pid_str, label in labels.items():
                pid = int(pid_str)
                rec = success_records.get(pid)
                if rec and rec["updates"].get("competition_level") != label:
                    conn.execute(
                        "UPDATE products SET competition_level=? WHERE id=?",
                        (label, pid),
                    )
                    rec["updates"]["competition_level"] = label
                    if pid in applied_outputs:
                        applied_outputs[pid]["competition_level"] = label
        conn.commit()
        logger.info(
            "ai_calibration_desire: dist=%s info=%s", dist_desire if desire_scores else {}, desire_info,
        )
        logger.info(
            "ai_calibration_comp: dist=%s info=%s", dist_comp if comp_scores else {}, comp_info,
        )

    for pid, rec in success_records.items():
        sig_hash = rec.get("sig_hash")
        updates = rec.get("updates", {})
        if sig_hash:
            database.upsert_ai_cache_entry(
                conn,
                sig_hash,
                model=model,
                version=cache_version,
                desire=updates.get("desire"),
                desire_magnitude=updates.get("desire_magnitude"),
                awareness_level=updates.get("awareness_level"),
                competition_level=updates.get("competition_level"),
            )
    conn.commit()

    pending_ids = sorted(pending_set)
    done_val = counts["ok"] + counts["cached"]
    if job_updates_enabled:
        database.update_import_job_ai_progress(conn, int(job_id), done_val)
        database.set_import_job_ai_counts(conn, int(job_id), counts_with_cost, pending_ids)
        if result_error:
            database.set_import_job_ai_error(conn, int(job_id), result_error)
    _emit_status(
        status_cb,
        phase="enrich",
        counts=counts_with_cost,
        total=total_items,
        done=done_val,
        message=f"IA columnas {done_val}/{total_items}",
    )

    logger.info(
        "run_ai_fill_job: job=%s model=%s parallelism=%d micro=%d total=%d ok=%d cached=%d ko=%d cost=%.4f cost_hit=%s pending=%d error=%s duration=%.2fs",
        job_id,
        model,
        parallelism,
        microbatch_size,
        total_items,
        counts["ok"],
        counts["cached"],
        counts["ko"],
        cost_spent,
        cost_limit_hit,
        len(pending_ids),
        result_error,
        time.perf_counter() - start_ts,
    )

    conn.close()
    return {
        "counts": counts_with_cost,
        "pending_ids": pending_ids,
        "error": result_error,
        "ok": applied_outputs,
        "ko": fail_reasons,
        "skipped_existing": skipped_existing,
        "total_requested": len(requested_ids),
    }


def fill_ai_columns(
    product_ids: Sequence[int],
    *,
    model: Optional[str] = None,
    batch_mode: Optional[bool] = None,
    cost_cap_usd: Optional[float] = None,
) -> Dict[str, Any]:
    # Compatibility wrapper used in a few legacy code paths.
    result = run_ai_fill_job(0, product_ids)
    counts = result.get("counts", {})
    total_requested = result.get("total_requested", len(product_ids))
    queued = counts.get("queued", 0)
    processed = counts.get("ok", 0) + counts.get("cached", 0)
    legacy_counts = {
        "n_importados": total_requested,
        "n_para_ia": queued,
        "n_procesados": processed,
        "n_omitidos_por_valor_existente": result.get("skipped_existing", 0),
        "n_reintentados": counts.get("retried", 0),
        "n_error_definitivo": counts.get("ko", 0),
        "truncated": result.get("error") == "cost_cap_reached",
        "cost_estimated_usd": counts.get("cost_spent_usd", 0.0),
    }
    return {
        "ok": {str(pid): data for pid, data in result.get("ok", {}).items()},
        "ko": {str(pid): reason for pid, reason in result.get("ko", {}).items()},
        "counts": legacy_counts,
        "pending_ids": result.get("pending_ids", []),
        "cost_estimated_usd": counts.get("cost_spent_usd", 0.0),
        "ui_cost_message": None,
        "error": result.get("error"),
    }
