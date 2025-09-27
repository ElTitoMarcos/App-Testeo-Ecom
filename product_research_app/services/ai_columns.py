from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
import sqlite3
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import httpx

from .. import config, database, gpt
from ..logging_setup import get_logger, get_log_dir
from ..utils.signature import compute_sig_hash
from .desire_utils import cleanse, looks_like_product_desc
from ..obs import (
    Stage,
    ReasonCode,
    log_ok,
    log_ko,
    ensure_dirs,
    dump_artifact,
    artifacts_enabled,
)

logger = get_logger(__name__)

APP_DIR = Path(__file__).resolve().parent.parent
DB_PATH = APP_DIR / "data.sqlite3"
CALIBRATION_CACHE_FILE = APP_DIR / "ai_calibration_cache.json"

AI_FIELDS = ("desire", "desire_magnitude", "awareness_level", "competition_level")
StatusCallback = Callable[..., None]


@dataclass
class Candidate:
    id: int
    sig_hash: str
    payload: Dict[str, Any]
    extra: Dict[str, Any]


@dataclass
class BatchRequest:
    req_id: str
    candidates: List[Candidate]
    user_text: str
    prompt_tokens_est: int


@dataclass
class ProductTrace:
    product_id: int
    job_id: str
    artifacts: List[str] = field(default_factory=list)
    req_id: Optional[str] = None
    retries: int = 0
    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    response_tokens: Optional[int] = None
    http_status: Optional[int] = None
    durations: Dict[str, float] = field(default_factory=dict)

    def add_artifact(self, path: Optional[str]) -> None:
        if path and path not in self.artifacts:
            self.artifacts.append(path)


def _job_label(job_id: Optional[int]) -> str:
    if job_id in {None, 0}:
        return "adhoc"
    return str(job_id)


def _artifacts_root(job_id: Optional[int]) -> Path:
    base = get_log_dir()
    return base / "ko" / _job_label(job_id)


def _artifact_path(job_id: Optional[int], req_id: str, product_id: int, filename: str) -> Path:
    root = _artifacts_root(job_id)
    return root / req_id / str(product_id) / filename


def _format_prompt(system_prompt: str, user_text: str) -> str:
    return f"SYSTEM:\n{system_prompt.strip()}\n\nUSER:\n{user_text.strip()}"


def _validation_report(payload: Dict[str, Any]) -> Dict[str, Any]:
    required = ["desire", "desire_magnitude", "awareness_level", "competition_level"]
    missing = [field for field in required if not payload.get(field)]
    present = [field for field in required if payload.get(field)]
    return {"required": required, "missing": missing, "present": present}


def _record_artifact(trace: ProductTrace, path: Path, data: Any) -> Optional[str]:
    artifact = dump_artifact(path, data)
    trace.add_artifact(artifact)
    return artifact


def _persist_request_inputs(
    job_id: Optional[int], request: BatchRequest, traces: Dict[int, ProductTrace]
) -> None:
    if not artifacts_enabled():
        return
    prompt_text = _format_prompt(SYSTEM_PROMPT, request.user_text)
    for cand in request.candidates:
        trace = traces.get(cand.id)
        if trace is None:
            continue
        req_id = trace.req_id or request.req_id
        trace.req_id = req_id
        prompt_path = _artifact_path(job_id, req_id, cand.id, "prompt.txt")
        payload_path = _artifact_path(job_id, req_id, cand.id, "payload.json")
        _record_artifact(trace, prompt_path, prompt_text)
        _record_artifact(trace, payload_path, cand.payload)


def _persist_response_artifacts(
    job_id: Optional[int],
    request: BatchRequest,
    traces: Dict[int, ProductTrace],
    *,
    raw_json: Optional[Dict[str, Any]] = None,
    response_text: Optional[str] = None,
) -> None:
    if not artifacts_enabled():
        return
    for cand in request.candidates:
        trace = traces.get(cand.id)
        if trace is None:
            continue
        req_id = trace.req_id or request.req_id
        if raw_json is not None:
            raw_path = _artifact_path(job_id, req_id, cand.id, "raw_response.json")
            _record_artifact(trace, raw_path, raw_json)
        if response_text:
            resp_path = _artifact_path(job_id, req_id, cand.id, "response_text.txt")
            _record_artifact(trace, resp_path, response_text)


def _persist_validation_artifacts(
    job_id: Optional[int],
    product_id: int,
    trace: ProductTrace,
    payload: Dict[str, Any],
) -> List[str]:
    artifacts: List[str] = []
    if not artifacts_enabled():
        return artifacts
    req_id = trace.req_id or "validation"
    report_path = _artifact_path(job_id, req_id, product_id, "validation_report.json")
    normalized_path = _artifact_path(job_id, req_id, product_id, "normalized_output.json")
    report = _validation_report(payload)
    rep_art = _record_artifact(trace, report_path, report)
    norm_art = _record_artifact(trace, normalized_path, payload)
    if rep_art:
        artifacts.append(rep_art)
    if norm_art:
        artifacts.append(norm_art)
    return artifacts


def _persist_db_failure_artifacts(
    job_id: Optional[int],
    product_id: int,
    trace: ProductTrace,
    payload: Dict[str, Any],
    exc: BaseException,
) -> List[str]:
    artifacts: List[str] = []
    if not artifacts_enabled():
        return artifacts
    req_id = trace.req_id or "write_db"
    payload_path = _artifact_path(job_id, req_id, product_id, "db_payload.json")
    exc_path = _artifact_path(job_id, req_id, product_id, "db_exception.txt")
    pay_art = _record_artifact(trace, payload_path, payload)
    exc_art = _record_artifact(trace, exc_path, f"{exc.__class__.__name__}: {exc}")
    if pay_art:
        artifacts.append(pay_art)
    if exc_art:
        artifacts.append(exc_art)
    return artifacts


def _reason_from_exception(exc: BaseException) -> ReasonCode:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code if exc.response is not None else None
        if status == 429:
            return ReasonCode.OPENAI_RATE_LIMITED
        if status in {408, 504, 524}:
            return ReasonCode.OPENAI_TIMEOUT
        return ReasonCode.OPENAI_HTTP_ERROR
    if isinstance(exc, httpx.TimeoutException):
        return ReasonCode.OPENAI_TIMEOUT
    if isinstance(exc, (json.JSONDecodeError, gpt.InvalidJSONError)):
        return ReasonCode.OPENAI_BAD_JSON
    if isinstance(exc, httpx.HTTPError):
        return ReasonCode.OPENAI_HTTP_ERROR
    return ReasonCode.UNKNOWN


def _http_status_from_exception(exc: BaseException) -> Optional[int]:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        return exc.response.status_code
    if isinstance(exc, httpx.HTTPError) and getattr(exc, "response", None) is not None:
        return exc.response.status_code  # type: ignore[attr-defined]
    return None


def _reason_from_string(reason: str) -> ReasonCode:
    mapping = {
        "missing": ReasonCode.MISSING_REQUIRED_KEYS,
        "validation_error": ReasonCode.VALIDATION_ERROR,
        "cost_cap_reached": ReasonCode.UNKNOWN,
    }
    return mapping.get(reason or "", ReasonCode.UNKNOWN)


def _extract_usage_tokens(usage: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    prompt = usage.get("prompt_tokens")
    if prompt is None:
        prompt = usage.get("input_tokens") or usage.get("tokens_in")
    completion = usage.get("completion_tokens")
    if completion is None:
        completion = usage.get("output_tokens") or usage.get("tokens_out")
    def _to_int(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(float(value))
        except Exception:
            return None

    return _to_int(prompt), _to_int(completion)


class _AsyncRateLimiter:
    def __init__(self, rpm_limit: Optional[int], tpm_limit: Optional[int]) -> None:
        self.rpm = max(0, int(rpm_limit or 0))
        self.tpm = max(0, int(tpm_limit or 0))
        self._lock = asyncio.Lock()
        self._events: deque[tuple[float, int]] = deque()

    async def acquire(self, tokens: int) -> None:
        if self.rpm <= 0 and self.tpm <= 0:
            return
        tokens = max(0, int(tokens or 0))
        window = 60.0
        async with self._lock:
            while True:
                now = time.monotonic()
                while self._events and now - self._events[0][0] >= window:
                    self._events.popleft()
                wait_time = 0.0
                if self.rpm and len(self._events) >= self.rpm:
                    oldest = self._events[0][0]
                    wait_time = max(wait_time, window - (now - oldest))
                if self.tpm:
                    token_sum = sum(tok for _, tok in self._events)
                    if token_sum + tokens > self.tpm:
                        deficit = token_sum + tokens - self.tpm
                        if deficit > 0 and self._events:
                            running = 0
                            target_wait = 0.0
                            for ts, tok in self._events:
                                running += tok
                                if token_sum - running + tokens <= self.tpm:
                                    target_wait = window - (now - ts)
                                    break
                            else:
                                target_wait = window - (now - self._events[-1][0])
                            wait_time = max(wait_time, max(0.0, target_wait))
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    continue
                self._events.append((time.monotonic(), tokens))
                return


SYSTEM_PROMPT = (
    "Eres un analista de marketing. Devuelve únicamente un JSON con claves de producto. "
    "Cada valor debe incluir desire (string corta), desire_magnitude (Low|Medium|High), "
    "awareness_level (Unaware|Problem-Aware|Solution-Aware|Product-Aware|Most Aware) y "
    "competition_level (Low|Medium|High)."
)
USER_INSTRUCTION = (
    "Analiza los siguientes productos y responde solo con un JSON cuyas claves sean los IDs. "
    "Cada entrada debe incluir desire, desire_magnitude, awareness_level y competition_level."
)

def _truncate_text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if limit and limit > 0 and len(text) > limit:
        trimmed = text[: max(0, limit - 1)].rstrip()
        return trimmed + "…"
    return text


def _join_bullets(value: Any, limit: int) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value if str(item).strip()]
        text = " • ".join(parts)
    elif isinstance(value, dict):
        parts = [str(item).strip() for item in value.values() if str(item).strip()]
        text = " • ".join(parts)
    else:
        text = str(value).strip()
    text = text.replace("\n", " ").replace("\r", " ")
    return _truncate_text(text, limit)


def _estimate_tokens_from_text(*texts: str) -> int:
    total_chars = sum(len(t) for t in texts if t)
    if total_chars <= 0:
        return 0
    return max(1, int(math.ceil(total_chars / 4)))


def _build_product_payload(
    candidate: Candidate,
    trunc_title: int,
    trunc_desc: int,
) -> Dict[str, Any]:
    payload = candidate.payload
    extra = candidate.extra or {}
    product: Dict[str, Any] = {"id": str(candidate.id)}

    title = _truncate_text(payload.get("name"), trunc_title)
    if title:
        product["title"] = title

    for key in ("category", "price", "rating", "units_sold", "revenue", "conversion_rate"):
        value = payload.get(key)
        if value not in {None, ""}:
            product[key] = value

    launch_date = payload.get("launch_date") or extra.get("launch_date")
    if launch_date:
        product["launch_date"] = launch_date

    date_range = payload.get("date_range")
    if date_range:
        product["date_range"] = date_range

    brand = extra.get("brand")
    if brand:
        product["brand"] = brand

    asin = extra.get("asin")
    if asin:
        product["asin"] = asin

    desc_source = (
        payload.get("description")
        or extra.get("description")
        or extra.get("body")
        or extra.get("long_description")
    )
    if isinstance(desc_source, (list, tuple)):
        desc_source = " ".join(str(item) for item in desc_source if item)
    description = _truncate_text(desc_source, trunc_desc)
    if description:
        product["description"] = description

    bullets_source = (
        extra.get("bullets")
        or extra.get("highlights")
        or extra.get("features")
        or payload.get("bullets")
    )
    bullets = _join_bullets(bullets_source, trunc_desc)
    if bullets:
        product["bullets"] = bullets

    return product


def _build_batch_request(
    req_id: str,
    candidates: List[Candidate],
    trunc_title: int,
    trunc_desc: int,
) -> BatchRequest:
    product_lines: List[str] = []
    for cand in candidates:
        product_payload = _build_product_payload(cand, trunc_title, trunc_desc)
        product_lines.append(
            json.dumps(product_payload, ensure_ascii=False, separators=(",", ":"))
        )
    user_text = USER_INSTRUCTION + "\n" + "\n".join(product_lines)
    prompt_tokens_est = _estimate_tokens_from_text(SYSTEM_PROMPT, user_text) + len(candidates) * 8
    return BatchRequest(
        req_id=req_id,
        candidates=candidates,
        user_text=user_text,
        prompt_tokens_est=prompt_tokens_est,
    )


def _extract_response_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    parsed_json, text_content = gpt._parse_message_content(raw)
    if isinstance(parsed_json, dict):
        return parsed_json
    content_text = text_content.strip() if isinstance(text_content, str) else ""
    if not content_text:
        raise gpt.InvalidJSONError("La respuesta IA no es JSON")
    try:
        obj = json.loads(content_text)
    except json.JSONDecodeError:
        obj, _ = gpt._extract_first_json_block(content_text)
    if not isinstance(obj, dict):
        raise gpt.InvalidJSONError("La respuesta IA no es JSON")
    return obj


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    lower = ordered[lo]
    upper = ordered[hi]
    return lower + (upper - lower) * (pos - lo)


def _load_calibration_cache() -> Dict[str, Any]:
    if not CALIBRATION_CACHE_FILE.exists():
        return {}
    try:
        with open(CALIBRATION_CACHE_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except Exception:
        logger.debug("failed to load calibration cache", exc_info=True)
    return {}


def _store_calibration_cache(data: Dict[str, Any]) -> None:
    try:
        tmp = CALIBRATION_CACHE_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        tmp.replace(CALIBRATION_CACHE_FILE)
    except Exception:
        logger.debug("failed to persist calibration cache", exc_info=True)


def _calibration_cache_key(weights_id: int, model: str) -> str:
    return f"{int(weights_id)}::{model}".strip()


def _get_cached_calibration(weights_id: int, model: str) -> Optional[Dict[str, Any]]:
    cache = _load_calibration_cache()
    key = _calibration_cache_key(weights_id, model)
    entry = cache.get(key)
    if isinstance(entry, dict):
        return entry
    return None


def _set_cached_calibration(weights_id: int, model: str, entry: Dict[str, Any]) -> None:
    cache = _load_calibration_cache()
    cache[_calibration_cache_key(weights_id, model)] = entry
    _store_calibration_cache(cache)


def _apply_thresholds_with_minimums(
    scores: List[Tuple[str, float]],
    thresholds: Dict[str, Any],
    *,
    min_low_pct: float,
    min_medium_pct: float,
    min_high_pct: float,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    labels: Dict[str, str] = {}
    dist = {"Low": 0, "Medium": 0, "High": 0}
    if not scores:
        return labels, dist

    q33 = thresholds.get("q33")
    q67 = thresholds.get("q67")
    pairs = list(scores)
    if q33 is None or q67 is None or (isinstance(q33, float) and isinstance(q67, float) and abs(q67 - q33) < 1e-6):
        sorted_pairs = sorted(pairs, key=lambda x: x[1])
        n = len(sorted_pairs)
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
    else:
        for pid, score in pairs:
            if score <= float(q33):
                lab = "Low"
            elif score >= float(q67):
                lab = "High"
            else:
                lab = "Medium"
            labels[pid] = lab
            dist[lab] += 1

    n = len(pairs)
    min_medium = math.ceil(min_medium_pct * n)
    min_low = math.ceil(min_low_pct * n)
    min_high = math.ceil(min_high_pct * n)

    if dist["Medium"] < min_medium:
        need = min_medium - dist["Medium"]
        candidates = [
            (abs(score - 0.5), pid)
            for pid, score in pairs
            if labels.get(pid) != "Medium"
        ]
        candidates.sort()
        for _, pid in candidates[:need]:
            prev = labels[pid]
            labels[pid] = "Medium"
            dist["Medium"] += 1
            dist[prev] -= 1

    available = max(0, dist["Medium"] - min_medium)
    if dist["Low"] < min_low and available > 0:
        need = min(min_low - dist["Low"], available)
        candidates = [
            (abs(score - (q33 if isinstance(q33, (int, float)) else 0.33)), pid)
            for pid, score in pairs
            if labels.get(pid) == "Medium"
        ]
        candidates.sort()
        for _, pid in candidates[:need]:
            labels[pid] = "Low"
            dist["Low"] += 1
            dist["Medium"] -= 1
        available = max(0, dist["Medium"] - min_medium)

    if dist["High"] < min_high and available > 0:
        need = min(min_high - dist["High"], available)
        candidates = [
            (abs(score - (q67 if isinstance(q67, (int, float)) else 0.67)), pid)
            for pid, score in pairs
            if labels.get(pid) == "Medium"
        ]
        candidates.sort()
        for _, pid in candidates[:need]:
            labels[pid] = "High"
            dist["High"] += 1
            dist["Medium"] -= 1

    return labels, dist


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


async def _call_batch_with_retries(
    client: httpx.AsyncClient,
    request: BatchRequest,
    *,
    model: str,
    max_retries: int,
    rate_limiter: _AsyncRateLimiter,
    semaphore: asyncio.Semaphore,
    stop_event: asyncio.Event,
    job_id: Optional[int],
    product_traces: Dict[int, ProductTrace],
    ko_reason_counts: Counter[str],
) -> Dict[str, Any]:
    if stop_event.is_set():
        now_iso = datetime.utcnow().isoformat()
        logger.info(
            "ai_columns.request: req_id=%s items=%d prompt_tokens_est=%d start=%s end=%s duration=0.00s status=%s retries=0",
            request.req_id,
            len(request.candidates),
            request.prompt_tokens_est,
            now_iso,
            now_iso,
            "skipped",
        )
        return {
            "req_id": request.req_id,
            "candidates": request.candidates,
            "skipped": True,
            "usage": {},
            "duration": 0.0,
            "retries": 0,
            "prompt_tokens_est": request.prompt_tokens_est,
        }

    async with semaphore:
        if stop_event.is_set():
            now_iso = datetime.utcnow().isoformat()
            logger.info(
                "ai_columns.request: req_id=%s items=%d prompt_tokens_est=%d start=%s end=%s duration=0.00s status=%s retries=0",
                request.req_id,
                len(request.candidates),
                request.prompt_tokens_est,
                now_iso,
                now_iso,
                "skipped",
            )
            return {
                "req_id": request.req_id,
                "candidates": request.candidates,
                "skipped": True,
                "usage": {},
                "duration": 0.0,
                "retries": 0,
                "prompt_tokens_est": request.prompt_tokens_est,
            }

        _persist_request_inputs(job_id, request, product_traces)
        for cand in request.candidates:
            trace = product_traces.get(cand.id)
            if trace:
                trace.model = model
                if trace.prompt_tokens is None:
                    trace.prompt_tokens = request.prompt_tokens_est

        attempt = 0
        while True:
            await rate_limiter.acquire(request.prompt_tokens_est)
            start_ts = time.perf_counter()
            start_iso = datetime.utcnow().isoformat()
            status = "ok"
            error_message: Optional[str] = None
            response_text: Optional[str] = None
            raw: Optional[Dict[str, Any]] = None
            http_status: Optional[int] = None
            try:
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": model,
                        "temperature": 0,
                        "top_p": 1,
                        "max_tokens": 380,
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": request.user_text},
                        ],
                    },
                )
                response_text = response.text
                http_status = response.status_code
                response.raise_for_status()
                raw = response.json()
                _persist_response_artifacts(job_id, request, product_traces, raw_json=raw)
                payload = _extract_response_payload(raw)
                if not isinstance(payload, dict):
                    raise gpt.InvalidJSONError("Respuesta IA no es JSON")
                data_map = {str(k): v for k, v in payload.items()}
                ok: Dict[str, Dict[str, Any]] = {}
                ko: Dict[str, str] = {}
                for cand in request.candidates:
                    pid = str(cand.id)
                    entry = data_map.get(pid)
                    if not isinstance(entry, dict):
                        ko[pid] = "missing"
                        continue
                    desire_payload = dict(entry)
                    draft_raw = str(
                        desire_payload.get("desire_statement")
                        or desire_payload.get("desire")
                        or ""
                    ).strip()
                    extra_title = (cand.extra or {}).get("title") if isinstance(cand.extra, dict) else None
                    title = str(cand.payload.get("name") or extra_title or "")
                    if len(draft_raw) < 280 or looks_like_product_desc(draft_raw, title):
                        refined = await _refine_desire_statement(cand, draft_raw)
                        if isinstance(refined, dict):
                            merged = dict(desire_payload)
                            merged.update(refined)
                            desire_payload = merged
                            draft_raw = str(
                                desire_payload.get("desire_statement")
                                or desire_payload.get("desire")
                                or ""
                            ).strip()
                    if not draft_raw:
                        draft_raw = str(entry.get("desire") or "")
                    cleaned_txt = cleanse(draft_raw)
                    if not cleaned_txt:
                        cleaned_txt = draft_raw
                    cleaned_txt = cleaned_txt.strip()
                    desire_primary = desire_payload.get("desire_primary")
                    if desire_primary is None:
                        desire_primary = entry.get("desire_primary")
                    desire_magnitude_raw = desire_payload.get("desire_magnitude")
                    if desire_magnitude_raw is None:
                        desire_magnitude_raw = entry.get("desire_magnitude")
                    if isinstance(desire_magnitude_raw, dict):
                        desire_magnitude_raw = desire_magnitude_raw.get("overall")
                    ok[pid] = {
                        "desire": cleaned_txt,
                        "desire_primary": desire_primary,
                        "desire_magnitude": gpt._norm_tri(desire_magnitude_raw),
                        "awareness_level": gpt._norm_awareness(
                            desire_payload.get("awareness_level")
                        ),
                        "competition_level": gpt._norm_tri(
                            desire_payload.get("competition_level")
                        ),
                    }
                duration = time.perf_counter() - start_ts
                end_iso = datetime.utcnow().isoformat()
                usage = raw.get("usage", {}) or {}
                prompt_tokens_val, completion_tokens_val = _extract_usage_tokens(usage)
                for cand in request.candidates:
                    trace = product_traces.get(cand.id)
                    if trace:
                        trace.retries = attempt
                        trace.http_status = http_status
                        trace.durations[Stage.CALL_OPENAI.value] = duration
                        if prompt_tokens_val is not None:
                            trace.prompt_tokens = prompt_tokens_val
                        elif trace.prompt_tokens is None:
                            trace.prompt_tokens = request.prompt_tokens_est
                        if completion_tokens_val is not None:
                            trace.response_tokens = completion_tokens_val
                logger.info(
                    "ai_columns.request: req_id=%s items=%d prompt_tokens_est=%d start=%s end=%s duration=%.2fs status=%s retries=%d http_status=%s model=%s prompt_tokens=%s response_tokens=%s",
                    request.req_id,
                    len(request.candidates),
                    request.prompt_tokens_est,
                    start_iso,
                    end_iso,
                    duration,
                    status,
                    attempt,
                    http_status,
                    model,
                    prompt_tokens_val if prompt_tokens_val is not None else request.prompt_tokens_est,
                    completion_tokens_val,
                )
                return {
                    "req_id": request.req_id,
                    "candidates": request.candidates,
                    "ok": ok,
                    "ko": ko,
                    "usage": usage,
                    "duration": duration,
                    "retries": attempt,
                    "prompt_tokens_est": request.prompt_tokens_est,
                }
            except (httpx.HTTPError, json.JSONDecodeError, gpt.InvalidJSONError) as exc:
                error_message = str(exc)
                status = "retry" if attempt < max_retries and not stop_event.is_set() else "error"
                duration = time.perf_counter() - start_ts
                end_iso = datetime.utcnow().isoformat()
                http_status = _http_status_from_exception(exc) or http_status
                _persist_response_artifacts(
                    job_id,
                    request,
                    product_traces,
                    raw_json=raw if isinstance(raw, dict) else None,
                    response_text=response_text or error_message,
                )
                logger.info(
                    "ai_columns.request: req_id=%s items=%d prompt_tokens_est=%d start=%s end=%s duration=%.2fs status=%s retries=%d error=%s http_status=%s model=%s",
                    request.req_id,
                    len(request.candidates),
                    request.prompt_tokens_est,
                    start_iso,
                    end_iso,
                    duration,
                    status,
                    attempt,
                    error_message,
                    http_status,
                    model,
                )
                if attempt < max_retries and not stop_event.is_set():
                    attempt += 1
                    sleep_for = min(10.0, 0.5 * (2**(attempt - 1))) + random.uniform(0.05, 0.25)
                    await asyncio.sleep(sleep_for)
                    continue
                reason_code = _reason_from_exception(exc)
                stage = Stage.PARSE_RESPONSE if reason_code == ReasonCode.OPENAI_BAD_JSON else Stage.CALL_OPENAI
                for cand in request.candidates:
                    trace = product_traces.get(cand.id)
                    if trace:
                        trace.retries = attempt
                        trace.http_status = http_status
                        trace.durations[stage.value] = duration
                        if trace.prompt_tokens is None:
                            trace.prompt_tokens = request.prompt_tokens_est
                        artifacts = list(trace.artifacts)
                        log_ko(
                            logger,
                            stage=stage,
                            job_id=job_id,
                            req_id=trace.req_id or request.req_id,
                            product_id=str(cand.id),
                            duration_ms=duration * 1000.0,
                            retries=attempt,
                            model=trace.model or model,
                            prompt_tokens=trace.prompt_tokens,
                            response_tokens=trace.response_tokens,
                            reason_code=reason_code,
                            reason_detail=error_message,
                            exception=exc,
                            http_status=http_status,
                            artifacts=artifacts,
                        )
                        ko_reason_counts[reason_code.value] += 1
                return {
                    "req_id": request.req_id,
                    "candidates": request.candidates,
                    "ok": {},
                    "ko": {str(cand.id): error_message for cand in request.candidates},
                    "usage": {},
                    "duration": duration,
                    "retries": attempt,
                    "error": error_message,
                    "prompt_tokens_est": request.prompt_tokens_est,
                }


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
    began_tx = False
    try:
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
            began_tx = True
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
        if began_tx:
            conn.commit()
    except Exception:
        if began_tx and conn.in_transaction:
            conn.rollback()
        raise


def _build_payload(row: Any, extra: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "category": row["category"],
        "price": row["price"],
        "rating": extra.get("rating"),
        "units_sold": extra.get("units_sold"),
        "revenue": extra.get("revenue"),
        "conversion_rate": extra.get("conversion_rate"),
        "launch_date": extra.get("launch_date"),
        "date_range": row["date_range"],
        "image_url": row["image_url"],
        "description": row.get("description"),
        "bullets": extra.get("bullets"),
        "highlights": extra.get("highlights"),
        "body": extra.get("body"),
        "long_description": extra.get("long_description"),
    }


DESIRE_REFINE_EXTRA_USER = (
    "El borrador parece una descripción de producto. Reescribe SOLO 'desire_statement' (280–420 chars) "
    "centrado en el DESEO HUMANO (resultado+emoción+escena+fricción), sin marcas, sin categoría, sin medidas."
)


def _candidate_to_desire_context(candidate: Candidate) -> Dict[str, Any]:
    payload = candidate.payload or {}
    extra = candidate.extra or {}
    product: Dict[str, Any] = {
        "id": candidate.id,
        "title": payload.get("name") or extra.get("title") or "",
        "category": payload.get("category") or extra.get("category") or "",
    }
    for key in (
        "price",
        "rating",
        "units_sold",
        "revenue",
        "conversion_rate",
        "launch_date",
        "date_range",
    ):
        value = payload.get(key)
        if value in {None, ""}:
            value = extra.get(key)
        if value not in {None, ""}:
            product[key] = value
    for text_key in ("description", "bullets", "highlights", "body", "long_description"):
        value = payload.get(text_key)
        if not value:
            value = extra.get(text_key)
        if value:
            product[text_key] = value
    return {"product": product}


async def _refine_desire_statement(
    candidate: Candidate,
    draft_text: str,
) -> Optional[Dict[str, Any]]:
    context = _candidate_to_desire_context(candidate)

    def _call() -> Optional[Dict[str, Any]]:
        try:
            result = gpt.call_gpt(
                "DESIRE",
                context_json=context,
                temperature=0,
                extra_user=(
                    f"Borrador previo:\n{draft_text.strip()}\n\n" + DESIRE_REFINE_EXTRA_USER
                ),
                mode="refine_no_product",
                max_tokens=450,
                stop=None,
            )
        except Exception:
            logger.exception("desire refine call failed for id=%s", candidate.id)
            return None
        content = result.get("content") if isinstance(result, dict) else None
        if isinstance(content, dict):
            return content
        return None

    return await asyncio.to_thread(_call)


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
    job_label = _job_label(job_id)
    ko_reason_counts: Counter[str] = Counter()
    job_artifacts_dir = _artifacts_root(job_id)
    artifacts_on = artifacts_enabled()
    if artifacts_on:
        ensure_dirs(job_artifacts_dir)
    conn = _ensure_conn()
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
        name = row["name"]
        if not name:
            skipped_existing += 1
            continue
        brand = extra.get("brand")
        asin = extra.get("asin")
        product_url = extra.get("product_url")
        sig_hash = row["sig_hash"] or compute_sig_hash(name, brand, asin, product_url)
        if sig_hash and not row["sig_hash"]:
            sig_updates.append((sig_hash, pid))
        payload = _build_payload(row, extra)
        candidates.append(Candidate(id=pid, sig_hash=sig_hash, payload=payload, extra=extra))

    product_traces: Dict[int, ProductTrace] = {
        cand.id: ProductTrace(product_id=cand.id, job_id=job_label)
        for cand in candidates
    }

    if sig_updates:
        cur = conn.cursor()
        skipped_sig_hash = 0
        for sig_hash, pid in sig_updates:
            cur.execute(
                "UPDATE OR IGNORE products SET sig_hash=? WHERE id=?",
                (sig_hash, pid),
            )
            if cur.rowcount == 0:
                skipped_sig_hash += 1
        if conn.in_transaction:
            conn.commit()
        if skipped_sig_hash:
            logger.debug(
                "skipped %d sig_hash updates due to existing conflicts",
                skipped_sig_hash,
            )

    total_items = len(candidates)

    runtime_cfg = config.get_ai_runtime_config()
    if parallelism is None:
        parallelism = int(runtime_cfg.get("parallelism", 8) or 8)
    parallelism = max(1, parallelism)

    microbatch_size = int(microbatch or runtime_cfg.get("microbatch", 12) or 12)
    microbatch_size = max(1, microbatch_size)

    cache_enabled = bool(runtime_cfg.get("cache_enabled", True))
    cache_version = int(runtime_cfg.get("version", 1) or 1)
    tpm_limit = runtime_cfg.get("tpm_limit")
    if tpm_limit is not None:
        try:
            tpm_limit = int(tpm_limit)
        except Exception:
            tpm_limit = None
    rpm_limit = runtime_cfg.get("rpm_limit")
    if rpm_limit is not None:
        try:
            rpm_limit = int(rpm_limit)
        except Exception:
            rpm_limit = None
    trunc_title = int(runtime_cfg.get("trunc_title", 180) or 180)
    trunc_desc = int(runtime_cfg.get("trunc_desc", 800) or 800)
    timeout_s = float(runtime_cfg.get("timeout", 45) or 45)

    batch_cfg = config.get_ai_batch_config()
    max_retries = int(batch_cfg.get("MAX_RETRIES", 3) or 3)

    cost_cfg = config.get_ai_cost_config()
    model = cost_cfg.get("model") or config.get_model()
    env_model = os.environ.get("AI_MODEL")
    if env_model:
        model = env_model
    cost_cap = cost_cfg.get("costCapUSD")
    price_map = cost_cfg.get("prices", {}).get(model, {})
    price_in = float(price_map.get("input", 0.0))
    price_out = float(price_map.get("output", 0.0))
    api_key = config.get_api_key() or os.environ.get("OPENAI_API_KEY")

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
            cached_desire_raw = cache_row["desire"] if cache_row["desire"] is not None else ""
            cached_title = str(cand.payload.get("name") or (cand.extra or {}).get("title") or "")
            cached_desire = cleanse(str(cached_desire_raw or "").strip())
            if not cached_desire:
                cached_desire = str(cached_desire_raw or "").strip()
            if len(cached_desire) < 280 or looks_like_product_desc(cached_desire, cached_title):
                remaining.append(cand)
                continue
            update_payload = {
                "desire": cached_desire,
                "desire_magnitude": cache_row["desire_magnitude"],
                "awareness_level": cache_row["awareness_level"],
                "competition_level": cache_row["competition_level"],
            }
            desire_primary_cached = cache_row["desire_primary"] if "desire_primary" in cache_row.keys() else None
            if desire_primary_cached is not None:
                update_payload["desire_primary"] = desire_primary_cached
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
        candidates = remaining
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
        _emit_status(status_cb, phase="enrich", counts=counts_with_cost, total=total_items, done=counts["cached"], message="IA pendiente")
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

    batches: List[BatchRequest] = []
    if remaining:
        index = 0
        req_counter = 0
        total_remaining = len(remaining)
        while index < total_remaining:
            size = min(microbatch_size, total_remaining - index)
            if size <= 0:
                break
            while size > 0:
                chunk = remaining[index : index + size]
                if not chunk:
                    break
                req_id = f"{req_counter + 1:03d}"
                batch = _build_batch_request(req_id, chunk, trunc_title, trunc_desc)
                if tpm_limit and batch.prompt_tokens_est > tpm_limit and size > 1:
                    size = max(1, size - 1)
                    continue
                if tpm_limit and batch.prompt_tokens_est > tpm_limit and size == 1:
                    logger.warning(
                        "ai_columns.prompt_estimate_exceeds_limit: req_id=%s est=%d limit=%d",
                        req_id,
                        batch.prompt_tokens_est,
                        int(tpm_limit),
                    )
                batches.append(batch)
                req_counter += 1
                for cand in chunk:
                    trace = product_traces.get(cand.id)
                    if trace:
                        trace.req_id = batch.req_id
                        trace.prompt_tokens = batch.prompt_tokens_est
                index += size
                break
            else:
                index += size or 1

    rate_limiter = _AsyncRateLimiter(rpm_limit, tpm_limit)
    stop_event = asyncio.Event()

    desire_scores: List[Tuple[str, float]] = []
    comp_scores: List[Tuple[str, float]] = []
    success_records: Dict[int, Dict[str, Any]] = {}
    request_latencies: List[float] = []
    processed_batches = 0

    def process_result(result: Dict[str, Any]) -> None:
        nonlocal cost_spent, counts_with_cost, processed_batches
        candidates_list: List[Candidate] = result.get("candidates", [])
        if result.get("skipped"):
            for cand in candidates_list:
                pending_set.add(cand.id)
                fail_reasons.setdefault(cand.id, "cost_cap_reached")
            return

        processed_batches += 1
        counts["sent"] += len(candidates_list)
        retries = int(result.get("retries", 0) or 0)
        counts["retried"] += retries

        duration = float(result.get("duration", 0.0) or 0.0)
        if duration > 0:
            request_latencies.append(duration)
        usage = result.get("usage", {}) or {}
        if usage:
            cost_spent += _calculate_cost(usage, price_in, price_out)

        ok_map: Dict[str, Dict[str, Any]] = result.get("ok", {}) or {}
        ko_map: Dict[str, str] = result.get("ko", {}) or {}

        success_updates: Dict[int, Dict[str, Any]] = {}
        validation_artifacts: Dict[int, List[str]] = {}
        for pid_str, payload in ok_map.items():
            try:
                pid = int(pid_str)
            except Exception:
                continue
            success_updates[pid] = payload
            pending_set.discard(pid)

        ko_entries: List[Tuple[int, str]] = []
        for pid_str, reason in ko_map.items():
            try:
                pid = int(pid_str)
            except Exception:
                continue
            pending_set.add(pid)
            fail_reasons[pid] = reason or "error"
            ko_entries.append((pid, reason or "error"))

        if ko_entries and not result.get("error"):
            for pid, reason in ko_entries:
                trace = product_traces.get(pid)
                reason_code = _reason_from_string(reason)
                detail = "Respuesta IA sin entrada para el producto" if reason == "missing" else reason
                if trace:
                    trace.durations[Stage.PARSE_RESPONSE.value] = duration
                    log_ko(
                        logger,
                        stage=Stage.PARSE_RESPONSE,
                        job_id=job_id,
                        req_id=trace.req_id or result.get("req_id"),
                        product_id=str(pid),
                        duration_ms=duration * 1000.0,
                        retries=trace.retries,
                        model=trace.model or model,
                        prompt_tokens=trace.prompt_tokens,
                        response_tokens=trace.response_tokens,
                        reason_code=reason_code,
                        reason_detail=detail,
                        http_status=trace.http_status,
                        artifacts=list(trace.artifacts),
                    )
                else:
                    log_ko(
                        logger,
                        stage=Stage.PARSE_RESPONSE,
                        job_id=job_id,
                        req_id=result.get("req_id"),
                        product_id=str(pid),
                        duration_ms=duration * 1000.0,
                        retries=retries,
                        model=model,
                        prompt_tokens=None,
                        response_tokens=None,
                        reason_code=reason_code,
                        reason_detail=detail,
                        http_status=None,
                        artifacts=[],
                    )
                ko_reason_counts[reason_code.value] += 1

        if success_updates:
            for pid, payload in success_updates.items():
                trace = product_traces.get(pid)
                if trace:
                    trace.durations[Stage.VALIDATE_OUTPUT.value] = duration
                    validation_artifacts[pid] = _persist_validation_artifacts(job_id, pid, trace, payload)
                    log_ok(
                        logger,
                        stage=Stage.VALIDATE_OUTPUT,
                        job_id=job_id,
                        req_id=trace.req_id or result.get("req_id"),
                        product_id=str(pid),
                        duration_ms=duration * 1000.0,
                        retries=trace.retries,
                        model=trace.model or model,
                        prompt_tokens=trace.prompt_tokens,
                        response_tokens=trace.response_tokens,
                        http_status=trace.http_status,
                        artifacts=validation_artifacts.get(pid, []),
                    )
                else:
                    validation_artifacts[pid] = []

            db_start = time.perf_counter()
            try:
                _apply_ai_updates(conn, success_updates)
            except Exception as exc:
                db_duration_ms = (time.perf_counter() - db_start) * 1000.0
                reason_code = (
                    ReasonCode.DB_CONSTRAINT_VIOLATION
                    if isinstance(exc, sqlite3.IntegrityError)
                    else ReasonCode.DB_WRITE_ERROR
                )
                for pid, payload in success_updates.items():
                    trace = product_traces.get(pid)
                    artifacts = _persist_db_failure_artifacts(job_id, pid, trace, payload, exc) if trace else []
                    log_ko(
                        logger,
                        stage=Stage.WRITE_DB,
                        job_id=job_id,
                        req_id=(trace.req_id if trace else result.get("req_id")),
                        product_id=str(pid),
                        duration_ms=db_duration_ms,
                        retries=trace.retries if trace else retries,
                        model=(trace.model if trace else model),
                        prompt_tokens=trace.prompt_tokens if trace else None,
                        response_tokens=trace.response_tokens if trace else None,
                        reason_code=reason_code,
                        reason_detail=str(exc),
                        exception=exc,
                        http_status=trace.http_status if trace else None,
                        artifacts=artifacts,
                    )
                    ko_reason_counts[reason_code.value] += 1
                raise

            db_duration_ms = (time.perf_counter() - db_start) * 1000.0
            per_item_duration = db_duration_ms / max(1, len(success_updates))
            for pid, payload in success_updates.items():
                candidate = candidate_map.get(pid)
                sig_hash = candidate.sig_hash if candidate else ""
                success_records[pid] = {
                    "sig_hash": sig_hash,
                    "updates": payload.copy(),
                }
                parsed_desire = _parse_score(payload.get("desire_magnitude"))
                parsed_comp = _parse_score(payload.get("competition_level"))
                if parsed_desire is not None:
                    desire_scores.append((str(pid), parsed_desire))
                    success_records[pid]["_desire_score"] = parsed_desire
                if parsed_comp is not None:
                    comp_scores.append((str(pid), parsed_comp))
                    success_records[pid]["_competition_score"] = parsed_comp
                applied_outputs[pid] = {k: v for k, v in payload.items() if v is not None}
                trace = product_traces.get(pid)
                artifacts = validation_artifacts.get(pid, [])
                if trace:
                    trace.durations[Stage.WRITE_DB.value] = per_item_duration / 1000.0
                    write_artifacts = [a for a in artifacts if a.endswith("normalized_output.json")]
                    log_ok(
                        logger,
                        stage=Stage.WRITE_DB,
                        job_id=job_id,
                        req_id=trace.req_id or result.get("req_id"),
                        product_id=str(pid),
                        duration_ms=per_item_duration,
                        retries=trace.retries,
                        model=trace.model or model,
                        prompt_tokens=trace.prompt_tokens,
                        response_tokens=trace.response_tokens,
                        http_status=trace.http_status,
                        artifacts=write_artifacts or artifacts,
                    )

        counts["ok"] += len(success_updates)
        counts["ko"] += len(ko_map)

        throughput = (len(candidates_list) / duration) if duration > 0 else 0.0
        if job_updates_enabled:
            database.append_ai_job_metric(
                conn,
                int(job_id),
                processed_batches,
                len(candidates_list),
                duration * 1000.0,
                throughput,
                cached_hits=0,
            )

        done_val = counts["ok"] + counts["cached"]
        counts_with_cost = {**counts, "cost_spent_usd": cost_spent}
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

        if cost_cap is not None and cost_spent >= float(cost_cap) and not stop_event.is_set():
            stop_event.set()

    if batches:
        semaphore = asyncio.Semaphore(parallelism)

        async def _run_batches() -> None:
            limits = httpx.Limits(max_connections=100, max_keepalive_connections=100)
            timeout_cfg = httpx.Timeout(timeout_s)
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            async with httpx.AsyncClient(
                base_url="https://api.openai.com",
                timeout=timeout_cfg,
                limits=limits,
                headers=headers,
            ) as client:
                tasks = [
                    asyncio.create_task(
                        _call_batch_with_retries(
                            client,
                            batch,
                            model=model,
                            max_retries=max_retries,
                            rate_limiter=rate_limiter,
                            semaphore=semaphore,
                            stop_event=stop_event,
                            job_id=job_id,
                            product_traces=product_traces,
                            ko_reason_counts=ko_reason_counts,
                        )
                    )
                    for batch in batches
                ]
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    process_result(result)

        asyncio.run(_run_batches())
    else:
        counts_with_cost = {**counts, "cost_spent_usd": cost_spent}
        if job_updates_enabled:
            database.set_import_job_ai_counts(conn, int(job_id), counts_with_cost, sorted(pending_set))
        _emit_status(
            status_cb,
            phase="enrich",
            counts=counts_with_cost,
            total=total_items,
            done=counts["ok"] + counts["cached"],
        )

    result_error: Optional[str] = None
    if cost_cap is not None and cost_spent >= float(cost_cap):
        result_error = "cost_cap_reached"

    cfg_calib = config.get_ai_calibration_config()
    calibration_enabled = cfg_calib.get("enabled", True)
    fallback_cfg = cfg_calib.get("fallback_percentiles", {}) or {}
    fallback_desire = fallback_cfg.get("desire") or [0.33, 0.66]
    fallback_comp = fallback_cfg.get("competition") or [0.33, 0.66]

    desire_info: Dict[str, Any] = {}
    comp_info: Dict[str, Any] = {}
    dist_desire: Dict[str, int] = {}
    dist_comp: Dict[str, int] = {}

    if calibration_enabled and success_records:
        wins = float(cfg_calib.get("winsorize_pct", 0.05) or 0.0)
        min_low = float(cfg_calib.get("min_low_pct", 0.05) or 0.0)
        min_med = float(cfg_calib.get("min_medium_pct", 0.05) or 0.0)
        min_high = float(cfg_calib.get("min_high_pct", 0.05) or 0.0)

        weights_id = config.get_weights_version()
        cached_entry = _get_cached_calibration(weights_id, model)
        desire_thresholds = (cached_entry or {}).get("desire") or {
            "q33": float(fallback_desire[0]),
            "q67": float(fallback_desire[1]),
            "fallback": True,
        }
        comp_thresholds = (cached_entry or {}).get("competition") or {
            "q33": float(fallback_comp[0]),
            "q67": float(fallback_comp[1]),
            "fallback": True,
        }

        if cached_entry is None:
            began_tx = False
            try:
                if not conn.in_transaction:
                    conn.execute("BEGIN IMMEDIATE")
                    began_tx = True
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
                            conn.execute(
                                "UPDATE products SET desire_magnitude=? WHERE id=?",
                                (label, pid),
                            )
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
                if began_tx:
                    conn.commit()
            except Exception:
                if began_tx and conn.in_transaction:
                    conn.rollback()
                raise

            cache_payload = {
                "created_at": datetime.utcnow().isoformat(),
                "desire": {
                    "q33": float(desire_info.get("q33")) if desire_info.get("q33") is not None else float(desire_thresholds["q33"]),
                    "q67": float(desire_info.get("q67")) if desire_info.get("q67") is not None else float(desire_thresholds["q67"]),
                    "fallback": bool(desire_info.get("fallback")),
                },
                "competition": {
                    "q33": float(comp_info.get("q33")) if comp_info.get("q33") is not None else float(comp_thresholds["q33"]),
                    "q67": float(comp_info.get("q67")) if comp_info.get("q67") is not None else float(comp_thresholds["q67"]),
                    "fallback": bool(comp_info.get("fallback")),
                },
            }
            _set_cached_calibration(weights_id, model, cache_payload)
            if not desire_info:
                desire_info = {"fallback": bool(desire_thresholds.get("fallback"))}
            else:
                desire_info.setdefault("fallback", bool(desire_info.get("fallback")))
            if not comp_info:
                comp_info = {"fallback": bool(comp_thresholds.get("fallback"))}
            else:
                comp_info.setdefault("fallback", bool(comp_info.get("fallback")))
            desire_info["cached"] = False
            comp_info["cached"] = False
        else:
            desire_info = {
                "cached": True,
                "fallback": bool(desire_thresholds.get("fallback")),
                "q33": desire_thresholds.get("q33"),
                "q67": desire_thresholds.get("q67"),
            }
            comp_info = {
                "cached": True,
                "fallback": bool(comp_thresholds.get("fallback")),
                "q33": comp_thresholds.get("q33"),
                "q67": comp_thresholds.get("q67"),
            }
            desire_labels: Dict[str, str] = {}
            comp_labels: Dict[str, str] = {}
            if desire_scores:
                desire_labels, dist_desire = _apply_thresholds_with_minimums(
                    desire_scores,
                    desire_thresholds,
                    min_low_pct=min_low,
                    min_medium_pct=min_med,
                    min_high_pct=min_high,
                )
            if comp_scores:
                comp_labels, dist_comp = _apply_thresholds_with_minimums(
                    comp_scores,
                    comp_thresholds,
                    min_low_pct=min_low,
                    min_medium_pct=min_med,
                    min_high_pct=min_high,
                )
            if desire_labels or comp_labels:
                began_tx = False
                try:
                    if not conn.in_transaction:
                        conn.execute("BEGIN IMMEDIATE")
                        began_tx = True
                    for pid_str, label in desire_labels.items():
                        pid = int(pid_str)
                        rec = success_records.get(pid)
                        if rec and rec["updates"].get("desire_magnitude") != label:
                            conn.execute(
                                "UPDATE products SET desire_magnitude=? WHERE id=?",
                                (label, pid),
                            )
                            rec["updates"]["desire_magnitude"] = label
                            if pid in applied_outputs:
                                applied_outputs[pid]["desire_magnitude"] = label
                    for pid_str, label in comp_labels.items():
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
                    if began_tx:
                        conn.commit()
                except Exception:
                    if began_tx and conn.in_transaction:
                        conn.rollback()
                    raise

        logger.info(
            "ai_calibration_desire: dist=%s info=%s",
            dist_desire if desire_scores else {},
            desire_info,
        )
        logger.info(
            "ai_calibration_comp: dist=%s info=%s",
            dist_comp if comp_scores else {},
            comp_info,
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
    counts_with_cost = {**counts, "cost_spent_usd": cost_spent}
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

    latency_p50 = _percentile(request_latencies, 0.5) if request_latencies else 0.0
    latency_p95 = _percentile(request_latencies, 0.95) if request_latencies else 0.0

    try:
        job_artifacts_rel = str(job_artifacts_dir.relative_to(get_log_dir().parent))
    except Exception:
        job_artifacts_rel = str(job_artifacts_dir)

    logger.info(
        "run_ai_fill_job: job=%s total=%d ok=%d cached=%d ko=%d cost=%.4f pending=%d error=%s duration=%.2fs latency_p50=%.2fs latency_p95=%.2fs requests=%d job_artifacts_dir=%s",
        job_id,
        total_items,
        counts["ok"],
        counts["cached"],
        counts["ko"],
        cost_spent,
        len(pending_ids),
        result_error,
        time.perf_counter() - start_ts,
        latency_p50,
        latency_p95,
        len(request_latencies),
        job_artifacts_rel,
    )
    logger.info(
        "run_ai_fill_job.breakdown: job=%s ko_breakdown=%s job_artifacts_dir=%s",
        job_id,
        dict(ko_reason_counts),
        job_artifacts_rel,
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


def recalc_desire_for_all(
    db_conn: Optional[Any] = None,
    ids: Optional[Sequence[int]] = None,
    *,
    batch_size: int = 20,
    parallel: int = 3,
) -> int:
    """Rellena DESIRE para todos los items seleccionados."""

    close_conn = False
    conn = None
    if db_conn is None:
        conn = _ensure_conn()
        close_conn = True
    elif hasattr(db_conn, "cursor"):
        conn = db_conn
    else:
        conn = getattr(db_conn, "connection", None)
        if conn is None:
            conn = _ensure_conn()
            close_conn = True

    try:
        rows = [dict(row) for row in database.iter_products(conn, ids=ids)]
    finally:
        if close_conn and conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    pending: List[int] = []
    for row in rows:
        try:
            pid = int(row.get("id"))
        except Exception:
            continue
        title = str(row.get("name") or row.get("title") or "")
        desire_text = str(row.get("desire") or "").strip()
        if len(desire_text) < 280 or looks_like_product_desc(desire_text, title):
            pending.append(pid)

    if not pending:
        return 0

    processed = 0
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        if not chunk:
            continue
        run_ai_fill_job(
            job_id=None,
            product_ids=chunk,
            microbatch=batch_size,
            parallelism=parallel,
            status_cb=None,
        )
        processed += len(chunk)

    return processed


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
