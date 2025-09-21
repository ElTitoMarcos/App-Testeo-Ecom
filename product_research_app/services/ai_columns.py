from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from openai import AsyncOpenAI

from .. import config, database
from ..utils.signature import compute_sig_hash

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent.parent
DB_PATH = APP_DIR / "data.sqlite3"

AI_FIELDS = ("desire", "desire_magnitude", "awareness_level", "competition_level")
StatusCallback = Callable[..., None]

SYSTEM_PROMPT = (
    "Eres un analista de marketing. Aplica marcos de Breakthrough Advertising sin citar texto. "
    "Devuelve exclusivamente un JSON cuyas claves son los IDs de producto, y cuyos valores incluyen: "
    "desire (string), desire_magnitude (Low|Medium|High), awareness_level (Unaware|Problem-Aware|Solution-Aware|Product-Aware|Most Aware), "
    "competition_level (Low|Medium|High). No devuelvas comentarios, ni Markdown, ni bloques de código."
)
FALLBACK_MODEL = "gpt-4.1-mini"
DEFAULT_RESPONSE_FORMAT = {"type": "json_object"}
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


@dataclass
class Candidate:
    id: int
    sig_hash: str
    payload: Dict[str, Any]
    extra: Dict[str, Any]


class _AsyncRateLimiter:
    def __init__(
        self,
        rpm_limit: Optional[int],
        tpm_limit: Optional[int],
        est_in: int,
        est_out: int,
    ) -> None:
        self.rpm = int(rpm_limit or 0)
        self.tpm = int(tpm_limit or 0)
        self.est_in = max(0, int(est_in))
        self.est_out = max(0, int(est_out))
        self._lock = asyncio.Lock()
        self._next_request_ts = 0.0
        self._next_tokens_ts = 0.0

    async def acquire(self, item_count: int) -> None:
        if (self.rpm <= 0 and self.tpm <= 0) or item_count <= 0:
            return
        tokens = (self.est_in + self.est_out) * max(1, item_count)
        async with self._lock:
            now = time.monotonic()
            wait_until = now
            if self.tpm > 0 and tokens > 0:
                delay_tokens = (tokens / self.tpm) * 60.0
                start_tokens = max(now, self._next_tokens_ts)
                self._next_tokens_ts = start_tokens + delay_tokens
                wait_until = max(wait_until, start_tokens)
            if self.rpm > 0:
                delay_req = 60.0 / self.rpm
                start_req = max(now, self._next_request_ts)
                self._next_request_ts = start_req + delay_req
                wait_until = max(wait_until, start_req)
            wait = max(0.0, wait_until - now)
        if wait > 0:
            await asyncio.sleep(wait)


def _ensure_conn(db_path: Optional[Path | str] = None):
    target = Path(db_path) if db_path else DB_PATH
    conn = database.get_connection(target)
    database.initialize_database(conn)
    return conn


def _canonical(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFD", str(value))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z]", "", text.lower())


def _norm_tri(value: Any, default: str = "Medium") -> str:
    mapping = {
        "low": "Low",
        "baja": "Low",
        "medium": "Medium",
        "medio": "Medium",
        "med": "Medium",
        "high": "High",
        "alta": "High",
    }
    return mapping.get(_canonical(value), default)


def _norm_awareness(value: Any) -> str:
    mapping = {
        "unaware": "Unaware",
        "problemaware": "Problem-Aware",
        "problemaaware": "Problem-Aware",
        "problemaconsciente": "Problem-Aware",
        "solutionaware": "Solution-Aware",
        "solucionaware": "Solution-Aware",
        "solucionconsciente": "Solution-Aware",
        "productaware": "Product-Aware",
        "productoaware": "Product-Aware",
        "productoconciente": "Product-Aware",
        "mostaware": "Most Aware",
        "masaware": "Most Aware",
        "masconsciente": "Most Aware",
        "muyaware": "Most Aware",
    }
    return mapping.get(_canonical(value), "Problem-Aware")


def _strip_code_fences(text: str) -> str:
    txt = text.strip()
    if not txt.startswith("```"):
        return txt
    txt = txt[3:]
    if "\n" in txt:
        txt = txt.split("\n", 1)[1]
    if txt.endswith("```"):
        txt = txt[:-3]
    return txt.strip()


def _extract_first_json_block(text: str) -> Any:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        return obj
    raise ValueError("no_json_found")


def _load_json_payload(text: str) -> Any:
    cleaned = _strip_code_fences(text)
    if not cleaned:
        raise ValueError("empty_response")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return _extract_first_json_block(cleaned)


def _build_user_payload(items: List[Dict[str, Any]]) -> str:
    payload = {"items": items}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def _parse_batch_output(
    data: Any,
    items: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    if not isinstance(data, dict):
        raise ValueError("invalid_json")
    ok: Dict[str, Dict[str, Any]] = {}
    ko: Dict[str, str] = {}
    for item in items:
        pid = item.get("id")
        pid_str = str(pid)
        entry = data.get(pid_str)
        if entry is None and isinstance(pid, (int, float)):
            entry = data.get(int(pid))
        if entry is None:
            ko[pid_str] = "missing"
            continue
        if not isinstance(entry, dict):
            ko[pid_str] = "invalid_entry"
            continue
        ok[pid_str] = {
            "desire": entry.get("desire"),
            "desire_magnitude": _norm_tri(entry.get("desire_magnitude")),
            "awareness_level": _norm_awareness(entry.get("awareness_level")),
            "competition_level": _norm_tri(entry.get("competition_level")),
        }
    return ok, ko


def _coerce_usage(resp: Any) -> Dict[str, Any]:
    usage = getattr(resp, "usage", None)
    if not usage:
        return {}
    if isinstance(usage, dict):
        return usage
    for attr in ("model_dump", "dict"):
        fn = getattr(usage, attr, None)
        if callable(fn):
            try:
                data = fn()
            except Exception:
                continue
            if isinstance(data, dict):
                return data
    result: Dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"):
        if hasattr(usage, key):
            result[key] = getattr(usage, key)
    return result


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


def _extract_status_code(exc: Exception) -> int:
    for attr in ("status_code", "status", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    message = str(exc)
    for token in message.replace(":", " ").split():
        if token.isdigit():
            try:
                return int(token)
            except Exception:
                continue
    return 0


async def _call_batch_with_retries(
    client: AsyncOpenAI,
    model: str,
    items: List[Dict[str, Any]],
    *,
    limiter: Optional[_AsyncRateLimiter],
    max_retries: int,
    temperature: float,
    top_p: float,
    response_format: Dict[str, Any],
    fallback_model: Optional[str],
) -> Dict[str, Any]:
    attempt = 0
    retries = 0
    current_model = model
    fallback_used = False
    last_error: Optional[str] = None
    fmt = response_format or DEFAULT_RESPONSE_FORMAT

    while attempt <= max_retries:
        if limiter:
            await limiter.acquire(len(items))
        payload = _build_user_payload(items)
        start = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model=current_model,
                temperature=temperature,
                top_p=top_p,
                response_format=fmt,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": payload},
                ],
            )
            duration = time.perf_counter() - start
            choices = getattr(resp, "choices", []) or []
            if not choices:
                raise ValueError("empty_choices")
            first_choice = choices[0]
            message = getattr(first_choice, "message", None)
            content = ""
            if isinstance(message, dict):
                content = message.get("content") or ""
            else:
                content = getattr(message, "content", "") if message is not None else ""
            if not content and isinstance(first_choice, dict):
                content = (
                    first_choice.get("message", {}).get("content")
                    if isinstance(first_choice.get("message"), dict)
                    else content
                )
            parsed = _load_json_payload(content or "")
            ok_map, ko_map = _parse_batch_output(parsed, items)
            usage = _coerce_usage(resp)
            return {
                "ok": ok_map,
                "ko": ko_map,
                "usage": usage,
                "duration": float(duration),
                "retries": retries,
                "model_used": current_model,
                "error": None,
            }
        except Exception as exc:
            duration = time.perf_counter() - start
            last_error = str(exc)
            status = _extract_status_code(exc)
            message = last_error.lower()
            if (
                fallback_model
                and not fallback_used
                and current_model == model
                and (status == 404 or "unsupported" in message or "not found" in message)
            ):
                current_model = fallback_model
                fallback_used = True
                continue
            if attempt < max_retries and (status in _RETRYABLE_STATUS or status >= 500 or status == 0):
                delay = min(2.0, 0.5 * (2**attempt))
                if delay > 0:
                    await asyncio.sleep(delay)
                attempt += 1
                retries += 1
                continue
            return {
                "ok": {},
                "ko": {str(item.get("id")): last_error for item in items},
                "usage": {},
                "duration": float(duration),
                "retries": retries,
                "model_used": current_model,
                "error": last_error,
            }
    # Exceeded retries without success.
    error_message = last_error or "max_retries_exceeded"
    return {
        "ok": {},
        "ko": {str(item.get("id")): error_message for item in items},
        "usage": {},
        "duration": 0.0,
        "retries": retries,
        "model_used": current_model,
        "error": error_message,
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



async def run_ai_fill_job(
    job_id: int,
    product_ids: Sequence[int],
    *,
    microbatch: int = 32,
    parallelism: Optional[int] = None,
    status_cb: Optional[StatusCallback] = None,
    db_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    start_ts = time.perf_counter()
    conn = _ensure_conn(db_path)
    try:
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

        if sig_updates:
            cur = conn.cursor()
            for sig_hash, pid in sig_updates:
                cur.execute("UPDATE products SET sig_hash=? WHERE id=?", (sig_hash, pid))
            conn.commit()

        total_items = len(candidates)

        runtime_cfg = config.get_ai_runtime_config()
        if parallelism is None:
            parallelism = int(runtime_cfg.get("parallelism", 8) or 8)
        parallelism = max(1, parallelism)

        microbatch_size = int(microbatch or runtime_cfg.get("microbatch", 32) or 32)
        if microbatch_size < 24:
            microbatch_size = 24
        if microbatch_size > 64:
            microbatch_size = 64

        cache_enabled = bool(runtime_cfg.get("cache_enabled", True))
        cache_version = int(runtime_cfg.get("version", 1) or 1)

        rpm_limit = runtime_cfg.get("rpm_limit")
        if rpm_limit is not None:
            try:
                rpm_limit = int(rpm_limit)
            except Exception:
                rpm_limit = None
            else:
                if rpm_limit <= 0:
                    rpm_limit = None

        tpm_limit = runtime_cfg.get("tpm_limit")
        if tpm_limit is not None:
            try:
                tpm_limit = int(tpm_limit)
            except Exception:
                tpm_limit = None
            else:
                if tpm_limit <= 0:
                    tpm_limit = None

        batch_cfg = config.get_ai_batch_config()
        max_retries = int(batch_cfg.get("MAX_RETRIES", 3) or 3)

        cost_cfg = config.get_ai_cost_config()
        runtime_model = str(runtime_cfg.get("model") or cost_cfg.get("model") or config.get_model())
        cost_cap_val = runtime_cfg.get("costCapUSD")
        if cost_cap_val is None:
            cost_cap_val = cost_cfg.get("costCapUSD")
        try:
            cost_cap = float(cost_cap_val) if cost_cap_val is not None else None
        except Exception:
            cost_cap = None
        price_table = cost_cfg.get("prices", {})
        est_tokens_in = int(cost_cfg.get("estTokensPerItemIn", 0) or 0)
        est_tokens_out = int(cost_cfg.get("estTokensPerItemOut", 0) or 0)

        api_key = (
            config.get_api_key()
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ENRICH_API_KEY")
        )

        client: Optional[AsyncOpenAI] = None

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
        last_error: Optional[str] = None

        if total_items == 0:
            if job_updates_enabled and skipped_existing:
                database.set_import_job_ai_counts(conn, int(job_id), counts_with_cost, [])
            _emit_status(status_cb, phase="enrich", counts=counts_with_cost, total=total_items, done=0)
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
                    model=runtime_model,
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
                        model=runtime_model,
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
            last_error = "missing_api_key"
            for pid in list(pending_set):
                fail_reasons[pid] = last_error
            counts_with_cost = {**counts, "cost_spent_usd": cost_spent}
            if job_updates_enabled:
                database.set_import_job_ai_counts(conn, int(job_id), counts_with_cost, sorted(pending_set))
                database.set_import_job_ai_error(conn, int(job_id), last_error)
            _emit_status(
                status_cb,
                phase="enrich",
                counts=counts_with_cost,
                total=total_items,
                done=counts["cached"],
                message="IA pendiente",
            )
            return {
                "counts": counts_with_cost,
                "pending_ids": sorted(pending_set),
                "error": last_error,
                "ok": applied_outputs,
                "ko": fail_reasons,
                "skipped_existing": skipped_existing,
                "total_requested": len(requested_ids),
            }

        batches: List[Tuple[int, List[Candidate]]] = []
        for idx in range(0, len(remaining), microbatch_size):
            chunk = remaining[idx : idx + microbatch_size]
            if not chunk:
                continue
            batches.append((len(batches) + 1, chunk))

        limiter = _AsyncRateLimiter(rpm_limit, tpm_limit, est_tokens_in, est_tokens_out)
        stop_event = asyncio.Event()

        desire_scores: List[Tuple[str, float]] = []
        comp_scores: List[Tuple[str, float]] = []
        success_records: Dict[int, Dict[str, Any]] = {}

        temperature = float(runtime_cfg.get("temperature", 0.0) or 0.0)
        top_p = float(runtime_cfg.get("top_p", 0.0) or 0.0)
        response_format_cfg = runtime_cfg.get("response_format")
        if isinstance(response_format_cfg, dict):
            response_format = response_format_cfg
        elif isinstance(response_format_cfg, str):
            response_format = DEFAULT_RESPONSE_FORMAT if response_format_cfg.lower() == "json" else DEFAULT_RESPONSE_FORMAT
        else:
            response_format = DEFAULT_RESPONSE_FORMAT
        fallback_model = FALLBACK_MODEL if runtime_model == "gpt-4o-mini" else None

        client = AsyncOpenAI(api_key=api_key)

        try:
            counts_with_cost = {**counts, "cost_spent_usd": cost_spent}
            if batches:
                sem = asyncio.Semaphore(parallelism)
                in_flight: set[asyncio.Task[Dict[str, Any]]] = set()
                next_batch_index = 0

                async def submit_batch(batch_no: int, cand_list: List[Candidate]) -> Dict[str, Any]:
                    items_payload = [cand.payload for cand in cand_list]
                    if stop_event.is_set():
                        return {"batch_no": batch_no, "items": items_payload, "skipped": True}
                    try:
                        async with sem:
                            if stop_event.is_set():
                                return {"batch_no": batch_no, "items": items_payload, "skipped": True}
                            result = await _call_batch_with_retries(
                                client,
                                runtime_model,
                                items_payload,
                                limiter=limiter,
                                max_retries=max_retries,
                                temperature=temperature,
                                top_p=top_p,
                                response_format=response_format,
                                fallback_model=fallback_model,
                            )
                    except Exception as exc:
                        error_text = str(exc)
                        return {
                            "batch_no": batch_no,
                            "items": items_payload,
                            "ok": {},
                            "ko": {str(item.get("id")): error_text for item in items_payload},
                            "usage": {},
                            "duration": 0.0,
                            "retries": 0,
                            "error": error_text,
                        }
                    result.setdefault("ok", {})
                    result.setdefault("ko", {})
                    result.setdefault("usage", {})
                    result.setdefault("duration", 0.0)
                    result.setdefault("retries", 0)
                    result["batch_no"] = batch_no
                    result["items"] = items_payload
                    return result

                while next_batch_index < len(batches) or in_flight:
                    while (
                        next_batch_index < len(batches)
                        and len(in_flight) < parallelism
                        and not stop_event.is_set()
                    ):
                        batch_no, cand_chunk = batches[next_batch_index]
                        task = asyncio.create_task(submit_batch(batch_no, cand_chunk))
                        in_flight.add(task)
                        next_batch_index += 1
                    if not in_flight:
                        break
                    done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        try:
                            result = task.result()
                        except Exception:
                            logger.exception("AI batch task failed")
                            continue
                        batch_items = result.get("items", [])
                        if result.get("skipped"):
                            for item in batch_items:
                                try:
                                    pid = int(item.get("id"))
                                except Exception:
                                    continue
                                pending_set.add(pid)
                                fail_reasons.setdefault(pid, "cost_cap_reached")
                            continue
                        counts["sent"] += len(batch_items)
                        retries_used = int(result.get("retries", 0) or 0)
                        counts["retried"] += retries_used
                        ok_map: Dict[str, Dict[str, Any]] = result.get("ok", {}) or {}
                        ko_map: Dict[str, str] = result.get("ko", {}) or {}
                        duration = float(result.get("duration", 0.0) or 0.0)
                        usage = result.get("usage", {}) or {}
                        model_used = result.get("model_used") or runtime_model
                        if usage:
                            price_info = price_table.get(model_used, {})
                            if not isinstance(price_info, dict):
                                price_info = {}
                            price_in = float(price_info.get("input", 0.0) or 0.0)
                            price_out = float(price_info.get("output", 0.0) or 0.0)
                            cost_spent += _calculate_cost(usage, price_in, price_out)
                        success_updates: Dict[int, Dict[str, Any]] = {}
                        for pid_str, payload in ok_map.items():
                            try:
                                pid = int(pid_str)
                            except Exception:
                                continue
                            success_updates[pid] = payload
                            pending_set.discard(pid)
                        for pid_str, reason in ko_map.items():
                            try:
                                pid = int(pid_str)
                            except Exception:
                                continue
                            pending_set.add(pid)
                            fail_reasons[pid] = reason or "error"
                            last_error = reason or "error"
                        if success_updates:
                            _apply_ai_updates(conn, success_updates)
                            for pid, payload in success_updates.items():
                                candidate = candidate_map.get(pid)
                                sig_hash = candidate.sig_hash if candidate else ""
                                success_records[pid] = {
                                    "sig_hash": sig_hash,
                                    "model_used": model_used,
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
                        if result.get("error"):
                            last_error = str(result.get("error"))
                        counts["ok"] += len(success_updates)
                        counts["ko"] += len(ko_map)
                        throughput = (len(batch_items) / duration) if duration > 0 else 0.0
                        if job_updates_enabled:
                            database.append_ai_job_metric(
                                conn,
                                int(job_id),
                                result.get("batch_no", 0),
                                len(batch_items),
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
                            last_error = "cost_cap_reached"
                counts_with_cost = {**counts, "cost_spent_usd": cost_spent}
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
        finally:
            if client is not None:
                close_fn = getattr(client, "close", None)
                if close_fn:
                    try:
                        result_close = close_fn()
                    except Exception:
                        result_close = None
                    if inspect.iscoroutine(result_close):
                        try:
                            await result_close
                        except Exception:
                            pass

        if cost_cap is not None and cost_spent >= float(cost_cap):
            last_error = last_error or "cost_cap_reached"

        cfg_calib = config.get_ai_calibration_config()
        if cfg_calib.get("enabled", True) and success_records:
            wins = float(cfg_calib.get("winsorize_pct", 0.05) or 0.0)
            min_low = float(cfg_calib.get("min_low_pct", 0.05) or 0.0)
            min_med = float(cfg_calib.get("min_medium_pct", 0.05) or 0.0)
            min_high = float(cfg_calib.get("min_high_pct", 0.05) or 0.0)
            desire_info: Dict[str, Any] = {}
            comp_info: Dict[str, Any] = {}
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
                cache_model = rec.get("model_used", runtime_model)
                database.upsert_ai_cache_entry(
                    conn,
                    sig_hash,
                    model=cache_model,
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
            if last_error:
                database.set_import_job_ai_error(conn, int(job_id), last_error)
        _emit_status(
            status_cb,
            phase="enrich",
            counts=counts_with_cost,
            total=total_items,
            done=done_val,
            message=f"IA columnas {done_val}/{total_items}",
        )

        logger.info(
            "run_ai_fill_job: job=%s total=%d ok=%d cached=%d ko=%d cost=%.4f pending=%d error=%s duration=%.2fs",
            job_id,
            total_items,
            counts["ok"],
            counts["cached"],
            counts["ko"],
            cost_spent,
            len(pending_ids),
            last_error,
            time.perf_counter() - start_ts,
        )

        return {
            "counts": counts_with_cost,
            "pending_ids": pending_ids,
            "error": last_error,
            "ok": applied_outputs,
            "ko": fail_reasons,
            "skipped_existing": skipped_existing,
            "total_requested": len(requested_ids),
        }
    finally:
        conn.close()


def fill_ai_columns(
    product_ids: Sequence[int],
    *,
    model: Optional[str] = None,
    batch_mode: Optional[bool] = None,
    cost_cap_usd: Optional[float] = None,
) -> Dict[str, Any]:
    result = asyncio.run(run_ai_fill_job(0, product_ids))
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
