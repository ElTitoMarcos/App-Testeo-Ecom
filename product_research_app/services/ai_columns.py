from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

from .. import config, database, gpt
from ..config import (
    AI_DEGRADE_FACTOR,
    AI_MAX_OUTPUT_TOKENS,
    AI_MAX_PRODUCTS_PER_CALL,
    AI_MIN_PRODUCTS_PER_CALL,
    get_ai_cost_config,
)
from ..ai.strict_jsonl import parse_jsonl_and_validate
from ..obs import log_partial_ko, log_recovered
from ..utils.signature import compute_sig_hash
from .desire_utils import cleanse, looks_like_product_desc
from .prompt_templates import MISSING_ONLY_JSONL_PROMPT, STRICT_JSONL_PROMPT

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent.parent
DB_PATH = APP_DIR / "data.sqlite3"
CALIBRATION_CACHE_FILE = APP_DIR / "ai_calibration_cache.json"

AI_FIELDS = ("desire", "desire_magnitude", "awareness_level", "competition_level")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


DEFAULT_BATCH_SIZE = max(8, _env_int("PRAPP_AI_COLUMNS_BATCH_SIZE", 32))
MIN_BATCH_SIZE = 8
MAX_RETRIES_MISSING = _env_int("PRAPP_AI_RETRIES_MISSING", 2)

RAW_MAX_CONCURRENCY = max(1, _env_int("PRAPP_OPENAI_MAX_CONCURRENCY", 8))
OPENAI_HEADROOM = max(0.1, min(0.99, _env_float("PRAPP_OPENAI_HEADROOM", 0.90)))
AI_MAX_CONCURRENCY = max(
    1,
    min(
        RAW_MAX_CONCURRENCY,
        int(max(1, math.floor(RAW_MAX_CONCURRENCY * OPENAI_HEADROOM))),
    ),
)
AI_BATCH_MAX_ITEMS = _env_int("AI_BATCH_MAX_ITEMS", 12)

STRICT_JSON_ENABLED = _env_bool("PRAPP_AI_COLUMNS_STRICT_JSON", True)
MAX_TOKENS_PER_ITEM = max(32, _env_int("PRAPP_AI_COLUMNS_MAX_TOKENS_PER_ITEM", 128))
SAFE_CONTEXT_TOKENS = max(4000, _env_int("PRAPP_OPENAI_CONTEXT_SAFE_TOKENS", 120000))
StatusCallback = Callable[..., None]

TRIAGE_ENABLED = _env_bool("PRAPP_AI_TRIAGE_ENABLED", True)
TRIAGE_MODEL = os.getenv("PRAPP_AI_TRIAGE_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
TRIAGE_CONFIDENCE = max(0.0, min(1.0, _env_float("PRAPP_AI_TRIAGE_CONFIDENCE", 0.80)))

AI_COST = get_ai_cost_config()
try:
    EST_OUT_PER_ITEM = int((AI_COST or {}).get("estTokensPerItemOut", 80))
except Exception:
    EST_OUT_PER_ITEM = 80


def _persist_rows(rows: List[Dict[str, Any]]) -> None:
    """Actualiza columnas de IA en la tabla product."""

    from product_research_app import db

    conn = db.get_conn()
    for r in rows:
        conn.execute(
            """
            UPDATE product SET
              desire = COALESCE(:desire, desire),
              desire_magnitude = COALESCE(:desire_magnitude, desire_magnitude),
              awareness_level = COALESCE(:awareness_level, awareness_level),
              competition_level = COALESCE(:competition_level, competition_level)
            WHERE id = :id
            """,
            r,
        )
    conn.commit()


def _max_tokens_for_batch(batch_len: int) -> int:
    est = EST_OUT_PER_ITEM * max(1, batch_len) + 200
    return min(max(800, est), max(800, AI_MAX_OUTPUT_TOKENS), 4000)


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
    product_map: Dict[int, str]
    depth: int = 0
    json_retry_count: int = 0
    adapted: bool = False
    trunc_title: int = 0
    trunc_desc: int = 0

class BatchAdaptationRequired(Exception):
    """Raised when a batch should be split before retrying."""

    def __init__(self, reason: str, message: str, *, allow_split: bool = True):
        super().__init__(message)
        self.reason = reason
        self.allow_split = allow_split


def _estimate_batch_tokens(batch: BatchRequest) -> int:
    """Estimate total tokens (prompt + completion headroom) for a batch."""

    completion_budget = len(batch.candidates) * MAX_TOKENS_PER_ITEM
    return max(0, int(batch.prompt_tokens_est)) + completion_budget


def _should_downshift(batch: BatchRequest) -> bool:
    if len(batch.candidates) <= MIN_BATCH_SIZE:
        return False
    estimated = _estimate_batch_tokens(batch)
    return estimated >= SAFE_CONTEXT_TOKENS


def _detect_adaptation_reason(exc: Exception) -> Optional[str]:
    message = str(exc).lower()
    status = getattr(exc, "status_code", None)
    try:
        status = int(status) if status is not None else None
    except Exception:
        status = None
    if status == 429 or "rate limit" in message or "too many requests" in message:
        return "rate_limit"
    context_markers = ("context", "maximum context", "max_tokens", "length")
    if any(marker in message for marker in context_markers):
        return "context"
    return None


def _format_job_id(job_id: Optional[int]) -> str:
    """Return a stable string label for job identifiers."""

    try:
        numeric = int(job_id) if job_id is not None else 0
    except Exception:
        return str(job_id)
    if numeric <= 0:
        return "adhoc"
    return str(numeric)


def _sanitize_reason(reason: Optional[str], *, limit: int = 160) -> str:
    """Normalize KO reason text for safe logging."""

    if reason is None:
        return ""
    text = str(reason).strip().replace("\n", " ").replace("\r", " ")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _classify_ko_reason(reason: Optional[str]) -> Dict[str, Any]:
    """Map raw KO reason strings to structured diagnostics."""

    text = str(reason or "").strip()
    lowered = text.lower()

    def _payload(code: str, message: str, *, retryable: bool, remediation: str) -> Dict[str, Any]:
        return {
            "code": code,
            "message": message,
            "retryable": retryable,
            "remediation": remediation,
        }

    if not text:
        return _payload(
            "ko.unknown",
            "No failure reason was provided by the upstream component.",
            retryable=True,
            remediation="Inspect upstream service logs and retry the job to reproduce the issue.",
        )

    if "missing_api_key" in lowered or "api key" in lowered:
        return _payload(
            "auth.missing_api_key",
            "AI request failed because no API credentials were configured.",
            retryable=False,
            remediation="Provide a valid API key in the environment or settings and restart the job.",
        )

    if "cost_cap" in lowered or "cost cap" in lowered or "budget" in lowered:
        return _payload(
            "quota.cost_cap_reached",
            "AI job halted after hitting the configured cost cap.",
            retryable=False,
            remediation="Increase the AI cost cap or resume once the budget window resets.",
        )

    if "rate limit" in lowered or "429" in lowered or "too many requests" in lowered:
        return _payload(
            "remote.rate_limited",
            "OpenAI rejected the request due to rate limiting.",
            retryable=True,
            remediation="Reduce concurrency or request smaller batches, then retry once the rate limit window clears.",
        )

    if "timeout" in lowered or "timed out" in lowered:
        return _payload(
            "remote.timeout",
            "The request to the AI provider timed out before completion.",
            retryable=True,
            remediation="Retry the job or increase the timeout in the AI runtime configuration.",
        )

    if any(keyword in lowered for keyword in ("connection", "network", "ssl", "handshake", "dns")):
        return _payload(
            "remote.network_error",
            "A network or TLS problem prevented the AI response.",
            retryable=True,
            remediation="Check network connectivity to the AI provider and retry once connectivity is restored.",
        )

    if "invalid" in lowered and "json" in lowered:
        return _payload(
            "response.invalid_json",
            "AI provider returned an invalid or unparsable JSON payload.",
            retryable=True,
            remediation="Inspect the AI response format; adjust the prompt or enable retries with smaller batches.",
        )

    if "unauthorized" in lowered or "401" in lowered:
        return _payload(
            "auth.unauthorized",
            "The AI provider rejected the credentials for this request.",
            retryable=False,
            remediation="Verify the configured API key has access to the requested model.",
        )

    if "forbidden" in lowered or "403" in lowered:
        return _payload(
            "auth.forbidden",
            "The AI provider blocked the request due to insufficient permissions.",
            retryable=False,
            remediation="Confirm the account is permitted to use the selected model or endpoint.",
        )

    if "not found" in lowered or "404" in lowered:
        return _payload(
            "remote.not_found",
            "The AI endpoint or resource referenced in the request was not found.",
            retryable=False,
            remediation="Verify the model identifier and endpoint URL in the configuration.",
        )

    if "missing" in lowered:
        return _payload(
            "data.missing_ai_output",
            "The AI response omitted one or more requested product entries.",
            retryable=True,
            remediation="Retry the batch; if the issue persists, review the prompt and source data for the affected IDs.",
        )

    if any(keyword in lowered for keyword in ("500", "502", "503", "504", "server error", "bad gateway")):
        return _payload(
            "remote.server_error",
            "The AI provider returned an internal server error.",
            retryable=True,
            remediation="Retry the request after a short delay; contact the provider if errors persist.",
        )

    return _payload(
        "ko.unclassified",
        "The KO reason did not match known failure categories.",
        retryable=True,
        remediation="Review upstream logs and input payloads to determine the unexpected failure pattern.",
    )


def _build_ko_diagnostics(
    reasons: Mapping[str, Any],
    *,
    candidate_lookup: Optional[Mapping[int, Candidate]] = None,
    req_id: Optional[str] = None,
    max_samples: int = 5,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """Aggregate KO reasons into structured summaries for logging."""

    summary: Dict[str, Dict[str, Any]] = {}
    detail_samples: List[Dict[str, Any]] = []
    candidate_lookup = candidate_lookup or {}

    for index, (pid_key, raw_reason) in enumerate(reasons.items()):
        full_reason = str(raw_reason or "")
        diag = _classify_ko_reason(full_reason)
        sanitized = _sanitize_reason(full_reason)
        entry = summary.setdefault(
            diag["code"],
            {
                "count": 0,
                "message": diag["message"],
                "retryable": diag["retryable"],
                "remediation": diag["remediation"],
                "sample_product_ids": [],
            },
        )
        entry["count"] += 1

        try:
            pid = int(pid_key)
        except Exception:
            pid = None

        if pid is not None and len(entry["sample_product_ids"]) < max_samples:
            entry["sample_product_ids"].append(pid)

        if index < max_samples:
            detail: Dict[str, Any] = {
                "code": diag["code"],
                "diagnostic": diag["message"],
                "raw_reason": sanitized,
                "retryable": diag["retryable"],
                "remediation": diag["remediation"],
                "req_id": req_id,
            }
            if pid is not None:
                detail["product_id"] = pid
                detail["data_path"] = f"{DB_PATH}#table=products&id={pid}"
                candidate = candidate_lookup.get(pid)
                detail["sig_hash_present"] = bool(candidate and candidate.sig_hash)
            else:
                detail["product_id"] = pid_key
                detail["data_path"] = str(DB_PATH)
                detail["sig_hash_present"] = False
            detail_samples.append(detail)

    return summary, detail_samples


def _log_ko_event(
    event: str,
    reasons: Mapping[str, Any],
    *,
    job_id: Optional[int],
    candidate_lookup: Optional[Mapping[int, Candidate]] = None,
    req_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if not reasons:
        return

    normalized: Dict[str, Any] = {str(key): value for key, value in reasons.items()}
    summary, samples = _build_ko_diagnostics(
        normalized,
        candidate_lookup=candidate_lookup,
        req_id=req_id,
    )
    if not summary and not samples:
        return

    payload: Dict[str, Any] = {
        "job_id": _format_job_id(job_id),
        "ko_total": sum(item["count"] for item in summary.values()),
        "data_path": str(DB_PATH),
        "source_table": "products",
        "summary": summary,
        "samples": samples,
    }
    if req_id:
        payload.setdefault("req_id", req_id)
    if extra:
        payload.update(extra)
    logger.warning("ai_columns.%s: %s", event, payload)


SYSTEM_PROMPT = (
    "Eres un analista de marketing. Devuelve exclusivamente un array JSON. "
    "Cada elemento debe corresponder al producto solicitado en el mismo orden y contener exactamente las claves: "
    "aw, aw_m, d, d_m, c, c_m, confidence. No añadas notas ni texto adicional. "
    "aw y c deben ser 'Low', 'Medium' o 'High'. aw_m, d_m y c_m son números entre 0 y 100. "
    "confidence es un número entre 0 y 1 (por ejemplo 0.87)."
)
USER_INSTRUCTION = (
    "Analiza los siguientes productos y responde únicamente con un array JSON siguiendo el formato indicado. "
    "El número de objetos debe coincidir con los product_id solicitados y mantener el mismo orden."
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
    *,
    depth: int = 0,
    json_retry_count: int = 0,
    adapted: bool = False,
) -> BatchRequest:
    product_lines: List[str] = []
    product_map: Dict[int, str] = {}
    for cand in candidates:
        product_payload = _build_product_payload(cand, trunc_title, trunc_desc)
        line = json.dumps(product_payload, ensure_ascii=False, separators=(",", ":"))
        product_lines.append(line)
        product_map[cand.id] = line
    ids = [cand.id for cand in candidates]
    instruction = STRICT_JSONL_PROMPT(ids, AI_FIELDS)
    user_parts = [instruction, USER_INSTRUCTION, "### PRODUCTOS", *product_lines]
    user_text = "\n".join(part for part in user_parts if part)
    prompt_tokens_est = _estimate_tokens_from_text(SYSTEM_PROMPT, user_text) + len(candidates) * 8
    return BatchRequest(
        req_id=req_id,
        candidates=candidates,
        user_text=user_text,
        prompt_tokens_est=prompt_tokens_est,
        product_map=product_map,
        depth=depth,
        json_retry_count=json_retry_count,
        adapted=adapted,
        trunc_title=trunc_title,
        trunc_desc=trunc_desc,
    )


def _extract_jsonl_payload(raw: Dict[str, Any]) -> str:
    parsed_json, text_content = gpt._parse_message_content(raw)
    if isinstance(parsed_json, dict):
        records: List[Dict[str, Any]] = []
        if isinstance(parsed_json.get("results"), list):
            for item in parsed_json.get("results", []):
                if isinstance(item, dict):
                    records.append(item)
        else:
            for key, value in parsed_json.items():
                if not isinstance(value, dict):
                    continue
                item = dict(value)
                try:
                    pid = int(key)
                except Exception:
                    pid = value.get("product_id")
                if pid is not None and "product_id" not in item:
                    item["product_id"] = pid
                records.append(item)
        if records:
            return "\n".join(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                for record in records
            )
    if isinstance(parsed_json, list):
        records = [
            json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            for item in parsed_json
            if isinstance(item, dict)
        ]
        if records:
            return "\n".join(records)
    if isinstance(text_content, str) and text_content.strip():
        return text_content.strip()
    raise gpt.InvalidJSONError("Respuesta IA no es JSONL")


def _parse_jsonl_loose(text: str) -> Dict[int, Dict[str, Any]]:
    result: Dict[int, Dict[str, Any]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        pid = obj.get("product_id")
        if isinstance(pid, int):
            result[pid] = obj
    return result


def _build_missing_prompt(batch: BatchRequest, missing_ids: List[int]) -> str:
    prompt = MISSING_ONLY_JSONL_PROMPT(missing_ids, AI_FIELDS)
    product_lines = [batch.product_map.get(pid, "") for pid in missing_ids]
    body = "\n".join(line for line in product_lines if line)
    return "\n".join(part for part in (prompt, "### PRODUCTOS", body) if part)


def _parse_strict_json_payload(
    raw: Dict[str, Any],
    expected_ids: Sequence[int],
    *,
    require_confidence: bool = False,
) -> Tuple[Dict[int, Dict[str, Any]], str]:
    parsed_json, text_content = gpt._parse_message_content(raw)
    payload_list: Optional[List[Any]] = None
    if isinstance(parsed_json, list):
        payload_list = list(parsed_json)
    elif isinstance(parsed_json, dict):
        maybe = parsed_json.get("results")
        if isinstance(maybe, list):
            payload_list = list(maybe)
    raw_text: Optional[str] = None
    if payload_list is None:
        source_text = text_content
        if not source_text and parsed_json is not None:
            try:
                source_text = json.dumps(parsed_json, ensure_ascii=False)
            except Exception:
                source_text = None
        if not source_text:
            raise ValueError("Respuesta vacía o sin JSON")
        try:
            loaded = json.loads(source_text)
        except Exception as exc:
            raise ValueError("Respuesta no es JSON válido") from exc
        if not isinstance(loaded, list):
            raise ValueError("Se esperaba un array JSON")
        payload_list = list(loaded)
        raw_text = source_text
    if payload_list is None:
        raise ValueError("Respuesta sin array JSON")
    if len(payload_list) != len(expected_ids):
        raise ValueError(
            f"Longitud incorrecta: respuesta={len(payload_list)} esperada={len(expected_ids)}"
        )
    result: Dict[int, Dict[str, Any]] = {}
    serialisable: List[Dict[str, Any]] = []
    for pid, item in zip(expected_ids, payload_list):
        if not isinstance(item, dict):
            raise ValueError("Cada elemento del array debe ser un objeto JSON")
        aw_raw = item.get("aw")
        c_raw = item.get("c")
        d_raw = item.get("d")
        aw_score = _coerce_percent(item.get("aw_m"), label="aw_m")
        desire_score = _coerce_percent(item.get("d_m"), label="d_m")
        comp_score = _coerce_percent(item.get("c_m"), label="c_m")
        confidence = _coerce_confidence(item.get("confidence"), required=require_confidence)
        if not isinstance(aw_raw, str):
            raise ValueError("aw debe ser string")
        if not isinstance(c_raw, str):
            raise ValueError("c debe ser string")
        if not isinstance(d_raw, (str, type(None))):
            raise ValueError("d debe ser string")
        aw_label = aw_raw.strip().title()
        comp_label = c_raw.strip().title()
        if aw_label not in {"Low", "Medium", "High"}:
            raise ValueError("aw fuera de rango")
        if comp_label not in {"Low", "Medium", "High"}:
            raise ValueError("c fuera de rango")
        desire_text = cleanse(d_raw or "").strip()
        if not desire_text and isinstance(d_raw, str):
            desire_text = d_raw.strip()
        desire_label = _tri_label_from_percent(desire_score)
        comp_level = _tri_label_from_percent(comp_score)
        awareness = _map_awareness_bucket(aw_label, aw_score)
        audit_payload = {
            "aw_bucket": aw_label,
            "aw_pct": aw_score,
            "desire_pct": desire_score,
            "competition_pct": comp_score,
        }
        if confidence is not None:
            audit_payload["confidence"] = confidence
        result[int(pid)] = {
            "product_id": int(pid),
            "desire": desire_text,
            "desire_statement": desire_text,
            "desire_magnitude": desire_label,
            "awareness_level": awareness,
            "competition_level": comp_level,
            "_audit": audit_payload,
        }
        serialisable.append(
            {
                "aw": aw_label,
                "aw_m": aw_score,
                "d": desire_text,
                "d_m": desire_score,
                "c": comp_label,
                "c_m": comp_score,
                **({"confidence": confidence} if confidence is not None else {}),
            }
        )
    if raw_text is None:
        raw_text = json.dumps(serialisable, ensure_ascii=False, separators=(",", ":"))
    return result, raw_text


def _parse_strict_json_text(text: str, expected_ids: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    fake_raw = {"choices": [{"message": {"content": text}}]}
    parsed, _ = _parse_strict_json_payload(fake_raw, expected_ids)
    return parsed


def _recover_missing_sync(
    batch: BatchRequest,
    api_key: str,
    model: str,
    missing_ids: List[int],
) -> Tuple[Dict[int, Dict[str, Any]], int]:
    retries_used = 0
    recovered: Dict[int, Dict[str, Any]] = {}
    warning_prefix = ""
    to_fill = list(missing_ids)
    while to_fill and retries_used < MAX_RETRIES_MISSING:
        prompt_text = _build_missing_prompt(batch, to_fill)
        if warning_prefix:
            prompt_text = f"{warning_prefix}\n\n{prompt_text}"
        est_tokens = max(
            200,
            int(batch.prompt_tokens_est * len(to_fill) / max(1, len(batch.candidates))),
        )
        max_tokens = _max_tokens_for_batch(len(to_fill))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        try:
            raw = gpt.call_gpt(
                messages=messages,
                api_key=api_key,
                model=model,
                temperature=0.2,
                max_tokens=max_tokens,
                tokens_estimate=est_tokens,
            )
        except gpt.OpenAIError:
            break
        try:
            text = _extract_jsonl_payload(raw)
            parsed = parse_jsonl_and_validate(text, to_fill)
        except (gpt.InvalidJSONError, ValueError):
            if warning_prefix:
                break
            warning_prefix = "Tu respuesta no cumplió JSONL. Repite solo JSONL."
            retries_used += 1
            continue
        recovered.update(parsed)
        to_fill = [pid for pid in missing_ids if pid not in recovered]
        retries_used += 1
    return recovered, retries_used


def _finalize_batch_payload(
    batch: BatchRequest, strict_map: Mapping[int, Dict[str, Any]]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    data_map = {str(pid): payload for pid, payload in strict_map.items()}
    ok: Dict[str, Dict[str, Any]] = {}
    ko: Dict[str, str] = {}

    remaining_missing = [
        cand.id for cand in batch.candidates if str(cand.id) not in data_map
    ]
    for missing_id in remaining_missing:
        ko[str(missing_id)] = "missing"

    for cand in batch.candidates:
        pid = str(cand.id)
        entry = data_map.get(pid)
        if not isinstance(entry, dict):
            if pid not in ko:
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
            refined = _refine_desire_statement(cand, draft_raw)
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
            "awareness_level": gpt._norm_awareness(desire_payload.get("awareness_level")),
            "competition_level": gpt._norm_tri(
                desire_payload.get("competition_level")
            ),
        }
    return ok, ko


def _merge_usage(*usage_maps: Mapping[str, Any]) -> Dict[str, Any]:
    combined: Dict[str, Any] = {}
    for usage in usage_maps:
        if not usage:
            continue
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                prev = combined.get(key)
                if isinstance(prev, (int, float)):
                    combined[key] = float(prev) + float(value)
                else:
                    combined[key] = float(value)
            else:
                combined.setdefault(key, value)
    return combined


def _extract_confidence(payload: Mapping[str, Any]) -> Optional[float]:
    if not isinstance(payload, Mapping):
        return None
    audit = payload.get("_audit")
    value = None
    if isinstance(audit, Mapping):
        value = audit.get("confidence")
    if value is None:
        value = payload.get("confidence")
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


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


def _coerce_percent(value: Any, *, label: str) -> float:
    if isinstance(value, (int, float)):
        num = float(value)
    elif isinstance(value, str):
        try:
            num = float(value.strip())
        except Exception as exc:
            raise ValueError(f"{label} debe ser numérico") from exc
    else:
        raise ValueError(f"{label} debe ser numérico")
    if not 0 <= num <= 100:
        raise ValueError(f"{label} fuera de rango (0-100)")
    return float(num)

def _coerce_confidence(value: Any, *, required: bool) -> Optional[float]:
    if value is None:
        if required:
            raise ValueError("confidence requerido")
        return None
    try:
        num = float(value)
    except Exception as exc:
        raise ValueError("confidence debe ser numérico") from exc
    if math.isnan(num) or num < 0.0 or num > 1.0:
        raise ValueError("confidence fuera de rango (0-1)")
    return float(num)

def _tri_label_from_percent(percent: float) -> str:
    if percent <= 33:
        return "Low"
    if percent >= 67:
        return "High"
    return "Medium"

def _map_awareness_bucket(bucket: str, score: float) -> str:
    canonical = (bucket or "").strip().lower()
    if canonical == "low":
        if score <= 10:
            return "Unaware"
        return "Problem-Aware"
    if canonical == "medium":
        return "Solution-Aware"
    if canonical == "high":
        if score >= 90:
            return "Most Aware"
        return "Product-Aware"
    return gpt._norm_awareness(bucket)


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


def _call_batch_with_retries(
    batch: BatchRequest,
    *,
    api_key: str,
    model: str,
    max_retries: int,
) -> Dict[str, Any]:
    def _error_payload(error_message: str, duration: float, retries: int) -> Dict[str, Any]:
        return {
            "req_id": batch.req_id,
            "candidates": batch.candidates,
            "ok": {},
            "ko": {str(cand.id): error_message for cand in batch.candidates},
            "usage": {},
            "duration": duration,
            "retries": retries,
            "error": error_message,
            "prompt_tokens_est": batch.prompt_tokens_est,
            "raw_text": None,
        }

    attempt = 0
    while True:
        attempt += 1
        tokens_est = batch.prompt_tokens_est
        max_tokens = _max_tokens_for_batch(len(batch.candidates))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": batch.user_text},
        ]
        start_ts = time.perf_counter()
        try:
            raw = gpt.call_gpt(
                messages=messages,
                api_key=api_key,
                model=model,
                temperature=0.2,
                max_tokens=max_tokens,
                tokens_estimate=tokens_est,
            )
        except gpt.OpenAIError as exc:
            duration = time.perf_counter() - start_ts
            reason = _detect_adaptation_reason(exc)
            if reason and len(batch.candidates) > 1:
                raise BatchAdaptationRequired(reason, str(exc), allow_split=True) from exc
            if attempt <= max_retries:
                sleep_for = min(10.0, 0.5 * (2 ** (attempt - 1))) + random.uniform(0.05, 0.25)
                time.sleep(sleep_for)
                continue
            return _error_payload(str(exc), duration, attempt)

        expected_ids = [cand.id for cand in batch.candidates]
        retries_missing_used = 0
        raw_payload_text: Optional[str] = None
        strict_map: Dict[int, Dict[str, Any]] = {}

        if STRICT_JSON_ENABLED:
            duration = time.perf_counter() - start_ts
            try:
                strict_map, raw_payload_text = _parse_strict_json_payload(raw, expected_ids)
            except ValueError as exc:
                if batch.json_retry_count < 1 and len(batch.candidates) > 1:
                    raise BatchAdaptationRequired("json_parse", str(exc), allow_split=True) from exc
                return _error_payload(str(exc), duration, attempt)
        else:
            try:
                jsonl_payload = _extract_jsonl_payload(raw)
            except gpt.InvalidJSONError as exc:
                duration = time.perf_counter() - start_ts
                if attempt <= max_retries:
                    sleep_for = min(10.0, 0.5 * (2 ** (attempt - 1))) + random.uniform(0.05, 0.25)
                    time.sleep(sleep_for)
                    continue
                return _error_payload(str(exc), duration, attempt)

            raw_payload_text = jsonl_payload
            try:
                strict_map = parse_jsonl_and_validate(jsonl_payload, expected_ids)
            except ValueError as exc:
                partial_map = _parse_jsonl_loose(jsonl_payload)
                returned_ids = list(partial_map.keys())
                missing_ids = [pid for pid in expected_ids if pid not in partial_map]
                log_partial_ko(
                    "call_openai",
                    expected_ids,
                    returned_ids,
                    len(batch.candidates),
                    model,
                    note=str(exc),
                )
                recovered_payload: Dict[int, Dict[str, Any]] = {}
                if missing_ids:
                    recovered_payload, retries_missing_used = _recover_missing_sync(
                        batch,
                        api_key,
                        model,
                        missing_ids,
                    )
                combined = {**partial_map, **recovered_payload}
                pending_after = [pid for pid in expected_ids if pid not in combined]
                if not pending_after:
                    strict_map = combined
                    if recovered_payload:
                        log_recovered(
                            "call_openai",
                            retries_missing_used,
                            len(batch.candidates),
                            sorted(recovered_payload.keys()),
                        )
                else:
                    strict_map = combined

        usage = raw.get("usage", {}) or {}
        ok, ko = _finalize_batch_payload(batch, strict_map)

        duration = time.perf_counter() - start_ts
        return {
            "req_id": batch.req_id,
            "candidates": batch.candidates,
            "ok": ok,
            "ko": ko,
            "usage": usage,
            "duration": duration,
            "retries": attempt + retries_missing_used,
            "prompt_tokens_est": tokens_est,
            "raw_text": raw_payload_text,
        }


def _call_triage_batch(
    batch: BatchRequest,
    *,
    api_key: str,
    model: str,
) -> Dict[str, Any]:
    tokens_est = batch.prompt_tokens_est
    max_tokens = _max_tokens_for_batch(len(batch.candidates))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": batch.user_text},
    ]
    start_ts = time.perf_counter()
    raw = gpt.call_gpt(
        messages=messages,
        api_key=api_key,
        model=model,
        temperature=0.2,
        max_tokens=max_tokens,
        tokens_estimate=tokens_est,
    )
    duration = time.perf_counter() - start_ts
    expected_ids = [cand.id for cand in batch.candidates]
    strict_map, raw_payload_text = _parse_strict_json_payload(
        raw,
        expected_ids,
        require_confidence=True,
    )
    usage = raw.get("usage", {}) or {}
    return {
        "req_id": batch.req_id,
        "candidates": batch.candidates,
        "strict_map": strict_map,
        "usage": usage,
        "duration": duration,
        "raw_text": raw_payload_text,
        "prompt_tokens_est": tokens_est,
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


def _apply_ai_updates(conn, updates: Dict[int, Dict[str, Any]]) -> int:
    if not updates:
        return 0
    now_iso = datetime.utcnow().isoformat()
    cur = conn.cursor()
    began_tx = False
    affected = 0
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
            affected += max(0, cur.rowcount)
        if began_tx:
            conn.commit()
    except Exception:
        if began_tx and conn.in_transaction:
            conn.rollback()
        raise
    return affected


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


def _refine_desire_statement(
    candidate: Candidate,
    draft_text: str,
) -> Optional[Dict[str, Any]]:
    context = _candidate_to_desire_context(candidate)
    try:
        result = gpt.call_prompt_task(
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
    except gpt.OpenAIError as exc:
        message = str(exc).lower()
        if "status 429" in message:
            logger.warning(
                "ai_columns.refine.retry_exhausted id=%s err=%s",
                candidate.id,
                exc,
            )
        else:
            logger.error(
                "ai_columns.refine.failed id=%s",
                candidate.id,
                exc_info=True,
            )
        return None
    except Exception:
        logger.exception("desire refine call failed for id=%s", candidate.id)
        return None

    content = result.get("content") if isinstance(result, dict) else None
    if isinstance(content, dict):
        return content
    return None


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

    cache_enabled = bool(runtime_cfg.get("cache_enabled", True))
    cache_version = int(runtime_cfg.get("version", 1) or 1)
    trunc_title = int(runtime_cfg.get("trunc_title", 180) or 180)
    trunc_desc = int(runtime_cfg.get("trunc_desc", 800) or 800)

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
            "json_retry": 0,
            "batch_adapted": 0,
            "triage_covered": 0,
            "fallback_big": 0,
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
        if fail_reasons:
            _log_ko_event(
                "job_ko_summary",
                {str(pid): reason for pid, reason in fail_reasons.items()},
                job_id=job_id,
                candidate_lookup=candidate_map,
                extra={
                    "result_error": result_error,
                    "pending_total": len(pending_set),
                    "sample_pending_ids": sorted(pending_set)[:5],
                },
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
            "json_retry": 0,
            "batch_adapted": 0,
            "triage_covered": 0,
            "fallback_big": 0,
        }

    desire_scores: List[Tuple[str, float]] = []
    comp_scores: List[Tuple[str, float]] = []
    success_records: Dict[int, Dict[str, Any]] = {}
    request_latencies: List[float] = []
    processed_batches = 0
    json_retry_total = 0
    batch_adaptations_total = 0
    triage_covered_total = 0
    fallback_big_total = 0

    def process_result(result: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal cost_spent, counts_with_cost, processed_batches
        summary: Dict[str, Any] = {
            "parsed_ids": [],
            "missing_ids": [],
            "ko_ids": [],
            "status": "ok",
            "candidates": result.get("candidates", []),
            "req_id": result.get("req_id"),
        }
        candidates_list: List[Candidate] = result.get("candidates", [])
        if result.get("skipped"):
            for cand in candidates_list:
                pending_set.add(cand.id)
                fail_reasons.setdefault(cand.id, "cost_cap_reached")
            summary["status"] = "skipped"
            summary["pending_after"] = len(pending_set)
            return summary

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

        pending_before = len(pending_set)
        candidate_ids = [cand.id for cand in candidates_list]

        success_updates: Dict[int, Dict[str, Any]] = {}
        for pid_str, payload in ok_map.items():
            try:
                pid = int(pid_str)
            except Exception:
                continue
            success_updates[pid] = payload
            pending_set.discard(pid)

        ko_ids: set[int] = set()
        for pid_str, reason in ko_map.items():
            try:
                pid = int(pid_str)
            except Exception:
                continue
            pending_set.add(pid)
            fail_reasons[pid] = reason or "error"
            ko_ids.add(pid)

        parsed_ids = list(success_updates.keys())
        missing_ids = [
            pid for pid in candidate_ids if pid not in success_updates and pid not in ko_ids
        ]

        updated_rows = _apply_ai_updates(conn, success_updates)
        if success_updates:
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

        counts["ok"] += len(success_updates)
        counts["ko"] += len(ko_map)

        if ko_map:
            _log_ko_event(
                "ko_diagnostics",
                ko_map,
                job_id=job_id,
                candidate_lookup=candidate_map,
                req_id=result.get("req_id"),
                extra={
                    "pending_after_batch": len(pending_set),
                    "retry_count": retries,
                    "batch_duration_s": duration,
                    "batch_size": len(candidates_list),
                },
            )

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

        if missing_ids:
            logger.warning("ai.batch.missing ids_sample=%s", missing_ids[:5])
        if not parsed_ids:
            raw_sample = str(result.get("raw_text") or "")[:200]
            if raw_sample:
                logger.error("ai.parse.empty raw_sample=%s", raw_sample)

        if cost_cap is not None and cost_spent >= float(cost_cap):
            summary["status"] = "skipped"
            return summary
        summary["parsed_ids"] = parsed_ids
        summary["missing_ids"] = missing_ids
        summary["ko_ids"] = list(ko_map.keys())
        summary["pending_after"] = len(pending_set)
        if result.get("error"):
            summary["status"] = "error"
        elif missing_ids or ko_map:
            summary["status"] = "partial"
        return summary

    if remaining:
        pending_candidates = list(remaining)
        max_batch_size = min(AI_MAX_PRODUCTS_PER_CALL, len(pending_candidates))
        max_batch_size = max(AI_MIN_PRODUCTS_PER_CALL, max_batch_size)
        max_batch_size = max(1, min(AI_BATCH_MAX_ITEMS, max_batch_size))
        logger.info(
            "ai.run start items_total=%d batch_init=%d concurrency_target=%d strict_json=%s",
            total_items,
            max_batch_size,
            AI_MAX_CONCURRENCY,
            STRICT_JSON_ENABLED,
        )
        req_counter = 0
        pending_groups: Deque[Tuple[List[Candidate], int, int, bool]] = deque()
        for start in range(0, len(pending_candidates), max_batch_size):
            chunk = pending_candidates[start : start + max_batch_size]
            if not chunk:
                continue
            pending_groups.append((chunk, 0, 0, False))

        async def _execute_batches(batches: List[BatchRequest]) -> List[Dict[str, Any]]:
            semaphore = asyncio.Semaphore(max(1, AI_MAX_CONCURRENCY))

            async def _run_single(batch: BatchRequest) -> Dict[str, Any]:
                nonlocal triage_covered_total, fallback_big_total
                async with semaphore:
                    original_batch = batch
                    triage_result: Optional[Dict[str, Any]] = None
                    triage_usage: Dict[str, Any] = {}
                    triage_duration = 0.0
                    triage_raw_text: Optional[str] = None
                    fallback_candidates: List[Candidate] = list(batch.candidates)
                    allow_triage = (
                        TRIAGE_ENABLED
                        and batch.candidates
                        and batch.depth == 0
                        and not batch.adapted
                    )

                    if allow_triage:
                        logger.info(
                            "ai_columns.triage start req_id=%s model=%s items=%d confidence_min=%.2f strict_json=%s",
                            batch.req_id,
                            TRIAGE_MODEL,
                            len(batch.candidates),
                            TRIAGE_CONFIDENCE,
                            STRICT_JSON_ENABLED,
                        )
                        try:
                            triage_payload = await asyncio.to_thread(
                                _call_triage_batch,
                                batch,
                                api_key=api_key,
                                model=TRIAGE_MODEL,
                            )
                            triage_duration = float(
                                triage_payload.get("duration", 0.0) or 0.0
                            )
                            triage_usage = triage_payload.get("usage", {}) or {}
                            triage_raw_text = triage_payload.get("raw_text")
                            strict_map = triage_payload.get("strict_map") or {}
                            accepted_candidates: List[Candidate] = []
                            accepted_map: Dict[int, Dict[str, Any]] = {}
                            fallback_candidates = []
                            for cand in batch.candidates:
                                entry = strict_map.get(cand.id)
                                if not isinstance(entry, dict):
                                    fallback_candidates.append(cand)
                                    continue
                                confidence = _extract_confidence(entry)
                                if confidence is None or confidence < TRIAGE_CONFIDENCE:
                                    fallback_candidates.append(cand)
                                    continue
                                accepted_candidates.append(cand)
                                accepted_map[cand.id] = entry
                            triage_covered = len(accepted_candidates)
                            triage_covered_total += triage_covered
                            logger.info(
                                "ai_columns.triage done req_id=%s accepted=%d fallback=%d duration=%.2f",
                                batch.req_id,
                                triage_covered,
                                len(fallback_candidates),
                                triage_duration,
                            )
                            if accepted_candidates:
                                accepted_batch = BatchRequest(
                                    req_id=f"{batch.req_id}-triage",
                                    candidates=accepted_candidates,
                                    user_text=batch.user_text,
                                    prompt_tokens_est=batch.prompt_tokens_est,
                                    product_map=batch.product_map,
                                    depth=batch.depth,
                                    json_retry_count=0,
                                    adapted=batch.adapted,
                                    trunc_title=batch.trunc_title,
                                    trunc_desc=batch.trunc_desc,
                                )
                                triage_ok, _ = _finalize_batch_payload(
                                    accepted_batch,
                                    accepted_map,
                                )
                                triage_result = {
                                    "req_id": accepted_batch.req_id,
                                    "candidates": accepted_candidates,
                                    "ok": triage_ok,
                                    "ko": {},
                                    "usage": triage_usage,
                                    "duration": triage_duration,
                                    "retries": 0,
                                    "prompt_tokens_est": triage_payload.get(
                                        "prompt_tokens_est", 0
                                    ),
                                    "raw_text": triage_raw_text,
                                }
                        except Exception as exc:
                            logger.warning(
                                "ai_columns.triage failed req_id=%s err=%s",
                                batch.req_id,
                                exc,
                            )
                            fallback_candidates = list(batch.candidates)
                            triage_result = None
                            triage_usage = {}
                            triage_duration = 0.0
                            triage_raw_text = None

                    if not allow_triage and TRIAGE_ENABLED:
                        triage_covered_total += 0

                    if not fallback_candidates:
                        result_payload = triage_result or {
                            "req_id": f"{batch.req_id}-triage",
                            "candidates": batch.candidates,
                            "ok": {},
                            "ko": {},
                            "usage": triage_usage,
                            "duration": triage_duration,
                            "retries": 0,
                            "prompt_tokens_est": batch.prompt_tokens_est,
                            "raw_text": triage_raw_text,
                        }
                        final_result = {
                            "req_id": batch.req_id,
                            "candidates": batch.candidates,
                            "ok": result_payload.get("ok", {}),
                            "ko": {},
                            "usage": result_payload.get("usage", {}),
                            "duration": result_payload.get("duration", triage_duration),
                            "retries": 0,
                            "prompt_tokens_est": result_payload.get(
                                "prompt_tokens_est", batch.prompt_tokens_est
                            ),
                            "raw_text": result_payload.get("raw_text"),
                        }
                        return {
                            "status": "ok",
                            "batch": batch,
                            "result": final_result,
                            "duration": float(final_result.get("duration", 0.0) or 0.0),
                            "error": None,
                        }

                    fallback_batch = _build_batch_request(
                        f"{batch.req_id}-main",
                        fallback_candidates,
                        batch.trunc_title,
                        batch.trunc_desc,
                        depth=batch.depth,
                        json_retry_count=batch.json_retry_count,
                        adapted=batch.adapted,
                    )

                    fallback_big_total += len(fallback_candidates)

                    logger.info(
                        "ai_columns.batch start req_id=%s model=%s items=%d prompt_tokens_est=%d strict_json=%s batch_adapted=%s",
                        fallback_batch.req_id,
                        model,
                        len(fallback_batch.candidates),
                        fallback_batch.prompt_tokens_est,
                        STRICT_JSON_ENABLED,
                        fallback_batch.adapted,
                    )
                    start_local = time.perf_counter()
                    try:
                        result = await asyncio.to_thread(
                            _call_batch_with_retries,
                            fallback_batch,
                            api_key=api_key,
                            model=model,
                            max_retries=max_retries,
                        )
                        duration = time.perf_counter() - start_local
                        if isinstance(result, dict) and "duration" not in result:
                            result["duration"] = duration
                        triage_ok = triage_result.get("ok", {}) if triage_result else {}
                        combined_ok: Dict[str, Dict[str, Any]] = {}
                        combined_ok.update(triage_ok)
                        combined_ok.update(result.get("ok", {}) or {})
                        combined_ko = dict(result.get("ko", {}) or {})
                        combined_usage = _merge_usage(
                            triage_result.get("usage") if triage_result else {},
                            result.get("usage", {}),
                        )
                        combined_duration = float(triage_duration) + float(
                            result.get("duration", duration) or duration
                        )
                        combined_prompt = 0
                        if triage_result:
                            combined_prompt += int(
                                triage_result.get("prompt_tokens_est") or 0
                            )
                        combined_prompt += int(result.get("prompt_tokens_est") or 0)
                        combined_result = {
                            "req_id": batch.req_id,
                            "candidates": list(batch.candidates),
                            "ok": combined_ok,
                            "ko": combined_ko,
                            "usage": combined_usage,
                            "duration": combined_duration,
                            "retries": result.get("retries", 0),
                            "prompt_tokens_est": combined_prompt or batch.prompt_tokens_est,
                            "raw_text": result.get("raw_text")
                            or (triage_result.get("raw_text") if triage_result else None),
                        }
                        return {
                            "status": "ok",
                            "batch": fallback_batch,
                            "result": combined_result,
                            "duration": combined_duration,
                            "error": None,
                        }
                    except BatchAdaptationRequired as exc:
                        duration = time.perf_counter() - start_local
                        logger.warning(
                            "ai_columns.batch split req_id=%s reason=%s duration=%.2f",
                            fallback_batch.req_id,
                            exc.reason,
                            duration,
                        )
                        return {
                            "status": "split",
                            "batch": fallback_batch,
                            "result": None,
                            "duration": duration,
                            "error": exc,
                            "reason": exc.reason,
                            "json_retry": 1 if exc.reason == "json_parse" else 0,
                            "allow_split": exc.allow_split,
                            "triage_result": triage_result,
                        }
                    except Exception as exc:
                        duration = time.perf_counter() - start_local
                        logger.exception(
                            "ai_columns.batch exception req_id=%s",
                            fallback_batch.req_id,
                        )
                        ko_payload = {str(c.id): str(exc) for c in fallback_batch.candidates}
                        fallback = {
                            "req_id": fallback_batch.req_id,
                            "candidates": fallback_batch.candidates,
                            "ok": {},
                            "ko": ko_payload,
                            "usage": {},
                            "duration": duration,
                            "error": str(exc),
                        }
                        return {
                            "status": "error",
                            "batch": fallback_batch,
                            "result": fallback,
                            "duration": duration,
                            "error": exc,
                            "triage_result": triage_result,
                        }

            tasks = [asyncio.create_task(_run_single(batch)) for batch in batches]
            gathered: List[Dict[str, Any]] = []
            for coro in asyncio.as_completed(tasks):
                gathered.append(await coro)
            return gathered

        while pending_groups:
            batches: List[BatchRequest] = []
            while pending_groups:
                chunk, depth, json_retry_count, adapted_flag = pending_groups.popleft()
                if not chunk:
                    continue
                req_counter += 1
                batch = _build_batch_request(
                    f"{req_counter:03d}",
                    chunk,
                    trunc_title,
                    trunc_desc,
                    depth=depth,
                    json_retry_count=json_retry_count,
                    adapted=adapted_flag,
                )
                if _should_downshift(batch):
                    batch_adaptations_total += 1
                    logger.info(
                        "ai_columns.batch adapt_pre req_id=%s items=%d prompt_tokens_est=%d est_tokens=%d limit=%d",
                        batch.req_id,
                        len(batch.candidates),
                        batch.prompt_tokens_est,
                        _estimate_batch_tokens(batch),
                        SAFE_CONTEXT_TOKENS,
                    )
                    if len(chunk) <= MIN_BATCH_SIZE:
                        batch.adapted = True
                        batches.append(batch)
                        continue
                    split_point = max(1, math.ceil(len(chunk) / 2))
                    if split_point >= len(chunk):
                        split_point = max(1, len(chunk) // 2)
                    first = chunk[:split_point]
                    second = chunk[split_point:]
                    if second:
                        pending_groups.appendleft((second, depth + 1, json_retry_count, True))
                    if first:
                        pending_groups.appendleft((first, depth + 1, json_retry_count, True))
                    continue
                batches.append(batch)
            if not batches:
                break

            loop_results = asyncio.run(_execute_batches(batches))
            made_progress = False
            for entry in loop_results:
                status = entry.get("status")
                batch = entry.get("batch")
                if not batch:
                    continue
                triage_payload = entry.get("triage_result")
                if triage_payload:
                    triage_summary = process_result(triage_payload)
                    if triage_summary.get("parsed_ids"):
                        made_progress = True
                if status == "split":
                    reason = entry.get("reason") or "unknown"
                    json_retry_inc = int(entry.get("json_retry", 0) or 0)
                    json_retry_total += json_retry_inc
                    batch_adaptations_total += 1
                    if len(batch.candidates) <= 1 or not entry.get("allow_split", True):
                        fallback = entry.get("result")
                        if isinstance(fallback, dict):
                            summary = process_result(fallback)
                            if summary.get("parsed_ids"):
                                made_progress = True
                        continue
                    split_point = max(1, math.ceil(len(batch.candidates) / 2))
                    if split_point >= len(batch.candidates):
                        split_point = max(1, len(batch.candidates) // 2)
                    first = batch.candidates[:split_point]
                    second = batch.candidates[split_point:]
                    next_json_retry = batch.json_retry_count + json_retry_inc
                    if second:
                        pending_groups.appendleft((second, batch.depth + 1, next_json_retry, True))
                    if first:
                        pending_groups.appendleft((first, batch.depth + 1, next_json_retry, True))
                    continue
                if status != "ok":
                    exc = entry.get("error")
                    logger.warning(
                        "ai_columns.batch.retryable req_id=%s status=%s err=%s",
                        batch.req_id,
                        status,
                        exc,
                    )
                    pending_groups.append((list(batch.candidates), batch.depth + 1, batch.json_retry_count, True))
                    continue

                result = entry.get("result") or {}
                summary = process_result(result)
                parsed_ids = {
                    int(pid)
                    for pid in summary.get("parsed_ids", [])
                    if isinstance(pid, int)
                }
                missing_ids_list = summary.get("missing_ids", []) or []
                missing_ids: List[int] = []
                for mid in missing_ids_list:
                    try:
                        missing_ids.append(int(mid))
                    except Exception:
                        continue
                if parsed_ids:
                    made_progress = True

                logger.info(
                    "ai_columns.request ok items=%d parsed=%d missing=%d pending=%d",
                    len(batch.candidates),
                    len(parsed_ids),
                    len(missing_ids),
                    len(pending_set),
                )

                status_label = summary.get("status") or status or "ok"
                logger.info(
                    "ai_columns.batch end req_id=%s status=%s duration=%.2f parsed=%d missing=%d pending=%d batch_adapted=%s",
                    batch.req_id,
                    status_label,
                    float(entry.get("duration") or 0.0),
                    len(parsed_ids),
                    len(missing_ids),
                    len(pending_set),
                    batch.adapted,
                )

                if missing_ids:
                    missing_candidates = [
                        candidate_map.get(pid)
                        for pid in missing_ids
                        if candidate_map.get(pid)
                    ]
                    if missing_candidates:
                        pending_groups.append(
                            (missing_candidates, batch.depth + 1, batch.json_retry_count, True)
                        )

                if cost_cap is not None and cost_spent >= float(cost_cap):
                    pending_groups.clear()
                    break

            if not made_progress and not pending_groups:
                break
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

    wall_time = max(time.perf_counter() - start_ts, 1e-6)
    total_processed = counts.get("ok", 0) + counts.get("cached", 0)
    remaining_total = len(pending_set)
    total_batches = processed_batches
    total_latency = sum(request_latencies)
    concurrency_eff = total_latency / wall_time if request_latencies else 0.0
    p50 = _quantile(request_latencies, 0.5) if request_latencies else 0.0
    p95 = _quantile(request_latencies, 0.95) if request_latencies else 0.0
    logger.info(
        "ai.run done total=%d processed=%d remaining=%d concurrency_eff=%.2f concurrency_target=%d batches=%d p50=%.2f p95=%.2f batch_adapted=%d json_retry=%d strict_json=%s triage_covered=%d fallback_big=%d",
        total_items,
        total_processed,
        remaining_total,
        concurrency_eff,
        AI_MAX_CONCURRENCY,
        total_batches,
        p50,
        p95,
        batch_adaptations_total,
        json_retry_total,
        STRICT_JSON_ENABLED,
        triage_covered_total,
        fallback_big_total,
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

    processed_count = len(applied_outputs)
    logger.info(
        "ai.run done total=%d processed=%d remaining=%d batch_adapted=%d json_retry=%d triage_covered=%d fallback_big=%d",
        total_items,
        processed_count,
        len(pending_ids),
        batch_adaptations_total,
        json_retry_total,
        triage_covered_total,
        fallback_big_total,
    )

    logger.info(
        "run_ai_fill_job: job=%s total=%d ok=%d cached=%d ko=%d cost=%.4f pending=%d error=%s duration=%.2fs latency_p50=%.2fs latency_p95=%.2fs requests=%d",
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
    )

    if fail_reasons:
        _log_ko_event(
            "job_ko_summary",
            {str(pid): reason for pid, reason in fail_reasons.items()},
            job_id=job_id,
            candidate_lookup=candidate_map,
            extra={
                "result_error": result_error,
                "pending_total": len(pending_ids),
                "sample_pending_ids": pending_ids[:5],
                "processed_total": counts["ok"] + counts["cached"],
                "cost_spent_usd": cost_spent,
            },
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
        "json_retry": json_retry_total,
        "batch_adapted": batch_adaptations_total,
        "metrics": {
            "triage_covered": triage_covered_total,
            "fallback_big": fallback_big_total,
        },
        "triage_covered": triage_covered_total,
        "fallback_big": fallback_big_total,
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


def _call_model_sync(model_client: Any, prompt: str) -> str:
    response = model_client.complete(prompt=prompt, temperature=0.2, top_p=0.9)
    text = getattr(response, "text", "")
    if not isinstance(text, str):
        raise ValueError("Modelo no devolvió texto JSONL")
    return text


def _retry_missing_sync(model_client: Any, missing_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    prompt = MISSING_ONLY_JSONL_PROMPT(missing_ids, AI_FIELDS)
    text = _call_model_sync(model_client, prompt)
    if STRICT_JSON_ENABLED:
        return _parse_strict_json_text(text, missing_ids)
    return parse_jsonl_and_validate(text, missing_ids)


def fill_ai_columns_with_recovery(model_client: Any, product_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    batch_size = max(MIN_BATCH_SIZE, min(DEFAULT_BATCH_SIZE, len(product_ids) or MIN_BATCH_SIZE))
    out: Dict[int, Dict[str, Any]] = {}
    consecutive_partial = 0
    index = 0
    ids = list(dict.fromkeys(int(pid) for pid in product_ids))
    while index < len(ids):
        batch_ids = ids[index : index + batch_size]
        instruction = STRICT_JSONL_PROMPT(batch_ids, AI_FIELDS)
        try:
            text = _call_model_sync(model_client, instruction)
            if STRICT_JSON_ENABLED:
                parsed = _parse_strict_json_text(text, batch_ids)
            else:
                parsed = parse_jsonl_and_validate(text, batch_ids)
            out.update(parsed)
            consecutive_partial = 0
            index += batch_size
        except ValueError as exc:
            if STRICT_JSON_ENABLED:
                parsed_loose: Dict[int, Dict[str, Any]] = {}
            else:
                parsed_loose = _parse_jsonl_loose(text if 'text' in locals() else "")
            if parsed_loose:
                out.update(parsed_loose)
            returned = list(parsed_loose.keys())
            missing = [pid for pid in batch_ids if pid not in parsed_loose]
            log_partial_ko(
                "sync_call",
                batch_ids,
                returned,
                batch_size,
                getattr(model_client, "model", "unknown"),
                note=str(exc),
            )
            retries_used = 0
            if missing:
                try:
                    recovered = _retry_missing_sync(model_client, missing)
                except Exception:
                    recovered = {}
                else:
                    out.update(recovered)
                    retries_used = 1
            remaining = [pid for pid in batch_ids if pid not in out]
            if not remaining:
                log_recovered(
                    "sync_call",
                    retries_used,
                    batch_size,
                    batch_ids,
                )
                consecutive_partial = 0
                index += batch_size
            else:
                consecutive_partial += 1
                if batch_size > MIN_BATCH_SIZE and consecutive_partial >= 2:
                    batch_size = max(MIN_BATCH_SIZE, batch_size // 2)
                else:
                    index += batch_size
    return out
