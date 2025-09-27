from __future__ import annotations

import json
import os
import re
import traceback
from enum import Enum
from pathlib import Path
from typing import Any, MutableMapping, Sequence

from .logging_setup import get_log_dir

_MAX_ARTIFACT_BYTES = 200_000
_SECRET_PATTERNS = [
    (re.compile(r"(sk-)[A-Za-z0-9]{10,}", re.IGNORECASE), 1),
    (re.compile(r"(bearer\s+)[A-Za-z0-9\-_=]{10,}", re.IGNORECASE), 1),
    (re.compile(r"(api[_-]?key\s*[:=]\s*)([A-Za-z0-9\-_=]{6,})", re.IGNORECASE), 2),
]


class Stage(str, Enum):
    FETCH_PRODUCTS = "fetch_products"
    BUILD_PROMPT = "build_prompt"
    CALL_OPENAI = "call_openai"
    PARSE_RESPONSE = "parse_response"
    VALIDATE_OUTPUT = "validate_output"
    WRITE_DB = "write_db"
    RECOMPUTE_SCORES = "recompute_scores"
    EMIT_UI = "emit_ui"


class ReasonCode(str, Enum):
    OPENAI_HTTP_ERROR = "openai_http_error"
    OPENAI_TIMEOUT = "openai_timeout"
    OPENAI_RATE_LIMITED = "openai_rate_limited"
    OPENAI_BAD_JSON = "openai_bad_json"
    MISSING_REQUIRED_KEYS = "missing_required_keys"
    VALIDATION_ERROR = "validation_error"
    DB_CONSTRAINT_VIOLATION = "db_constraint_violation"
    DB_WRITE_ERROR = "db_write_error"
    PARSE_ERROR = "parse_error"
    UNKNOWN = "unknown"


_ARTIFACTS_ENABLED = os.environ.get("PRAPP_ARTIFACTS", "1") not in {"0", "false", "False"}


def artifacts_enabled() -> bool:
    """Return True when diagnostic artifacts should be stored."""

    return _ARTIFACTS_ENABLED


def ensure_dirs(*paths: Path | str) -> None:
    """Create directories for provided paths if they do not exist."""

    for path in paths:
        if not path:
            continue
        path_obj = Path(path)
        try:
            path_obj.mkdir(parents=True, exist_ok=True)
        except Exception:
            continue


def _redact_secrets(value: str) -> str:
    text = value
    for pattern, group_index in _SECRET_PATTERNS:
        def _replace(match: re.Match[str]) -> str:
            original = match.group(0)
            if group_index <= match.lastindex:
                token = match.group(group_index)
                if token:
                    return original.replace(token, "***")
            if len(original) <= 4:
                return "***"
            return original[:4] + "***"

        text = pattern.sub(_replace, text)
    return text


def _truncate_bytes(data: bytes, target_path: Path) -> tuple[bytes, Path]:
    if len(data) <= _MAX_ARTIFACT_BYTES:
        return data, target_path
    truncated = data[:_MAX_ARTIFACT_BYTES]
    new_name = f"{target_path.stem}_truncated{target_path.suffix or '.bin'}"
    return truncated, target_path.with_name(new_name)


def _relative_path(path: Path) -> str:
    base = get_log_dir().parent
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path)


def dump_artifact(path: Path | str, data: Any) -> str | None:
    """Persist diagnostic artifacts and return the relative path."""

    if not artifacts_enabled():
        return None
    target = Path(path)
    parent = target.parent
    ensure_dirs(parent)
    try:
        if isinstance(data, (dict, list)):
            if target.suffix != ".json":
                target = target.with_suffix(".json")
            payload = json.dumps(data, ensure_ascii=False, indent=2)
            payload = _redact_secrets(payload)
            raw = payload.encode("utf-8")
        elif isinstance(data, (bytes, bytearray)):
            raw = bytes(data)
            if target.suffix == "":
                target = target.with_suffix(".bin")
        else:
            text = _redact_secrets(str(data))
            if target.suffix == "":
                target = target.with_suffix(".txt")
            raw = text.encode("utf-8")
        raw, target = _truncate_bytes(raw, target)
        with open(target, "wb") as fh:
            fh.write(raw)
        return _relative_path(target)
    except Exception:
        return None


def _base_payload(
    *,
    stage: Stage | str,
    job_id: str | int | None,
    req_id: str | int | None,
    product_id: str | int | None,
    duration_ms: float | int | None,
    retries: int | None,
    model: str | None,
    prompt_tokens: int | None,
    response_tokens: int | None,
    http_status: int | None,
    artifacts: Sequence[str] | None,
) -> MutableMapping[str, Any]:
    base: MutableMapping[str, Any] = {
        "stage": stage.value if isinstance(stage, Stage) else str(stage),
        "job_id": str(job_id) if job_id is not None else None,
        "req_id": str(req_id) if req_id is not None else None,
        "product_id": str(product_id) if product_id is not None else None,
        "duration_ms": float(duration_ms) if duration_ms is not None else None,
        "retries": int(retries or 0),
        "model": model,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "http_status": http_status,
        "artifacts": list(artifacts or []),
    }
    return base


def log_ok(
    logger: Any,
    *,
    stage: Stage | str,
    job_id: str | int | None,
    req_id: str | int | None,
    product_id: str | int | None,
    duration_ms: float | int | None,
    retries: int | None,
    model: str | None,
    prompt_tokens: int | None,
    response_tokens: int | None,
    http_status: int | None = None,
    artifacts: Sequence[str] | None = None,
) -> None:
    payload = _base_payload(
        stage=stage,
        job_id=job_id,
        req_id=req_id,
        product_id=product_id,
        duration_ms=duration_ms,
        retries=retries,
        model=model,
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        http_status=http_status,
        artifacts=artifacts,
    )
    payload["status"] = "OK"
    logger.info("product_event", extra=payload)


def log_ko(
    logger: Any,
    *,
    stage: Stage | str,
    job_id: str | int | None,
    req_id: str | int | None,
    product_id: str | int | None,
    duration_ms: float | int | None,
    retries: int | None,
    model: str | None,
    prompt_tokens: int | None,
    response_tokens: int | None,
    reason_code: ReasonCode | str,
    reason_detail: str,
    exception: BaseException | None = None,
    exception_class: str | None = None,
    traceback_short: str | None = None,
    http_status: int | None = None,
    artifacts: Sequence[str] | None = None,
) -> None:
    payload = _base_payload(
        stage=stage,
        job_id=job_id,
        req_id=req_id,
        product_id=product_id,
        duration_ms=duration_ms,
        retries=retries,
        model=model,
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        http_status=http_status,
        artifacts=artifacts,
    )
    payload["status"] = "KO"
    code = reason_code.value if isinstance(reason_code, ReasonCode) else str(reason_code)
    payload["reason_code"] = code
    detail = reason_detail or ""
    if len(detail) > 500:
        detail = detail[:500] + "…"
    payload["reason_detail"] = detail
    exc_cls = exception_class
    if exception is not None:
        exc_cls = exception.__class__.__name__
    payload["exception_class"] = exc_cls
    if traceback_short:
        payload["traceback_short"] = traceback_short
    elif exception is not None:
        tb_lines = traceback.format_exception(exception.__class__, exception, exception.__traceback__)
        payload["traceback_short"] = "".join(tb_lines[:8]).strip()
    logger.error("product_event", extra=payload)
