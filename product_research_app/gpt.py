# product_research_app/gpt.py
"""Utilities for interacting with the OpenAI Chat Completions API.

This module centralises the logic required to talk to the OpenAI-compatible
endpoint used by the application.  It keeps compatibility with older code
paths that still call :func:`call_gpt` / :func:`call_gpt_async` while
providing the newer behaviour required by the latest GPT-4.1/GPT-5 models,
which demand the ``max_completion_tokens`` parameter instead of
``max_tokens``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY") or os.getenv("OPENAI_KEY")
OPENAI_BASE_URL = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com").rstrip("/")


class InvalidJSONError(ValueError):
    """Raised when the assistant response cannot be interpreted as JSON."""


class OpenAIError(RuntimeError):
    """Errores controlados para llamadas al API de OpenAI."""


# Modelos que exigen 'max_completion_tokens' (y suelen aceptar JSON mode con response_format)
_NEEDS_MAX_COMPLETION = re.compile(r"^(gpt-5|gpt-4\.1)", re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def _needs_max_completion_tokens(model: str) -> bool:
    return bool(_NEEDS_MAX_COMPLETION.match(model or ""))


def _resolve_api_key(explicit: Optional[str]) -> str:
    key = (explicit or OPENAI_API_KEY or "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY no configurada")
    return key


def _auth_headers(
    *, api_key: Optional[str] = None, extra: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    base = {
        "Authorization": f"Bearer {_resolve_api_key(api_key)}",
        "Content-Type": "application/json",
    }
    if extra:
        base.update(extra)
    return base


def _post_json(
    path: str,
    payload: Dict[str, Any],
    *,
    timeout: float = 60.0,
    api_key: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> httpx.Response:
    url = f"{OPENAI_BASE_URL}{path}"
    with httpx.Client(timeout=timeout) as client:
        return client.post(
            url,
            headers=_auth_headers(api_key=api_key, extra=extra_headers),
            json=payload,
        )


async def _post_json_async(
    path: str,
    payload: Dict[str, Any],
    *,
    timeout: float = 60.0,
    api_key: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> httpx.Response:
    url = f"{OPENAI_BASE_URL}{path}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.post(
            url,
            headers=_auth_headers(api_key=api_key, extra=extra_headers),
            json=payload,
        )


_DEFAULT_TEMPERATURE_ONLY = re.compile(r"^(gpt-5-mini)", re.IGNORECASE)


def _sanitize_temperature(model: str, temperature: Optional[float]) -> Optional[float]:
    """Return the temperature value that should be sent to the API.

    Some newer models (like ``gpt-5-mini``) currently reject any explicit
    ``temperature`` value other than the default (``1``).  When we detect one
    of those models and the caller asked for a different value we simply omit
    the parameter so that the platform fallback applies instead of raising a
    ``400`` error.
    """

    if temperature is None:
        return None

    if _DEFAULT_TEMPERATURE_ONLY.match(model or "") and temperature not in (1, 1.0):
        logger.warning(
            "gpt.temperature adjusted to default for model=%s requested=%s",
            model,
            temperature,
        )
        return None

    return temperature


def _to_chat_payload(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_output_tokens: Optional[int],
    strict_json: bool,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    sanitized_temperature = _sanitize_temperature(model, temperature)
    if sanitized_temperature is not None:
        payload["temperature"] = sanitized_temperature

    # JSON mode si lo piden
    if strict_json:
        payload["response_format"] = {"type": "json_object"}

    # Herramientas si se usan en algún flujo
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice

    # Escoger nombre del parámetro de tokens según el modelo
    if max_output_tokens is not None:
        param_name = (
            "max_completion_tokens" if _needs_max_completion_tokens(model) else "max_tokens"
        )
        payload[param_name] = int(max_output_tokens)

    return payload


def _extract_error_message(response: httpx.Response) -> Tuple[Dict[str, Any], str]:
    try:
        detail = response.json()
    except Exception:
        detail = {"error": {"message": response.text}}

    message = ""
    if isinstance(detail, dict):
        err = detail.get("error")
        if isinstance(err, dict):
            candidates = [
                err.get("message"),
                err.get("detail"),
                err.get("code"),
            ]
            inner = err.get("innererror")
            if isinstance(inner, dict):
                candidates.append(inner.get("message"))
            for candidate in candidates:
                if isinstance(candidate, str) and candidate.strip():
                    message = candidate.strip()
                    break
        if not message:
            try:
                message = json.dumps(detail, ensure_ascii=False)
            except Exception:
                message = response.text.strip()
    else:
        message = response.text.strip()
    return detail if isinstance(detail, dict) else {"error": {"message": message}}, message


def _should_retry_max_tokens(message: str, response_text: str) -> bool:
    haystack = " ".join(filter(None, [message, response_text])).lower()
    return "max_tokens" in haystack and "max_completion_tokens" in haystack


def _ensure_completion_param(payload: Dict[str, Any], max_tokens: Optional[int]) -> None:
    payload.pop("max_tokens", None)
    payload.pop("max_completion_tokens", None)
    if max_tokens is not None:
        payload["max_completion_tokens"] = int(max_tokens)


def _strip_code_fences(text: str) -> str:
    """Remove Markdown code fences that commonly wrap JSON payloads."""

    stripped = text.strip()
    match = _CODE_BLOCK_RE.fullmatch(stripped)
    if match:
        inner = match.group(1).strip()
        if inner:
            return inner
    return stripped


def _extract_code_fence(text: str) -> Optional[str]:
    """Return the first JSON-ish fenced block found within *text* if present."""

    for match in _CODE_BLOCK_RE.finditer(text):
        candidate = (match.group(1) or "").strip()
        if candidate:
            return candidate
    return None


def _json_dumps(value: Any) -> Optional[str]:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return None


def _normalise_content_piece(item: Any) -> Optional[str]:
    if isinstance(item, str):
        return item.strip()

    if isinstance(item, (dict, list)):
        if isinstance(item, dict):
            for key in ("text", "content", "data", "arguments", "partial_json"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            json_value = item.get("json")
            if isinstance(json_value, (dict, list)):
                dumped = _json_dumps(json_value)
                if dumped:
                    return dumped
            if item.get("type") == "tool_call":
                fn = item.get("function")
                if isinstance(fn, dict):
                    args = fn.get("arguments")
                    if isinstance(args, str) and args.strip():
                        return args.strip()
                    if isinstance(args, (dict, list)):
                        dumped = _json_dumps(args)
                        if dumped:
                            return dumped
        dumped = _json_dumps(item)
        if dumped:
            return dumped

    return None


def _collect_message_strings(content: Any) -> List[str]:
    if content is None:
        return []

    if isinstance(content, list):
        pieces: List[str] = []
        for item in content:
            normalised = _normalise_content_piece(item)
            if normalised:
                pieces.append(normalised)
        return pieces

    normalised = _normalise_content_piece(content)
    return [normalised] if normalised else []


def _select_message(raw: Dict[str, Any]) -> Dict[str, Any]:
    choices = raw.get("choices")
    if isinstance(choices, Sequence):
        for choice in choices:
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict):
                    return message
                # Algunos proveedores pueden omitir la clave "message" y colocar
                # el contenido directamente en la elección.
                fallback_fields = {
                    key: value
                    for key, value in choice.items()
                    if key in {"content", "tool_calls", "refusal", "parsed"}
                }
                if fallback_fields:
                    return fallback_fields
    message = raw.get("message")
    return message if isinstance(message, dict) else {}


def _join_content(pieces: Sequence[str]) -> Optional[str]:
    joined = "\n".join(piece.strip() for piece in pieces if piece and piece.strip())
    return joined.strip() or None


def _parse_message_content(raw: Dict[str, Any]) -> Tuple[Optional[Any], Optional[str]]:
    """Extract JSON (if possible) and plain text from a chat completion payload."""

    if not isinstance(raw, dict):
        return None, None

    message = _select_message(raw)

    parsed: Optional[Any] = None
    if isinstance(message.get("parsed"), (dict, list)):
        parsed = message.get("parsed")

    pieces = _collect_message_strings(message.get("content"))

    refusal = message.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        pieces.append(refusal.strip())

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if isinstance(call, dict):
                fn = call.get("function")
                if isinstance(fn, dict):
                    args = fn.get("arguments")
                    if isinstance(args, str) and args.strip():
                        pieces.append(args.strip())
                    elif isinstance(args, (dict, list)):
                        dumped = _json_dumps(args)
                        if dumped:
                            pieces.append(dumped)

    text_content = _join_content(pieces)

    candidate_for_json = text_content
    if candidate_for_json:
        candidate_for_json = _strip_code_fences(candidate_for_json)
    if candidate_for_json and parsed is None:
        try:
            parsed = json.loads(candidate_for_json)
        except Exception:
            fenced = _extract_code_fence(text_content or "")
            if fenced:
                try:
                    parsed = json.loads(fenced)
                    candidate_for_json = _strip_code_fences(fenced)
                except Exception:
                    pass

    if parsed is not None and text_content is None:
        text_content = candidate_for_json or _json_dumps(parsed)
    elif candidate_for_json and parsed is not None:
        text_content = candidate_for_json

    return parsed, text_content


def chat(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.2,
    # Aceptamos ambos nombres para retrocompatibilidad:
    max_output_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,  # legacy
    strict_json: bool = False,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
    extra_headers: Optional[Dict[str, str]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Realiza una llamada síncrona a /v1/chat/completions."""
    effective_max = max_output_tokens if max_output_tokens is not None else max_tokens

    payload = _to_chat_payload(
        model=model,
        messages=messages,
        temperature=temperature,
        max_output_tokens=effective_max,
        strict_json=strict_json,
        tools=tools,
        tool_choice=tool_choice,
    )

    path = "/v1/chat/completions"

    try:
        resp = _post_json(
            path,
            payload,
            timeout=timeout,
            api_key=api_key,
            extra_headers=extra_headers,
        )
        if resp.status_code == 400:
            _, message = _extract_error_message(resp)
            if _should_retry_max_tokens(message, resp.text):
                logger.warning(
                    "OpenAI 400: cambiando a max_completion_tokens y reintentando (model=%s)",
                    model,
                )
                _ensure_completion_param(payload, effective_max)
                resp = _post_json(
                    path,
                    payload,
                    timeout=timeout,
                    api_key=api_key,
                    extra_headers=extra_headers,
                )
                if resp.status_code == 400:
                    _, message = _extract_error_message(resp)
                    logger.error(
                        "gpt.error status=%s detail=%s",
                        resp.status_code,
                        message,
                    )
                    raise OpenAIError(
                        f"OpenAI API returned status {resp.status_code}: {message}"
                    )
            else:
                logger.error(
                    "gpt.error status=%s detail=%s",
                    resp.status_code,
                    message,
                )
                raise OpenAIError(
                    f"OpenAI API returned status {resp.status_code}: {message}"
                )

        resp.raise_for_status()
        return resp.json()

    except httpx.HTTPStatusError as e:
        _, message = _extract_error_message(e.response)
        logger.error("gpt.error status=%s detail=%s", e.response.status_code, message)
        raise OpenAIError(
            f"OpenAI API returned status {e.response.status_code}: {message}"
        ) from e

    except httpx.HTTPError as e:
        logger.exception("gpt.error unexpected HTTP: %s", e)
        raise OpenAIError(f"OpenAI request failed: {e}") from e

    except Exception as e:
        logger.exception("gpt.error unexpected: %s", e)
        raise


async def chat_async(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.2,
    max_output_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    strict_json: bool = False,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
    extra_headers: Optional[Dict[str, str]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Variante asíncrona de :func:`chat`."""
    effective_max = max_output_tokens if max_output_tokens is not None else max_tokens

    payload = _to_chat_payload(
        model=model,
        messages=messages,
        temperature=temperature,
        max_output_tokens=effective_max,
        strict_json=strict_json,
        tools=tools,
        tool_choice=tool_choice,
    )

    path = "/v1/chat/completions"

    try:
        resp = await _post_json_async(
            path,
            payload,
            timeout=timeout,
            api_key=api_key,
            extra_headers=extra_headers,
        )
        if resp.status_code == 400:
            _, message = _extract_error_message(resp)
            if _should_retry_max_tokens(message, resp.text):
                logger.warning(
                    "OpenAI 400: cambiando a max_completion_tokens y reintentando (model=%s)",
                    model,
                )
                _ensure_completion_param(payload, effective_max)
                resp = await _post_json_async(
                    path,
                    payload,
                    timeout=timeout,
                    api_key=api_key,
                    extra_headers=extra_headers,
                )
                if resp.status_code == 400:
                    _, message = _extract_error_message(resp)
                    logger.error(
                        "gpt.error status=%s detail=%s",
                        resp.status_code,
                        message,
                    )
                    raise OpenAIError(
                        f"OpenAI API returned status {resp.status_code}: {message}"
                    )
            else:
                logger.error(
                    "gpt.error status=%s detail=%s",
                    resp.status_code,
                    message,
                )
                raise OpenAIError(
                    f"OpenAI API returned status {resp.status_code}: {message}"
                )

        resp.raise_for_status()
        return resp.json()

    except httpx.HTTPStatusError as e:
        _, message = _extract_error_message(e.response)
        logger.error("gpt.error status=%s detail=%s", e.response.status_code, message)
        raise OpenAIError(
            f"OpenAI API returned status {e.response.status_code}: {message}"
        ) from e

    except httpx.HTTPError as e:
        logger.exception("gpt.error unexpected HTTP: %s", e)
        raise OpenAIError(f"OpenAI request failed: {e}") from e

    except Exception as e:
        logger.exception("gpt.error unexpected: %s", e)
        raise


def call_gpt(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    strict_json: bool = False,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
    extra_headers: Optional[Dict[str, str]] = None,
    estimated_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Compatibilidad sincrónica con el API histórico."""
    del estimated_tokens  # mantenido por compatibilidad
    return chat(
        model=model,
        messages=messages,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        max_tokens=max_tokens,
        strict_json=strict_json,
        tools=tools,
        tool_choice=tool_choice,
        timeout=timeout,
        extra_headers=extra_headers,
        api_key=api_key,
    )


async def call_gpt_async(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    strict_json: bool = False,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
    extra_headers: Optional[Dict[str, str]] = None,
    estimated_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Compatibilidad asíncrona con el API histórico."""
    del estimated_tokens
    return await chat_async(
        model=model,
        messages=messages,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        max_tokens=max_tokens,
        strict_json=strict_json,
        tools=tools,
        tool_choice=tool_choice,
        timeout=timeout,
        extra_headers=extra_headers,
        api_key=api_key,
    )
