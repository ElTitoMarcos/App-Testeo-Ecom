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
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY") or os.getenv("OPENAI_KEY")
OPENAI_BASE_URL = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com").rstrip("/")


class OpenAIError(RuntimeError):
    """Errores controlados para llamadas al API de OpenAI."""


# Modelos que exigen 'max_completion_tokens' (y suelen aceptar JSON mode con response_format)
_NEEDS_MAX_COMPLETION = re.compile(r"^(gpt-5|gpt-4\.1)", re.IGNORECASE)


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
        "temperature": temperature,
    }

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
