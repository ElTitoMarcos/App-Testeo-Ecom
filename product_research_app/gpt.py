# product_research_app/gpt.py
import os
import re
import json
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY") or os.getenv("OPENAI_KEY")
OPENAI_BASE_URL = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com").rstrip("/")

# Modelos que exigen 'max_completion_tokens' (y suelen aceptar JSON mode con response_format)
_NEEDS_MAX_COMPLETION = re.compile(r"^(gpt-5|gpt-4\.1)", re.IGNORECASE)

def _needs_max_completion_tokens(model: str) -> bool:
    return bool(_NEEDS_MAX_COMPLETION.match(model or ""))

def _auth_headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY no configurada")
    base = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    if extra:
        base.update(extra)
    return base

def _post_json(path: str, payload: Dict[str, Any], timeout: float = 60.0) -> httpx.Response:
    url = f"{OPENAI_BASE_URL}{path}"
    with httpx.Client(timeout=timeout) as client:
        return client.post(url, headers=_auth_headers(), json=payload)

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
        param_name = "max_completion_tokens" if _needs_max_completion_tokens(model) else "max_tokens"
        payload[param_name] = int(max_output_tokens)

    return payload

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
) -> Dict[str, Any]:
    """
    Wrapper centralizado. Siempre usa /v1/chat/completions pero elige el nombre correcto
    del parámetro de límite de tokens. Si llega un 400 indicando que 'max_tokens' no es
    soportado, reintenta automáticamente con 'max_completion_tokens'.
    """
    # Resolver compat: si max_output_tokens no viene, usar max_tokens legacy
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
        resp = _post_json(path, payload, timeout=timeout)
        if resp.status_code == 400:
            # Inspeccionar si es el caso del parámetro no soportado
            try:
                err = resp.json()
            except Exception:
                err = {"error": {"message": resp.text}}

            msg = (err.get("error") or {}).get("message", "")
            if "Unsupported parameter: 'max_tokens'" in msg or "Use 'max_completion_tokens' instead" in msg:
                # Forzar conversión y reintentar 1 vez
                logger.warning("OpenAI 400: cambiando a max_completion_tokens y reintentando (model=%s)", model)
                # Limpiar ambos por si acaso
                payload.pop("max_tokens", None)
                payload.pop("max_completion_tokens", None)
                if effective_max is not None:
                    payload["max_completion_tokens"] = int(effective_max)
                resp = _post_json(path, payload, timeout=timeout)
            else:
                resp.raise_for_status()

        resp.raise_for_status()
        data = resp.json()
        return data

    except httpx.HTTPStatusError as e:
        # Log detallado y re-raise para que servicios aguas arriba reporten bien
        try:
            detail = e.response.json()
        except Exception:
            detail = {"error": {"message": e.response.text}}
        logger.error("gpt.error status=%s detail=%s", e.response.status_code, (detail.get("error") or {}).get("message", detail))
        raise

    except Exception as e:
        logger.exception("gpt.error unexpected: %s", e)
        raise
