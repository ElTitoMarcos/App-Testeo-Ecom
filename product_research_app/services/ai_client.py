"""Wrapper del cliente OpenAI para respuestas JSON estrictas.

Variables de entorno soportadas:
- ``PRAPP_OPENAI_MAX_TOKENS`` (por defecto 2048)
- ``PRAPP_OPENAI_TEMPERATURE`` (por defecto 0.2)
- ``PRAPP_OPENAI_TOP_P`` (por defecto 1.0)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except Exception as e:
    # Mensaje explícito cuando falta la lib o está rota
    raise RuntimeError(
        "El paquete 'openai' no está instalado o es incompatible.\n"
        "Instálalo/actualízalo y vuelve a ejecutar:\n"
        "  .\\.venv\\Scripts\\pip install --upgrade \"openai>=1.45.0,<2\""
    ) from e

from product_research_app.utils.json_extract import coerce_json, message_parts_to_text

_RAW_DIR = Path("./logs/ai_raw")
_RAW_DIR.mkdir(parents=True, exist_ok=True)

PLACEHOLDERS = {"tu_openai", "tu_openai_api_key", "your_openai_api_key", "your_openai_key", "xxxxx", "abc123"}


def _get_api_key() -> str:
    """
    Fuentes por orden:
      1) Var de entorno OPENAI_API_KEY
      2) Módulo opcional product_research_app.services.secrets.get('openai_api_key')
      3) Fichero PRAPP_API_KEY_FILE (por defecto product_research_app/.secrets/openai.key)
    """

    key = (os.getenv("OPENAI_API_KEY") or "").strip()

    if not key or key.lower() in PLACEHOLDERS:
        try:
            from product_research_app.services import secrets as pr_secrets  # type: ignore

            candidate = pr_secrets.get("openai_api_key")
            if candidate:
                key = str(candidate).strip()
        except Exception:
            pass

    if not key or key.lower() in PLACEHOLDERS:
        path = (os.getenv("PRAPP_API_KEY_FILE") or "product_research_app/.secrets/openai.key").strip()
        try:
            with open(path, "r", encoding="utf-8") as handler:
                candidate = handler.read().strip()
                if candidate:
                    key = candidate
        except Exception:
            pass

    if not key or key.lower() in PLACEHOLDERS:
        hint = f"{key[:6]}…" if key else "(vacía)"
        raise RuntimeError(
            "OPENAI_API_KEY no está configurada correctamente (valor actual: "
            f"{hint}).\nGuárdala desde la UI, define la variable de entorno o coloca la clave en "
            "product_research_app/.secrets/openai.key (cambia la ruta con PRAPP_API_KEY_FILE)."
        )
    return key


def _make_client() -> OpenAI:
    api_key = _get_api_key()
    kwargs: Dict[str, Any] = {"api_key": api_key}

    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    if base_url:
        kwargs["base_url"] = base_url

    organization = (os.getenv("OPENAI_ORG") or "").strip()
    if organization:
        kwargs["organization"] = organization

    project = (os.getenv("OPENAI_PROJECT") or "").strip()
    if project:
        kwargs["project"] = project

    return OpenAI(**kwargs)


def _dump_raw(prefix: str, payload: Dict[str, Any], resp: Dict[str, Any]) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    path = _RAW_DIR / f"{prefix}_{timestamp}.json"
    try:
        with path.open("w", encoding="utf-8") as handler:
            json.dump({"request": payload, "response": resp}, handler, ensure_ascii=False, indent=2)
    except Exception:
        return ""
    return str(path)


def chat_json(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    json_schema: Optional[Dict[str, Any]] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = 7,
    req_id: Optional[str] = None,
    timeout: float = 90.0,
) -> Tuple[Any, Dict[str, Any]]:
    """Llama a Chat Completions asegurando salida JSON y devolviendo el payload parseado y el raw."""

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(os.getenv("PRAPP_OPENAI_TEMPERATURE", temperature if temperature is not None else 0.2)),
        "top_p": float(os.getenv("PRAPP_OPENAI_TOP_P", top_p if top_p is not None else 1.0)),
    }

    if seed is not None:
        payload["seed"] = seed

    if max_tokens is None:
        max_tokens = int(os.getenv("PRAPP_OPENAI_MAX_TOKENS", 2048))
    payload["max_tokens"] = max_tokens

    if json_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": json_schema.get("name", "ai_columns_payload"),
                "schema": json_schema.get("schema", {"type": "object"}),
                "strict": True,
            },
        }
    else:
        payload["response_format"] = {"type": "json_object"}

    retries = [0.0, 0.5, 1.0, 2.0, 4.0]
    last_error: Optional[Exception] = None
    for delay in retries:
        if delay:
            time.sleep(delay)
        try:
            client = _make_client()
            response = client.chat.completions.create(**payload, timeout=timeout)
            response_dict = json.loads(response.model_dump_json())
            _dump_raw(f"chat_{req_id or 'noid'}", payload, response_dict)

            choice = (response_dict.get("choices") or [{}])[0]
            message = choice.get("message", {}) or {}

            content_text = message_parts_to_text(message.get("content"))
            if content_text:
                return coerce_json(content_text), response_dict

            for tool_call in message.get("tool_calls") or []:
                fn = tool_call.get("function") or {}
                arguments = fn.get("arguments")
                if arguments:
                    return coerce_json(arguments), response_dict

            function_call = message.get("function_call") or {}
            arguments = function_call.get("arguments")
            if arguments:
                return coerce_json(arguments), response_dict

            raise ValueError("empty_or_nonjson")
        except Exception as exc:  # pragma: no cover - rutas de error difíciles de forzar
            last_error = exc
            continue
    raise last_error or ValueError("empty_or_nonjson")


__all__ = ["chat_json"]

