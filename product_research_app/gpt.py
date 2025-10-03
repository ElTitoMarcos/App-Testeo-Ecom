# product_research_app/gpt.py
import os
import re
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY") or os.getenv("OPENAI_KEY")
OPENAI_BASE_URL = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com").rstrip("/")

# Modelos que exigen 'max_completion_tokens'
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

    if strict_json:
        payload["response_format"] = {"type": "json_object"}

    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice

    if max_output_tokens is not None:
        param_name = "max_completion_tokens" if _needs_max_completion_tokens(model) else "max_tokens"
        payload[param_name] = int(max_output_tokens)

    return payload

def chat(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.2,
    max_output_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,  # legacy compat
    strict_json: bool = False,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Wrapper centralizado para /v1/chat/completions. Acepta max_tokens (legacy) y
    lo traduce a max_completion_tokens en modelos nuevos. Reintenta si el 400
    es por 'Unsupported parameter: max_tokens'.
    """
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
            try:
                err = resp.json()
            except Exception:
                err = {"error": {"message": resp.text}}
            msg = (err.get("error") or {}).get("message", "")
            if "Unsupported parameter: 'max_tokens'" in msg or "Use 'max_completion_tokens' instead" in msg:
                logger.warning("OpenAI 400: cambiando a max_completion_tokens y reintentando (model=%s)", model)
                payload.pop("max_tokens", None)
                payload.pop("max_completion_tokens", None)
                if effective_max is not None:
                    payload["max_completion_tokens"] = int(effective_max)
                resp = _post_json(path, payload, timeout=timeout)
            else:
                resp.raise_for_status()

        resp.raise_for_status()
        return resp.json()

    except httpx.HTTPStatusError as e:
        try:
            detail = e.response.json()
        except Exception:
            detail = {"error": {"message": e.response.text}}
        logger.error("gpt.error status=%s detail=%s", e.response.status_code, (detail.get("error") or {}).get("message", detail))
        raise
    except Exception as e:
        logger.exception("gpt.error unexpected: %s", e)
        raise

# ==========================
# Helpers de parsing robusto
# ==========================

class InvalidJSONError(ValueError):
    """El contenido no contiene JSON válido tras normalizarlo."""

_JSON_FENCE_RE = re.compile(r"```(?:json|javascript|js|ts|txt|md)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)

def _strip_code_fences(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # Si hay fences pero sin lenguaje
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            # contenido entre el primer y segundo fence
            return parts[1].strip()
    return text.strip()

def _find_balanced_json_block(s: str) -> Optional[str]:
    """
    Intenta extraer el primer bloque JSON (objeto o array) balanceado dentro de s.
    Ignora llaves dentro de strings de comillas dobles con escapes simples.
    """
    if not s:
        return None
    start_idxs = [i for i, ch in enumerate(s) if ch in "{["]
    for start in start_idxs:
        stack = []
        in_str = False
        escape = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                    continue
                if ch in "{[":
                    stack.append(ch)
                elif ch in "]}":
                    if not stack:
                        break
                    top = stack.pop()
                    if (top == "{" and ch != "}") or (top == "[" and ch != "]"):
                        break
                    if not stack:
                        # bloque completo
                        return s[start : i + 1].strip()
        # si no cerró, probar siguiente posible inicio
    return None

def _loads_loose_json(maybe_json: str) -> Union[Dict[str, Any], List[Any]]:
    try:
        return json.loads(maybe_json)
    except Exception as e:
        raise InvalidJSONError(str(e))

def _message_text_from_parts(content: Union[str, List[Any], None]) -> str:
    """
    Une content en texto. Maneja content parts tipo:
    [{"type":"text","text":"..."}, {"type":"input_text","text":"..."}]
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    chunks: List[str] = []
    for part in content:
        if isinstance(part, dict):
            # modelos modernos usan {'type': 'text', 'text': '...'}
            t = part.get("text")
            if isinstance(t, str):
                chunks.append(t)
        elif isinstance(part, str):
            chunks.append(part)
    return "\n".join(chunks).strip()

def _extract_json_from_tool_calls(msg: Dict[str, Any]) -> Optional[Union[Dict[str, Any], List[Any]]]:
    tool_calls = msg.get("tool_calls") or []
    for tc in tool_calls:
        fn = (tc or {}).get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str) and args.strip():
            cleaned = _strip_code_fences(args)
            block = _find_balanced_json_block(cleaned) or cleaned
            try:
                return _loads_loose_json(block)
            except InvalidJSONError:
                continue
    return None

def _extract_message_obj(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Devuelve un dict 'message' a partir de varias formas de respuesta:
    - {"choices":[{"message": {...}}]}
    - {"choices":[{"delta": {...}}]} (stream chunks)
    - {"message": {...}}
    - {"content": "..."} (degradación)
    """
    if not isinstance(raw, dict):
        return {}
    if "choices" in raw and raw["choices"]:
        ch0 = raw["choices"][0] or {}
        if isinstance(ch0, dict):
            if "message" in ch0 and isinstance(ch0["message"], dict):
                return ch0["message"]
            if "delta" in ch0 and isinstance(ch0["delta"], dict):
                return ch0["delta"]
    if "message" in raw and isinstance(raw["message"], dict):
        return raw["message"]
    if "content" in raw:
        return {"content": raw.get("content")}
    return {}

def _parse_message_content(raw: Dict[str, Any]) -> Tuple[Optional[Union[Dict[str, Any], List[Any]]], str]:
    """
    Extrae (parsed_json, text_content) de una respuesta cruda de Chat Completions.
    - parsed_json: dict/list si se localiza JSON en tool_calls o en el contenido, si no None.
    - text_content: texto plano concatenado (sin fences), útil para logs o fallback.
    """
    msg = _extract_message_obj(raw)
    text = _message_text_from_parts(msg.get("content"))
    # 1) Intentar primero tool_calls (si el flujo usó funciones)
    parsed: Optional[Union[Dict[str, Any], List[Any]]] = _extract_json_from_tool_calls(msg)
    # 2) Si no hubo JSON en tool_calls, intentar desde el contenido (fences/fragmentos)
    if parsed is None and text:
        cleaned = _strip_code_fences(text)
        block = _find_balanced_json_block(cleaned) or cleaned
        try:
            parsed = _loads_loose_json(block)
        except InvalidJSONError:
            parsed = None
    return parsed, (text or "").strip()

# Alias público por si alguna parte del código lo usa sin guion bajo
parse_message_content = _parse_message_content

# Stub inocuo para tests antiguos que esperan esta función
def build_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compat: en algunos tests antiguos se invoca; devolvemos tal cual."""
    return messages
