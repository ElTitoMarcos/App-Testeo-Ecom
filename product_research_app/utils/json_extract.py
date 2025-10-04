"""Utilidades para extraer y parsear contenido JSON de respuestas de IA."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _stack_find_json(s: str) -> Optional[str]:
    """Busca el primer objeto o array JSON balanceado dentro de ``s``."""

    if not s:
        return None

    start_idxs = [idx for idx, char in enumerate(s) if char in ("{", "[")]

    for start in start_idxs:
        stack: list[str] = []
        in_string = False
        escaped = False
        for idx in range(start, len(s)):
            char = s[idx]
            if in_string:
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char in ("{", "["):
                stack.append(char)
            elif char in "]}":
                if not stack:
                    break
                top = stack.pop()
                if (top == "{" and char != "}") or (top == "[" and char != "]"):
                    break
                if not stack:
                    candidate = s[start : idx + 1].strip()
                    try:
                        json.loads(candidate)
                    except Exception:
                        break
                    return candidate
    return None


def message_parts_to_text(content: Any) -> str:
    """Convierte los posibles formatos de ``content`` en texto plano."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content)


def extract_first_json_blob(text: str) -> Optional[str]:
    """Intenta localizar el primer fragmento JSON válido dentro de ``text``."""

    if not text:
        return None

    text = text.strip()
    if text and text[0] in "{[":
        try:
            json.loads(text)
        except Exception:
            pass
        else:
            return text

    match = FENCE_RE.search(text)
    if match:
        fenced = match.group(1).strip()
        try:
            json.loads(fenced)
        except Exception:
            pass
        else:
            return fenced

    return _stack_find_json(text)


def coerce_json(value: Any) -> Any:
    """Convierte ``value`` a JSON si es necesario."""

    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        blob = extract_first_json_blob(value)
        if blob is None:
            raise ValueError("empty_or_nonjson")
        return json.loads(blob)
    raise ValueError("unsupported_content_type")


__all__ = [
    "coerce_json",
    "extract_first_json_blob",
    "message_parts_to_text",
]

