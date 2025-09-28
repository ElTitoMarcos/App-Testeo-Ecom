"""
Integration with OpenAI Chat Completion API.

This module wraps calls to the OpenAI chat completion endpoint using the
requests library.  It constructs prompts based on the Breakthrough Advertising
framework for product evaluation and returns structured scores and
justifications.  The user must supply a valid API key and choose which
model to call (for example, ``gpt-4o``, ``gpt-4`` or future ``gpt-5``).  The
calls are synchronous; if network errors occur the caller is responsible
for retrying or handling the exception.

Because the openai Python package may not be available in the target
environment, we use direct HTTP calls to ``https://api.openai.com/v1/chat/completions``.

The evaluate_product function takes the product metadata and returns a
dictionary with scores and explanations for the six evaluation axes as
defined in the provided document.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from . import database, config
from .ratelimit import decorrelated_jitter_sleep, reserve
from .prompts.registry import (
    get_json_schema,
    get_system_prompt,
    get_task_prompt,
    is_json_only,
    normalize_task,
)
from .services import winner_score as winner_calc

logger = logging.getLogger(__name__)
log = logger

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "data.sqlite3"

AI_API_VERBOSE = int(os.getenv("PRAPP_AI_API_VERBOSE", "0"))
LIMIT_NEAR_FRAC = float(os.getenv("PRAPP_AI_LIMIT_NEAR_FRAC", "0.90"))

# Cache for baseline arrays recalculated every 10 minutes
_BASELINE_CACHE: Dict[str, Any] = {"ts": 0, "data": None}

STOPWORDS = {
    "the",
    "and",
    "de",
    "la",
    "el",
    "y",
    "a",
    "para",
    "por",
    "con",
    "sin",
    "en",
    "un",
    "una",
    "los",
    "las",
    "lo",
    "al",
    "del",
}


class OpenAIError(Exception):
    """Custom exception for OpenAI API errors."""
    pass


class InvalidJSONError(OpenAIError):
    """Raised when the model response is not valid JSON."""
    pass


class RateLimitWouldDegrade(OpenAIError):
    """Raised when batch size should be degraded after repeated 429s."""


def _parse_retry_after_seconds(response: Optional[requests.Response], message: str) -> Optional[float]:
    retry_after_header: Optional[str] = None
    if response is not None:
        retry_after_header = response.headers.get("Retry-After")
    retry_after_seconds: Optional[float] = None
    if retry_after_header:
        try:
            retry_after_seconds = float(retry_after_header)
        except ValueError:
            try:
                from email.utils import parsedate_to_datetime

                parsed = parsedate_to_datetime(retry_after_header)
                if parsed is not None:
                    retry_after_seconds = max(
                        0.0,
                        (parsed - datetime.utcnow()).total_seconds(),
                    )
            except Exception:
                retry_after_seconds = None
    if retry_after_seconds is not None:
        return max(0.0, retry_after_seconds)

    patterns = [
        r"try again in\s*([0-9]+(?:\.[0-9]+)?)\s*s",
        r"in\s*([0-9]+(?:\.[0-9]+)?)\s*seconds",
        r"in\s*([0-9]+(?:\.[0-9]+)?)\s*sec",
        r"in\s*([0-9]+)\s*ms",
    ]
    lowered = message or ""
    for pattern in patterns:
        match = re.search(pattern, lowered, re.I)
        if not match:
            continue
        try:
            value = float(match.group(1))
            if pattern.endswith("ms"):
                return max(0.0, value / 1000.0)
            return max(0.0, value)
        except Exception:
            continue
    return None


def _dumps_payload(payload: Any | None) -> str:
    data = {} if payload is None else payload
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def _looks_like_response_format_error(message: str) -> bool:
    lowered = message.lower()
    return "response_format" in lowered or "json_schema" in lowered or "schema" in lowered


def _extract_first_json_block(text: str) -> Tuple[Any, str]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            obj, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        remainder = text[index + end :].strip()
        return obj, remainder
    raise InvalidJSONError("No se encontró JSON válido en la respuesta")


def _parse_message_content(raw: Dict[str, Any]) -> Tuple[Optional[Any], str]:
    choices = raw.get("choices") or []
    if not choices:
        raise OpenAIError("Respuesta de OpenAI sin choices")
    message = choices[0].get("message", {}) or {}
    content = message.get("content", "")
    parsed_json: Optional[Any] = None
    text_parts: List[str] = []
    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type") or part.get("role")
            if part_type in {"text", "output_text"}:
                text_parts.append(str(part.get("text", "")))
            elif part_type in {"json", "output_json"}:
                parsed_json = part.get("json")
    elif isinstance(content, dict):
        part_type = content.get("type")
        if part_type in {"json", "output_json"}:
            parsed_json = content.get("json")
        if "text" in content:
            text_parts.append(str(content.get("text", "")))
    text = "\n".join(part for part in text_parts if part).strip()
    return parsed_json, text


def _parse_json_content(text: str) -> Any:
    if not text:
        raise InvalidJSONError("La respuesta JSON está vacía")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj, remainder = _extract_first_json_block(text)
        if remainder:
            raise InvalidJSONError("La respuesta JSON contiene texto adicional")
    if not isinstance(obj, (dict, list)):
        raise InvalidJSONError("La respuesta JSON debe ser un objeto o lista")
    return obj


def build_messages(
    task: str,
    context_json: Optional[Dict[str, Any]] = None,
    aggregates: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    *,
    extra_user: Optional[str] = None,
    mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Construye los mensajes para Prompt Maestro v3."""

    try:
        canonical = normalize_task(task)
    except KeyError as exc:
        raise ValueError(f"Tarea desconocida: {task}") from exc

    system_prompt = get_system_prompt(canonical)
    task_prompt = get_task_prompt(canonical)
    sections = [task_prompt]

    if canonical in {"A", "C", "D", "E"}:
        sections.append("### CONTEXT_JSON\n" + _dumps_payload(context_json))
    elif canonical == "B":
        sections.append("### AGGREGATES\n" + _dumps_payload(aggregates))
    elif canonical == "E_auto":
        sections.append("### DATA\n" + _dumps_payload(data))

    if mode:
        sections.append(f"### MODE\n{mode}")
    if extra_user:
        sections.append(extra_user.strip())

    user_content = "\n\n".join(sections).strip()
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


MAX_429 = int(os.getenv("PRAPP_OPENAI_MAX_RETRIES_429", "6"))
MAX_5XX = int(os.getenv("PRAPP_OPENAI_MAX_RETRIES_5XX", "4"))
BACKOFF_CAP = float(os.getenv("PRAPP_OPENAI_BACKOFF_CAP_S", "8"))

# Si no existe en este módulo, definimos un fallback para tipar el except
try:  # pragma: no cover - protección en tiempo de ejecución
    OpenAIError  # type: ignore[name-defined]
except NameError:  # pragma: no cover
    class OpenAIError(Exception): ...


def _estimate_tokens(text: str) -> int:
    # heurística: 1 token ~ 4 chars aprox. (inglés/español corto)
    return max(1, int(len(text) / 4))


def _extract_retry_after_seconds(msg: str) -> float | None:
    # soporta "Please try again in 1.55s" o "in 102ms"
    m = re.search(r"in\s+([0-9]+(?:\.[0-9]+)?)\s*s", msg, re.I)
    if m: return float(m.group(1))
    m = re.search(r"in\s+([0-9]+)\s*ms", msg, re.I)
    if m: return float(m.group(1)) / 1000.0
    return None


def call_gpt(*, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    """
    F U N E L  Ú N I C O:
    - Concurrencia acotada + RPM/TPM por token bucket (reserve).
    - Reintentos 429 con Retry-After / parseo del mensaje y backoff con jitter.
    Mantiene la API pública existente.
    """
    # Estimación simple de tokens del prompt (si no hay otra función ya en uso)
    try:
        prompt_text = "\n".join(
            m.get("content", "") for m in (messages or []) if isinstance(m, dict)
        )
    except Exception:
        prompt_text = ""
    tokens_est = _estimate_tokens(prompt_text)
    extra_est = kwargs.get("tokens_estimate")
    if extra_est is not None:
        try:
            tokens_est = max(tokens_est, int(extra_est))
        except Exception:
            pass

    overall_attempt = 0
    rate_attempts = 0
    server_attempts = 0
    sleep_prev_rate = 0.0
    sleep_prev_server = 0.0

    while True:
        overall_attempt += 1
        # Embudo global: RPM/TPM + concurrencia
        with reserve(tokens_est):
            try:
                raw = call_openai_chat(messages=messages, **kwargs)
                if overall_attempt > 1:
                    log.info(
                        "gpt.call_gpt.recovered status=OK_RECOVERED attempts=%s",
                        overall_attempt,
                    )
                return raw
            except OpenAIError as e:
                msg = str(e)
                low = msg.lower()
                resp = getattr(e, "response", None)
                status_code = None
                if resp is not None:
                    try:
                        status_code = int(getattr(resp, "status_code", None) or 0) or None
                    except Exception:
                        status_code = None

                is_rate_limit = (
                    (status_code == 429)
                    or ("status 429" in low)
                    or ("rate limit" in low)
                )

                if is_rate_limit:
                    rate_attempts += 1
                    if rate_attempts > MAX_429:
                        log.error(
                            "gpt.call_gpt.gave_up_429 attempts=%s",
                            rate_attempts,
                        )
                        raise

                    retry_after = getattr(e, "retry_after", None)
                    if retry_after is None:
                        try:
                            headers = getattr(resp, "headers", None) if resp is not None else None
                            if headers:
                                ra = headers.get("Retry-After") or headers.get("retry-after")
                                if ra:
                                    retry_after = float(ra)
                        except Exception:
                            retry_after = None

                    wait_s = (
                        retry_after
                        if retry_after is not None
                        else _extract_retry_after_seconds(msg) or sleep_prev_rate
                    )
                    sleep_prev_rate = decorrelated_jitter_sleep(wait_s, BACKOFF_CAP)
                    log.warning(
                        "gpt.call_gpt.retry_429 attempt=%s sleep=%.2fs",
                        rate_attempts,
                        sleep_prev_rate,
                    )
                    continue

                transient_markers = (
                    "failed to connect",
                    "timeout",
                    "timed out",
                    "connection reset",
                    "temporarily unavailable",
                    "service unavailable",
                    "upstream connect error",
                )
                is_server_error = False
                if status_code is not None:
                    if 500 <= status_code < 600:
                        is_server_error = True
                if not is_server_error:
                    if any(marker in low for marker in transient_markers):
                        is_server_error = True

                if is_server_error:
                    server_attempts += 1
                    if server_attempts > MAX_5XX:
                        log.error(
                            "gpt.call_gpt.gave_up_5xx status=%s attempts=%s",
                            status_code if status_code is not None else "n/a",
                            server_attempts,
                        )
                        raise

                    retry_after = getattr(e, "retry_after", None)
                    wait_base = retry_after if retry_after is not None else (
                        sleep_prev_server if sleep_prev_server > 0 else 0.5
                    )
                    sleep_prev_server = decorrelated_jitter_sleep(wait_base, BACKOFF_CAP)
                    log.warning(
                        "gpt.call_gpt.retry_5xx status=%s attempt=%s sleep=%.2fs",
                        status_code if status_code is not None else "n/a",
                        server_attempts,
                        sleep_prev_server,
                    )
                    continue

                raise


def _message_text(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text:
                        parts.append(str(text))
        elif isinstance(content, dict):
            text = content.get("text")
            if text:
                parts.append(str(text))
    return "\n".join(parts)


def call_prompt_task(
    task: str,
    context_json: Optional[Dict[str, Any]] = None,
    aggregates: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    temperature: float = 0,
    *,
    extra_user: Optional[str] = None,
    mode: Optional[str] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[Any] = None,
) -> Dict[str, Any]:
    """Ejecuta una llamada estándar a Prompt Maestro v3."""

    try:
        canonical = normalize_task(task)
    except KeyError as exc:
        raise ValueError(f"Tarea desconocida: {task}") from exc

    messages = build_messages(
        canonical,
        context_json,
        aggregates,
        data,
        extra_user=extra_user,
        mode=mode,
    )
    api_key = config.get_api_key() or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise OpenAIError("No hay API key configurada")
    model = config.get_model()

    schema = get_json_schema(canonical)
    response_format: Optional[Dict[str, Any]] = None
    if is_json_only(canonical) and schema:
        response_format = {"type": "json_schema", "json_schema": schema}

    default_max_tokens = 450 if canonical == "DESIRE" else None
    call_max_tokens = max_tokens if max_tokens is not None else default_max_tokens

    prompt_text = _message_text(messages)
    prompt_tokens_est = _estimate_tokens(prompt_text)
    if call_max_tokens:
        try:
            prompt_tokens_est += int(call_max_tokens)
        except Exception:
            prompt_tokens_est += 0

    raw = call_gpt(
        messages=messages,
        api_key=api_key,
        model=model,
        temperature=temperature,
        response_format=response_format,
        max_tokens=call_max_tokens,
        stop=stop,
        tokens_estimate=prompt_tokens_est,
    )

    parsed_json, text_content = _parse_message_content(raw)
    if is_json_only(canonical):
        content = parsed_json if parsed_json is not None else _parse_json_content(text_content)
        if not isinstance(content, (dict, list)):
            raise InvalidJSONError("La respuesta JSON debe ser un objeto o lista")
    else:
        if text_content:
            content = text_content
        elif parsed_json is not None:
            content = json.dumps(parsed_json, ensure_ascii=False)
        else:
            content = ""

    return {"ok": True, "task": canonical, "content": content, "raw": raw}


def _build_image_message(image_bytes: bytes, instructions: str, filename: str) -> list:
    """
    Construct a vision‑enabled message payload for the OpenAI Chat API.

    This helper converts binary image data into a base64 data URL and combines it
    with the provided textual instructions.  The returned list can be used as
    the value of the ``content`` field for a message.

    Args:
        image_bytes: Raw binary contents of the image.
        instructions: Instructions for the model describing what to extract.
        filename: Original filename (used to infer MIME type).

    Returns:
        A list suitable for use in the ``content`` field of a message dict.
    """
    import base64
    from pathlib import Path

    ext = Path(filename).suffix.lower().lstrip('.') or 'png'
    mime = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"
    b64 = base64.b64encode(image_bytes).decode('utf-8')
    data_url = f"data:{mime};base64,{b64}"
    return [
        {"type": "text", "text": instructions},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]


def extract_products_from_image(
    api_key: str,
    model: str,
    image_path: str,
    *,
    instructions: Optional[str] = None,
    temperature: float = 0.2,
) -> List[Dict[str, Any]]:
    """
    Use a vision‑capable OpenAI model to extract product information from an image.

    The function reads the image at ``image_path``, encodes it as a data URL and
    sends it to the Chat Completion API along with natural language instructions
    requesting a list of products present in the screenshot.  The model should
    respond with a JSON array of objects, each containing at least a ``name``
    field and optional ``description``, ``category`` and ``price`` fields.

    Args:
        api_key: Your OpenAI API key.
        model: The vision‑capable model to call (e.g. "gpt-4o").
        image_path: Path to the image file on disk.
        instructions: Optional custom instructions for the model.  If omitted,
            a default Spanish instruction will be used.
        temperature: Sampling temperature for the generation.

    Returns:
        A list of dictionaries representing the products extracted from the image.
        Each dict should contain at least a ``name`` key.  If the model does not
        return valid JSON or no products are detected an empty list is returned.

    Raises:
        OpenAIError: If the API call fails or returns an error.
    """
    default_instructions = (
        "Analiza detenidamente la imagen proporcionada y extrae toda la información útil sobre "
        "anuncios o productos que contenga. Para cada elemento identifica campos relevantes como "
        "nombre del producto o anuncio ('name'), descripción corta ('description'), categoría ('category'), "
        "precio o ingreso ('price' o 'revenue') y cualquier otra métrica que aparezca (por ejemplo, unidades vendidas, "
        "ratio de conversión, fecha de lanzamiento). Devuelve únicamente un array JSON de objetos donde cada objeto "
        "incluya al menos la clave 'name' y tantas otras claves como se puedan deducir. No añadas ningún comentario ni "
        "explicación fuera del JSON."
    )
    instr = instructions or default_instructions
    # read image bytes
    try:
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
    except Exception as exc:
        raise OpenAIError(f"No se pudo leer la imagen: {exc}") from exc
    messages = [
        {"role": "system", "content": "Eres un asistente experto en investigación de productos."},
        {
            "role": "user",
            "content": _build_image_message(img_bytes, instr, image_path),
        },
    ]
    # call the API
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    except requests.RequestException as exc:
        raise OpenAIError(f"Error al conectar con OpenAI: {exc}") from exc
    if response.status_code != 200:
        try:
            err = response.json()
            msg = err.get("error", {}).get("message", response.text)
        except Exception:
            msg = response.text
        raise OpenAIError(f"La API de OpenAI devolvió un error {response.status_code}: {msg}")
    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        # attempt to parse JSON
        products = json.loads(content)
        if isinstance(products, dict):
            # sometimes the model returns a dict with "products" key
            products = products.get("products", [])
        if not isinstance(products, list):
            return []
        # ensure each item is a dict with name
        cleaned = []
        for item in products:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("nombre") or item.get("title")
            if not name:
                continue
            cleaned.append({
                "name": name.strip(),
                "description": item.get("description") or item.get("descripcion"),
                "category": item.get("category") or item.get("categoria"),
                "price": item.get("price") or item.get("precio"),
            })
        return cleaned
    except Exception:
        # if parsing fails, return empty list
        return []


def call_openai_chat(
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    *,
    temperature: float = 0.2,
    response_format: Optional[Dict[str, Any]] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[Any] = None,
    tokens_estimate: Optional[int] = None,
) -> Dict[str, Any]:
    """Send a chat completion request to the OpenAI API.

    Args:
        api_key: The user's OpenAI API key.
        model: The identifier of the model to call, e.g. ``gpt-4o`` or ``gpt-3.5-turbo``.
        messages: A list of message dicts, each containing ``role`` and ``content`` fields.
        temperature: The sampling temperature; lower values produce more deterministic output.

    Returns:
        The parsed JSON response from OpenAI.

    Raises:
        OpenAIError: If the API responds with an error or unexpected content.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if stop is not None:
        payload["stop"] = stop
    if response_format is not None:
        payload["response_format"] = response_format

    if AI_API_VERBOSE >= 2:
        logger.debug(
            "gpt.pre model=%s est_tokens=%s",
            model,
            str(tokens_estimate or ""),
        )

    try:
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            timeout=60,
        )
    except requests.RequestException as exc:
        raise OpenAIError(f"Failed to connect to OpenAI API: {exc}") from exc

    if response.status_code == 200:
        try:
            data = response.json()
        except Exception as exc:
            raise OpenAIError(f"Invalid JSON response from OpenAI: {exc}") from exc
        if AI_API_VERBOSE >= 2:
            usage = data.get("usage") if isinstance(data, dict) else None
            if isinstance(usage, dict):
                logger.debug(
                    "gpt.post model=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                    model,
                    usage.get("prompt_tokens"),
                    usage.get("completion_tokens"),
                    usage.get("total_tokens"),
                )
        return data

    try:
        err = response.json()
        msg = err.get("error", {}).get("message", response.text)
    except Exception:
        msg = response.text
    status = int(response.status_code)
    message = f"OpenAI API returned status {status}: {msg}"
    error = OpenAIError(message)
    setattr(error, "response", response)
    retry_hint = _parse_retry_after_seconds(response, msg)
    if retry_hint is not None:
        setattr(error, "retry_after", retry_hint)
    if status == 429:
        pass
    elif 500 <= status < 600:
        logger.warning("gpt.http status=%s detail=%s", status, msg)
    else:
        logger.error("gpt.error status=%s detail=%s", status, msg)
    raise error


def build_evaluation_prompt(product: Dict[str, Any]) -> str:
    """Construct the evaluation prompt for a given product.

    The prompt follows the guidelines from Breakthrough Advertising: it asks the
    model to assess the product across six dimensions—Momentum, Saturation,
    Differentiation, Social Proof, Estimated Margin and Logistic Complexity—
    returning numerical scores (0–10) and explanations for each.  The model is
    instructed to respond strictly in JSON so that it can be parsed reliably.

    Args:
        product: A dict containing the product fields ``name``, ``description``,
            ``category``, and optional ``price``.

    Returns:
        A string containing the evaluation prompt.
    """
    name = product.get("name", "")
    description = product.get("description", "") or ""
    category = product.get("category", "") or ""
    price = product.get("price", None)
    price_str = f"Precio: {price}" if price is not None else ""
    prompt = f"""
Eres un analista de productos experto en marketing y dropshipping.  Te voy a dar
información sobre un producto y debes evaluarlo siguiendo el marco mental del libro
"Breakthrough Advertising" de Eugene Schwartz.  Debes puntuar los siguientes
aspectos del producto con un número entre 0 y 10 (donde 10 es excelente y 0 es
pobre) y proporcionar una explicación breve (1–3 frases) para cada puntuación:

1. Momentum: ¿Qué tan fuerte es la tendencia reciente de interés o ventas del
   producto en los últimos 7, 14 y 30 días?
2. Saturación: ¿Cuántos competidores están vendiendo productos similares y qué
   tan saturado parece el mercado?
3. Diferenciación: ¿Qué tan único o diferenciado es este producto respecto a
   otros competidores?  Considera ángulos de marketing o características
   únicas.
4. PruebaSocial: ¿Qué indicadores de aceptación (reseñas, interacciones,
   compartidos) sugieren que el público confía o está interesado en este producto?
5. Margen: ¿Cuál podría ser el margen de beneficio estimado comparando precio
   de venta y coste aproximado?  10 significa margen excelente, 0 significa
   margen muy bajo.
6. Logística: ¿Qué tan compleja es la logística para cumplir con este producto?
   Teniendo en cuenta peso, fragilidad, variantes/tallas y requerimientos de envío.

Además de las puntuaciones y explicaciones, proporciona un campo "summary" donde
resumas en 2–4 frases los puntos clave y recomiendes si merece la pena seguir
investigando este producto.  Calcula un campo "totalScore" como la media
aritmética de las seis puntuaciones.

Datos del producto:
Nombre: {name}
Descripción: {description}
Categoría: {category}
{price_str}

Responde **únicamente** con un objeto JSON válido con las siguientes claves:
{
  "momentum_score": <número>,
  "momentum_explanation": "...",
  "saturation_score": <número>,
  "saturation_explanation": "...",
  "differentiation_score": <número>,
  "differentiation_explanation": "...",
  "social_proof_score": <número>,
  "social_proof_explanation": "...",
  "margin_score": <número>,
  "margin_explanation": "...",
  "logistics_score": <número>,
  "logistics_explanation": "...",
  "totalScore": <número>,
  "summary": "..."
}

Asegúrate de no añadir ningún comentario fuera del objeto JSON.
"""
    return prompt.strip()


def evaluate_product(
    api_key: str,
    model: str,
    product: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate a product using the OpenAI model and return structured scores.

    Args:
        api_key: User's OpenAI API key.
        model: The model identifier to call.
        product: A dict containing product fields.  At minimum ``name`` should be provided.

    Returns:
        A dict containing numeric scores and explanations for each metric, along
        with a total score and summary.  The structure mirrors the JSON
        specification in the prompt.

    Raises:
        OpenAIError: If the API call fails or returns invalid content.
    """
    prompt = build_evaluation_prompt(product)
    messages = [
        {"role": "system", "content": "Eres un asistente inteligente que responde en español."},
        {"role": "user", "content": prompt},
    ]
    resp_json = call_gpt(messages=messages, api_key=api_key, model=model)
    try:
        content = resp_json["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        raise OpenAIError(f"Respuesta inesperada de OpenAI: {exc}") from exc
    # The content should be pure JSON; attempt to parse it
    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        # Log the problematic content for debugging
        logger.error("No se pudo analizar la respuesta de la IA como JSON: %s", content)
        raise OpenAIError("La respuesta de la IA no está en formato JSON válido.") from exc
    return result


def _canonical(s: Optional[str]) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z]", "", s.lower())


def _norm_tri(val: Optional[str], default: str = "Medium") -> str:
    mapping = {
        "low": "Low",
        "baja": "Low",
        "medium": "Medium",
        "medio": "Medium",
        "med": "Medium",
        "high": "High",
        "alta": "High",
    }
    return mapping.get(_canonical(val), default)


def _norm_awareness(val: Optional[str]) -> str:
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
    return mapping.get(_canonical(val), "Problem-Aware")


def _to_float(val: Any) -> Optional[float]:
    try:
        return float(str(val).strip().replace(",", "."))
    except Exception:
        return None


def _tokenize(name: str) -> List[str]:
    return [
        tok
        for tok in re.findall(r"[a-zA-Z0-9]+", name.lower())
        if tok not in STOPWORDS
    ]


def pct_rank(x: Optional[float], arr: List[Optional[float]]) -> Optional[float]:
    if x is None or not arr:
        return None
    s = sorted(a for a in arr if a is not None)
    if not s:
        return None
    import bisect

    i = bisect.bisect_left(s, x)
    return i / max(1, len(s) - 1)


def nz(x: Optional[float], d: float = 0.0) -> float:
    return d if x is None else x


def _load_baseline_data() -> Dict[str, Any]:
    now = time.time()
    cached = _BASELINE_CACHE.get("data")
    if cached and now - _BASELINE_CACHE.get("ts", 0) < 600:
        return cached
    conn = database.get_connection(DB_PATH)
    rows = database.list_products(conn)
    units: List[float] = []
    conv: List[float] = []
    rating_vals: List[float] = []
    revenue: List[float] = []
    cat_counts: Dict[str, int] = {}
    token_index: Dict[str, set] = {}
    for p in rows:
        extra = p["extra"] if "extra" in p.keys() else {}
        try:
            extra = json.loads(extra) if isinstance(extra, str) else (extra or {})
        except Exception:
            extra = {}
        u = _to_float(extra.get("units_sold"))
        if u is not None:
            units.append(u)
        c = _to_float(extra.get("conversion_rate"))
        if c is not None:
            conv.append(c)
        r = _to_float(extra.get("rating"))
        if r is not None:
            r_norm = max(0.0, min(1.0, (r - 3.0) / 2.0))
            rating_vals.append(r_norm)
        rev = _to_float(extra.get("revenue"))
        if rev is not None:
            revenue.append(rev)
        cat = (p["category"] or "").split(">")[0].strip()
        if cat:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        for tok in _tokenize(p["name"] or ""):
            token_index.setdefault(tok, set()).add(p["id"])
    data = {
        "units": units,
        "conv": conv,
        "rating": rating_vals,
        "revenue": revenue,
        "cat_counts": cat_counts,
        "cat_count_list": list(cat_counts.values()),
        "token_index": token_index,
    }
    _BASELINE_CACHE["ts"] = now
    _BASELINE_CACHE["data"] = data
    return data


def _compute_baselines(product: Dict[str, Any]) -> Tuple[str, str]:
    data = _load_baseline_data()
    all_units = data["units"]
    all_conv = data["conv"]
    p_units = pct_rank(_to_float(product.get("units_sold")), all_units)
    p_conv = pct_rank(_to_float(product.get("conversion_rate")), all_conv)
    rating_val = _to_float(product.get("rating"))
    if rating_val is not None:
        p_rate = max(0.0, min(1.0, (rating_val - 3.0) / 2.0))
    else:
        p_rate = None
    desire_signal = 0.45 * nz(p_units, 0) + 0.35 * nz(p_conv, 0) + 0.20 * nz(p_rate, 0)
    if desire_signal > 0.66:
        desire_baseline = "High"
    elif desire_signal < 0.33:
        desire_baseline = "Low"
    else:
        desire_baseline = "Medium"
    if p_units is None or p_conv is None or p_rate is None:
        if desire_baseline == "High":
            desire_baseline = "Medium"
    cat = (product.get("category") or "").split(">")[0].strip()
    cat_counts = data["cat_counts"]
    cat_count = cat_counts.get(cat)
    cat_freq_pct = (
        pct_rank(cat_count, data["cat_count_list"]) if cat_count is not None else None
    )
    tokens = _tokenize(product.get("name", ""))
    candidate_counts: Dict[int, int] = {}
    pid = product.get("id")
    for tok in tokens:
        for other in data["token_index"].get(tok, set()):
            if other == pid:
                continue
            candidate_counts[other] = candidate_counts.get(other, 0) + 1
    name_sim_count = sum(1 for v in candidate_counts.values() if v >= 2)
    if (cat_freq_pct is not None and cat_freq_pct >= 0.70) or name_sim_count >= 8:
        competition_baseline = "High"
    elif (cat_freq_pct is not None and cat_freq_pct <= 0.30) and name_sim_count <= 2:
        competition_baseline = "Low"
    else:
        competition_baseline = "Medium"
    return desire_baseline, competition_baseline


def generate_ba_insights(api_key: str, model: str, product: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    sys_msg = (
        "Eres estratega de marketing. Aplicas Breakthrough Advertising sin citar texto. "
        "Devuelve solo JSON con el esquema pedido. Español claro."
    )

    desire_baseline, competition_baseline = _compute_baselines(product)

    fields = [
        "id",
        "name",
        "category",
        "price",
        "rating",
        "units_sold",
        "revenue",
        "conversion_rate",
        "launch_date",
        "date_range",
        "desire",
        "desire_magnitude",
        "awareness_level",
        "competition_level",
    ]
    lines = [f"{k}: {product.get(k)}" for k in fields]
    baseline_line = (
        f"Baselines cuantitativos sugeridos: desire_magnitude={desire_baseline}, "
        f"competition_level={competition_baseline}. Si discrepas, sé conservador."
    )
    url = (product.get("image_url") or "").strip()
    if url and re.match(r"^https?://", url):
        text = (
            "Responde estrictamente con JSON siguiendo este esquema:\n"
            "{ \"grid_updates\": { \"desire\": \"...\", \"desire_magnitude\": \"Low|Medium|High\", "
            "\"awareness_level\": \"Unaware|Problem-Aware|Solution-Aware|Product-Aware|Most Aware\", "
            "\"competition_level\": \"Low|Medium|High\" } }\n" +
            baseline_line + "\n\n" +
            "Datos del producto:\n" + "\n".join(lines)
        )
        content = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": url}},
        ]
    else:
        if url:
            lines.append(f"Image URL: {url}")
        text = (
            "Responde estrictamente con JSON siguiendo este esquema:\n"
            "{ \"grid_updates\": { \"desire\": \"...\", \"desire_magnitude\": \"Low|Medium|High\", "
            "\"awareness_level\": \"Unaware|Problem-Aware|Solution-Aware|Product-Aware|Most Aware\", "
            "\"competition_level\": \"Low|Medium|High\" } }\n" +
            baseline_line + "\n\n" +
            "Datos del producto:\n" + "\n".join(lines)
        )
        content = [{"type": "text", "text": text}]

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": content},
    ]

    start = time.time()
    resp = call_gpt(
        messages=messages,
        api_key=api_key,
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    duration = time.time() - start
    usage = resp.get("usage", {})

    try:
        raw = resp["choices"][0]["message"]["content"].strip()
        if raw.startswith("```") and raw.endswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        data = json.loads(raw)
    except Exception as exc:
        logger.error("Respuesta BA no es JSON: %s", resp)
        raise InvalidJSONError("Respuesta BA no es JSON") from exc

    grid = data.get("grid_updates", {})
    gpt_desire = _norm_tri(grid.get("desire_magnitude"), "Medium")
    gpt_comp = _norm_tri(grid.get("competition_level"), "Medium")
    norm = {
        "desire": grid.get("desire"),
        "desire_magnitude": gpt_desire,
        "awareness_level": _norm_awareness(grid.get("awareness_level")),
        "competition_level": gpt_comp,
    }

    levels = ["Low", "Medium", "High"]

    def _idx(v: str) -> int:
        try:
            return levels.index(v)
        except ValueError:
            return 1

    norm["desire_magnitude"] = levels[
        min(_idx(gpt_desire), _idx(desire_baseline))
    ]
    norm["competition_level"] = levels[
        max(_idx(gpt_comp), _idx(competition_baseline))
    ]

    logger.info(
        "BA insights tokens=%s duration=%.2fs",
        usage.get("total_tokens"),
        duration,
    )
    return norm, usage, duration


def generate_batch_columns(api_key: str, model: str, items: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, Any], float]:
    sys_msg = (
        "Eres un analista de marketing. Aplica marcos de Breakthrough Advertising sin citar texto. "
        "Devuelve exclusivamente un JSON cuyas claves son los IDs de producto, y cuyos valores incluyen: "
        "desire (string), desire_magnitude (Low|Medium|High), awareness_level (Unaware|Problem-Aware|Solution-Aware|Product-Aware|Most Aware), "
        "competition_level (Low|Medium|High). No devuelvas comentarios, ni Markdown, ni bloques de código."
    )

    intro_text = (
        "Responde SOLO con un JSON.\n"
        "Raíz: objeto cuyas claves son IDs de producto (string o número).\n"
        "Por cada ID: { \"desire\": string,\n"
        "               \"desire_magnitude\": \"Low|Medium|High\",\n"
        "               \"awareness_level\": \"Unaware|Problem-Aware|Solution-Aware|Product-Aware|Most Aware\",\n"
        "               \"competition_level\": \"Low|Medium|High\" }.\n"
    )

    content: List[Dict[str, Any]] = [{"type": "text", "text": intro_text}]
    for it in items:
        lines = [
            "BEGIN_PRODUCT",
            f"id: {it.get('id')}",
            f"name: {it.get('name')}",
            f"category: {it.get('category')}",
            f"price: {it.get('price')}",
            f"rating: {it.get('rating')}",
            f"units_sold: {it.get('units_sold')}",
            f"revenue: {it.get('revenue')}",
            f"conversion_rate: {it.get('conversion_rate')}",
            f"launch_date: {it.get('launch_date')}",
            f"date_range: {it.get('date_range')}",
        ]
        url = (it.get("image_url") or "").strip()
        if not re.match(r"^https?://", url):
            if url:
                lines.append(f"image_url: {url}")
        lines.append("END_PRODUCT")
        content.append({"type": "text", "text": "\n".join(lines)})
        if url and re.match(r"^https?://", url):
            content.append({"type": "image_url", "image_url": {"url": url}})

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": content},
    ]

    start = time.time()
    resp = call_gpt(
        messages=messages,
        api_key=api_key,
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    duration = time.time() - start
    usage = resp.get("usage", {})

    try:
        raw = resp["choices"][0]["message"]["content"].strip()
        if raw.startswith("```") and raw.endswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        data = json.loads(raw)
    except Exception as exc:
        logger.error("Respuesta IA no es JSON: %s", resp)
        raise InvalidJSONError("Respuesta IA no es JSON") from exc

    if not isinstance(data, dict):
        raise InvalidJSONError("Respuesta IA no es JSON")

    ok: Dict[str, Dict[str, Any]] = {}
    ko: Dict[str, str] = {}
    for it in items:
        pid = str(it.get("id"))
        entry = data.get(pid)
        if not isinstance(entry, dict):
            ko[pid] = "missing"
            continue
        ok[pid] = {
            "desire": entry.get("desire"),
            "desire_magnitude": _norm_tri(entry.get("desire_magnitude")),
            "awareness_level": _norm_awareness(entry.get("awareness_level")),
            "competition_level": _norm_tri(entry.get("competition_level")),
        }

    return ok, ko, usage, duration


# ---------------- Winner Score evaluation -----------------


WINNER_SCORE_FIELDS = [
    "magnitud_deseo",
    "nivel_consciencia",
    "saturacion_mercado",
    "facilidad_anuncio",
    "facilidad_logistica",
    "escalabilidad",
    "engagement_shareability",
    "durabilidad_recurrencia",
]

NUMERIC_FIELD_MAP = {
    "orders": ("magnitud_deseo", False),
    "sellers": ("saturacion_mercado", True),
    "weight": ("facilidad_logistica", True),
}


def compute_numeric_scores(
    metrics: Dict[str, Any],
    ranges: Dict[str, Tuple[float, float]],
) -> Tuple[Dict[str, int], Dict[str, str], Dict[str, str]]:
    """Derive Winner Score sub-scores from numeric metrics.

    Args:
        metrics: Mapping of raw metric values for a product.
        ranges: Precomputed ``min``/``max`` pairs for each metric.

    Returns:
        Three dictionaries: ``scores`` with integer values 1-5, ``justifications``
        with short explanations, and ``sources`` marking variables as
        "data".
    """

    out_scores: Dict[str, int] = {}
    out_justifs: Dict[str, str] = {}
    out_sources: Dict[str, str] = {}

    for metric, (field, reverse) in NUMERIC_FIELD_MAP.items():
        raw_val = metrics.get(metric)
        if raw_val is None or raw_val == "":
            continue
        try:
            val = float(str(raw_val).replace(",", "."))
        except Exception:
            continue
        min_val, max_val = ranges.get(metric, (val, val))
        if max_val == min_val:
            score = 3.0
        else:
            ratio = (val - min_val) / (max_val - min_val)
            if reverse:
                ratio = 1.0 - ratio
            score = 1.0 + ratio * 4.0
        out_scores[field] = int(round(score))
        out_justifs[field] = f"Basado en {metric}: {raw_val}"[:120]
        out_sources[field] = "data"

    return out_scores, out_justifs, out_sources


def build_winner_score_prompt(product: Dict[str, Any]) -> str:
    """Construct the Winner Score prompt for a product.
    The prompt asks the model to rate eight qualitative variables between 1 and
    5 and provide a brief justification for each.  Optional metrics can be
    supplied to give the model additional context.

    Args:
        product: Mapping with keys ``title``/``name``, ``description`` and
            ``category`` describing the product.  An optional ``metrics``
            mapping may contain additional numeric information (e.g. orders,
            revenue).
    Returns:
        A Spanish prompt string to send to the model.
    """

    title = product.get("title") or product.get("name") or ""
    description = product.get("description") or ""
    category = product.get("category") or ""
    metrics = product.get("metrics") or {}
    metrics_lines = []
    if isinstance(metrics, dict) and metrics:
        metrics_lines.append("Métricas opcionales:")
        for k, v in metrics.items():
            metrics_lines.append(f"- {k}: {v}")

    metrics_block = "\n".join(metrics_lines)
    prompt = f"""
Eres un analista de producto experto en e-commerce y dropshipping.
Te doy datos de un producto (título, descripción, categoría, métricas opcionales).
Evalúa del 1 al 5 cada una de estas variables:
- Magnitud del deseo
- Nivel de consciencia del mercado
- Saturación / sofisticación de mercado
- Facilidad de explicar en un anuncio
- Facilidad logística
- Escalabilidad
- Engagement / shareability
- Durabilidad / recurrencia

Título: {title}
Descripción: {description}
Categoría: {category}
{metrics_block}

Devuelve solo en JSON con este formato:
{{
  "magnitud_deseo": X,
  "nivel_consciencia": X,
  "saturacion_mercado": X,
  "facilidad_anuncio": X,
  "facilidad_logistica": X,
  "escalabilidad": X,
  "engagement_shareability": X,
  "durabilidad_recurrencia": X,
  "justificacion": {{
    "magnitud_deseo": "...",
    "nivel_consciencia": "...",
    "saturacion_mercado": "...",
    "facilidad_anuncio": "...",
    "facilidad_logistica": "...",
    "escalabilidad": "...",
    "engagement_shareability": "...",
    "durabilidad_recurrencia": "..."
  }}
}}

Las justificaciones deben ser frases cortas (máx 15 palabras).
"""
    return prompt.strip()


def evaluate_winner_score(
    api_key: str, model: str, product: Dict[str, Any]
) -> Dict[str, Any]:
    """Call OpenAI to obtain Winner Score sub-scores for a product.

    The function returns a mapping with two keys:

    ``scores`` – dictionary of the eight variables with integer values 1–5.

    ``justifications`` – dictionary of short textual explanations for each
    variable (maximum 15 words, trimmed if necessary).
    Args:
        api_key: OpenAI API key.
        model: Identifier of the chat model to use.
        product: Mapping with product information.
    Raises:
        OpenAIError: If the API call fails or returns invalid content.
    """

    metrics = product.get("metrics") or {}
    required = [
        "price",
        "rating",
        "units_sold",
        "revenue",
        "desire",
        "competition",
        "oldness",
    ]
    missing = [k for k in required if metrics.get(k) is None]
    if missing:
        logger.warning("Winner Score missing_fields=%s", missing)
    metrics_filtered = {k: metrics.get(k) for k in required if metrics.get(k) is not None}
    product = product.copy()
    product["metrics"] = metrics_filtered

    prompt = build_winner_score_prompt(product)
    image_url = (product.get("image_url") or "").strip()
    include_image = True
    reason = ""
    cost_est = 0.0
    if image_url:
        include_image = config.include_image_in_ai()
        if not include_image:
            reason = "config"
        else:
            token_est = 85
            cost_est = token_est / 1000.0 * 0.002
            if cost_est > config.get_ai_image_cost_max_usd():
                include_image = False
                reason = "cost"
    if image_url and not include_image:
        logger.info("include_image=false reason=%s", reason)
    elif image_url and include_image:
        logger.info("include_image=true est_cost=%.4f", cost_est)

    if image_url and include_image:
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    else:
        user_content = prompt

    messages = [
        {
            "role": "system",
            "content": "Eres un asistente que responde únicamente con JSON válido.",
        },
        {"role": "user", "content": user_content},
    ]
    resp_json = call_gpt(messages=messages, api_key=api_key, model=model)
    try:
        content = resp_json["choices"][0]["message"]["content"].strip()
        raw = json.loads(content)
    except Exception as exc:
        raise OpenAIError(
            f"La respuesta de la IA no está en formato JSON válido: {exc}"
        ) from exc

    scores: Dict[str, int] = {}
    justifs_raw = raw.get("justificacion") or {}
    justifs: Dict[str, str] = {}
    for field in WINNER_SCORE_FIELDS:
        val = raw.get(field)
        try:
            ival = int(val)
        except Exception:
            ival = 3
        if ival < 1:
            ival = 1
        if ival > 5:
            ival = 5
        scores[field] = ival

        jtxt = ""
        if isinstance(justifs_raw, dict):
            jtxt = justifs_raw.get(field, "")
        if isinstance(jtxt, str):
            words = jtxt.strip().split()
            if len(words) > 15:
                jtxt = " ".join(words[:15])
            justifs[field] = jtxt

    return {"scores": scores, "justifications": justifs}

def simplify_product_names(api_key: str, model: str, names: List[str], *, temperature: float = 0.2) -> Dict[str, str]:
    """
    Simplify a list of product names by removing brand names and extra descriptors.

    This helper sends a single request to the Chat Completion API asking the
    model to return a JSON object mapping each original name to a simplified
    version.  It limits the number of names to avoid exceeding token limits.

    Args:
        api_key: OpenAI API key.
        model: Model identifier, e.g. "gpt-4o".
        names: List of full product names to simplify.
        temperature: Temperature parameter for the model.

    Returns:
        A dictionary mapping original names to simplified names.  If parsing fails
        or an error occurs, an empty dict is returned.
    """
    if not names:
        return {}
    # Limit to first 50 names to stay within token limits
    limited = names[:50]
    prompt_lines = []
    prompt_lines.append(
        "Simplifica los siguientes nombres de productos de comercio electrónico. "
        "Para cada nombre, deja solo el término del producto principal, sin marcas, tamaños ni especificaciones. "
        "Devuelve un objeto JSON donde cada clave sea el nombre original y el valor sea el nombre simplificado."
    )
    for n in limited:
        prompt_lines.append(f"- {n}")
    prompt = "\n".join(prompt_lines)
    messages = [
        {"role": "system", "content": "Eres un asistente experto en síntesis de nombres de productos."},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = call_gpt(
            messages=messages,
            api_key=api_key,
            model=model,
            temperature=temperature,
        )
        # We expect a JSON response in the assistant's content
        content = resp['choices'][0]['message']['content']
        simplified = json.loads(content)
        if not isinstance(simplified, dict):
            return {}
        return simplified
    except Exception:
        # If parsing or the API call fails, return an empty mapping
        return {}


def recommend_winner_weights(
    api_key: str,
    model: str,
    samples: List[Dict[str, Any]],
    success_key: str,
) -> Dict[str, Any]:
    """
    Devuelve pesos 0..100 (independientes, NO normalizados) y, si la IA falla o devuelve
    basura, calcula un fallback estadístico por correlación con la señal de éxito.
    Incluye SIEMPRE todas las variables permitidas (incluida 'revenue').
    """
    try:
        allowed = list(winner_calc.ALLOWED_FIELDS)
    except Exception:
        allowed = ["price","rating","units_sold","revenue","desire","competition","oldness","awareness"]

    def _stat_fallback(rows: List[Dict[str, float]], target: str) -> Dict[str, int]:
        # Pearson simple sin numpy, robusto a valores faltantes.
        def _corr(xs, ys):
            n = len(xs)
            if n < 2: return 0.0
            mx = sum(xs)/n; my = sum(ys)/n
            num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
            denx = sum((x-mx)**2 for x in xs)**0.5
            deny = sum((y-my)**2 for y in ys)**0.5
            return 0.0 if denx==0 or deny==0 else (num/(denx*deny))
        tgt = [float(r.get(target, 0.0)) for r in rows]
        raw = {}
        for k in allowed:
            if k == target:  # aún queremos ponderar target si lo deseas; aquí usamos solo como señal
                # lo dejamos, pero correlación de la variable consigo misma sería 1.0 — no usarla para inflarla
                pass
            xs = [float(r.get(k, 0.0)) for r in rows]
            c = abs(_corr(xs, tgt))
            raw[k] = c
        # Reescala 0..100; si todo cero, da un perfil razonable
        mx = max(raw.values() or [0.0])
        if mx <= 0:
            profile = {"revenue":80,"units_sold":70,"rating":60,"price":55,"desire":45,"competition":35,"oldness":25,"awareness":15}
            return {k:int(profile.get(k,50)) for k in allowed}
        return {k: int(round((v/mx)*100.0)) for k,v in raw.items()}

    def _extract_json_block(text: str) -> dict:
        import json, re
        # código entre ```json ... ``` o ``` ... ```
        m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if m:
            try: return json.loads(m.group(1))
            except Exception: pass
        # primera llave balanceada
        s = text
        try:
            start = s.index("{")
            depth = 0
            for i,ch in enumerate(s[start:], start):
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return json.loads(s[start:i+1])
        except Exception:
            pass
        # último intento directo
        return json.loads(text)

    # Si no hay muestras, devuelve algo neutro pero válido
    if not samples:
        return {"weights": {k: 50 for k in allowed}, "justification": ""}

    # Construir prompt
    sample_json = json.dumps(samples[:50], ensure_ascii=False)
    prompt = (
        "Eres un optimizador de modelos de e-commerce.\n"
        f"Variables del Winner Score: {allowed}\n"
        f"Señal de éxito a maximizar: '{success_key}'.\n"
        "Devuelve JSON ESTRICTO con pesos 0..100 por variable (independientes, NO normalizados):\n"
        "{\n"
        '  "pesos_0_100": {\n'
        '    "price": 0..100,\n'
        '    "rating": 0..100,\n'
        '    "units_sold": 0..100,\n'
        '    "revenue": 0..100,\n'
        '    "desire": 0..100,\n'
        '    "competition": 0..100,\n'
        '    "oldness": 0..100,\n'
        '    "awareness": 0..100\n'
        "  },\n"
        '  "orden": ["revenue","price",...],\n'
        '  "justificacion": "1-2 frases"\n'
        "}\n"
        "- No normalices para que sumen 100; cada peso es una intensidad 0..100.\n\n"
        "Muestra parcial:\n" + sample_json
    )
    messages = [
        {"role": "system", "content": "Eres un optimizador de modelos para e-commerce."},
        {"role": "user", "content": prompt},
    ]

    parsed = {}
    try:
        resp = call_gpt(messages=messages, api_key=api_key, model=model)
        content = resp["choices"][0]["message"]["content"].strip()
        parsed = _extract_json_block(content)
    except Exception:
        parsed = {}

    # Leer posibles claves de pesos
    raw = {}
    if isinstance(parsed, dict):
        raw = parsed.get("pesos_0_100") or parsed.get("pesos") or parsed.get("weights") or {}
    justification = (parsed.get("justificacion") or parsed.get("justification") or "") if isinstance(parsed, dict) else ""

    # Reescalar si vino en 0..1 o suma≈1
    try:
        vals = [float(v) for v in raw.values()]
    except Exception:
        vals = []
    if vals:
        all_01 = all(0.0 <= float(v) <= 1.0 for v in vals)
        sum_is_1 = abs(sum(vals) - 1.0) < 1e-6
        if all_01 and (sum_is_1 or max(vals) <= 1.0):
            raw = {k: float(v) * 100.0 for k, v in raw.items()}

    # Completar 0..100, clamp y enteros
    out: Dict[str, int] = {}
    for k in allowed:
        v = raw.get(k, None)
        if v is None:
            continue
        try: v = float(v)
        except Exception: v = None
        if v is None: continue
        v = max(0.0, min(100.0, v))
        out[k] = int(round(v))

    # Si no hay nada útil o todos iguales → fallback estadístico por correlación
    if not out or len(set(out.values())) == 1:
        out = _stat_fallback(samples, success_key)

    # Garantiza todas las claves (si falta alguna tras fallback, rellena 0)
    for k in allowed:
        out.setdefault(k, 0)

    return {"weights": out, "justification": justification}


def summarize_top_products(api_key: str, model: str, products: List[Dict[str, Any]]) -> str:
    """Generate a brief report highlighting top products based on their scores.

    Args:
        api_key: OpenAI API key.
        model: Chat model identifier.
        products: List of product dicts. Each item should include ``name`` (or
            ``title``) and a mapping of scores under ``scores`` or ``variables``.

    Returns:
        A short natural language summary in Spanish suitable for display in an
        "Insights IA" panel.

    Raises:
        OpenAIError: If the OpenAI API call fails.
    """

    # Build a concise description of each product and its variables
    lines: List[str] = []
    for idx, prod in enumerate(products, 1):
        name = prod.get("name") or prod.get("title") or f"Producto {idx}"
        scores = prod.get("scores") or prod.get("variables") or {}
        score_str = ", ".join(f"{k}: {v}" for k, v in scores.items())
        lines.append(f"{idx}. {name} - {score_str}")
    products_block = "\n".join(lines)

    prompt = (
        "Eres un consultor de producto.\n"
        "A continuación se listan varios productos con sus puntuaciones.\n"
        "Resume en un informe breve cuáles son los 3 productos con mayor potencial según sus scores.\n"
        "Explica en pocas frases qué variables son las más decisivas en su puntuación.\n\n"
        f"{products_block}\n\nInforme breve:"
    )
    messages = [
        {
            "role": "system",
            "content": "Eres un analista que redacta conclusiones claras en español.",
        },
        {"role": "user", "content": prompt},
    ]
    resp = call_gpt(messages=messages, api_key=api_key, model=model)
    try:
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        raise OpenAIError(f"Respuesta inesperada de OpenAI: {exc}") from exc
