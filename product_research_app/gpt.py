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
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class OpenAIError(Exception):
    """Custom exception for OpenAI API errors."""
    pass


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


def call_openai_chat(api_key: str, model: str, messages: List[Dict[str, str]], temperature: float = 0.2) -> Dict[str, Any]:
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
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    except requests.RequestException as exc:
        raise OpenAIError(f"Failed to connect to OpenAI API: {exc}") from exc
    if response.status_code != 200:
        try:
            err = response.json()
            msg = err.get("error", {}).get("message", response.text)
        except Exception:
            msg = response.text
        raise OpenAIError(f"OpenAI API returned status {response.status_code}: {msg}")
    try:
        return response.json()
    except Exception as exc:
        raise OpenAIError(f"Invalid JSON response from OpenAI: {exc}") from exc


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
    resp_json = call_openai_chat(api_key, model, messages)
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


# ---------------- Winner Score v2 evaluation -----------------


WINNER_SCORE_V2_FIELDS = [
    "magnitud_deseo",
    "nivel_consciencia",
    "saturacion_mercado",
    "facilidad_anuncio",
    "facilidad_logistica",
    "escalabilidad",
    "engagement_shareability",
    "durabilidad_recurrencia",
]


def build_winner_score_prompt(product: Dict[str, Any]) -> str:
    """Construct the Winner Score v2 prompt for a product.

    The prompt requests the model to rate eight variables from 1 to 5 based on
    the product's title, description and category. The model must answer with a
    JSON object containing the integer scores for each variable.

    Args:
        product: Mapping with optional keys ``title``/``name``, ``description``
            and ``category`` describing the product.

    Returns:
        A Spanish prompt string to send to the model.
    """

    title = product.get("title") or product.get("name") or ""
    description = product.get("description") or ""
    category = product.get("category") or ""
    prompt = f"""
Eres un analista de producto para e-commerce.
Te doy un título, descripción y categoría.
Evalúa del 1 al 5 (solo números enteros) cada una de estas variables:
- Magnitud del deseo
- Nivel de consciencia del mercado
- Saturación / sofisticación (estima según categoría y competencia implícita)
- Facilidad de explicar en un anuncio
- Facilidad logística (peso/envío/fragilidad, estima por categoría)
- Escalabilidad (aplica a mucha gente o a pocos)
- Engagement / shareability (atractivo visual o viral)
- Durabilidad / recurrencia de compra

Título: {title}
Descripción: {description}
Categoría: {category}

Devuelve en formato JSON usando para cada variable un objeto con "score" (entero 1-5) y "justificacion" (frase breve de una línea):
{{
  "magnitud_deseo": {{"score": x, "justificacion": "..."}},
  "nivel_consciencia": {{"score": x, "justificacion": "..."}},
  "saturacion_mercado": {{"score": x, "justificacion": "..."}},
  "facilidad_anuncio": {{"score": x, "justificacion": "..."}},
  "facilidad_logistica": {{"score": x, "justificacion": "..."}},
  "escalabilidad": {{"score": x, "justificacion": "..."}},
  "engagement_shareability": {{"score": x, "justificacion": "..."}},
  "durabilidad_recurrencia": {{"score": x, "justificacion": "..."}}
}}
"""
    return prompt.strip()


def evaluate_winner_score(
    api_key: str, model: str, product: Dict[str, Any]
) -> Dict[str, Any]:
    """Call OpenAI to obtain Winner Score v2 sub-scores for a product.

    Args:
        api_key: OpenAI API key.
        model: Identifier of the chat model to use.
        product: Mapping with product information.

    Returns:
        Parsed JSON dictionary with the eight variables. Missing or invalid
        values are left as-is for the caller to handle.

    Raises:
        OpenAIError: If the API call fails or returns invalid content.
    """

    prompt = build_winner_score_prompt(product)
    messages = [
        {
            "role": "system",
            "content": "Eres un asistente que responde únicamente con JSON válido.",
        },
        {"role": "user", "content": prompt},
    ]
    resp_json = call_openai_chat(api_key, model, messages)
    try:
        content = resp_json["choices"][0]["message"]["content"].strip()
        result = json.loads(content)
    except Exception as exc:
        raise OpenAIError(f"La respuesta de la IA no está en formato JSON válido: {exc}") from exc
    return result

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
        resp = call_openai_chat(api_key, model, messages, temperature=temperature)
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
) -> Dict[str, float]:
    """Ask GPT to propose weights for Winner Score variables.

    This helper sends a list of sample products, each containing the eight
    Winner Score variables and a real-world success metric (``success_key``),
    and asks the model to return a JSON object with normalized weights that
    best correlate with the provided metric.

    Args:
        api_key: OpenAI API key.
        model: Chat model identifier.
        samples: List of mappings with the eight variables and a success value.
        success_key: Name of the success metric (e.g. ``orders`` or ``revenue``)
            included in each sample.

    Returns:
        Mapping of variable name to weight, normalized so the sum equals 1. If
        the model does not return valid weights, a uniform distribution is
        returned instead.
    """

    if not samples:
        # no data -> uniform weights
        return {k: 1.0 / len(WINNER_SCORE_V2_FIELDS) for k in WINNER_SCORE_V2_FIELDS}

    sample_json = json.dumps(samples[:20], ensure_ascii=False)
    prompt = (
        "Analiza la siguiente muestra de productos representada como un array JSON. "
        f"Cada producto incluye un valor '{success_key}' que indica su éxito real y las ocho subpuntuaciones de Winner Score v2. "
        "Devuelve únicamente un objeto JSON con pesos normalizados (suma=1) para las claves: "
        + ", ".join(WINNER_SCORE_V2_FIELDS) + "."
        " Los pesos deben maximizar la correlación con el éxito."\
    )
    prompt += "\nMuestra:\n" + sample_json
    messages = [
        {"role": "system", "content": "Eres un analista experto en estadística de productos."},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = call_openai_chat(api_key, model, messages)
        content = resp["choices"][0]["message"]["content"].strip()
        weights = json.loads(content)
        if not isinstance(weights, dict):
            raise ValueError("Respuesta no es un objeto JSON")
    except Exception:
        # fallback to uniform weights
        return {k: 1.0 / len(WINNER_SCORE_V2_FIELDS) for k in WINNER_SCORE_V2_FIELDS}

    total = 0.0
    cleaned: Dict[str, float] = {}
    for key in WINNER_SCORE_V2_FIELDS:
        try:
            val = float(weights.get(key, 0.0))
            if val < 0:
                val = 0.0
        except Exception:
            val = 0.0
        cleaned[key] = val
        total += val
    if total <= 0:
        return {k: 1.0 / len(WINNER_SCORE_V2_FIELDS) for k in WINNER_SCORE_V2_FIELDS}
    return {k: v / total for k, v in cleaned.items()}

