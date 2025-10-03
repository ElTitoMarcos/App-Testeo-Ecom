from __future__ import annotations

import json
from typing import Any, Dict, List

AI_FIELDS = ["desire", "desire_magnitude", "awareness_level", "competition_level"]


# NUEVO
def score_item_schema():
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "integer"},
            "desire": {"type": "number", "minimum": 0, "maximum": 1},
            "desire_label": {"type": "string", "minLength": 1},
            "desire_magnitude": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "competition_level": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "price": {"type": "number"},
        },
        "required": ["id", "desire", "desire_label", "desire_magnitude"],
    }


# NUEVO
def build_score_json_schema():
    return {
        "name": "score_batch",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "items": score_item_schema(),
                }
            },
            "required": ["items"],
        },
        "strict": True,
    }


def build_triage_messages(batch: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Construye mensajes para decidir si un producto requiere puntuación completa."""
    sys = (
        "Eres un clasificador. Procura responder en JSON limpio cuando sea posible. "
        "Entrada: lista de productos. Salida: array con objetos {id, needs_scoring}."
    )
    items = []
    for p in batch:
        items.append(
            {
                "id": p["id"],
                "title": p.get("title") or p.get("name") or "",
                "category": p.get("category") or "",
                "desc": (p.get("description") or "")[:600],
            }
        )
    user = (
        "Decide si se requiere evaluación completa (needs_scoring=true) "
        "solo cuando el título/desc no permitan inferir con reglas obvias.\n"
        "Devuelve una lista JSON si puedes, con objetos {\"id\": <int>, \"needs_scoring\": <bool>}.\n"
        f"INPUT={json.dumps(items, ensure_ascii=False)}"
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def parse_triage(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parsea respuesta de triaje a lista de {id, needs_scoring}."""
    content = raw.get("content") or raw.get("message") or raw
    if isinstance(content, dict):
        txt = json.dumps(content, ensure_ascii=False)
    else:
        txt = str(content)
    try:
        data = json.loads(txt)
        assert isinstance(data, list)
        rows = []
        for x in data:
            if isinstance(x, dict) and "id" in x and "needs_scoring" in x:
                rows.append({"id": int(x["id"]), "needs_scoring": bool(x["needs_scoring"])})
        return rows
    except Exception:
        return []


def build_score_messages(batch: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Construye mensajes para puntuar productos."""
    sys = (
        "Eres un analista. Devuelve únicamente un objeto JSON con esta forma: {\"items\":[ ... ]}. "
        "Nada de texto adicional, ni markdown. Cada objeto del array debe corresponder al producto solicitado en el mismo orden e incluir como mínimo: "
        "id, desire, desire_label, desire_magnitude. Añade competition_level y price cuando puedas inferirlos. Prohibidas las explicaciones."
    )
    items = []
    for p in batch:
        items.append(
            {
                "id": p["id"],
                "title": p.get("title") or p.get("name") or "",
                "category": p.get("category") or "",
                "desc": (p.get("description") or "")[:1200],
            }
        )
    user = (
        "Evalúa cada producto y devuelve únicamente un objeto JSON con clave items cuyo valor sea un array en el mismo orden. "
        "Prohibido añadir comentarios o texto antes o después. Usa desire_magnitude entre 0 y 1 y etiqueta corta en desire_label. "
        f"INPUT={json.dumps(items, ensure_ascii=False)}"
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def parse_score(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parsea respuesta de puntuación."""
    content = raw.get("content") or raw.get("message") or raw
    if isinstance(content, dict):
        txt = json.dumps(content, ensure_ascii=False)
    else:
        txt = str(content)
    try:
        data = json.loads(txt)
        assert isinstance(data, list)
        rows = []
        for x in data:
            if not isinstance(x, dict) or "id" not in x:
                continue
            row = {"id": int(x["id"])}
            for k in AI_FIELDS:
                if k in x:
                    row[k] = x[k]
            rows.append(row)
        return rows
    except Exception:
        return []
