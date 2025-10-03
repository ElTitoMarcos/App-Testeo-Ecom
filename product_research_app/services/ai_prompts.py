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
            "desire_reason": {"type": "string", "minLength": 1},
            "competition": {"type": "number", "minimum": 0, "maximum": 1},
            "competition_level": {
                "type": "string",
                "enum": ["low", "medium", "high"],
            },
            "revenue": {"type": "number", "minimum": 0},
            "units_sold": {"type": "number", "minimum": 0},
            "price": {"type": "number", "minimum": 0},
            "oldness": {"type": "number", "minimum": 0, "maximum": 1},
            "rating": {"type": "number", "minimum": 0, "maximum": 5},
        },
        "required": [
            "id",
            "desire",
            "desire_reason",
            "competition",
            "competition_level",
            "revenue",
            "units_sold",
            "price",
            "oldness",
            "rating",
        ],
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
    from .prompt_templates import STRICT_JSONL_PROMPT

    ids = [p["id"] for p in batch]
    items: List[Dict[str, Any]] = []
    for p in batch:
        items.append(
            {
                "id": p["id"],
                "title": p.get("title") or p.get("name") or "",
                "category": p.get("category") or "",
                "desc": (p.get("description") or "")[:1200],
            }
        )
    instruction = STRICT_JSONL_PROMPT(ids, tuple(AI_FIELDS))
    user_content = instruction + json.dumps(items, ensure_ascii=False, indent=2)
    sys_prompt = (
        "Eres un motor determinista de transformación de datos. Devuelves SOLO JSON válido que cumpla exactamente el esquema indicado. No añades texto adicional."
    )
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_content}]


def _bucket_from_fraction(score: float) -> str:
    if score <= 0.33:
        return "Low"
    if score >= 0.67:
        return "High"
    return "Medium"


def _awareness_from_fraction(score: float) -> str:
    if score >= 0.85:
        return "Most Aware"
    if score >= 0.65:
        return "Product-Aware"
    if score >= 0.45:
        return "Solution-Aware"
    return "Problem-Aware"


def _normalise_competition(label: str, score: float) -> str:
    mapping = {
        "low": "Low",
        "bajo": "Low",
        "medium": "Medium",
        "medio": "Medium",
        "med": "Medium",
        "high": "High",
        "alto": "High",
    }
    key = label.strip().lower()
    if key in mapping:
        return mapping[key]
    return _bucket_from_fraction(score)


def _coerce_fraction(value: Any) -> float:
    if isinstance(value, (int, float)):
        num = float(value)
    else:
        try:
            num = float(str(value).strip())
        except Exception:
            raise ValueError("invalid fraction")
    if num > 1:
        num /= 100.0
    if not 0.0 <= num <= 1.0:
        raise ValueError("fraction out of range")
    return num


def parse_score(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parsea respuesta de puntuación."""
    content = raw.get("content") or raw.get("message") or raw
    if isinstance(content, dict):
        txt = json.dumps(content, ensure_ascii=False)
    else:
        txt = str(content)
    try:
        data = json.loads(txt)
        if isinstance(data, dict):
            items = data.get("items")
        else:
            items = data
        if not isinstance(items, list):
            return []
        rows: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict) or "id" not in item:
                continue
            try:
                pid = int(item["id"])
            except Exception:
                continue
            try:
                desire_score = _coerce_fraction(item.get("desire"))
                comp_score = _coerce_fraction(item.get("competition"))
            except Exception:
                continue
            reason = str(item.get("desire_reason") or "").strip()
            if not reason:
                continue
            comp_label_raw = str(item.get("competition_level") or "")
            comp_label = _normalise_competition(comp_label_raw, comp_score)
            rows.append(
                {
                    "id": pid,
                    "desire": reason,
                    "desire_magnitude": _bucket_from_fraction(desire_score),
                    "awareness_level": _awareness_from_fraction(desire_score),
                    "competition_level": comp_label,
                }
            )
        return rows
    except Exception:
        return []
