from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

from flask import Blueprint, current_app, jsonify, request

from . import app
from product_research_app.ai import gpt_orchestrator
from product_research_app.services.aggregates import (
    build_weighting_aggregates,
    sample_product_titles,
)

_GPT_API = Blueprint("gpt_api", __name__)

_ALLOWED_PRODUCT_FIELDS = {
    "price",
    "rating",
    "units_sold",
    "revenue",
    "desire",
    "competition",
    "oldness",
    "awareness",
    "category",
    "title",
    "description",
    "id",
    "dateAdded",
    "store",
}

_DEFAULT_PROMPTS: Mapping[str, str] = {
    "consulta": "Analiza el conjunto de productos y entrega hallazgos accionables.",
    "pesos": "Revisa los agregados y sugiere cómo ajustar los pesos del scoring.",
    "tendencias": "Identifica tendencias clave y oportunidades dentro del contexto.",
    "imputacion": "Propón imputaciones plausibles para los campos faltantes.",
    "desire": "Evalúa el nivel de deseo comercial de cada producto.",
}

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "ai" / "prompts"
_PROMPT_FILENAMES: Mapping[str, str] = {
    "consulta": "consulta.md",
    "pesos": "pesos.md",
    "tendencias": "tendencias.md",
    "imputacion": "imputacion.md",
    "desire": "desire.md",
}


def _register_routes() -> None:
    app.register_blueprint(_GPT_API, url_prefix="/api/gpt")


def _handle_task(task: str) -> tuple[dict, int]:
    body = request.get_json(force=True, silent=True)
    if not isinstance(body, dict):
        return {"ok": False, "error": "JSON inválido"}, 400

    prompt_text = body.get("prompt_text")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        prompt_text = _DEFAULT_PROMPTS.get(task, "")
    prompt_text = prompt_text.strip()

    context_raw = body.get("context")
    if not isinstance(context_raw, MutableMapping):
        context_raw = {}

    params_raw = body.get("params")
    params: Dict[str, Any]
    if isinstance(params_raw, MutableMapping):
        params = dict(params_raw)
    else:
        params = {}

    sanitized_context = _sanitize_context(context_raw)
    payload = dict(sanitized_context)
    if task == "pesos":
        payload = _inject_weighting_summary(payload)
    if params:
        payload["params"] = params

    result = gpt_orchestrator.run_task(
        task, prompt_text=prompt_text, json_payload=payload, system_prompt=_get_system_prompt(task)
    )

    meta = result.get("meta") or {}
    info = {
        "task": task,
        "items": len(sanitized_context.get("products") or []),
        "group_id": sanitized_context.get("group_id"),
        "time_window": sanitized_context.get("time_window"),
        "chunks": meta.get("chunks"),
        "model": result.get("model"),
    }
    current_app.logger.info("gpt_endpoint %s", json.dumps(info, ensure_ascii=False))

    response = {
        "ok": bool(result.get("ok")),
        "text": result.get("text"),
        "data": result.get("data"),
        "warnings": result.get("warnings") or [],
        "meta": meta,
        "model": result.get("model"),
    }
    return response, 200


def _normalize_group_id(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return str(value)


def _sanitize_context(context: Mapping[str, Any]) -> Dict[str, Any]:
    group_id_raw = context.get("group_id", context.get("groupId"))
    group_id = _normalize_group_id(group_id_raw)
    time_window = context.get("time_window")
    if isinstance(time_window, str):
        time_window = time_window.strip() or None
    else:
        time_window = None

    visible_ids_raw = context.get("visible_ids", context.get("visibleIds"))
    visible_ids: Optional[List[str]] = None
    if isinstance(visible_ids_raw, Iterable) and not isinstance(visible_ids_raw, (str, bytes)):
        tmp: List[str] = []
        for item in visible_ids_raw:
            if item in (None, ""):
                continue
            tmp.append(str(item))
        visible_ids = tmp

    products_raw = context.get("products")
    sanitized_products: List[Dict[str, Any]] = []
    if isinstance(products_raw, list):
        for entry in products_raw:
            if not isinstance(entry, MutableMapping):
                continue
            if group_id is not None:
                entry_group = entry.get("group_id") or entry.get("groupId") or entry.get("group")
                if _normalize_group_id(entry_group) != group_id:
                    continue
            sanitized = {}
            for key in _ALLOWED_PRODUCT_FIELDS:
                if key in entry:
                    value = entry[key]
                    if key == "id" and value not in (None, ""):
                        sanitized[key] = str(value)
                    else:
                        sanitized[key] = value
            if sanitized:
                sanitized_products.append(sanitized)

    sanitized_context: Dict[str, Any] = {
        "group_id": group_id,
        "time_window": time_window,
        "products": sanitized_products,
    }
    if visible_ids is not None:
        sanitized_context["visible_ids"] = visible_ids

    return sanitized_context


def _inject_weighting_summary(context: Dict[str, Any]) -> Dict[str, Any]:
    products = context.get("products")
    if not isinstance(products, list):
        context["products"] = {"aggregates": {"metrics": {}, "total_products": 0}, "sample_titles": []}
        return context

    aggregates = build_weighting_aggregates(products)
    titles = sample_product_titles(products, limit=20)
    context["products"] = {"aggregates": aggregates, "sample_titles": titles}
    return context


@lru_cache(maxsize=None)
def _get_system_prompt(task: str) -> str:
    filename = _PROMPT_FILENAMES.get(task)
    if not filename:
        return gpt_orchestrator.SYSTEM_PROMPT
    path = _PROMPT_DIR / filename
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        current_app.logger.warning("prompt_template_missing task=%s path=%s", task, path)
        return gpt_orchestrator.SYSTEM_PROMPT
    cleaned = text.strip()
    return cleaned or gpt_orchestrator.SYSTEM_PROMPT


@_GPT_API.route("/consulta", methods=["POST"])
def consulta_endpoint():
    payload, status = _handle_task("consulta")
    return jsonify(payload), status


@_GPT_API.route("/pesos", methods=["POST"])
def pesos_endpoint():
    payload, status = _handle_task("pesos")
    return jsonify(payload), status


@_GPT_API.route("/tendencias", methods=["POST"])
def tendencias_endpoint():
    payload, status = _handle_task("tendencias")
    return jsonify(payload), status


@_GPT_API.route("/imputacion", methods=["POST"])
def imputacion_endpoint():
    payload, status = _handle_task("imputacion")
    return jsonify(payload), status


@_GPT_API.route("/desire", methods=["POST"])
def desire_endpoint():
    payload, status = _handle_task("desire")
    return jsonify(payload), status


_register_routes()
