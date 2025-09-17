from __future__ import annotations

import json
import math
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

_IMPUTABLE_FIELDS = {"review_count", "image_count", "profit_margin"}

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

    warnings = list(result.get("warnings") or [])
    ok = bool(result.get("ok"))
    data = result.get("data")

    if task == "imputacion":
        reshaped, reshape_warnings = _reshape_imputation_payload(
            data, sanitized_context.get("products") or []
        )
        warnings.extend(reshape_warnings)
        data = reshaped
        if not data or not data.get("imputed"):
            ok = False

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
        "ok": ok,
        "text": result.get("text"),
        "data": data or None,
        "warnings": warnings,
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


def _reshape_imputation_payload(
    data: Any, products: List[Dict[str, Any]]
) -> tuple[Optional[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    if not isinstance(data, Mapping):
        return None, ["Respuesta de imputación sin estructura JSON"]

    cleaned: Dict[str, Any] = {}
    for key, value in data.items():
        if key in {"results", "items", "imputed"}:
            continue
        cleaned[key] = value

    source_map: Optional[Dict[str, Any]] = None
    raw_imputed = data.get("imputed")
    if isinstance(raw_imputed, Mapping):
        source_map = dict(raw_imputed)
    elif isinstance(data.get("results"), Mapping):
        source_map = dict(data["results"])
    elif isinstance(data.get("items"), Iterable):
        temp: Dict[str, Any] = {}
        for entry in data["items"]:
            if not isinstance(entry, Mapping):
                continue
            pid = entry.get("id") or entry.get("product_id") or entry.get("asin")
            if pid in (None, ""):
                continue
            details = {
                key: value
                for key, value in entry.items()
                if key not in {"id", "product_id", "asin"}
            }
            temp[str(pid)] = details
        if temp:
            source_map = temp
    else:
        fallback: Dict[str, Any] = {}
        for key, value in data.items():
            if key == "prompt_version":
                continue
            if isinstance(value, Mapping):
                fallback[str(key)] = value
        if fallback:
            source_map = fallback

    imputed_map, map_warnings = _build_imputed_map(source_map, products)
    warnings.extend(map_warnings)
    cleaned["imputed"] = imputed_map
    return cleaned, warnings


def _build_imputed_map(
    source_map: Optional[Mapping[str, Any]], products: List[Dict[str, Any]]
) -> tuple[Dict[str, Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    if not isinstance(source_map, Mapping):
        if source_map not in (None, {}):
            warnings.append("Formato inesperado en datos de imputación; se ignoró el bloque")
        return {}, warnings

    allowed_ids = {
        str(entry.get("id"))
        for entry in products
        if isinstance(entry, Mapping) and entry.get("id") not in (None, "")
    }

    imputed: Dict[str, Dict[str, Any]] = {}
    for raw_id, raw_fields in source_map.items():
        pid = str(raw_id).strip() if raw_id not in (None, "") else ""
        if not pid:
            warnings.append("Se omitió una imputación sin identificador válido")
            continue
        if allowed_ids and pid not in allowed_ids:
            warnings.append(f"Producto {pid} fuera del contexto recibido; se omitió")
            continue
        if not isinstance(raw_fields, Mapping):
            warnings.append(f"Producto {pid}: estructura de campos inválida")
            continue

        entry: Dict[str, Any] = {}
        for field in _IMPUTABLE_FIELDS:
            if field not in raw_fields:
                continue
            value, value_warnings = _normalise_imputed_value(field, raw_fields[field])
            if value is None:
                for msg in value_warnings:
                    warnings.append(f"Producto {pid}: {msg}")
                continue
            for msg in value_warnings:
                warnings.append(f"Producto {pid}: {msg}")
            entry[field] = value

        if entry:
            imputed[pid] = entry

    return imputed, warnings


def _normalise_imputed_value(field: str, raw_value: Any) -> tuple[Optional[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    candidate = raw_value
    confidence: Optional[str] = None
    notes: Optional[str] = None

    if isinstance(raw_value, Mapping):
        for key in ("value", "valor", "suggested", "imputed", field):
            if key in raw_value:
                candidate = raw_value[key]
                break
        conf = raw_value.get("confidence")
        if isinstance(conf, str) and conf.strip():
            confidence = conf.strip()
        note = raw_value.get("notes") or raw_value.get("justification")
        if isinstance(note, str) and note.strip():
            notes = note.strip()

    numeric = _parse_numeric(candidate)
    if numeric is None:
        warnings.append(f"{field} sin valor numérico utilizable")
        return None, warnings

    adjusted = False
    if numeric < 0:
        numeric = 0.0
        adjusted = True

    if field in {"review_count", "image_count"}:
        numeric = round(numeric)
        if numeric < 0:
            numeric = 0.0
        value: Any = int(numeric)
    else:
        value = float(numeric)

    result: Dict[str, Any] = {"value": value}
    if notes:
        result["notes"] = notes
    if field == "profit_margin":
        result["confidence"] = confidence or "low_confidence"
    elif confidence:
        result["confidence"] = confidence

    if adjusted:
        warnings.append(f"{field} ajustado a 0 por valor negativo")

    return result, warnings


def _parse_numeric(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except Exception:
            return None
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("%"):
            text = text[:-1]
        text = text.replace(",", ".")
        try:
            number = float(text)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


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
