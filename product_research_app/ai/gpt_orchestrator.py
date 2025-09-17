from __future__ import annotations

import json
import logging
import math
import os
import re
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import requests

logger = logging.getLogger(__name__)

CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
SYSTEM_PROMPT = (
    "Eres un analista experto que trabaja con grandes listados de productos. "
    "Debes entregar conclusiones claras y siempre terminar con un bloque JSON "
    "dentro de triple acento grave que incluya la clave obligatoria 'prompt_version'."
)
JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)

_TaskName = Literal["consulta", "pesos", "tendencias", "imputacion", "desire"]

_TASK_MODEL_MAP: Dict[_TaskName, Tuple[str, str]] = {
    "consulta": ("A", "gpt-4o-mini"),
    "pesos": ("B", "gpt-4o"),
    "tendencias": ("C", "gpt-4o-mini"),
    "imputacion": ("D", "gpt-4o-mini"),
    "desire": ("E", "gpt-4o-mini"),
}


def run_task(
    task: _TaskName,
    *,
    prompt_text: str,
    json_payload: Optional[Dict[str, Any]],
    model_hint: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute an AI task orchestrating chunking and response parsing."""

    if task not in _TASK_MODEL_MAP:
        raise ValueError(f"Unknown task '{task}'")

    model = _resolve_model(task, model_hint)
    api_key = _resolve_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    max_items = _get_max_items()
    timeout = _get_timeout()

    payload = dict(json_payload) if isinstance(json_payload, dict) else {}
    products = payload.get("products") if isinstance(payload.get("products"), list) else None

    warnings: List[str] = []
    text_parts: List[str] = []
    combined_json: Dict[str, Any] = {}
    result_map: Dict[str, Any] = {}
    prompt_versions: List[str] = []

    call_count = 0
    chunk_sizes: List[int] = []
    estimated_tokens = 0.0

    chosen_system_prompt = system_prompt.strip() if isinstance(system_prompt, str) and system_prompt.strip() else SYSTEM_PROMPT
    if task == "pesos":
        context = dict(payload)
        if "products" in context:
            products_list = context.pop("products")
        else:
            products_list = []
        summary = _summarise_products_for_weights(products_list or [])
        context["summary_stats"] = summary
        prompt = _build_prompt(prompt_text, context)
        response = _call_openai(model, prompt, api_key, timeout, chosen_system_prompt)
        call_count += 1
        chunk_sizes.append(len(products_list or []))
        content = response["content"]
        estimated_tokens += _estimate_tokens(prompt, content, response.get("usage"))
        text, data, chunk_warnings = _parse_model_response(content)
        warnings.extend(chunk_warnings)
        if text:
            text_parts.append(text)
        if data:
            combined_json = data
            pv = data.get("prompt_version")
            if isinstance(pv, str):
                prompt_versions.append(pv)
        ok = bool(data) and not chunk_warnings
    else:
        if not products:
            chunks = [None]
        else:
            if task in {"imputacion", "desire"}:
                chunk_size = max(1, min(max_items, 100))
            else:
                chunk_size = max(1, max_items)
            chunks = list(_chunk_sequence(products, chunk_size))
        if not chunks:
            chunks = [None]

        ok = True
        for chunk in chunks:
            context = dict(payload)
            if chunk is not None:
                context["products"] = chunk
                chunk_sizes.append(len(chunk))
            elif products:
                # this happens when we have products but chunking returned nothing
                chunk_sizes.append(0)
            else:
                chunk_sizes.append(0)
            prompt = _build_prompt(prompt_text, context)
            response = _call_openai(model, prompt, api_key, timeout, chosen_system_prompt)
            call_count += 1
            content = response["content"]
            estimated_tokens += _estimate_tokens(prompt, content, response.get("usage"))
            text, data, chunk_warnings = _parse_model_response(content)
            warnings.extend(chunk_warnings)
            if text:
                text_parts.append(text)
            if not data:
                ok = False
                continue
            pv = data.get("prompt_version")
            if isinstance(pv, str):
                prompt_versions.append(pv)

            if task in {"consulta", "tendencias"}:
                combined_json = _merge_chunk_data(combined_json, data)
            elif task in {"imputacion", "desire"}:
                mapping = _extract_mapping(data)
                if mapping:
                    result_map.update(mapping)
                else:
                    ok = False
                combined_json = {"prompt_version": pv} if pv else {}
            else:
                combined_json = data

            if chunk_warnings:
                ok = False

    if task in {"imputacion", "desire"}:
        if prompt_versions:
            combined_json["prompt_version"] = prompt_versions[-1]
        combined_json["results"] = result_map
        ok = ok and bool(result_map)

    if task in {"consulta", "tendencias"} and prompt_versions:
        combined_json["prompt_version"] = prompt_versions[-1]

    meta = {
        "calls": call_count or len(chunk_sizes),
        "chunks": len(chunk_sizes),
        "chunk_sizes": chunk_sizes,
        "estimated_tokens": int(round(estimated_tokens)),
        "model": model,
    }

    logger.info(
        "gpt_orchestrator %s",
        json.dumps(
            {
                "task": task,
                "model": model,
                "calls": meta["calls"],
                "chunks": meta["chunks"],
                "tokens": meta["estimated_tokens"],
            },
            ensure_ascii=False,
        ),
    )

    return {
        "ok": ok,
        "task": task,
        "model": model,
        "text": "\n\n".join(p for p in text_parts if p).strip() or None,
        "data": combined_json or None,
        "warnings": warnings,
        "meta": meta,
    }


def _resolve_model(task: _TaskName, model_hint: Optional[str]) -> str:
    letter, default_model = _TASK_MODEL_MAP[task]
    if model_hint:
        return model_hint
    env_model = os.environ.get(f"GPT_MODEL_{letter}")
    return env_model or default_model


def _resolve_api_key() -> Optional[str]:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    try:
        from product_research_app import config
    except Exception:  # pragma: no cover - fallback when config is unavailable
        return None
    return config.get_api_key()


def _get_max_items() -> int:
    try:
        from product_research_app import config
    except Exception:  # pragma: no cover - fallback when config import fails
        return _DEFAULT_MAX_ITEMS
    return config.get_env_max_items(_DEFAULT_MAX_ITEMS)


def _get_timeout() -> float:
    try:
        from product_research_app import config
    except Exception:  # pragma: no cover - fallback when config import fails
        return _DEFAULT_TIMEOUT
    return config.get_gpt_timeout_seconds(_DEFAULT_TIMEOUT)


_DEFAULT_MAX_ITEMS = 300
_DEFAULT_TIMEOUT = 60.0


def _chunk_sequence(seq: Sequence[Any], chunk_size: int) -> Iterable[List[Any]]:
    for idx in range(0, len(seq), chunk_size):
        yield list(seq[idx : idx + chunk_size])


def _build_prompt(prompt_text: str, context: Optional[Dict[str, Any]]) -> str:
    prompt = prompt_text.strip()
    if context:
        prompt += "\n\n### CONTEXTO JSON\n"
        prompt += json.dumps(context, ensure_ascii=False)
    prompt += (
        "\n\n### INSTRUCCIONES DE FORMATO\n"
        "Responde en español y finaliza siempre con un bloque ```json"  # noqa: B950
        "\n{...}\n``` que incluya la clave 'prompt_version'."
    )
    return prompt

def _call_openai(
    model: str,
    prompt: str,
    api_key: str,
    timeout: float,
    system_prompt: str,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    try:
        response = requests.post(
            CHAT_COMPLETIONS_URL,
            headers=headers,
            json=body,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:  # pragma: no cover - network errors
        raise RuntimeError(f"OpenAI request failed: {exc}") from exc

    choices = payload.get("choices")
    if not choices:  # pragma: no cover - defensive
        raise RuntimeError("OpenAI response missing choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):  # pragma: no cover - defensive
        raise RuntimeError("OpenAI response missing content")

    usage = None
    usage_payload = payload.get("usage")
    if isinstance(usage_payload, dict):
        usage_val = usage_payload.get("total_tokens")
        if isinstance(usage_val, (int, float)):
            usage = float(usage_val)

    return {"content": content, "usage": usage}


def _parse_model_response(content: str) -> Tuple[str, Optional[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    match = JSON_BLOCK_RE.search(content)
    data: Optional[Dict[str, Any]] = None
    if not match:
        warnings.append("Respuesta sin bloque JSON")
        text = content.strip()
        return text, None, warnings

    json_text = match.group(1)
    try:
        parsed = json.loads(json_text)
        if not isinstance(parsed, dict):
            warnings.append("Bloque JSON no es un objeto")
        else:
            if "prompt_version" not in parsed:
                warnings.append("JSON sin prompt_version")
            data = parsed
    except json.JSONDecodeError as exc:
        warnings.append(f"JSON inválido: {exc}")

    text = JSON_BLOCK_RE.sub("", content).strip()
    return text, data, warnings


def _merge_chunk_data(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    if not base:
        base = {}
    if not isinstance(incoming, dict):
        return base
    refs = incoming.get("refs")
    if isinstance(refs, list):
        base_refs = base.get("refs")
        if not isinstance(base_refs, list):
            base_refs = []
        base_refs = _merge_refs(base_refs, refs)
        base["refs"] = base_refs
    for key, value in incoming.items():
        if key == "refs":
            continue
        base[key] = value
    return base


def _merge_refs(existing: List[Dict[str, Any]], new_refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    merged: List[Dict[str, Any]] = []
    for ref in existing + new_refs:
        if not isinstance(ref, dict):
            continue
        ref_id = ref.get("id")
        ref_cat = ref.get("category") or ref.get("categoria")
        key = (str(ref_id) if ref_id is not None else None, str(ref_cat) if ref_cat is not None else None)
        if key in seen:
            continue
        seen.add(key)
        merged.append(ref)
    return merged


def _extract_mapping(data: Dict[str, Any]) -> Dict[str, Any]:
    mapping: Dict[str, Any] = {}
    if not isinstance(data, dict):
        return mapping
    if isinstance(data.get("results"), dict):
        for key, value in data["results"].items():
            mapping[str(key)] = value
        return mapping
    if isinstance(data.get("items"), list):
        for entry in data["items"]:
            if not isinstance(entry, dict):
                continue
            pid = entry.get("id") or entry.get("product_id") or entry.get("asin")
            if pid is None:
                continue
            copy_entry = dict(entry)
            copy_entry.pop("id", None)
            copy_entry.pop("product_id", None)
            mapping[str(pid)] = copy_entry
        return mapping
    for key, value in data.items():
        if key == "prompt_version":
            continue
        mapping[str(key)] = value
    return mapping


def _summarise_products_for_weights(products: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    metric_values: Dict[str, List[Tuple[str, float]]] = {}
    for item in products:
        if not isinstance(item, dict):
            continue
        pid = _extract_product_id(item)
        for metric, value in _gather_numeric_metrics(item).items():
            metric_values.setdefault(metric, [])
            metric_values[metric].append((pid, value))

    summary: Dict[str, Any] = {"metrics": {}}
    for metric, entries in metric_values.items():
        values = [val for _, val in entries]
        if not values:
            continue
        stats = _metric_summary(entries, values)
        summary["metrics"][metric] = stats
    summary["total_products"] = len(products)
    return summary


def _extract_product_id(item: Dict[str, Any]) -> str:
    for key in ("id", "product_id", "asin", "sku", "code", "name"):
        val = item.get(key)
        if val not in (None, ""):
            return str(val)
    return ""


def _gather_numeric_metrics(item: Dict[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, value in item.items():
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    nested = item.get("metrics")
    if isinstance(nested, dict):
        for key, value in nested.items():
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
    return metrics


def _metric_summary(entries: List[Tuple[str, float]], values: List[float]) -> Dict[str, Any]:
    values_sorted = sorted(values)
    min_val = float(values_sorted[0])
    max_val = float(values_sorted[-1])
    mean_val = float(sum(values_sorted) / len(values_sorted))
    p25 = float(_percentile(values_sorted, 0.25))
    p50 = float(_percentile(values_sorted, 0.50))
    p75 = float(_percentile(values_sorted, 0.75))
    std_val = float(_stddev(values_sorted))

    top_entries = [
        {"id": str(pid), "value": val}
        for pid, val in sorted(entries, key=lambda x: x[1], reverse=True)
        if pid
    ][:10]
    bottom_entries = [
        {"id": str(pid), "value": val}
        for pid, val in sorted(entries, key=lambda x: x[1])
        if pid
    ][:10]

    return {
        "count": len(values_sorted),
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "std": std_val,
        "top": top_entries,
        "bottom": bottom_entries,
        "top_ids": [entry["id"] for entry in top_entries],
        "bottom_ids": [entry["id"] for entry in bottom_entries],
    }


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return float(values[0])
    idx = (len(values) - 1) * pct
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return float(values[int(idx)])
    lower_val = values[lower]
    upper_val = values[upper]
    return float(lower_val + (upper_val - lower_val) * (idx - lower))


def _stddev(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean_val = sum(values) / n
    variance = sum((val - mean_val) ** 2 for val in values) / (n - 1)
    return variance ** 0.5


def _estimate_tokens(prompt: str, content: str, usage: Optional[float]) -> float:
    if usage is not None:
        return float(usage)
    approx = (len(prompt) + len(content)) / 4.0
    return max(1.0, approx)
