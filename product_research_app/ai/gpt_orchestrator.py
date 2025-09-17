from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import requests

try:  # pragma: no cover - defensive when services package is unavailable
    from product_research_app.services.aggregates import (
        build_weighting_aggregates,
        sample_product_titles,
    )
except Exception:  # pragma: no cover
    build_weighting_aggregates = None
    sample_product_titles = None

logger = logging.getLogger(__name__)

CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
SYSTEM_PROMPT = (
    "Todos producen texto legible y terminan con un bloque JSON dentro de ```json DATA_JSON``` "
    "con la clave obligatoria 'prompt_version'. Tu backend separará la última fence, "
    "así que respeta ese formato. Regla global: ignora filtros de UI y usa exactamente "
    "el conjunto recibido (si hay group_id, céntrate en él; si no, usa todos los productos)."
)
JSON_BLOCK_RE = re.compile(
    r"```json(?:\s+DATA_JSON)?\s*(\{.*?\})\s*```",
    re.DOTALL,
)

_TaskName = Literal["consulta", "pesos", "tendencias", "imputacion", "desire"]

_TASK_MODEL_MAP: Dict[_TaskName, Tuple[str, str]] = {
    "consulta": ("A", "gpt-4o-mini"),
    "pesos": ("B", "gpt-4o"),
    "tendencias": ("C", "gpt-4o-mini"),
    "imputacion": ("D", "gpt-4o-mini"),
    "desire": ("E", "gpt-4o-mini"),
}


def _legacy_run_task(
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
        context, original_count = _prepare_weights_context(payload)
        prompt = _build_prompt(prompt_text, context)
        response = _call_openai(model, prompt, api_key, timeout, chosen_system_prompt)
        call_count += 1
        chunk_sizes.append(original_count)
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

def _prepare_weights_context(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    context = dict(payload)
    raw_products = payload.get("products")
    product_list: List[Dict[str, Any]]
    if isinstance(raw_products, list):
        product_list = [item for item in raw_products if isinstance(item, dict)]
    else:
        product_list = []
    product_count = len(product_list)

    aggregates_payload = None
    sample_titles: List[str] = []

    if isinstance(raw_products, dict):
        aggregates_payload = _normalise_aggregates(raw_products)
        sample_titles = _normalise_titles(raw_products.get("sample_titles"))
    elif product_list:
        aggregates_payload = _build_weighting_aggregates_from_list(product_list)
        sample_titles = _derive_sample_titles(product_list)

    context.pop("products", None)

    if aggregates_payload is None:
        candidate = context.pop("aggregates", None)
        aggregates_payload = _normalise_aggregates(candidate)

    if aggregates_payload is None and product_list:
        aggregates_payload = _build_weighting_aggregates_from_list(product_list)
    if aggregates_payload is None:
        aggregates_payload = {"metrics": {}, "total_products": product_count}

    if not sample_titles:
        sample_titles = _normalise_titles(context.get("sample_titles"))
        if not sample_titles:
            sample_titles = _derive_sample_titles(product_list)

    if sample_titles:
        context["sample_titles"] = sample_titles
    else:
        context.pop("sample_titles", None)

    context["aggregates"] = aggregates_payload

    if not product_count and isinstance(aggregates_payload, dict):
        total_hint = aggregates_payload.get("total_products") or aggregates_payload.get("total_items")
        if isinstance(total_hint, (int, float)):
            product_count = int(total_hint)

    return context, product_count


def _normalise_aggregates(candidate: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(candidate, dict):
        return None
    if isinstance(candidate.get("aggregates"), dict):
        return dict(candidate["aggregates"])
    if isinstance(candidate.get("summary_stats"), dict):
        return dict(candidate["summary_stats"])
    metrics = candidate.get("metrics")
    if isinstance(metrics, dict):
        result = dict(candidate)
        result["metrics"] = dict(metrics)
        return result
    return None


def _normalise_titles(raw_titles: Any, limit: int = 20) -> List[str]:
    if not isinstance(raw_titles, Iterable) or isinstance(raw_titles, (str, bytes)):
        return []
    cleaned: List[str] = []
    seen = set()
    for title in raw_titles:
        if not isinstance(title, str):
            continue
        trimmed = title.strip()
        if not trimmed or trimmed in seen:
            continue
        cleaned.append(trimmed)
        seen.add(trimmed)
        if len(cleaned) >= limit:
            break
    return cleaned


def _derive_sample_titles(products: Sequence[Dict[str, Any]], limit: int = 20) -> List[str]:
    if not products:
        return []
    if sample_product_titles is not None:
        return sample_product_titles(list(products), limit=limit)
    return _legacy_sample_titles(products, limit=limit)


def _legacy_sample_titles(products: Sequence[Dict[str, Any]], limit: int = 20) -> List[str]:
    if limit <= 0:
        return []
    titles: List[str] = []
    seen = set()
    for product in products:
        if not isinstance(product, dict):
            continue
        title = product.get("title")
        if not isinstance(title, str):
            continue
        trimmed = title.strip()
        if not trimmed or trimmed in seen:
            continue
        titles.append(trimmed)
        seen.add(trimmed)

    if len(titles) <= limit:
        return titles

    if limit == 1:
        return titles[:1]

    span = len(titles) - 1
    indices = []
    for i in range(limit):
        idx = round(i * span / (limit - 1))
        if idx not in indices:
            indices.append(idx)
    selected = [titles[idx] for idx in indices]
    for title in titles:
        if len(selected) >= limit:
            break
        if title not in selected:
            selected.append(title)
    return selected[:limit]


def _build_weighting_aggregates_from_list(products: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if build_weighting_aggregates is not None:
        return build_weighting_aggregates(list(products))
    return _summarise_products_for_weights(products)

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
        "Responde en español y finaliza siempre con un bloque DATA_JSON usando la sintaxis"
        " ```json DATA_JSON\n{...}\n``` e incluye la clave obligatoria 'prompt_version'."
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
    matches = list(JSON_BLOCK_RE.finditer(content))
    data: Optional[Dict[str, Any]] = None
    if not matches:
        warnings.append("Respuesta sin bloque JSON")
        text = content.strip()
        return text, None, warnings

    match = matches[-1]
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

    text = (content[: match.start()] + content[match.end() :]).strip()
    return text, data, warnings


def _merge_chunk_data(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(base, dict):
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
    elif isinstance(refs, dict):
        existing_refs = base.get("refs") if isinstance(base.get("refs"), dict) else {}
        base["refs"] = _merge_refs_dict(existing_refs, refs)

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

def _merge_refs_dict(existing: Dict[str, Any], new_refs: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if isinstance(existing, dict):
        for key, value in existing.items():
            if isinstance(value, list):
                result[key] = list(value)
            else:
                result[key] = value

    for key, value in new_refs.items():
        if key == "product_ids" and isinstance(value, list):
            current = result.get(key, []) if isinstance(result.get(key), list) else []
            merged_ids: List[str] = []
            seen: set[str] = set()
            for source in (current, value):
                for item in source:
                    if item in (None, ""):
                        continue
                    string_id = str(item)
                    if string_id in seen:
                        continue
                    seen.add(string_id)
                    merged_ids.append(string_id)
            result[key] = merged_ids
        elif isinstance(value, list):
            result[key] = list(value)
        else:
            result[key] = value

    return result
  
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


class ChatCompletionError(RuntimeError):
    """Error raised when the JSON-only chat orchestration fails."""

    def __init__(self, message: str, *, status_code: Optional[int] = None, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


_JSON_RESPONSE_FORMAT = {"type": "json_object"}
_DESIRE_SYSTEM_PROMPT = (
    "Resume deseo para dropshipping WW. SOLO JSON:\n"
    '{"items":[{"id":"...","normalized_text":["l1","l2","l3?"],"class":"alto|medio|bajo","magnitude":0-100,"keywords":["k1","k2"]}]}'
)
_IMPUTACION_SYSTEM_PROMPT = (
    "Imputa métricas faltantes SOLO JSON: {\"items\":[{\"id\":\"...\",\"review_count\":int|null,\"image_count\":int|null}]}"
)
_WEIGHTS_SYSTEM_PROMPT = (
    "Ajusta pesos ganador SOLO JSON: {\"weights\":{...},\"order\":[...]}. Pesos en 0-100, order prioriza primero."
)

_DEFAULT_SIMPLE_TIMEOUT = 20
_DEFAULT_SIMPLE_RETRY = 1


def _get_simple_timeout_retry() -> tuple[int, int]:
    try:
        from product_research_app import config  # type: ignore circular import

        settings = config.get_ai_auto_settings()
    except Exception:
        return _DEFAULT_SIMPLE_TIMEOUT, _DEFAULT_SIMPLE_RETRY
    timeout = int(settings.get("GPT_TIMEOUT", _DEFAULT_SIMPLE_TIMEOUT) or _DEFAULT_SIMPLE_TIMEOUT)
    retry = int(settings.get("GPT_MAX_RETRY", _DEFAULT_SIMPLE_RETRY) or _DEFAULT_SIMPLE_RETRY)
    timeout = max(1, timeout)
    retry = max(0, retry)
    return timeout, retry


def _parse_retry_after(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(int(str(value)))
        except Exception:
            return None


def _run_chat_once(
    system: str,
    user: str,
    *,
    model: str,
    timeout: int,
    temperature: float = 0.0,
) -> str:
    api_key = _resolve_api_key()
    if not api_key:
        raise ChatCompletionError("OPENAI_API_KEY is not configured")
    system_prompt = system.strip() or "Responde únicamente con JSON válido."
    user_message = user if isinstance(user, str) else json.dumps(user, ensure_ascii=False)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": max(0.0, float(temperature)),
        "response_format": _JSON_RESPONSE_FORMAT,
    }
    try:
        response = requests.post(
            CHAT_COMPLETIONS_URL,
            headers=headers,
            json=payload,
            timeout=max(1, int(timeout)),
        )
    except requests.RequestException as exc:  # pragma: no cover - network failure
        raise ChatCompletionError(f"OpenAI request error: {exc}") from exc

    retry_after = _parse_retry_after(response.headers.get("Retry-After"))
    if response.status_code != 200:
        message = response.text
        try:
            err = response.json()
            message = err.get("error", {}).get("message", message)
        except Exception:
            pass
        raise ChatCompletionError(
            f"OpenAI API error {response.status_code}: {message}",
            status_code=response.status_code,
            retry_after=retry_after,
        )

    try:
        payload_json = response.json()
    except Exception as exc:  # pragma: no cover - unexpected payload
        raise ChatCompletionError("OpenAI response is not valid JSON") from exc

    choices = payload_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ChatCompletionError("OpenAI response missing choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ChatCompletionError("OpenAI response missing message")
    content = message.get("content")
    if not isinstance(content, str):
        raise ChatCompletionError("OpenAI response missing content")
    return content.strip()


def _call_chat_with_retry(
    system: str,
    user: str,
    *,
    model: str,
    timeout: int,
    max_retry: int,
    temperature: float = 0.0,
) -> str:
    attempts = 0
    while True:
        try:
            return _run_chat_once(system, user, model=model, timeout=timeout, temperature=temperature)
        except ChatCompletionError as exc:
            status = exc.status_code or 0
            should_retry = status == 429 or 500 <= status < 600
            if attempts >= max_retry or not should_retry:
                raise
            delay = exc.retry_after
            if delay is None:
                delay = min(2 ** attempts, 10)
            else:
                delay = max(0.0, float(delay))
            if delay:
                time.sleep(min(delay, 30.0))
            attempts += 1


def _ensure_list_of_str(values: object) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for item in values:
        if item in (None, ""):
            continue
        out.append(str(item).strip())
    return out
def _ensure_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(round(float(value)))
    except Exception:
        return None


class DesireBatchResult(dict):
    """Mapping of product id -> normalized desire output with raw payload."""

    def __init__(self, data: Mapping[str, Dict[str, Any]], raw_response: str) -> None:
        super().__init__(data)
        self.raw_response = raw_response


def _extract_desire_json_block(raw_text: str) -> Optional[str]:
    text = (raw_text or "").strip()
    if not text:
        return None
    match = JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1)
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : index + 1]
    return None


def _normalize_desire_lines(entry: Mapping[str, Any]) -> List[str]:
    lines = _ensure_list_of_str(entry.get("normalized_text"))
    if not lines:
        lines = _ensure_list_of_str(entry.get("lines"))
    if not lines:
        text_value = entry.get("text")
        if isinstance(text_value, str) and text_value.strip():
            lines = [segment.strip() for segment in text_value.splitlines() if segment.strip()]
    normalized: List[str] = []
    for line in lines:
        if not line:
            continue
        normalized.append(line[:90])
        if len(normalized) >= 3:
            break
    if len(normalized) == 1:
        splits = [seg.strip() for seg in normalized[0].replace(";", ".").split(".") if seg.strip()]
        if len(splits) >= 2:
            normalized = [segment[:90] for segment in splits[:3]]
        elif normalized[0]:
            normalized.append(normalized[0])
    return normalized


def _normalize_desire_label(
    raw_label: Any,
    raw_keywords: Any,
    lines: Sequence[str],
    magnitude_hint: Optional[int] = None,
) -> str:
    valid = {"alto", "medio", "bajo"}
    if isinstance(raw_label, str):
        cleaned = raw_label.strip().lower()
        if cleaned in valid:
            return cleaned
    if magnitude_hint is not None:
        if magnitude_hint >= 70:
            return "alto"
        if magnitude_hint >= 45:
            return "medio"
        return "bajo"
    keywords = _ensure_list_of_str(raw_keywords)
    keyword_count = len(keywords)
    blob = " ".join(lines).lower()
    if any(token in blob for token in ("muy alta", "alta demanda", "gran demanda", "hot", "tendencia", "trend")):
        return "alto"
    if any(token in blob for token in ("baja demanda", "poca demanda", "low demand", "sin demanda", "fría", "fria")):
        return "bajo"
    if keyword_count >= 4 or len(blob) >= 200:
        return "alto"
    if keyword_count <= 1 and len(blob) < 80:
        return "bajo"
    if keyword_count == 3:
        return "alto"
    if keyword_count == 2:
        return "medio"
    return "medio"


def _normalize_desire_magnitude(
    raw_value: Any,
    label: str,
    magnitude_hint: Optional[int] = None,
) -> int:
    magnitude: Optional[int] = None
    candidates: List[Any] = []
    if raw_value not in (None, ""):
        candidates.append(raw_value)
    if magnitude_hint is not None:
        candidates.append(magnitude_hint)
    for candidate in candidates:
        try:
            magnitude = int(round(float(candidate)))
            break
        except Exception:
            continue
    if magnitude is None:
        defaults = {"alto": 80, "medio": 50, "bajo": 25}
        magnitude = defaults.get(label, 50)
    return max(0, min(100, int(magnitude)))


def run_desire_batch(
    items: List[Dict[str, Any]],
    *,
    timeout: Optional[int] = None,
    max_retry: Optional[int] = None,
) -> DesireBatchResult:
    if not items:
        return DesireBatchResult({}, "")
    default_timeout, default_retry = _get_simple_timeout_retry()
    timeout = timeout or default_timeout
    max_retry = max_retry if max_retry is not None else default_retry
    payload = {"items": items}
    user = "### DATA\n" + json.dumps(payload, ensure_ascii=False)
    response = _call_chat_with_retry(
        _DESIRE_SYSTEM_PROMPT,
        user,
        model="gpt-4o-mini",
        timeout=timeout,
        max_retry=max_retry,
        temperature=0.0,
    )
    json_block = _extract_desire_json_block(response)
    if not json_block:
        preview = (response or "")[:200]
        logger.warning("desire_batch_invalid_payload preview=%s", preview)
        return DesireBatchResult({}, response)
    try:
        data = json.loads(json_block)
    except json.JSONDecodeError as exc:
        preview = json_block[:200]
        logger.warning("desire_batch_invalid_json error=%s preview=%s", exc, preview)
        return DesireBatchResult({}, response)
    entries = data.get("items")
    if not isinstance(entries, list):
        preview = json_block[:200]
        logger.warning("desire_batch_invalid_items preview=%s", preview)
        return DesireBatchResult({}, response)
    result: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        pid = entry.get("id")
        if pid in (None, ""):
            continue
        key = str(pid)
        lines = _normalize_desire_lines(entry)
        if not lines:
            continue
        magnitude_hint = _ensure_int(entry.get("magnitude"))
        label = _normalize_desire_label(
            entry.get("class"), entry.get("keywords"), lines, magnitude_hint
        )
        magnitude = _normalize_desire_magnitude(entry.get("magnitude"), label, magnitude_hint)
        result[key] = {
            "text": "\n".join(lines[:3]),
            "label": label,
            "magnitude": magnitude,
        }
    return DesireBatchResult(result, response)


def run_imputacion_batch(
    items: List[Dict[str, Any]],
    *,
    timeout: Optional[int] = None,
    max_retry: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    if not items:
        return {}
    default_timeout, default_retry = _get_simple_timeout_retry()
    timeout = timeout or default_timeout
    max_retry = max_retry if max_retry is not None else default_retry
    payload = {"items": items}
    user = "### DATA\n" + json.dumps(payload, ensure_ascii=False)
    response = _call_chat_with_retry(
        _IMPUTACION_SYSTEM_PROMPT,
        user,
        model="gpt-4o-mini",
        timeout=timeout,
        max_retry=max_retry,
        temperature=0.0,
    )
    try:
        data = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ChatCompletionError(f"Invalid JSON from imputacion batch: {exc}") from exc
    entries = data.get("items")
    if not isinstance(entries, list):
        raise ChatCompletionError("Imputación batch response missing items list")
    result: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        pid = entry.get("id")
        if pid in (None, ""):
            continue
        key = str(pid)
        review_count = _ensure_int(entry.get("review_count"))
        image_count = _ensure_int(entry.get("image_count"))
        result[key] = {}
        if review_count is not None:
            result[key]["review_count"] = review_count
        if image_count is not None:
            result[key]["image_count"] = image_count
    return result


def run_weights_once(
    aggregates: Dict[str, Any],
    *,
    timeout: Optional[int] = None,
    max_retry: Optional[int] = None,
) -> Dict[str, Any]:
    default_timeout, default_retry = _get_simple_timeout_retry()
    timeout = timeout or default_timeout
    max_retry = max_retry if max_retry is not None else default_retry
    payload = {"aggregates": aggregates}
    user = "### DATA\n" + json.dumps(payload, ensure_ascii=False)
    response = _call_chat_with_retry(
        _WEIGHTS_SYSTEM_PROMPT,
        user,
        model="gpt-4o",
        timeout=timeout,
        max_retry=max_retry,
        temperature=0.0,
    )
    try:
        data = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ChatCompletionError(f"Invalid JSON from weights request: {exc}") from exc
    weights = data.get("weights")
    order = data.get("order")
    if not isinstance(weights, dict):
        raise ChatCompletionError("Weights response missing 'weights'")
    if not isinstance(order, list):
        raise ChatCompletionError("Weights response missing 'order'")
    normalized = {str(k): _ensure_int(v) for k, v in weights.items()}
    normalized = {k: (v if v is not None else 0) for k, v in normalized.items()}
    order_list = [str(item) for item in order if item not in (None, "")]
    return {"weights": normalized, "order": order_list}


def run_task(*args, **kwargs):  # type: ignore[override]
    """Dispatcher supporting both legacy and JSON-only execution styles."""

    if args:
        first = args[0]
    else:
        first = None

    if (
        first in _TASK_MODEL_MAP
        or "prompt_text" in kwargs
        or "json_payload" in kwargs
        or "system_prompt" in kwargs
    ):
        return _legacy_run_task(*args, **kwargs)

    if {"system", "user", "model", "timeout"} <= set(kwargs.keys()):
        system = kwargs["system"]
        user = kwargs["user"]
        model = kwargs["model"]
        timeout = kwargs["timeout"]
        max_retry = kwargs.get("max_retry")
        temperature = kwargs.get("temperature", 0.0)
        default_timeout, default_retry = _get_simple_timeout_retry()
        timeout_val = int(timeout or default_timeout)
        retry_val = max_retry if max_retry is not None else default_retry
        return _call_chat_with_retry(
            str(system),
            str(user),
            model=str(model),
            timeout=max(1, timeout_val),
            max_retry=max(0, int(retry_val)),
            temperature=float(temperature or 0.0),
        )

    if len(args) >= 4 and "system" not in kwargs and "prompt_text" not in kwargs:
        system, user, model, timeout = args[:4]
        max_retry = kwargs.get("max_retry")
        temperature = kwargs.get("temperature", 0.0)
        default_timeout, default_retry = _get_simple_timeout_retry()
        timeout_val = int(timeout or default_timeout)
        retry_val = max_retry if max_retry is not None else default_retry
        return _call_chat_with_retry(
            str(system),
            str(user),
            model=str(model),
            timeout=max(1, timeout_val),
            max_retry=max(0, int(retry_val)),
            temperature=float(temperature or 0.0),
        )

    raise TypeError("Unsupported arguments for run_task")
