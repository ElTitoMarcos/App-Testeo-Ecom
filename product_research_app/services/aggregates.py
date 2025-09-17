"""Utilities for building aggregate statistics over product datasets."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple


MetricEntry = Tuple[str, float]


def build_weighting_aggregates(products: List[dict]) -> Dict[str, dict]:
    """Return aggregate statistics for each numeric metric in ``products``.

    The result maps every metric name to a dictionary containing descriptive
    statistics plus identifiers of the top and bottom performers. Only numeric
    values (``int`` or ``float``) are considered for aggregation and coverage
    ratios are expressed as values between 0.0 and 1.0.
    """

    metrics: Dict[str, List[MetricEntry]] = {}
    total_products = len(products)

    for product in products:
        if not isinstance(product, dict):
            continue
        product_id = _coerce_id(product.get("id"))
        for key, value in product.items():
            if _is_numeric(value):
                metrics.setdefault(key, []).append((product_id, float(value)))
        nested = product.get("metrics")
        if isinstance(nested, dict):
            for key, value in nested.items():
                if _is_numeric(value):
                    metrics.setdefault(key, []).append((product_id, float(value)))

    aggregates: Dict[str, dict] = {}
    for metric, entries in metrics.items():
        values = [value for _, value in entries]
        if not values:
            continue
        aggregates[metric] = _summarise_metric(entries, values, total_products)

    return {"metrics": aggregates, "total_products": total_products}


def sample_product_titles(products: List[dict], limit: int = 20) -> List[str]:
    """Return up to ``limit`` unique product titles distributed across the list."""

    if limit <= 0:
        return []

    unique_titles: List[str] = []
    seen = set()
    for product in products:
        if not isinstance(product, dict):
            continue
        title = product.get("title")
        if not isinstance(title, str):
            continue
        cleaned = title.strip()
        if not cleaned or cleaned in seen:
            continue
        unique_titles.append(cleaned)
        seen.add(cleaned)

    if len(unique_titles) <= limit:
        return unique_titles

    if limit == 1:
        return unique_titles[:1]

    span = len(unique_titles) - 1
    step_positions = [round(i * span / (limit - 1)) for i in range(limit)]

    selected: List[str] = []
    used_positions = set()
    for pos in step_positions:
        if pos in used_positions:
            continue
        selected.append(unique_titles[pos])
        used_positions.add(pos)

    if len(selected) < limit:
        for title in unique_titles:
            if title in selected:
                continue
            selected.append(title)
            if len(selected) >= limit:
                break

    return selected[:limit]


def _coerce_id(value: object) -> str:
    if value in (None, ""):
        return ""
    return str(value)


def _is_numeric(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _summarise_metric(entries: Sequence[MetricEntry], values: Sequence[float], total: int) -> dict:
    values_sorted = sorted(values)
    count = len(values_sorted)
    coverage = (count / total) if total else 0.0

    min_val = float(values_sorted[0])
    max_val = float(values_sorted[-1])
    mean_val = float(sum(values_sorted) / count)
    p25 = float(_percentile(values_sorted, 0.25))
    median = float(_percentile(values_sorted, 0.50))
    p75 = float(_percentile(values_sorted, 0.75))
    std_val = float(_stddev(values_sorted))

    top_ids = _rank_ids(entries, reverse=True)
    bottom_ids = _rank_ids(entries, reverse=False)

    return {
        "min": min_val,
        "p25": p25,
        "median": median,
        "p75": p75,
        "max": max_val,
        "mean": mean_val,
        "std": std_val,
        "top_ids": top_ids,
        "bottom_ids": bottom_ids,
        "coverage": coverage,
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
    return float(math.sqrt(variance))


def _rank_ids(entries: Iterable[MetricEntry], *, reverse: bool) -> List[str]:
    ranked: List[str] = []
    seen = set()
    for product_id, _ in sorted(entries, key=lambda item: item[1], reverse=reverse):
        if not product_id:
            continue
        if product_id in seen:
            continue
        ranked.append(product_id)
        seen.add(product_id)
        if len(ranked) >= 10:
            break
    return ranked
