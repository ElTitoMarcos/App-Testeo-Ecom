"""Utilities to compute aggregate statistics for imported products.

This module exposes helpers that summarise the current dataset (or a
restricted subset of products) so higher level services can build prompts
for the AI orchestrator without iterating over every product on the Python
side.  The output focuses on the metrics relevant for the automatic Winner
Score calibration pipeline: price, rating, sales proxies, desire and
competition labels as well as product "oldness" and awareness levels.

All functions operate on a SQLite connection and avoid mutating the
database.  Returned structures are plain dictionaries ready to be serialised
as JSON.
"""

from __future__ import annotations

import json
from collections import Counter
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from .. import database
from ..utils.db import row_to_dict
from . import winner_score


NumericMetric = Mapping[str, Any]


def _load_extra(payload: Mapping[str, Any]) -> MutableMapping[str, Any]:
    raw = payload.get("extra")
    if isinstance(raw, dict):
        return dict(raw)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except Exception:
        return None


def _normalise_label(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value).strip().lower() or "unknown"


def _percentiles(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    if len(ordered) == 1:
        v = ordered[0]
        return {"p05": v, "p25": v, "p50": v, "p75": v, "p95": v}

    def pick(q: float) -> float:
        idx = int(round((len(ordered) - 1) * q))
        idx = max(0, min(len(ordered) - 1, idx))
        return ordered[idx]

    return {
        "p05": pick(0.05),
        "p25": pick(0.25),
        "p50": pick(0.50),
        "p75": pick(0.75),
        "p95": pick(0.95),
    }


def _numeric_summary(values: Iterable[Optional[float]]) -> Dict[str, Any]:
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return {"count": 0}
    stats = {
        "count": len(cleaned),
        "min": min(cleaned),
        "max": max(cleaned),
        "mean": mean(cleaned),
    }
    stats.update(_percentiles(cleaned))
    return stats


def _categorical_summary(values: Iterable[str]) -> Dict[str, Any]:
    counter = Counter(_normalise_label(v) for v in values if v is not None)
    total = sum(counter.values())
    data = dict(counter)
    data["count"] = total
    return data


def _extract_metrics(product: Mapping[str, Any]) -> Dict[str, Any]:
    data = row_to_dict(product)
    extra = _load_extra(data)
    metrics: Dict[str, Any] = {}
    metrics["price"] = _to_float(data.get("price") or extra.get("price"))
    metrics["rating"] = _to_float(data.get("rating") or extra.get("rating"))
    metrics["units_sold"] = _to_float(
        extra.get("units_sold")
        or extra.get("orders")
        or extra.get("sales")
    )
    metrics["revenue"] = _to_float(extra.get("revenue") or extra.get("gmv"))

    desire = data.get("desire") or data.get("desire_magnitude") or extra.get("desire")
    if desire is None:
        desire = extra.get("magnitud_deseo")
    metrics["desire"] = _normalise_label(desire)

    competition = data.get("competition_level") or extra.get("competition_level")
    if competition is None:
        competition = extra.get("saturacion_mercado")
    metrics["competition"] = _normalise_label(competition)

    awareness = (
        data.get("awareness_level")
        or extra.get("awareness_level")
        or extra.get("nivel_consciencia")
    )
    metrics["awareness"] = _normalise_label(awareness)

    merged = dict(data)
    merged.update(extra)
    metrics["oldness_days"] = winner_score._oldness_days(merged)  # type: ignore[attr-defined]
    metrics["winner_score"] = _to_float(data.get("winner_score"))
    metrics["conversion_rate"] = _to_float(extra.get("conversion_rate"))
    metrics["profit_margin"] = _to_float(extra.get("profit_margin"))
    return metrics


def compute_dataset_aggregates(
    conn,
    *,
    scope_ids: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    """Return aggregated statistics for the given set of products."""

    rows: List[Mapping[str, Any]]
    if scope_ids:
        unique_ids = [int(r) for r in dict.fromkeys(scope_ids) if str(r).strip()]
        if not unique_ids:
            rows = []
        else:
            rows = database.get_products_by_ids(conn, unique_ids)
    else:
        rows = database.list_products(conn)

    metrics = [_extract_metrics(row) for row in rows]
    summary = {
        "total_products": len(metrics),
        "numeric": {
            "price": _numeric_summary(m.get("price") for m in metrics),
            "rating": _numeric_summary(m.get("rating") for m in metrics),
            "units_sold": _numeric_summary(m.get("units_sold") for m in metrics),
            "revenue": _numeric_summary(m.get("revenue") for m in metrics),
            "oldness_days": _numeric_summary(m.get("oldness_days") for m in metrics),
            "winner_score": _numeric_summary(m.get("winner_score") for m in metrics),
            "conversion_rate": _numeric_summary(m.get("conversion_rate") for m in metrics),
            "profit_margin": _numeric_summary(m.get("profit_margin") for m in metrics),
        },
        "categorical": {
            "desire": _categorical_summary(m.get("desire") for m in metrics),
            "competition": _categorical_summary(m.get("competition") for m in metrics),
            "awareness": _categorical_summary(m.get("awareness") for m in metrics),
        },
    }
    if scope_ids:
        summary["scope_ids"] = list(dict.fromkeys(scope_ids))
    return summary


__all__ = ["compute_dataset_aggregates"]

