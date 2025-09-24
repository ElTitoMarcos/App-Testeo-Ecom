"""Winner weight estimation from aggregate dataset statistics.

This module implements the behaviour described in prompt "B" of the agent
workflow: given aggregate statistics for the available metrics it must emit a
set of Winner Score weights (0-100 scale, independent) together with ordering
and diagnostic notes.  When the aggregates block is missing or invalid the
specification requires returning zero weights and explaining the limitation.

The heuristics implemented here are deterministic and do not rely on any
external model.  They roughly follow the qualitative guidance from the prompt:

* Traction metrics (revenue, units_sold, rating) receive the highest base
  weights.
* Desire and awareness have medium weight, modulated by how centred or extreme
  their distribution looks.
* Competition and price are moderated so they do not dominate the score.
* Oldness rewards recency (lower values) but is capped by the coverage of the
  signal.
* For every metric with coverage below 0.35 the resulting weight is capped at
  15 and a diagnostic note is emitted, as mandated by the spec.

The output format mirrors exactly what the autoprompt expects so the calling
code (either tests or future pipelines) can persist the structure without
additional transformations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from . import winner_score as winner_calc

PROMPT_VERSION = "B.v2"
ALLOWED_FIELDS = list(winner_calc.ALLOWED_FIELDS)
BASE_ORDER = ALLOWED_FIELDS

# Base emphasis for each metric before adjustments.  These numbers are kept
# intentionally ordered by relative importance rather than summing to 100.
BASE_WEIGHTS: Dict[str, float] = {
    "revenue": 78.0,
    "units_sold": 74.0,
    "rating": 62.0,
    "desire": 60.0,
    "awareness": 54.0,
    "competition": 46.0,
    "oldness": 42.0,
    "price": 34.0,
}

# Preferred distribution tendencies per metric.
PREFERENCES = {
    "price": "mid",  # avoid extremes; balance affordability vs. perceived value
    "rating": "high",
    "units_sold": "high",
    "revenue": "high",
    "desire": "high",
    "competition": "low",
    "oldness": "low",  # newer listings should be favoured
    "awareness": "mid",  # prefer mid stages (problem/solution aware)
}

COVERAGE_THRESHOLD = 0.35
LOW_COVERAGE_MAX_WEIGHT = 15


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _extract_metrics(payload: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return the metrics block from an aggregates payload."""

    if not isinstance(payload, dict):
        return {}

    for key in ("metrics", "aggregates", "data", "stats"):
        block = payload.get(key)
        if isinstance(block, dict):
            filtered = {k: v for k, v in block.items() if isinstance(v, dict)}
            if filtered:
                return filtered

    # Fallback: assume the metrics are stored at the top level.
    filtered = {
        k: v
        for k, v in payload.items()
        if isinstance(v, dict)
        and {"mean", "min", "max", "coverage", "p50"}.intersection(v.keys())
    }
    return filtered


def _distribution_position(stats: Dict[str, Any]) -> Optional[float]:
    min_v = _to_float(stats.get("min"))
    max_v = _to_float(stats.get("max"))
    if min_v is None or max_v is None or max_v <= min_v:
        return None
    center = _to_float(stats.get("p50"))
    if center is None:
        center = _to_float(stats.get("median"))
    if center is None:
        center = _to_float(stats.get("mean"))
    if center is None:
        center = min_v + (max_v - min_v) / 2.0
    pos = (center - min_v) / (max_v - min_v)
    return _clamp(pos, 0.0, 1.0)


def _spread_ratio(stats: Dict[str, Any]) -> float:
    q1 = _to_float(stats.get("p25"))
    q3 = _to_float(stats.get("p75"))
    min_v = _to_float(stats.get("min"))
    max_v = _to_float(stats.get("max"))
    if (
        q1 is not None
        and q3 is not None
        and min_v is not None
        and max_v is not None
        and max_v > min_v
        and q3 >= q1
    ):
        return _clamp((q3 - q1) / (max_v - min_v + 1e-9), 0.0, 1.0)
    std = _to_float(stats.get("std"))
    if std is not None and min_v is not None and max_v is not None and max_v > min_v:
        return _clamp(abs(std) / (max_v - min_v + 1e-9), 0.0, 1.0)
    return 0.5


def _signal_strength(stats: Dict[str, Any], preference: str) -> float:
    """Return a 0..1 strength score based on distribution preference."""

    pos = _distribution_position(stats)
    spread = _spread_ratio(stats)
    if pos is None:
        base = 0.5
    else:
        if preference == "high":
            base = pos
        elif preference == "low":
            base = 1.0 - pos
        elif preference == "mid":
            base = 1.0 - abs(pos - 0.5) * 2.0
        else:
            base = 0.5
    base = _clamp(base, 0.0, 1.0)
    spread = _clamp(spread, 0.0, 1.0)
    # Combine base preference with how much dispersion exists (avoids flat metrics
    # receiving a very high weight just because the centre matches the preference).
    return _clamp(0.65 * base + 0.35 * (0.3 + 0.7 * spread), 0.0, 1.0)


def _zero_result(note: str) -> Dict[str, Any]:
    return {
        "weights": {k: 0 for k in ALLOWED_FIELDS},
        "order": [],
        "notes": [note] if note else [],
        "prompt_version": PROMPT_VERSION,
    }


def calculate_weights_from_aggregates(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Derive Winner Score weights from aggregate statistics."""

    metrics = _extract_metrics(payload)
    if not metrics:
        return _zero_result("sin agregados: no se pueden fijar pesos sin inventar")

    notes: list[str] = []
    weights: Dict[str, int] = {k: 0 for k in ALLOWED_FIELDS}

    for key in ALLOWED_FIELDS:
        stats = metrics.get(key)
        if not isinstance(stats, dict) or not stats:
            notes.append(f"{key}: sin datos en agregados")
            weights[key] = 0
            continue

        preference = PREFERENCES.get(key, "high")
        base_weight = BASE_WEIGHTS.get(key, 40.0)

        coverage = _to_float(stats.get("coverage"))
        if coverage is None:
            coverage = 0.0
        coverage = _clamp(coverage, 0.0, 1.0)
        coverage_factor = 0.1 + 0.9 * coverage

        strength = _signal_strength(stats, preference)
        weight = base_weight * (0.6 + 0.8 * strength)
        weight *= coverage_factor

        pos = _distribution_position(stats)

        if key == "competition" and pos is not None:
            # High competition (pos -> 1) should reduce the weight, but allow
            # some upside when competition is scarce.
            weight *= 0.5 + 0.5 * (1.0 - pos)
            if pos >= 0.7:
                notes.append("competition: promedio alto, peso moderado")

        if key == "oldness" and pos is not None:
            # Reward novelty (smaller oldness) while still keeping some weight when
            # catalogue is mature.
            novelty = 1.0 - pos
            weight *= 0.8 + 0.4 * novelty

        if key == "price" and pos is not None:
            # Aim for affordable but not ultra-cheap items; emphasise mid-low band.
            mid_bias = 1.0 - abs(pos - 0.4) * 1.6
            mid_bias = _clamp(mid_bias, 0.3, 1.15)
            weight *= mid_bias

        if key == "awareness" and pos is not None:
            # Prefer middle stages (problem/solution aware) and avoid saturation at
            # the extremes.
            mid_stage = 1.0 - abs(pos - 0.55) * 1.2
            mid_stage = _clamp(mid_stage, 0.35, 1.1)
            weight *= mid_stage

        if coverage < COVERAGE_THRESHOLD:
            weight = min(weight, LOW_COVERAGE_MAX_WEIGHT)
            notes.append(f"{key}: cobertura baja ({coverage:.2f})")

        weight = _clamp(weight, 0.0, 100.0)
        weights[key] = int(round(weight))

    order = [k for k in ALLOWED_FIELDS if weights.get(k, 0) > 0]
    order.sort(key=lambda k: (-weights[k], BASE_ORDER.index(k)))

    return {
        "weights": weights,
        "order": order,
        "notes": notes,
        "prompt_version": PROMPT_VERSION,
    }


__all__ = ["calculate_weights_from_aggregates", "PROMPT_VERSION"]

