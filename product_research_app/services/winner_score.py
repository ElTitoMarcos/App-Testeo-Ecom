"""Winner Score calculation utilities."""
from __future__ import annotations
import json
import math
from typing import Dict, Any, Iterable, Optional

from ..utils.db import rget

# Mapping tables for categorical metrics
MAGNITUD_DESEO = {"low":0.33, "medium":0.66, "high":1.0}
NIVEL_CONSCIENCIA_HEADROOM = {"unaware":1.0, "problem":0.8, "solution":0.6, "product":0.4, "most":0.2}
COMPETITION_LEVEL_INVERTIDO = {"low":1.0, "medium":0.5, "high":0.0}
FACILIDAD = {"low":0.33, "med":0.66, "medium":0.66, "high":1.0}
ESCALABILIDAD = FACILIDAD
DURABILIDAD = {"consumible":1.0, "durable":0.0, "intermedio":0.5}

MAPS = {
    "magnitud_deseo": MAGNITUD_DESEO,
    "nivel_consciencia_headroom": NIVEL_CONSCIENCIA_HEADROOM,
    "competition_level_invertido": COMPETITION_LEVEL_INVERTIDO,
    "facilidad_anuncio": FACILIDAD,
    "escalabilidad": ESCALABILIDAD,
    "durabilidad_recurrencia": DURABILIDAD,
}

ALL_METRICS = [
    "magnitud_deseo",
    "nivel_consciencia_headroom",
    "evidencia_demanda",
    "tasa_conversion",
    "ventas_por_dia",
    "recencia_lanzamiento",
    "competition_level_invertido",
    "facilidad_anuncio",
    "escalabilidad",
    "durabilidad_recurrencia",
]

def clamp(v: float) -> float:
    return 0.0 if v < 0 else 1.0 if v > 1 else v

def _percentiles(values: Iterable[float]) -> Dict[str, float]:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"p5":0.0, "p95":1.0}
    def p(q: float) -> float:
        idx = int(q * (len(vals)-1))
        return vals[idx]
    return {"p5": p(0.05), "p95": p(0.95)}

def compute_ranges(products: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str,float]]:
    ev = []
    vpd = []
    for p in products:
        if p.get("evidencia_demanda") is not None:
            ev.append(math.log1p(float(p.get("evidencia_demanda", 0))))
        if p.get("ventas_por_dia") is not None:
            vpd.append(float(p.get("ventas_por_dia", 0)))
    return {
        "evidencia_demanda": _percentiles(ev),
        "ventas_por_dia": _percentiles(vpd),
    }

def normalize_metric(name: str, value: Any, ranges: Dict[str, Dict[str,float]]) -> float | None:
    if value is None:
        return None
    if name in MAPS:
        return MAPS[name].get(str(value).lower())
    if name == "evidencia_demanda":
        v = math.log1p(float(value))
        r = ranges.get(name, {})
        return clamp((v - r.get("p5",0.0)) / (r.get("p95",1.0) - r.get("p5",0.0) or 1))
    if name == "tasa_conversion":
        v = float(value)
        if v > 1:
            v /= 100.0
        return clamp(v)
    if name == "ventas_por_dia":
        v = float(value)
        r = ranges.get(name, {})
        return clamp((v - r.get("p5",0.0)) / (r.get("p95",1.0) - r.get("p5",0.0) or 1))
    if name == "recencia_lanzamiento":
        return math.exp(-float(value)/180.0)
    return None

def score_product(
    prod: Dict[str, Any],
    weights: Dict[str, float],
    ranges: Dict[str, Dict[str, float]] | None = None,
    missing: list[str] | None = None,
    used: list[str] | None = None,
) -> float | None:
    if ranges is None:
        ranges = compute_ranges([prod])
    total_w = 0.0
    score = 0.0
    for k, w in weights.items():
        if w <= 0:
            continue
        val = normalize_metric(k, prod.get(k), ranges)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            if missing is not None:
                missing.append(k)
            continue
        if used is not None:
            used.append(k)
        score += w * val
        total_w += w
    if total_w <= 0:
        return None
    return score / total_w


# ---- V2 utilities ----

FeatureDict = Dict[str, Optional[float]]

def extract_features_v2(product_row: Any) -> FeatureDict:
    """Extract Winner Score features from a product row safely.

    Missing keys or non-numeric values yield ``None``; no exceptions are
    raised. Values may be sourced from the row itself or from an ``extra``/``extras``
    JSON column.
    """

    extras = rget(product_row, "extras")
    if extras is None:
        extras = rget(product_row, "extra")
    if isinstance(extras, str):
        try:
            extras = json.loads(extras)
        except Exception:
            extras = None

    def get_val(key: str) -> Optional[float]:
        val = rget(product_row, key)
        if val is None and isinstance(extras, dict):
            val = extras.get(key)
        try:
            return float(val)
        except Exception:
            return None

    return {k: get_val(k) for k in ALL_METRICS}


def compute_winner_score_v2(product_row: Any, weights: Dict[str, float]) -> Dict[str, int | bool]:
    """Compute Winner Score V2 using available features only.

    Args:
        product_row: Source of metrics (dict or sqlite3.Row).
        weights: Weight mapping for all metrics (will be renormalised).

    Returns:
        Dict with integer score 0-100, number of used/missing metrics and
        fallback flag.
    """

    feats = extract_features_v2(product_row)
    present = {k: v for k, v in feats.items() if v is not None}
    used = len(present)
    missing = len(ALL_METRICS) - used
    if not present:
        return {"score": 50, "used": 0, "missing": len(ALL_METRICS), "fallback": True}

    weights_used = {k: weights.get(k, 0.0) for k in present}
    total_w = sum(weights_used.values())
    if total_w <= 0:
        weights_used = {k: 1.0 / used for k in present}
    else:
        weights_used = {k: v / total_w for k, v in weights_used.items()}

    for k, v in list(present.items()):
        if v <= 1:
            v = v * 100
        if v < 0:
            v = 0
        elif v > 100:
            v = 100
        present[k] = v

    score = int(round(sum(present[k] * weights_used.get(k, 0.0) for k in present)))
    return {"score": score, "used": used, "missing": missing, "fallback": False}
