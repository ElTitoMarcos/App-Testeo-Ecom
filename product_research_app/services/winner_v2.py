"""Winner Score v2 calculation utilities."""
from __future__ import annotations
import math
from typing import Dict, Any, Iterable

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
