"""Winner Score calculation utilities."""
from __future__ import annotations
import json
import math
import hashlib
import logging
import sqlite3
from datetime import datetime
from typing import Dict, Any, Iterable, Optional, Callable

from ..utils.db import rget
from .. import config, database

logger = logging.getLogger(__name__)

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

# --- New Winner Score V2 (feature based on real product fields) ---

DESIRE_LABELS = {"low": 0.2, "medium": 0.5, "med": 0.5, "high": 0.8}
COMPETITION_LABELS = {"low": 0.8, "medium": 0.5, "med": 0.5, "high": 0.2}

FEATURES = [
    "price",
    "rating",
    "units_sold",
    "revenue",
    "review_count",
    "image_count",
    "shipping_days",
    "profit_margin",
    "desire",
    "competition",
]

def _get_extras(product_row: Any) -> Dict[str, Any]:
    extras = rget(product_row, "extras")
    if extras is None:
        extras = rget(product_row, "extra")
    if isinstance(extras, str):
        try:
            extras = json.loads(extras)
        except Exception:
            extras = None
    return extras if isinstance(extras, dict) else {}


def _to_float(val: Any) -> Optional[float]:
    try:
        if val is None:
            return None
        if isinstance(val, str):
            val = val.strip()
            if not val:
                return None
        num = float(val)
        return num
    except Exception:
        return None


def _get_from_sources(product_row: Any, extras: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    for k in keys:
        v = rget(product_row, k)
        if v is None:
            v = extras.get(k)
        v = _to_float(v)
        if v is not None:
            return v
    return None


def _feat_desire(product_row: Any, extras: Dict[str, Any]) -> Optional[float]:
    num = _get_from_sources(product_row, extras, ["ai_desire_score", "desire_score", "_desire_score"])
    if num is not None:
        if num > 1:
            num /= 100.0
        return clamp(num)
    lab = rget(product_row, "ai_desire_label") or rget(product_row, "desire_magnitude") or rget(product_row, "desire")
    if lab is None:
        lab = extras.get("ai_desire_label") or extras.get("desire_magnitude") or extras.get("desire")
    if isinstance(lab, str):
        return DESIRE_LABELS.get(lab.strip().lower())
    return None


def _feat_competition(product_row: Any, extras: Dict[str, Any]) -> Optional[float]:
    num = _get_from_sources(product_row, extras, ["ai_competition_score", "competition_score", "_competition_score"])
    if num is not None:
        if num > 1:
            num /= 100.0
        return clamp(num)
    lab = rget(product_row, "ai_competition_label") or rget(product_row, "competition_level")
    if lab is None:
        lab = extras.get("ai_competition_label") or extras.get("competition_level")
    if isinstance(lab, str):
        return COMPETITION_LABELS.get(lab.strip().lower())
    return None


FeatureGetter = Callable[[Any, Dict[str, Any]], Optional[float]]

FEATURE_MAP: Dict[str, FeatureGetter] = {
    "price": lambda p, e: _get_from_sources(p, e, ["price"]),
    "rating": lambda p, e: _get_from_sources(p, e, ["rating", "Product Rating"]),
    "units_sold": lambda p, e: _get_from_sources(p, e, ["units_sold", "Item Sold"]),
    "revenue": lambda p, e: _get_from_sources(p, e, ["revenue", "Revenue($)"]),
    "review_count": lambda p, e: _get_from_sources(p, e, ["review_count", "reviews", "reviewcount"]),
    "image_count": lambda p, e: _get_from_sources(p, e, ["image_count", "images"]),
    "shipping_days": lambda p, e: _get_from_sources(p, e, ["shipping_days", "shipping_time", "delivery_days"]),
    "profit_margin": lambda p, e: _get_from_sources(p, e, ["profit_margin", "margin"]),
    "desire": _feat_desire,
    "competition": _feat_competition,
}


def _norm_price(v: float) -> float:
    return clamp(v / 1000.0)


def _norm_rating(v: float) -> float:
    return clamp(v / 5.0)


def _norm_units(v: float) -> float:
    return clamp(math.log1p(v) / math.log1p(10000.0))


def _norm_revenue(v: float) -> float:
    return clamp(math.log1p(v) / math.log1p(1_000_000.0))


def _norm_review(v: float) -> float:
    return clamp(math.log1p(v) / math.log1p(10000.0))


def _norm_image(v: float) -> float:
    return clamp(v / 10.0)


def _norm_shipping(v: float) -> float:
    return clamp(1.0 - v / 30.0)


def _norm_margin(v: float) -> float:
    if v > 1:
        v /= 100.0
    return clamp(v)


def _norm_identity(v: float) -> float:
    return clamp(v)


NORMALIZERS: Dict[str, Callable[[float], float]] = {
    "price": _norm_price,
    "rating": _norm_rating,
    "units_sold": _norm_units,
    "revenue": _norm_revenue,
    "review_count": _norm_review,
    "image_count": _norm_image,
    "shipping_days": _norm_shipping,
    "profit_margin": _norm_margin,
    "desire": _norm_identity,
    "competition": _norm_identity,
}


def load_winner_weights() -> Dict[str, float]:
    """Load Winner Score weights from persistent configuration.

    The configuration is read from disk on each call to avoid stale caches.
    Any missing metric receives a weight of ``0.0`` and all values are
    coerced to ``float``.
    """

    cfg = config.load_config()
    stored = cfg.get("weights", {})
    weights: Dict[str, float] = {}
    for key in FEATURES:
        try:
            weights[key] = float(stored.get(key, 0.0))
        except Exception:
            weights[key] = 0.0
    return weights


def compute_winner_score_v2(product_row: Any, weights: Dict[str, float]) -> Dict[str, Any]:
    """Compute Winner Score using available features only.

    Returns a dict with keys ``score`` (or ``None``), ``used``, ``missing``,
    ``missing_fields`` and ``fallback``.
    """

    extras = _get_extras(product_row)
    feats: Dict[str, float] = {}
    missing_fields: list[str] = []
    for name, getter in FEATURE_MAP.items():
        val = getter(product_row, extras)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            missing_fields.append(name)
            continue
        norm = NORMALIZERS[name](val)
        feats[name] = norm

    present = list(feats.keys())
    missing_fields = list(set(FEATURE_MAP.keys()) - set(present))
    used = len(present)

    if used == 0:
        return {
            "score": 0,
            "score_float": 0.0,
            "used": 0,
            "missing": len(FEATURE_MAP),
            "missing_fields": missing_fields,
            "present_fields": present,
            "effective_weights": {},
            "fallback": True,
        }

    filtered = {k: weights.get(k, 0.0) for k in present}
    sum_filtered = sum(max(0.0, v) for v in filtered.values())
    if sum_filtered > 0:
        weights_used = {k: max(0.0, filtered.get(k, 0.0)) / sum_filtered for k in present}
    else:
        weights_used = {k: 1.0 / used for k in present}

    score_float = sum(feats[k] * weights_used.get(k, 0.0) for k in present)
    score_int = int(round(score_float * 100))
    return {
        "score": score_int,
        "score_float": score_float,
        "used": used,
        "missing": len(FEATURE_MAP) - used,
        "missing_fields": missing_fields,
        "present_fields": present,
        "effective_weights": weights_used,
        "fallback": False,
    }


def generate_winner_scores(
    conn: sqlite3.Connection,
    product_ids: Optional[Iterable[Any]] = None,
    weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, int | str]:
    """Recalculate and persist Winner Scores for the given products.

    Args:
        conn: Database connection.
        product_ids: Iterable of product IDs to update. If ``None`` or empty,
            all products are processed.
        weights: Optional weighting factors. If ``None``, weights are loaded
            fresh from configuration on each call.

    Returns a dict with counts and weight metadata.
    """

    if weights is None:
        weights = load_winner_weights()

    weights_hash_all = hashlib.sha1(
        json.dumps(weights, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]

    pos = {k: max(0.0, v) for k, v in weights.items()}
    sum_pos = sum(pos.values())
    if sum_pos > 0:
        eff_global = {k: v / sum_pos for k, v in pos.items() if v > 0}
    else:
        eff_global = {k: 1.0 / len(weights) for k in weights}
    weights_hash_eff = hashlib.sha1(
        json.dumps(eff_global, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]

    ids: Optional[set[int]] = None
    if product_ids:
        ids = {int(pid) for pid in product_ids if str(pid).strip()}

    if ids:
        placeholders = ",".join("?" for _ in ids)
        cur = conn.execute(f"SELECT * FROM products WHERE id IN ({placeholders})", tuple(ids))
        rows = cur.fetchall()
    else:
        rows = database.list_products(conn)

    processed = 0
    updated_int = 0
    now = datetime.utcnow().isoformat()
    for row in rows:
        pid = row["id"]
        old_score = row["winner_score"]
        res = compute_winner_score_v2(row, weights)
        sf = res.get("score_float") or 0.0
        present = set(res.get("present_fields", []))
        missing = set(res.get("missing_fields", []))
        eff_w = {k: round(v, 3) for k, v in res.get("effective_weights", {}).items()}
        eff_hash = hashlib.sha1(
            json.dumps(res.get("effective_weights", {}), sort_keys=True).encode("utf-8")
        ).hexdigest()[:8]

        score_raw_0_100 = max(0.0, min(1.0, sf)) * 100.0
        new_score = int(round(score_raw_0_100))

        if new_score != old_score or row["winner_score_raw"] != score_raw_0_100:
            conn.execute(
                "UPDATE products SET winner_score = ?, winner_score_raw = ?, winner_score_updated_at = ? WHERE id = ?",
                (new_score, score_raw_0_100, now, pid),
            )
            if new_score != old_score:
                updated_int += 1

        if not present:
            logger.warning(
                "Winner Score: product=%s score_int=%s score_raw=%.3f weights_all=%s weights_eff=%s present=%s missing=%s effective_weights=%s no_features_present",
                pid,
                new_score,
                score_raw_0_100,
                weights_hash_all,
                eff_hash,
                present,
                missing,
                eff_w,
            )
        else:
            logger.info(
                "Winner Score: product=%s score_int=%s score_raw=%.3f weights_all=%s weights_eff=%s present=%s missing=%s effective_weights=%s",
                pid,
                new_score,
                score_raw_0_100,
                weights_hash_all,
                eff_hash,
                present,
                missing,
                eff_w,
            )
        processed += 1

    conn.commit()
    return {
        "processed": processed,
        "updated": updated_int,
        "weights_all": weights_hash_all,
        "weights_eff": weights_hash_eff,
    }
