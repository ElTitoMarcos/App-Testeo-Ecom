"""Winner Score calculation utilities."""
from __future__ import annotations
import json
import math
import hashlib
import logging
import sqlite3
import unicodedata
from datetime import datetime, date
from typing import Dict, Any, Iterable, Optional, Callable

from ..utils.db import rget
from .. import database
from .config import (
    get_winner_weights_raw,
    set_winner_weights_raw,
    get_winner_order_raw,
    DB_PATH as CONFIG_DB_PATH,
)
from ..config import load_config, save_config

logger = logging.getLogger(__name__)


def load_settings() -> Dict[str, Any]:
    return load_config()


def save_settings(settings: Dict[str, Any]) -> None:
    save_config(settings)

# Winner Score allowed fields and compatibility aliases
ALLOWED_FIELDS = (
    "price",
    "rating",
    "units_sold",
    "revenue",
    "desire",
    "competition",
    "oldness",
    "awareness",
)
DEFAULT_WEIGHTS = {k: 1.0 for k in ALLOWED_FIELDS}

WEIGHT_KEYS = list(ALLOWED_FIELDS)

ALIASES = {
    "unitsSold": "units_sold",
    "orders": "units_sold",
}

AWARE_STAGES = [
    "unaware",
    "problem aware",
    "solution aware",
    "product aware",
    "most aware",
]
AWARE_INDEX = {name: i for i, name in enumerate(AWARE_STAGES)}
AWARE_CENTERS = [10, 30, 50, 70, 90]  # centros de cada tramo (0-19, 20-39, ...)


def _norm_awareness(s: str) -> str:
    return (s or "").strip().lower()


def awareness_stage_index_from_product(prod) -> int:
    cand = None
    extras = _get_extras(prod)
    for attr in ("awareness_type", "awareness_level", "awareness", "awarenessLabel"):
        cand = rget(prod, attr)
        if cand is None:
            cand = extras.get(attr)
        if cand is not None:
            break
    if isinstance(cand, (int, float)):
        i = int(round(float(cand)))
        return min(4, max(0, i))
    label = unicodedata.normalize("NFKD", str(cand))
    label = "".join(ch for ch in label if not unicodedata.combining(ch))
    return AWARE_INDEX.get(_norm_awareness(label), 2)


def awareness_pref_segment_from_weight(w: int) -> int:
    w = max(0, min(100, int(round(float(w)))))
    return min(4, w // 20)  # 0..4


def awareness_priority_order_from_weight(w: int) -> list[int]:
    """Ordena 0..4 por cercanía del valor w a los centros [10,30,50,70,90]. Ties: prioriza el menor índice."""
    w = max(0, min(100, int(round(float(w)))))
    return sorted(range(5), key=lambda i: (abs(w - AWARE_CENTERS[i]), i))


def awareness_closeness(w_slider_0_100: int, stage_idx: int) -> float:
    w = max(0, min(100, int(round(float(w_slider_0_100)))))
    dist = abs(w - AWARE_CENTERS[stage_idx])  # máx 80 (10↔90)
    return max(0.0, 1.0 - (dist / 80.0))  # 1.0..0.0


def awareness_closeness_from_weight(w: int, stage_idx: int) -> float:
    """Backward-compatible alias."""
    return awareness_closeness(w, stage_idx)


def awareness_feature_value(prod, w_slider_0_100: int) -> float:
    stage_idx = awareness_stage_index_from_product(prod)
    return awareness_closeness(w_slider_0_100, stage_idx)


def build_features(prod, settings):
    feats: Dict[str, float] = {}
    w_aw = (settings.get("winner_weights") or {}).get("awareness", 50)
    stage_idx = awareness_stage_index_from_product(prod)
    feats["awareness"] = awareness_closeness(w_aw, stage_idx)
    return feats


def _minmax_norm(pop, v):
    vals = [x for x in pop if x is not None]
    if not vals:
        return 0.0
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        return 0.5  # neutro si todos son iguales
    return (v - vmin) / (vmax - vmin)


def _parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _oldness_days(product: dict) -> int | None:
    dr = (product.get("date_range") or product.get("Date Range") or "").strip()
    start = None
    if "~" in dr:
        left = dr.split("~", 1)[0].strip()
        start = _parse_date(left)
    if start is None:
        for k in ("first_seen", "created_at", "createdAt"):
            start = _parse_date(product.get(k))
            if start:
                break
    return None if start is None else max(0, (date.today() - start).days)


OLDNESS_MIN = 0.0
OLDNESS_MAX = 0.0


def prepare_oldness_bounds(rows: Iterable[Any]) -> None:
    global OLDNESS_MIN, OLDNESS_MAX
    vals: list[int] = []
    for row in rows:
        try:
            data = dict(row)
        except Exception:
            data = {}
        extras = _get_extras(row)
        data.update(extras)
        val = _oldness_days(data)
        if val is not None:
            vals.append(val)
    if vals:
        OLDNESS_MIN = float(min(vals))
        OLDNESS_MAX = float(max(vals))
    else:
        OLDNESS_MIN = OLDNESS_MAX = 0.0


WEIGHTS_CACHE: Dict[str, float] | None = None
ORDER_CACHE: list[str] | None = None
WEIGHTS_VERSION: int = 0


def compute_effective_weights(weights: dict[str, int | float], order: list[str]) -> dict[str, float]:
    order = list(order)
    if "awareness" in weights and "awareness" not in order:
        order.append("awareness")
    n = len(order)
    if n == 0:
        return {}
    sum_ranks = n * (n + 1) / 2.0
    pri = {k: (n - i) for i, k in enumerate(order)}
    pri_factor = {k: pri[k] / (sum_ranks / n) for k in order}
    eff = {k: float(weights.get(k, 0)) * pri_factor[k] for k in order}
    s = sum(eff.values()) or 1.0
    return {k: v / s for k, v in eff.items()}

def sanitize_weights(weights: dict | None) -> dict:
    w = (weights or {}).copy()
    w = {k: float(v) for k, v in w.items() if k in ALLOWED_FIELDS}
    if not w:
        w = DEFAULT_WEIGHTS.copy()
    s = sum(w.values()) or 1.0
    return {k: v / s for k, v in w.items()}


def invalidate_weights_cache() -> None:
    global WEIGHTS_CACHE, ORDER_CACHE, WEIGHTS_VERSION
    WEIGHTS_CACHE = None
    ORDER_CACHE = None
    WEIGHTS_VERSION += 1


def normalize_weight_key(key: str) -> str:
    """Return canonical weight key or raise ``ValueError``.

    The function applies legacy alias mapping and accepts keys with an
    optional ``_weight`` suffix.  Unknown keys raise ``ValueError`` with a
    descriptive message.
    """

    k = str(key or "").strip()
    if k.endswith("_weight"):
        k = k[:-7]
    k = ALIASES.get(k, k)
    if k not in WEIGHT_KEYS:
        raise ValueError(f"Invalid weight key: {key}")
    return k


def load_winner_weights_raw() -> Dict[str, Any]:
    """Return persisted weights with minimal metadata."""

    w, o = load_winner_settings()
    return {"weights": w, "order": o}


def save_winner_weights_raw(data: Dict[str, Any]) -> None:
    set_winner_weights_raw(data.get("weights", {}))
    if "order" in data:
        from .config import set_winner_order_raw

        set_winner_order_raw(data.get("order", []))
    invalidate_weights_cache()


def load_winner_settings() -> tuple[Dict[str, float], list[str]]:
    global WEIGHTS_CACHE, ORDER_CACHE
    if WEIGHTS_CACHE is not None and ORDER_CACHE is not None:
        return WEIGHTS_CACHE, ORDER_CACHE
    weights = get_winner_weights_raw()
    order = get_winner_order_raw()
    WEIGHTS_CACHE = weights
    ORDER_CACHE = order
    return weights, order


def load_winner_weights() -> Dict[str, float]:
    """Load Winner Score weights (RAW integers)."""

    weights, _ = load_winner_settings()
    return weights


def update_winner_weight(key: str, value: float) -> None:
    """Update a single weight in persistent storage."""

    norm = normalize_weight_key(key)
    set_winner_weights_raw({norm: value})
    invalidate_weights_cache()


def set_winner_weights(weights: Dict[str, float]) -> None:
    """Replace or update multiple weights at once."""

    cleaned: Dict[str, float] = {}
    for k, v in weights.items():
        try:
            cleaned[normalize_weight_key(k)] = float(v)
        except ValueError:
            continue
    set_winner_weights_raw(cleaned)
    invalidate_weights_cache()

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

# Backwards-compatible alias for legacy imports
FEATURES = WEIGHT_KEYS

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


def _feat_oldness(product_row: Any, extras: Dict[str, Any]) -> Optional[float]:
    data: Dict[str, Any] = {}
    try:
        data.update(dict(product_row))
    except Exception:
        pass
    data.update(extras)
    val = _oldness_days(data)
    if val is None:
        return None
    return float(val)


FeatureGetter = Callable[[Any, Dict[str, Any]], Optional[float]]

FEATURE_MAP: Dict[str, FeatureGetter] = {
    "price": lambda p, e: _get_from_sources(p, e, ["price"]),
    "rating": lambda p, e: _get_from_sources(p, e, ["rating", "Product Rating"]),
    "units_sold": lambda p, e: _get_from_sources(p, e, ["units_sold", "Item Sold"]),
    "revenue": lambda p, e: _get_from_sources(p, e, ["revenue", "Revenue($)"]),
    "desire": _feat_desire,
    "competition": _feat_competition,
    "oldness": lambda p, e: _feat_oldness(p, e),
    "awareness": lambda p, e: (
        float(idx) if (idx := awareness_stage_index_from_product(p)) is not None else None
    ),
}


def _norm_price(v: float) -> float:
    return clamp(v / 1000.0)


def _norm_rating(v: float) -> float:
    return clamp(v / 5.0)


def _norm_units(v: float) -> float:
    return clamp(math.log1p(v) / math.log1p(10000.0))


def _norm_revenue(v: float) -> float:
    return clamp(math.log1p(v) / math.log1p(1_000_000.0))


def _norm_oldness(v: float) -> float:
    return clamp(_minmax_norm([OLDNESS_MIN, OLDNESS_MAX], v))

def _norm_identity(v: float) -> float:
    return clamp(v)


def _oldness_pref_and_weight(cfg_weights: dict[str, float]) -> tuple[float, float]:
    # Entrada UI 0..100 (la misma barra que ves en el modal)
    v = float(cfg_weights.get("oldness", 50))
    # Intensidad 0..1 (50 => 0, 0/100 => 1)
    intensity = abs(v - 50.0) / 50.0
    # Dirección: -1 recientes, +1 antiguos (50 => 0)
    direction = (v - 50.0) / 50.0
    return direction, intensity


NORMALIZERS: Dict[str, Callable[[float], float]] = {
    "price": _norm_price,
    "rating": _norm_rating,
    "units_sold": _norm_units,
    "revenue": _norm_revenue,
    "desire": _norm_identity,
    "competition": _norm_identity,
    "oldness": _norm_oldness,
    "awareness": _norm_identity,
}


def compute_winner_score_v2(
    product_row: Any, user_weights: Dict[str, float], order: Optional[list[str]] = None
) -> Dict[str, Any]:
    """Compute Winner Score using available features only."""

    extras = _get_extras(product_row)
    raw_vals: Dict[str, float] = {}
    for name, getter in FEATURE_MAP.items():
        val = getter(product_row, extras)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        raw_vals[name] = val

    w_aw = user_weights.get("awareness", 50)
    raw_vals["awareness"] = awareness_feature_value(product_row, w_aw)

    present = list(raw_vals.keys())
    missing_fields = [f for f in ALLOWED_FIELDS if f not in present]

    dir_old, w_old_intensity = _oldness_pref_and_weight(user_weights)

    norms: Dict[str, float] = {}
    for k, v in raw_vals.items():
        n = NORMALIZERS[k](v)
        if k == "oldness" and dir_old < 0:
            n = 1.0 - n
        norms[k] = n

    weights_for_priority = {k: int(round(user_weights.get(k, 0))) for k in ALLOWED_FIELDS}
    weights_for_priority["oldness"] = int(round(w_old_intensity * 100))
    if order is None:
        order = list(weights_for_priority.keys())
    eff_all = compute_effective_weights(weights_for_priority, order)
    eff_weights = {k: eff_all.get(k, 0.0) for k in norms.keys()}

    sum_filtered = sum(eff_weights.values())
    score_float = sum(eff_weights.get(k, 0.0) * norms[k] for k in norms)
    score_int = int(round(score_float * 100))

    eff_all_full = {k: eff_all.get(k, 0.0) for k in ALLOWED_FIELDS}
    return {
        "score": score_int,
        "score_float": score_float,
        "used": len(present),
        "missing": len(ALLOWED_FIELDS) - len(present),
        "missing_fields": missing_fields,
        "present_fields": present,
        "effective_weights": eff_all_full,
        "sum_filtered": sum_filtered,
        "fallback": sum_filtered == 0,
        "order": order,
    }


def generate_winner_scores(
    conn: sqlite3.Connection,
    product_ids: Optional[Iterable[Any]] = None,
    weights: Optional[Dict[str, float]] = None,
    debug: bool = False,
    ) -> Dict[str, Any]:
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
        weights, order = load_winner_settings()
    else:
        order = get_winner_order_raw()

    dir_old, w_old_intensity = _oldness_pref_and_weight(weights)
    oldness_ui = float(weights.get("oldness", 50))
    dir_old_label = "+1" if dir_old >= 0 else "-1"

    weights_hash_all = hashlib.sha1(
        json.dumps(weights, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    weights_hash_eff = ""

    ids: Optional[set[int]] = None
    if product_ids:
        ids = {int(pid) for pid in product_ids if str(pid).strip()}

    if ids:
        placeholders = ",".join("?" for _ in ids)
        cur = conn.execute(f"SELECT * FROM products WHERE id IN ({placeholders})", tuple(ids))
        rows = cur.fetchall()
    else:
        rows = database.list_products(conn)

    prepare_oldness_bounds(rows)

    processed = 0
    updated_int = 0
    now = datetime.utcnow().isoformat()
    diag_present: Optional[Iterable[str]] = None
    diag_missing: Optional[Iterable[str]] = None
    diag_sum_filtered: float = 0.0
    diag_eff: Dict[str, float] = {}
    for row in rows:
        pid = row["id"]
        old_score = row["winner_score"]
        res = compute_winner_score_v2(row, weights, order)
        sf = res.get("score_float") or 0.0
        present = set(res.get("present_fields", []))
        missing = set(res.get("missing_fields", []))
        eff_w = {k: round(v, 3) for k, v in res.get("effective_weights", {}).items()}
        eff_hash = hashlib.sha1(
            json.dumps(res.get("effective_weights", {}), sort_keys=True).encode("utf-8")
        ).hexdigest()[:8]
        weights_hash_eff = eff_hash
        eff_w_int = {k: int(round(v * 100)) for k, v in res.get("effective_weights", {}).items()}
        ord_list = res.get("order", order)

        if debug and diag_present is None:
            diag_present = present
            diag_missing = missing
            diag_sum_filtered = float(res.get("sum_filtered", 0.0))
            diag_eff = {k: round(v, 6) for k, v in res.get("effective_weights", {}).items()}

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
                "Winner Score: product=%s score_int=%s score_raw=%.3f weights_all=%s weights_eff=%s present=%s missing=%s "
                "effective_weights=%s order=%s weights_effective_int=%s oldness_dir=%s oldness_intensity=%.3f oldness_ui=%.1f no_features_present",
                pid,
                new_score,
                score_raw_0_100,
                weights_hash_all,
                eff_hash,
                present,
                missing,
                eff_w,
                ord_list,
                eff_w_int,
                dir_old_label,
                w_old_intensity,
                oldness_ui,
            )
        else:
            logger.info(
                "Winner Score: product=%s score_int=%s score_raw=%.3f weights_all=%s weights_eff=%s present=%s missing=%s "
                "effective_weights=%s order=%s weights_effective_int=%s oldness_dir=%s oldness_intensity=%.3f oldness_ui=%.1f",
                pid,
                new_score,
                score_raw_0_100,
                weights_hash_all,
                eff_hash,
                present,
                missing,
                eff_w,
                ord_list,
                eff_w_int,
                dir_old_label,
                w_old_intensity,
                oldness_ui,
            )
        processed += 1

    conn.commit()
    result: Dict[str, Any] = {
        "processed": processed,
        "updated": updated_int,
        "weights_all": weights_hash_all,
        "weights_eff": weights_hash_eff,
    }
    if debug:
        result["diag"] = {
            "present": sorted(diag_present or []),
            "missing": sorted(diag_missing or []),
            "sum_filtered": diag_sum_filtered,
            "effective_weights": diag_eff,
        }
    return result


def recompute_scores_for_all_products(scope: str = "all") -> int:
    """Recalcula y persiste el score 0..100 de todos los productos."""
    conn = database.get_connection(CONFIG_DB_PATH)
    try:
        res = generate_winner_scores(conn)
        updated = int(res.get("updated", 0))
        logger.info("winner_score recompute updated=%s", updated)
        return updated
    finally:
        conn.close()
