"""Simplified Winner Score calculator using enriched fields."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "review_count": 1.0,
    "image_count": 1.0,
    "shipping_days_median": 1.0,
    "profit_margin_pct": 1.0,
}


def _norm(value: float, lo: float, hi: float, invert: bool = False) -> float:
    if value is None:
        return 0.0
    if invert:
        value = hi - value + lo
    rng = hi - lo
    if rng <= 0:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / rng))


def score_product(prod: Dict[str, Any], weights: Dict[str, float]) -> Tuple[int, List[str], List[str], bool]:
    """Return (score_pct, used_fields, missing_fields, fallback)."""

    used: List[str] = []
    missing: List[str] = []
    total_w = 0.0
    score = 0.0

    def add(field: str, val: float, norm_params: Tuple[float, float, bool]):
        nonlocal total_w, score
        w = weights.get(field, 0.0)
        if w <= 0:
            return
        used.append(field)
        lo, hi, inv = norm_params
        norm = _norm(val, lo, hi, invert=inv)
        if prod.get(f"{field}_estimated") and (prod.get(f"{field}_confidence") or 0) < 0.4:
            norm *= 0.9
        score += w * norm
        total_w += w

    # review_count
    rc = prod.get("review_count")
    if rc is None:
        rc = 0
        missing.append("review_count")
    add("review_count", float(rc), (0.0, 1000.0, False))

    # image_count
    ic = prod.get("image_count")
    if ic is None:
        ic = 3
        missing.append("image_count")
    add("image_count", float(ic), (3.0, 8.0, False))

    # shipping_days median
    sd = prod.get("shipping_days_median")
    if sd is None:
        sd = 10
        missing.append("shipping_days_median")
    add("shipping_days_median", float(sd), (3.0, 15.0, True))

    # profit_margin_pct
    pm = prod.get("profit_margin_pct")
    if pm is None:
        pm = 0.10
        missing.append("profit_margin_pct")
    add("profit_margin_pct", float(pm), (0.05, 0.70, False))

    fallback = bool(missing)
    pct = 0
    if total_w > 0:
        pct = round((score / total_w) * 100)
        pct = max(0, min(100, pct))
    return pct, used, missing, fallback
