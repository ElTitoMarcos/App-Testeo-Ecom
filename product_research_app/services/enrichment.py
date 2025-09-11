import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .. import database

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = ROOT_DIR / "config"
MARGIN_CFG_PATH = CONFIG_DIR / "margin_baselines.yaml"

try:
    with open(MARGIN_CFG_PATH, "r", encoding="utf-8") as fh:
        MARGIN_BASELINES = yaml.safe_load(fh) or {}
except Exception:
    MARGIN_BASELINES = {}

IMAGE_COUNT_BY_CATEGORY = {
    "accesorios": 5,
    "moda": 5,
    "hogar": 5,
    "cocina": 5,
    "belleza": 4,
    "electronica": 6,
    "juguetes": 4,
}

SHIPPING_DAYS_BY_CATEGORY = {
    "belleza": (5, 9, 7),
    "accesorios": (5, 9, 7),
    "hogar": (6, 12, 8),
    "cocina": (6, 12, 8),
    "juguetes": (6, 12, 8),
    "electronica": (7, 15, 10),
    "voluminoso": (7, 15, 10),
}

REVIEW_RATE_BY_CATEGORY = {
    "belleza": 0.08,
    "hogar": 0.06,
    "cocina": 0.06,
    "juguetes": 0.05,
    "accesorios": 0.04,
    "moda": 0.04,
    "electronica": 0.03,
}


@dataclass
class ProductEnrichmentResult:
    product_id: int
    enriched_fields: Dict[str, Any]
    still_missing: List[str]
    used_rules: List[str]
    estimated_fields: int = 0
    ai_used: bool = False


def _category_key(cat: str | None) -> str:
    return (cat or "").strip().lower()


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _signature(prod: Dict[str, Any]) -> str:
    key = f"{(prod.get('name') or '').lower()}|{_category_key(prod.get('category'))}|{round(prod.get('price') or 0.0, 2):.2f}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


# deterministic enrichment rules

def _apply_rules(prod: Dict[str, Any]) -> ProductEnrichmentResult:
    enriched: Dict[str, Any] = {}
    missing: List[str] = []
    used_rules: List[str] = []
    est = 0
    cat = _category_key(prod.get("category"))

    # image_count
    if prod.get("image_count") in (None, ""):
        val = IMAGE_COUNT_BY_CATEGORY.get(cat)
        if val is not None:
            enriched["image_count"] = int(_clamp(val, 3, 8))
            enriched["image_count_estimated"] = True
            enriched["image_count_confidence"] = 0.6
            used_rules.append("image_count_category")
            est += 1
        else:
            missing.append("image_count")

    # shipping_days
    if prod.get("shipping_days_median") in (None, ""):
        tpl = SHIPPING_DAYS_BY_CATEGORY.get(cat)
        if tpl:
            mn, mx, md = tpl
            enriched.update(
                {
                    "shipping_days_min": mn,
                    "shipping_days_max": mx,
                    "shipping_days_median": md,
                    "shipping_days_estimated": True,
                    "shipping_days_confidence": 0.6,
                }
            )
            used_rules.append("shipping_days_category")
            est += 1
        else:
            missing.append("shipping_days")

    # review_count
    if prod.get("review_count") in (None, ""):
        units = prod.get("units_sold") or 0
        if units:
            r = REVIEW_RATE_BY_CATEGORY.get(cat, 0.05)
            rating = prod.get("rating")
            if rating is not None:
                if rating >= 4.7:
                    r *= 1.2
                elif rating <= 3.8:
                    r *= 0.8
            val = round(units * r)
            val = int(_clamp(val, 0, units))
            enriched["review_count"] = val
            enriched["review_count_estimated"] = True
            enriched["review_count_confidence"] = 0.6
            used_rules.append("review_count_units_sold")
            est += 1
        else:
            missing.append("review_count")

    # profit_margin_pct
    if prod.get("profit_margin_pct") in (None, ""):
        commission = prod.get("commission_rate")
        if commission is not None:
            baseline = MARGIN_BASELINES.get(cat)
            if baseline is not None:
                pm = baseline - commission - 0.05
                pm = _clamp(pm, 0.05, 0.70)
                enriched["profit_margin_pct"] = pm
                price = prod.get("price") or 0
                enriched["profit_margin_abs"] = price * pm
                enriched["profit_margin_estimated"] = True
                enriched["profit_margin_confidence"] = 0.6
                used_rules.append("profit_margin_baseline")
                est += 1
            else:
                missing.append("profit_margin_pct")
        else:
            missing.append("profit_margin_pct")

    return ProductEnrichmentResult(
        product_id=int(prod.get("id")),
        enriched_fields=enriched,
        still_missing=missing,
        used_rules=used_rules,
        estimated_fields=est,
    )


def enrich_missing_fields(products: List[Dict[str, Any]], weights: Dict[str, float]) -> List[ProductEnrichmentResult]:
    """Enrich missing fields using deterministic rules and optional AI."""

    results = [_apply_rules(p) for p in products]
    need_ai = [p for p, r in zip(products, results) if r.still_missing]
    if not need_ai:
        return results

    # try cache lookup
    conn = database.get_connection(database.DB_PATH) if hasattr(database, "DB_PATH") else None
    cache: Dict[str, Dict[str, Any]] = {}
    if conn:
        now = datetime.utcnow()
        for prod in need_ai:
            sig = _signature(prod)
            cached = database.get_enrichment_cache(conn, sig)
            if cached and now - datetime.fromisoformat(cached.get("created_at")) < timedelta(days=90):
                cache[sig] = cached

    to_query = []
    for prod in need_ai:
        sig = _signature(prod)
        if sig not in cache:
            to_query.append(prod)

    ai_results: Dict[int, Dict[str, Any]] = {}
    if to_query:
        try:
            from . import ia_client

            ai_list = ia_client.enrich_with_ai_batched(to_query, weights)
            for item in ai_list:
                pid = int(item.get("product_id"))
                ai_results[pid] = item
                if conn:
                    sig = _signature(next(p for p in to_query if int(p.get("id")) == pid))
                    database.set_enrichment_cache(conn, sig, item)
        except Exception:
            logger.exception("AI enrichment failed")

    # merge cache results
    for prod in need_ai:
        sig = _signature(prod)
        cached = cache.get(sig)
        if cached:
            ai_results[int(prod.get("id"))] = cached

    # apply AI results
    for res in results:
        pid = res.product_id
        ai = ai_results.get(pid)
        if not ai:
            continue
        res.ai_used = True
        for field in ["review_count", "image_count", "profit_margin_pct"]:
            val = ai.get(field)
            conf = ai.get("confidence", {}).get(field)
            if val is not None:
                res.enriched_fields[field] = val
                res.enriched_fields[f"{field}_estimated"] = True
                res.enriched_fields[f"{field}_confidence"] = conf
                if field in res.still_missing:
                    res.still_missing.remove(field)
        ship = ai.get("shipping_days") or {}
        if ship.get("median") is not None:
            res.enriched_fields.update(
                {
                    "shipping_days_min": ship.get("min"),
                    "shipping_days_max": ship.get("max"),
                    "shipping_days_median": ship.get("median"),
                    "shipping_days_estimated": True,
                    "shipping_days_confidence": ai.get("confidence", {}).get("shipping_days"),
                }
            )
            if "shipping_days" in res.still_missing:
                res.still_missing.remove("shipping_days")
        res.estimated_fields = sum(
            1 for k in res.enriched_fields if k.endswith("_estimated") and res.enriched_fields[k]
        )

    return results
