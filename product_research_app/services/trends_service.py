import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from .. import database

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parents[1] / "data.sqlite3"


def _parse_extra(extra: str | bytes | None) -> Dict[str, Any]:
    if not extra:
        return {}
    try:
        return json.loads(extra)
    except Exception:
        try:
            return json.loads(extra.decode("utf-8"))
        except Exception:
            return {}


def get_trends_summary(start: datetime, end: datetime, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return aggregated metrics and timeseries for the given period.

    Args:
        start: Start of range (inclusive).
        end: End of range (exclusive).
        filters: Optional dict of filters (e.g., {"category": "Electronics"}).

    Returns:
        Dict with categories aggregation and timeseries.
    """
    filters = filters or {}
    t0 = time.perf_counter()
    conn = database.get_connection(DB_PATH)
    cur = conn.cursor()

    params = [start.isoformat(), end.isoformat()]
    where = "WHERE import_date >= ? AND import_date < ?"
    cat_filter = filters.get("category")
    if cat_filter:
        where += " AND category LIKE ?"
        params.append(f"{cat_filter}%")

    rows = cur.execute(
        f"SELECT id, category, price, import_date, extra FROM products {where}",
        params,
    ).fetchall()
    logger.info("trends_summary_rows=%s", len(rows))

    duration = end - start
    prev_start = start - duration
    prev_end = start
    prev_rows = cur.execute(
        "SELECT category, extra, price FROM products WHERE import_date >= ? AND import_date < ?",
        [prev_start.isoformat(), prev_end.isoformat()],
    ).fetchall()

    prev_rev: Dict[str, float] = {}
    prev_units: Dict[str, float] = {}
    prev_total_rev = 0.0
    prev_total_units = 0.0
    for prow in prev_rows:
        extra = _parse_extra(prow[1])
        units = float(extra.get("units_sold") or 0)
        revenue = extra.get("revenue")
        if revenue is None:
            price = prow[2] or 0
            revenue = price * units
        cat = prow[0] or ""
        prev_rev[cat] = prev_rev.get(cat, 0.0) + float(revenue or 0)
        prev_units[cat] = prev_units.get(cat, 0.0) + units
        prev_total_rev += float(revenue or 0)
        prev_total_units += units

    categories: list[Dict[str, Any]] = []
    cat_data: Dict[str, Dict[str, Any]] = {}
    timeseries: Dict[str, Dict[str, float]] = {}
    granularity = "day" if duration <= timedelta(days=31) else "week"

    for row in rows:
        cat = row[1] or ""
        extra = _parse_extra(row[4])
        units = float(extra.get("units_sold") or 0)
        revenue = extra.get("revenue")
        price = row[2] or 0
        if revenue is None:
            revenue = price * units
        rating = extra.get("rating")
        import_dt = datetime.fromisoformat(row[3])

        c = cat_data.setdefault(
            cat,
            {
                "products": set(),
                "units": 0.0,
                "revenue": 0.0,
                "price_sum": 0.0,
                "price_count": 0,
                "rating_sum": 0.0,
                "rating_count": 0,
            },
        )
        c["products"].add(row[0])
        c["units"] += units
        c["revenue"] += float(revenue or 0)
        if row[2] is not None:
            c["price_sum"] += row[2]
            c["price_count"] += 1
        if rating is not None:
            try:
                c["rating_sum"] += float(rating)
                c["rating_count"] += 1
            except Exception:
                pass

        if granularity == "week":
            key = (import_dt - timedelta(days=import_dt.weekday())).date().isoformat()
        else:
            key = import_dt.date().isoformat()
        ts = timeseries.setdefault(key, {"units": 0.0, "revenue": 0.0})
        ts["units"] += units
        ts["revenue"] += float(revenue or 0)

    total_products = 0
    total_units = 0.0
    total_revenue = 0.0
    price_sum = 0.0
    price_count = 0
    rating_sum = 0.0
    rating_count = 0
    
    for cat, data in cat_data.items():
        units = data["units"]
        revenue = data["revenue"]
        prev = prev_rev.get(cat, 0.0)
        delta_pct = ((revenue - prev) / prev * 100.0) if prev else 0.0
        avg_price = data["price_sum"] / data["price_count"] if data["price_count"] else 0.0
        avg_rating = data["rating_sum"] / data["rating_count"] if data["rating_count"] else 0.0
        rev_per_unit = revenue / units if units else 0.0
        categories.append(
            {
                "category": cat,
                "unique_products": len(data["products"]),
                "units": units,
                "revenue": revenue,
                "avg_price": avg_price,
                "avg_rating": avg_rating,
                "rev_per_unit": rev_per_unit,
                "delta_revenue_pct": delta_pct,
            }
        )
        total_products += len(data["products"])
        total_units += units
        total_revenue += revenue
        price_sum += data["price_sum"]
        price_count += data["price_count"]
        rating_sum += data["rating_sum"]
        rating_count += data["rating_count"]

    categories.sort(key=lambda x: x["revenue"], reverse=True)
    ts_list = [
        {"date": k, "units": v["units"], "revenue": v["revenue"]}
        for k, v in sorted(timeseries.items())
    ]

    avg_price = price_sum / price_count if price_count else 0.0
    avg_rating = rating_sum / rating_count if rating_count else 0.0
    rev_per_unit = total_revenue / total_units if total_units else 0.0
    totals = {
        "unique_products": total_products,
        "units": total_units,
        "revenue": total_revenue,
        "avg_price": avg_price,
        "avg_rating": avg_rating,
        "rev_per_unit": rev_per_unit,
    }
    totals["delta_revenue_pct"] = (
        (total_revenue - prev_total_rev) / prev_total_rev * 100.0
        if prev_total_rev
        else 0.0
    )
    totals["delta_units_pct"] = (
        (total_units - prev_total_units) / prev_total_units * 100.0
        if prev_total_units
        else 0.0
    )

    logger.info(
        "trends_summary_done categories=%s points=%s duration_ms=%.2f",
        len(categories),
        len(ts_list),
        (time.perf_counter() - t0) * 1000,
    )
    return {
        "categories": categories,
        "timeseries": ts_list,
        "granularity": granularity,
        "totals": totals,
    }
