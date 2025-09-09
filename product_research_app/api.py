from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List

from flask import Blueprint, jsonify, request

from . import database

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "data.sqlite3"

api_bp = Blueprint("api", __name__, url_prefix="/api")

# Default weights and order used when no settings exist yet
DEFAULT_WEIGHTS: Dict[str, float] = {
    "desire_magnitude": 70,
    "awareness_level": 60,
    "desire": 50,
    "conversion_rate": 40,
    "sales_per_day": 30,
    "launch_date": 20,
    "competition_level": 10,
    "ad_ease": 10,
    "scalability": 10,
    "durability": 10,
}
DEFAULT_ORDER: List[str] = list(DEFAULT_WEIGHTS.keys())


def _get_conn():
    return database.get_connection(DB_PATH)


def _parse_float(val: Any) -> float | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    try:
        s = str(val).strip().replace("%", "").replace(",", ".")
        return float(s) if s else None
    except Exception:
        return None


def _parse_metric(row: Dict[str, Any], extras: Dict[str, Any], name: str) -> float | None:
    value = row.get(name)
    if value is None:
        value = extras.get(name)
    if value is None:
        return None
    try:
        if name == "launch_date":
            # Convert to recency (more recent => larger value)
            if isinstance(value, (int, float)):
                dt = datetime.fromtimestamp(float(value))
            else:
                dt = datetime.fromisoformat(str(value))
            delta = (datetime.utcnow() - dt).days
            return -float(delta)
        if name == "competition_level":
            v = _parse_float(value)
            return 1.0 / (1.0 + v) if v is not None else None
        if name in {"conversion_rate"}:
            return _parse_float(value)
        if name in {"sales_per_day"}:
            return _parse_float(value)
        if name in {"desire", "ad_ease", "scalability", "durability"}:
            return _parse_float(value)
    except Exception:
        return None
    s = str(value).strip().lower()
    if name == "desire_magnitude":
        return {"low": 1, "medium": 2, "high": 3}.get(s)
    if name == "awareness_level":
        mapping = {
            "unaware": 1,
            "problem-aware": 2,
            "solution-aware": 3,
            "product-aware": 4,
            "most aware": 5,
        }
        return mapping.get(s)
    return None


def _pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = sum((a - mean_x) ** 2 for a in x)
    den_y = sum((b - mean_y) ** 2 for b in y)
    denom = (den_x * den_y) ** 0.5
    return num / denom if denom else 0.0


def _compute_ai_weights(products: List[Dict[str, Any]]) -> Dict[str, float]:
    metrics = DEFAULT_ORDER
    scores: Dict[str, float] = {m: 0.0 for m in metrics}
    for metric in metrics:
        x_rev: List[float] = []
        y_rev: List[float] = []
        x_units: List[float] = []
        y_units: List[float] = []
        for row in products:
            extras = row.get("extra")
            try:
                extras_dict = json.loads(extras) if extras else {}
            except Exception:
                extras_dict = {}
            val = _parse_metric(row, extras_dict, metric)
            if val is None:
                continue
            rev = _parse_float(row.get("revenue") or extras_dict.get("revenue") or extras_dict.get("Revenue($)"))
            units = _parse_float(row.get("units_sold") or extras_dict.get("Item Sold") or extras_dict.get("units"))
            if rev is not None:
                x_rev.append(val)
                y_rev.append(rev)
            if units is not None:
                x_units.append(val)
                y_units.append(units)
        score = 0.0
        if len(x_rev) >= 2:
            score += abs(_pearson(x_rev, y_rev))
        if len(x_units) >= 2:
            score += abs(_pearson(x_units, y_units))
        scores[metric] = score
    max_score = max(scores.values()) or 1.0
    weights = {m: 20 + (scores[m] / max_score) * 70 for m in metrics}
    return weights


def recalc_winner_scores(conn, weights: Dict[str, float], order: List[str]) -> int:
    products = database.list_products(conn)
    metrics = order or list(weights.keys())
    rows: List[Dict[str, Any]] = []
    ranges: Dict[str, List[float]] = {m: [] for m in metrics}
    for r in products:
        row = dict(r)
        try:
            extras = json.loads(row.get("extra") or "{}")
        except Exception:
            extras = {}
        rows.append((row, extras))
        for m in metrics:
            val = _parse_metric(row, extras, m)
            if val is not None:
                ranges[m].append(val)
    minmax: Dict[str, tuple[float, float]] = {}
    for m, vals in ranges.items():
        minmax[m] = (min(vals), max(vals)) if vals else (0.0, 0.0)
    updated = 0
    for row, extras in rows:
        total_w = 0.0
        accum = 0.0
        for m in metrics:
            val = _parse_metric(row, extras, m)
            if val is None:
                continue
            mn, mx = minmax[m]
            norm = 0.0 if mx == mn else (val - mn) / (mx - mn)
            w = weights.get(m, 0.0)
            total_w += w
            accum += w * norm
        if total_w > 0:
            final = accum / total_w
            cur = conn.cursor()
            cur.execute("UPDATE products SET winner_score = ? WHERE id = ?", (final, row["id"]))
            updated += 1
    conn.commit()
    return updated


@api_bp.get("/settings")
def get_settings() -> Any:
    conn = _get_conn()
    settings = database.get_settings(conn)
    conn.close()
    return jsonify(settings)


@api_bp.post("/settings")
def save_settings() -> Any:
    data = request.get_json(force=True) or {}
    conn = _get_conn()
    if "weights" in data:
        database.set_setting(conn, "winner_weights", data["weights"])
    if "order" in data:
        database.set_setting(conn, "winner_order", data["order"])
    if "openai_api_key" in data:
        database.set_setting(conn, "openai_api_key", data["openai_api_key"])
    conn.close()
    return jsonify({"ok": True})


@api_bp.post("/weights/ai")
def ai_weights() -> Any:
    conn = _get_conn()
    products = [dict(r) for r in database.list_products(conn)]
    weights = _compute_ai_weights(products)
    order = sorted(weights.keys(), key=lambda k: weights[k], reverse=True)
    database.set_setting(conn, "winner_weights", weights)
    database.set_setting(conn, "winner_order", order)
    conn.close()
    return jsonify({"weights": weights, "order": order})


@api_bp.post("/winner-score/generate")
def generate_winner_scores() -> Any:
    conn = _get_conn()
    settings = database.get_settings(conn)
    weights = settings.get("weights", DEFAULT_WEIGHTS)
    order = settings.get("order", DEFAULT_ORDER)
    updated = recalc_winner_scores(conn, weights, order)
    conn.close()
    return jsonify({"ok": True, "updated": updated})
