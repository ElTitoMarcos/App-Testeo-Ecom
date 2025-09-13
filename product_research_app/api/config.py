from flask import request, jsonify, current_app
from . import app

from product_research_app.services.winner_score import (
    recompute_scores_for_all_products,
    load_settings,
    save_settings,
)
from product_research_app.services.config import compute_effective_int


def _coerce_weights(raw: dict | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for k, v in (raw or {}).items():
        try:
            iv = int(round(float(v)))
        except Exception:
            iv = 0
        out[k] = max(0, min(100, iv))
    return out


def _merge_winner_weights(current: dict | None, incoming: dict | None) -> dict:
    cur = dict(current or {})
    cur.update(incoming or {})
    return cur


# GET /api/config/winner-weights
@app.route("/api/config/winner-weights", methods=["GET"])
def api_get_winner_weights():
    settings = load_settings()
    weights = settings.get("winner_weights") or {}
    order = settings.get("winner_order") or list(weights.keys())
    eff_int = compute_effective_int(weights, order)
    resp = jsonify(
        {
            "winner_weights": weights,
            "winner_order": order,
            "weights": weights,
            "order": order,
            "effective": {"int": eff_int},
        }
    )
    resp.headers["Cache-Control"] = "no-store"
    return resp, 200


# PATCH /api/config/winner-weights
@app.route("/api/config/winner-weights", methods=["PATCH"])
def api_patch_winner_weights():
    payload = request.get_json(force=True) or {}
    incoming = _coerce_weights(
        payload.get("winner_weights")
        or payload.get("weights")
        or payload
    )

    settings = load_settings()
    current = _coerce_weights(settings.get("winner_weights"))

    if "awareness" not in incoming and "awareness" not in current:
        incoming["awareness"] = 50

    settings["winner_weights"] = _merge_winner_weights(current, incoming)

    order_in = (
        payload.get("winner_order")
        or payload.get("order")
        or settings.get("winner_order")
        or list(settings["winner_weights"].keys())
    )
    order = [k for k in order_in if k in settings["winner_weights"]]
    for k in settings["winner_weights"].keys():
        if k not in order:
            order.append(k)
    if "awareness" not in order:
        order.append("awareness")
    settings["winner_order"] = order

    save_settings(settings)

    updated = 0
    try:
        updated = recompute_scores_for_all_products(scope="all")
    except Exception as e:
        current_app.logger.warning("recompute on save failed: %s", e)

    resp = jsonify(
        {
            "ok": True,
            "updated": updated,
            "winner_weights": settings["winner_weights"],
            "winner_order": order,
            "weights": settings["winner_weights"],
            "order": order,
        }
    )
    resp.headers["Cache-Control"] = "no-store"
    return resp, 200
