from flask import request, jsonify, current_app
from . import app

from product_research_app.services.config import (
    get_winner_weights_raw,
    get_winner_order_raw,
    update_winner_settings,
    compute_effective_int,
)
from product_research_app.services.winner_score import recompute_scores_for_all_products


# GET /api/config/winner-weights
@app.route("/api/config/winner-weights", methods=["GET"])
def api_get_winner_weights():
    raw = get_winner_weights_raw()
    order = get_winner_order_raw()
    resp = jsonify({
        "weights": raw,
        "order": order,
        "effective": {"int": compute_effective_int(raw, order)},
    })
    resp.headers["Cache-Control"] = "no-store"
    return resp


# PATCH /api/config/winner-weights
@app.route("/api/config/winner-weights", methods=["PATCH"])
def api_patch_winner_weights():
    data = request.get_json(force=True) or {}
    raw_in = data.get("winner_weights") or data.get("weights") or {}
    order_in = data.get("winner_order") or data.get("order")
    saved_weights, saved_order = update_winner_settings(raw_in, order_in)
    try:
        recompute_scores_for_all_products(async_ok=True)
    except Exception as e:
        current_app.logger.warning(f"winner-score recompute deferred: {e}")
    app.logger.info(
        "settings_saved winner_weights=%s winner_order_len=%s",
        len(saved_weights),
        len(saved_order),
    )
    resp = jsonify({
        "weights": saved_weights,
        "order": saved_order,
        "effective": {"int": compute_effective_int(saved_weights, saved_order)},
    })
    resp.headers["Cache-Control"] = "no-store"
    return resp
