from flask import request, jsonify
from . import app

from product_research_app.services.config import (
    get_winner_weights_raw,
    set_winner_weights_raw,
    get_winner_order_raw,
    set_winner_order_raw,
    compute_effective_int,
)


# GET /api/config/winner-weights
@app.route("/api/config/winner-weights", methods=["GET"])
def api_get_winner_weights():
    raw = get_winner_weights_raw()
    order = get_winner_order_raw()
    return jsonify({
        "weights": raw,
        "order": order,
        "effective": {"int": compute_effective_int(raw, order)},
    })


# PATCH /api/config/winner-weights
@app.route("/api/config/winner-weights", methods=["PATCH"])
def api_patch_winner_weights():
    data = request.get_json(force=True) or {}
    raw_in = data.get("weights", {}) or {}
    order_in = data.get("order")
    saved_weights = set_winner_weights_raw(raw_in)
    if order_in is None:
        order_in = list(saved_weights.keys())
    saved_order = set_winner_order_raw(order_in)
    return jsonify({
        "weights": saved_weights,
        "order": saved_order,
        "effective": {"int": compute_effective_int(saved_weights, saved_order)},
    })
