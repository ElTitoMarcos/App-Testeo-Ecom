from flask import request, jsonify
from . import app

from product_research_app.services.config import get_winner_weights_raw, set_winner_weights_raw


# GET /api/config/winner-weights
@app.route("/api/config/winner-weights", methods=["GET"])
def api_get_winner_weights():
    raw = get_winner_weights_raw()
    return jsonify({"weights": raw})


# PATCH /api/config/winner-weights
@app.route("/api/config/winner-weights", methods=["PATCH"])
def api_patch_winner_weights():
    data = request.get_json(force=True) or {}
    raw_in = data.get("weights", {}) or {}
    saved = set_winner_weights_raw(raw_in)
    return jsonify({"weights": saved})
