from flask import request, jsonify
from . import app

# GET /api/config/weights
@app.route("/api/config/weights", methods=["GET"])
def api_get_weights():
    from product_research_app.services.config import get_winner_weights
    return jsonify({"weights": get_winner_weights()})

# PUT /api/config/weights
@app.route("/api/config/weights", methods=["PUT"])
def api_put_weights():
    from product_research_app.services.config import set_winner_weights
    data = request.get_json(force=True) or {}
    saved = set_winner_weights(data.get("weights", {}))
    return jsonify({"weights": saved})
