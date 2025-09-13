from flask import Blueprint, request, jsonify
from product_research_app.services.winner_score import recompute_scores_for_all_products

# Dedicated blueprint for winner score operations.  It will be mounted under
# the ``/api`` prefix by the application factory.
winner_score_api = Blueprint("winner_score_api", __name__)

@winner_score_api.route("/winner-score/recompute", methods=["POST"])
def recompute_api():
    scope = (request.get_json(silent=True) or {}).get("scope", "all")
    n = recompute_scores_for_all_products(scope=scope)
    return jsonify({"updated": n}), 200
