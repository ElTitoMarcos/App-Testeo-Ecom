from flask import request, jsonify, current_app
from . import app

from product_research_app.config import DEFAULT_ORDER, ensure_winner_order
from product_research_app.services.winner_score import (
    recompute_scores_for_all_products,
    load_settings,
    save_settings,
    invalidate_weights_cache,
)
from product_research_app.services.config import (
    ALLOWED_FIELDS,
    compute_effective_int,
)


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


"""Endpoints for winner weight configuration.

These routes must maintain backwards compatibility with older clients that
expect a flat mapping of weight keys, while newer UIs may send or receive a
``{"weights": {...}, "order": [...]}`` structure.  The GET endpoint therefore
returns both the legacy flat map and the richer wrapper.  The PATCH endpoint
accepts any of the shapes described above.
"""


# GET /api/config/winner-weights
@app.route("/api/config/winner-weights", methods=["GET"])
def api_get_winner_weights():
    settings = load_settings()
    prev_order = list(settings.get("winner_order", [])) if isinstance(settings.get("winner_order"), list) else None
    ensure_winner_order(settings)
    if prev_order != settings.get("winner_order"):
        save_settings(settings)
    weights = _coerce_weights(settings.get("winner_weights"))
    order = settings.get("winner_order") or list(DEFAULT_ORDER)
    enabled_raw = settings.get("weights_enabled") if isinstance(settings.get("weights_enabled"), dict) else {}
    enabled = {k: bool(enabled_raw.get(k, True)) for k in weights.keys()}
    weights_eff = {k: (weights.get(k, 0) if enabled.get(k, True) else 0) for k in weights.keys()}
    eff = compute_effective_int(weights_eff, order)
    resp = jsonify({
        **weights,
        "weights": weights,
        "order": order,
        "effective": {"int": eff},
        "weights_enabled": enabled,
        "weights_order": settings.get("weights_order") or order,
    })
    current_app.logger.info("winner_order served = %s", order)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp, 200


# PATCH /api/config/winner-weights
@app.route("/api/config/winner-weights", methods=["PATCH"])
def api_patch_winner_weights():
    body = request.get_json(force=True) or {}
    payload_map = (
        body.get("winner_weights")
        or body.get("weights")
        or {k: v for k, v in body.items() if k in ALLOWED_FIELDS}
    )
    incoming = _coerce_weights(payload_map)

    settings = load_settings()
    current = _coerce_weights(settings.get("winner_weights"))
    if "awareness" not in incoming and "awareness" not in current:
        incoming["awareness"] = 50
    settings["winner_weights"] = _merge_winner_weights(current, incoming)

    order_in = body.get("order") or body.get("weights_order")
    if isinstance(order_in, list):
        order = [k for k in order_in if k in ALLOWED_FIELDS]
    else:
        order = settings.get("winner_order") or list(settings["winner_weights"].keys())
    if "awareness" not in order:
        order.append("awareness")
    settings["winner_order"] = order
    settings["weights_order"] = order

    en_in = body.get("weights_enabled")
    if isinstance(en_in, dict):
        current_en = settings.get("weights_enabled", {})
        enabled = {k: bool(en_in.get(k, current_en.get(k, True))) for k in ALLOWED_FIELDS}
        settings["weights_enabled"] = enabled

    ensure_winner_order(settings)
    save_settings(settings)
    invalidate_weights_cache()

    updated = 0
    try:
        updated = recompute_scores_for_all_products(scope="all")
    except Exception as e:
        current_app.logger.warning("recompute on save failed: %s", e)

    resp = jsonify({"ok": True, "winner_weights": settings["winner_weights"], "winner_order": order, "updated": updated})
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp, 200
