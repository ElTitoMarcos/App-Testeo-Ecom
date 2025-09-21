import os

from flask import request, jsonify, current_app
from . import app

from product_research_app.services.winner_score import (
    recompute_scores_for_all_products,
    load_settings,
    save_settings,
    invalidate_weights_cache,
)
from product_research_app import config as app_config, gpt
from product_research_app.services.config import (
    ALLOWED_FIELDS,
    compute_effective_int,
    DEFAULT_ORDER,
    DEFAULT_WEIGHTS_RAW,
    validate_order,
)


def _coerce_weights(raw: dict | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for k, v in (raw or {}).items():
        if k not in ALLOWED_FIELDS:
            continue
        try:
            iv = int(round(float(v)))
        except Exception:
            iv = 0
        if iv < 0:
            iv = 0
        if iv > 100:
            iv = 100
        if iv > 50:
            iv = int(round(iv * 0.5))
        out[k] = max(0, min(50, iv))
    return out


def _merge_winner_weights(current: dict | None, incoming: dict | None) -> dict:
    cur = dict(current or {})
    cur.update(incoming or {})
    return cur


def _load_weights_state():
    settings = load_settings()
    base_weights = {k: DEFAULT_WEIGHTS_RAW.get(k, 0) for k in ALLOWED_FIELDS}
    weights = base_weights.copy()
    weights.update(_coerce_weights(settings.get("winner_weights")))
    try:
        order = validate_order(settings.get("winner_order"))
    except ValueError:
        order = DEFAULT_ORDER.copy()
    enabled = {k: bool((settings.get("weights_enabled") or {}).get(k, True)) for k in ALLOWED_FIELDS}
    return settings, weights, order, enabled


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
    settings, weights, order, enabled = _load_weights_state()
    weights_eff = {k: (weights.get(k, 0) if enabled.get(k, True) else 0) for k in ALLOWED_FIELDS}
    eff = compute_effective_int(weights_eff, order)
    payload = {k: weights.get(k, 0) for k in ALLOWED_FIELDS}
    resp = jsonify({
        **payload,
        "weights": payload,
        "order": order,
        "effective": {"int": eff},
        "weights_enabled": enabled,
        "weights_order": settings.get("weights_order") or order,
        "version": "v2",
    })
    resp.headers["Cache-Control"] = "no-store"
    return resp, 200


# POST /api/config/winner-weights/ai
@app.route("/api/config/winner-weights/ai", methods=["POST"])
def api_ai_winner_weights():
    body = request.get_json(force=True) or {}
    can_reorder = str(request.args.get("can_reorder", "false")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    data_sample = body.get("data_sample") or []
    target = (body.get("target") or "").strip()
    if not data_sample or not target:
        return jsonify({"error": "insufficient_data"}), 400

    features = [f for f in (body.get("features") or ALLOWED_FIELDS) if f in ALLOWED_FIELDS]
    samples: list[dict[str, float]] = []
    for row in data_sample:
        if not isinstance(row, dict):
            continue
        sample: dict[str, float] = {}
        for key in features:
            try:
                sample[key] = float(row.get(key, 0.0))
            except Exception:
                sample[key] = 0.0
        try:
            sample["target"] = float(row.get("target", 0.0))
        except Exception:
            sample["target"] = 0.0
        samples.append(sample)

    if not samples:
        return jsonify({"error": "insufficient_data"}), 400

    api_key = app_config.get_api_key() or os.environ.get("OPENAI_API_KEY")
    model = app_config.get_model()
    if not api_key or not model:
        return jsonify({"error": "missing_api_key"}), 400

    try:
        result = gpt.recommend_winner_weights(api_key, model, samples, target)
    except Exception as exc:  # pragma: no cover - defensive
        current_app.logger.warning("AI winner weights failed: %s", exc)
        return jsonify({"error": "ai_error", "message": str(exc)}), 500

    recommended = _coerce_weights(result.get("weights"))
    settings, current_weights, current_order, current_enabled = _load_weights_state()
    weights = {k: DEFAULT_WEIGHTS_RAW.get(k, 0) for k in ALLOWED_FIELDS}
    weights.update(current_weights)
    weights.update(recommended)

    order_candidate = result.get("order") if isinstance(result, dict) else None
    proposed_order = None
    if isinstance(order_candidate, list):
        try:
            proposed_order = validate_order(order_candidate)
        except ValueError:
            proposed_order = None
    if proposed_order is None:
        pos = {k: idx for idx, k in enumerate(current_order)}
        proposed_order = sorted(
            ALLOWED_FIELDS,
            key=lambda key: (-weights.get(key, 0), pos.get(key, len(ALLOWED_FIELDS))),
        )

    updated = 0
    persisted = False
    diagnostics = {"notes": result.get("justification", "") if isinstance(result, dict) else ""}

    if can_reorder:
        weight_changes = {
            k: (current_weights.get(k, 0), weights.get(k, 0))
            for k in ALLOWED_FIELDS
            if current_weights.get(k, 0) != weights.get(k, 0)
        }
        previous_positions = {k: idx for idx, k in enumerate(current_order)}
        order_moves = []
        for idx, key in enumerate(proposed_order):
            prev_idx = previous_positions.get(key)
            if prev_idx is not None and prev_idx != idx:
                order_moves.append({"metric": key, "from": prev_idx, "to": idx})
        if weight_changes or order_moves:
            current_app.logger.info(
                "winner_weights.diff source=ai weights=%s order=%s",
                weight_changes,
                order_moves,
            )
        settings["winner_weights"] = {k: weights.get(k, 0) for k in ALLOWED_FIELDS}
        settings["winner_weights_schema_version"] = 2
        settings["winner_order"] = proposed_order
        settings["weights_order"] = proposed_order
        settings["weights_enabled"] = current_enabled
        save_settings(settings)
        invalidate_weights_cache()
        try:
            updated = recompute_scores_for_all_products(scope="all")
        except Exception as exc:  # pragma: no cover - defensive
            current_app.logger.warning("recompute on ai save failed: %s", exc)
        persisted = True
    weights_eff = {k: (weights.get(k, 0) if current_enabled.get(k, True) else 0) for k in ALLOWED_FIELDS}
    order_for_effect = proposed_order if can_reorder else current_order
    effective = compute_effective_int(weights_eff, order_for_effect)
    payload = {k: weights.get(k, 0) for k in ALLOWED_FIELDS}
    stored_order = proposed_order if can_reorder else current_order
    response = {
        "weights": payload,
        "winner_weights": payload,
        "order": proposed_order,
        "winner_order": stored_order,
        "weights_enabled": current_enabled,
        "weights_order": stored_order,
        "effective": {"int": effective},
        "version": "v2",
        "method": result.get("method", "gpt") if isinstance(result, dict) else "gpt",
        "diagnostics": diagnostics,
        "updated": updated,
        "persisted": persisted,
    }
    return jsonify(response), 200


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

    settings, current_weights, current_order, current_enabled = _load_weights_state()
    new_weights = current_weights.copy()
    new_weights.update(incoming)
    for key in ALLOWED_FIELDS:
        new_weights.setdefault(key, DEFAULT_WEIGHTS_RAW.get(key, 0))

    order_in = body.get("order") or body.get("weights_order")
    order_provided = isinstance(order_in, list)
    if order_provided:
        try:
            order = validate_order(order_in)
        except ValueError:
            return jsonify({"error": "invalid_order"}), 400
    else:
        order = current_order

    en_in = body.get("weights_enabled")
    if isinstance(en_in, dict):
        enabled = {k: bool(en_in.get(k, current_enabled.get(k, True))) for k in ALLOWED_FIELDS}
    else:
        enabled = current_enabled

    weight_changes = {
        k: (current_weights.get(k, 0), new_weights.get(k, 0))
        for k in ALLOWED_FIELDS
        if current_weights.get(k, 0) != new_weights.get(k, 0)
    }
    order_moves = []
    if order_provided:
        previous_positions = {k: idx for idx, k in enumerate(current_order)}
        for idx, key in enumerate(order):
            prev_idx = previous_positions.get(key)
            if prev_idx is not None and prev_idx != idx:
                order_moves.append({"metric": key, "from": prev_idx, "to": idx})
    if weight_changes or order_moves:
        current_app.logger.info(
            "winner_weights.diff weights=%s order=%s",
            weight_changes,
            order_moves,
        )

    settings["winner_weights"] = new_weights
    settings["winner_weights_schema_version"] = 2
    settings["winner_order"] = order
    settings["weights_order"] = order
    settings["weights_enabled"] = enabled

    save_settings(settings)
    invalidate_weights_cache()

    updated = 0
    try:
        updated = recompute_scores_for_all_products(scope="all")
    except Exception as e:
        current_app.logger.warning("recompute on save failed: %s", e)

    weights_eff = {k: (new_weights.get(k, 0) if enabled.get(k, True) else 0) for k in ALLOWED_FIELDS}
    eff = compute_effective_int(weights_eff, order)
    resp_payload = {k: new_weights.get(k, 0) for k in ALLOWED_FIELDS}
    resp = jsonify(
        {
            "ok": True,
            **resp_payload,
            "weights": resp_payload,
            "winner_weights": new_weights,
            "winner_order": order,
            "order": order,
            "weights_enabled": enabled,
            "weights_order": order,
            "effective": {"int": eff},
            "version": "v2",
            "updated": updated,
        }
    )
    resp.headers["Cache-Control"] = "no-store"
    return resp, 200
