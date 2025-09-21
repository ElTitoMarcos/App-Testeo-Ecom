from flask import request, jsonify, current_app
import time

from . import app

from product_research_app.services.winner_score import (
    recompute_scores_for_all_products,
    load_settings,
    save_settings,
    invalidate_weights_cache,
)
from product_research_app.services.config import (
    ALLOWED_FIELDS,
    compute_effective_int,
    get_default_winner_weights,
)
from product_research_app.config import DEFAULT_WINNER_ORDER


def _coerce_weights(raw: dict | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for k, v in (raw or {}).items():
        if k not in ALLOWED_FIELDS:
            continue
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


def _normalize_order(order, weights: dict[str, int]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for key in order or []:
        if key in weights and key not in seen:
            out.append(key)
            seen.add(key)
    for key in DEFAULT_WINNER_ORDER:
        if key in weights and key not in seen:
            out.append(key)
            seen.add(key)
    for key in weights.keys():
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


def _sanitize_enabled(raw_enabled, keys: list[str]) -> dict[str, bool]:
    if not isinstance(raw_enabled, dict):
        return {k: True for k in keys}
    return {k: bool(raw_enabled.get(k, True)) for k in keys}

def _apply_reset(settings: dict | None = None) -> dict:
    cfg = dict(settings or load_settings() or {})
    default_weights = get_default_winner_weights()
    cfg["winner_weights"] = dict(default_weights)
    order = list(DEFAULT_WINNER_ORDER)
    cfg["winner_order"] = order[:]
    cfg["weights_order"] = order[:]
    raw_enabled = cfg.get("weights_enabled")
    if isinstance(raw_enabled, dict):
        enabled = {k: bool(raw_enabled.get(k, True)) for k in default_weights.keys()}
    else:
        enabled = {k: True for k in default_weights.keys()}
    cfg["weights_enabled"] = enabled
    cfg["weightsUpdatedAt"] = int(time.time())
    save_settings(cfg)
    try:
        invalidate_weights_cache()
    except Exception as exc:  # pragma: no cover - defensive logging
        current_app.logger.warning("reset invalidate failed: %s", exc)
    return cfg


def _build_reset_payload(cfg: dict) -> dict:
    payload = {
        "ok": True,
        "weights": dict(cfg.get("winner_weights", {})),
        "winner_weights": dict(cfg.get("winner_weights", {})),
        "order": list(cfg.get("winner_order", [])),
        "winner_order": list(cfg.get("winner_order", [])),
        "weights_order": list(cfg.get("weights_order", [])),
        "weights_enabled": dict(cfg.get("weights_enabled", {})),
    }
    for key in ("weightsUpdatedAt", "weightsVersion"):
        if key in cfg:
            payload[key] = cfg[key]
    return payload

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
    changed = False
    weights = _coerce_weights(settings.get("winner_weights"))
    if not weights:
        weights = {k: 50 for k in DEFAULT_WINNER_ORDER}
    else:
        for key in DEFAULT_WINNER_ORDER:
            weights.setdefault(key, 50)
    if weights != settings.get("winner_weights"):
        settings["winner_weights"] = dict(weights)
        changed = True

    order = settings.get("winner_order")
    if not isinstance(order, list) or len(order) != 8:
        order = list(DEFAULT_WINNER_ORDER)
        settings["winner_order"] = order[:]
        settings["weights_order"] = order[:]
        changed = True

    weights_order = settings.get("weights_order")
    if not isinstance(weights_order, list) or len(weights_order) != 8:
        settings["weights_order"] = order[:]
        weights_order = settings["weights_order"]
        changed = True

    order = _normalize_order(weights_order, weights)
    if order != settings.get("winner_order"):
        settings["winner_order"] = order[:]
        settings["weights_order"] = order[:]
        changed = True

    weights_enabled = _sanitize_enabled(settings.get("weights_enabled"), order)
    if weights_enabled != settings.get("weights_enabled"):
        settings["weights_enabled"] = weights_enabled
        changed = True

    if changed:
        save_settings(settings)

    weights_eff = {
        k: (weights.get(k, 0) if weights_enabled.get(k, True) else 0)
        for k in weights.keys()
    }
    eff = compute_effective_int(weights_eff, order)
    current_app.logger.info(
        "CONFIG served weights_order=%s", settings.get("weights_order")
    )
    resp = jsonify({
        **weights,
        "weights": weights,
        "order": order,
        "effective": {"int": eff},
        "weights_enabled": weights_enabled,
        "weights_order": order,
    })
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp, 200


# PATCH /api/config/winner-weights
@app.route("/api/config/winner-weights", methods=["PATCH"])
def api_patch_winner_weights():
    body = request.get_json(force=True) or {}
    if body.get("reset"):
        cfg = _apply_reset()
        current_app.logger.info(
            "RESET applied weights_order=%s", cfg.get("weights_order")
        )
        payload = _build_reset_payload(cfg)
        resp = jsonify(payload)
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp, 200
    payload_map = (
        body.get("winner_weights")
        or body.get("weights")
        or {k: v for k, v in body.items() if k in ALLOWED_FIELDS}
    )
    incoming = _coerce_weights(payload_map)

    settings = load_settings()
    current = _coerce_weights(settings.get("winner_weights"))
    merged = _merge_winner_weights(current, incoming)
    for key in DEFAULT_WINNER_ORDER:
        merged.setdefault(key, merged.get(key, 50))
    settings["winner_weights"] = merged

    order_in = body.get("order") or body.get("weights_order")
    if not isinstance(order_in, list) or not order_in:
        order_in = settings.get("winner_order") or list(DEFAULT_WINNER_ORDER)
    order = _normalize_order(order_in, merged)
    settings["winner_order"] = order[:]
    settings["weights_order"] = order[:]

    en_in = body.get("weights_enabled")
    if isinstance(en_in, dict):
        weights_enabled = _sanitize_enabled(en_in, order)
    else:
        weights_enabled = _sanitize_enabled(settings.get("weights_enabled"), order)
    settings["weights_enabled"] = weights_enabled

    settings["weightsUpdatedAt"] = int(time.time())
    save_settings(settings)
    invalidate_weights_cache()

    updated = 0
    try:
        updated = recompute_scores_for_all_products(scope="all")
    except Exception as e:
        current_app.logger.warning("recompute on save failed: %s", e)

    resp_payload = dict(settings)
    resp_payload.pop("api_key", None)
    resp_payload.update(
        {
            "weights": dict(settings["winner_weights"]),
            "order": list(order),
            "weights_order": list(order),
            "ok": True,
            "updated": updated,
        }
    )
    resp = jsonify(resp_payload)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp, 200


# POST /api/config/winner-weights/reset
@app.route("/api/config/winner-weights/reset", methods=["POST"])
def api_reset_winner_weights():
    cfg = _apply_reset()
    current_app.logger.info(
        "RESET applied weights_order=%s", cfg.get("weights_order")
    )
    payload = _build_reset_payload(cfg)
    resp = jsonify(payload)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp, 200
