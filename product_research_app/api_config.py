from flask import Blueprint, jsonify, request
from .config_store import read_config, write_config, get_api_key

bp = Blueprint("config_api", __name__, url_prefix="/api/config")

@bp.get("")
def get_basic_config():
    return jsonify({"has_api_key": bool(get_api_key())})

@bp.post("/api-key")
def set_api_key():
    payload = request.get_json(silent=True) or {}
    api_key = (payload.get("api_key") or "").strip()
    if not api_key:
        return jsonify({"ok": False, "error": "API key vacía"}), 400
    cfg = read_config()
    cfg["openai_api_key"] = api_key
    write_config(cfg)
    return jsonify({"ok": True})
