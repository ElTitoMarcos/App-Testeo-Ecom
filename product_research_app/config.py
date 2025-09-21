"""Configuration management for Product Research Copilot.

The application stores user configuration such as the OpenAI API key and
preferred model in a JSON file (config.json) located in the application's
directory. These helpers encapsulate loading and saving this configuration
file. If the file does not exist, default values are returned.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_WINNER_ORDER = [
    "awareness",
    "desire",
    "revenue",
    "competition",
    "units_sold",
    "price",
    "oldness",
    "rating",
]


DEFAULT_CONFIG: Dict[str, Any] = {
    "autoFillIAOnImport": True,
    "aiBatch": {
        "BATCH_SIZE": 10,
        "MAX_CONCURRENCY": 2,
        "MAX_RETRIES": 3,
        "TIME_LIMIT_SECONDS": 300,
    },
    "aiCost": {
        "model": "gpt-4.1-mini",
        "useBatchWhenCountGte": 300,
        "costCapUSD": 5.0,
        "estTokensPerItemIn": 300,
        "estTokensPerItemOut": 80,
    },
    "aiCalibration": {
        "enabled": True,
        "mode": "terciles",
        "winsorize_pct": 0.05,
        "min_low_pct": 0.05,
        "min_medium_pct": 0.05,
        "min_high_pct": 0.05,
    },
    "ai": {
        "model": "gpt-4o-mini",
        "parallelism": 32,
        "microbatch": 512,
        "temperature": 0.0,
        "topP": 0.1,
        "maxOutputTokensPerItem": 8,
        "truncate": {"title": 120, "description": 240},
        "enableCache": True,
    },
    "includeImageInAI": True,
    "aiImageCostMaxUSD": 0.02,
    "weightsVersion": 0,
    "weightsUpdatedAt": 0,
    "oldness_preference": "newer",
}


CONFIG_FILE = Path(__file__).resolve().parent / "config.json"


def load_config() -> Dict[str, Any]:
    """Load configuration from disk.

    Returns a dictionary with at least the keys ``api_key`` and ``model``.
    If the file does not exist, an empty configuration is returned.
    """

    data: Dict[str, Any] = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                data = raw
        except Exception:
            data = {}

    changed = False
    if "weights" not in data and isinstance(data.get("weights_v2"), dict):
        data["weights"] = data["weights_v2"]
        changed = True
    for obsolete in ["weights_v1", "use_v1_by_default", "winner_score_version"]:
        if obsolete in data:
            data.pop(obsolete, None)
            changed = True
    if "weights_v2" in data:
        data.pop("weights_v2", None)
        changed = True
    if _merge_defaults(data, DEFAULT_CONFIG):
        changed = True

    # ensure weights_enabled defaulting to True for all weight keys
    weights_map = data.get("winner_weights")
    if isinstance(weights_map, dict):
        enabled = data.get("weights_enabled")
        if not isinstance(enabled, dict):
            data["weights_enabled"] = {k: True for k in weights_map.keys()}
            changed = True
        else:
            for k in weights_map.keys():
                if k not in enabled:
                    enabled[k] = True
                    changed = True
    winner_order = data.get("winner_order")
    if not isinstance(winner_order, list) or not winner_order:
        data["winner_order"] = list(DEFAULT_WINNER_ORDER)
        winner_order = data["winner_order"]
        changed = True

    if "weights_order" not in data and isinstance(winner_order, list):
        data["weights_order"] = list(winner_order)
        changed = True

    if changed:
        save_config(data)
    return data


def save_config(config: Dict[str, Any]) -> None:
    """Persist configuration to disk atomically."""

    tmp_path = CONFIG_FILE.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    tmp_path.replace(CONFIG_FILE)


def _merge_defaults(dst: Dict[str, Any], src: Dict[str, Any]) -> bool:
    changed = False
    for k, v in src.items():
        if k not in dst:
            dst[k] = v
            changed = True
        elif isinstance(v, dict) and isinstance(dst.get(k), dict):
            if _merge_defaults(dst[k], v):
                changed = True
    return changed


def get_api_key() -> Optional[str]:
    """Return the configured OpenAI API key giving precedence to the environment."""

    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    config = load_config()
    return config.get("api_key")


def get_model() -> str:
    """Return the configured model or default to 'gpt-4o'."""

    config = load_config()
    model = config.get("model")
    if not model:
        ai_cfg = config.get("ai") or {}
        if isinstance(ai_cfg, dict):
            model = ai_cfg.get("model")
    if not model:
        return "gpt-4o-mini"
    return model


def get_ai_batch_config() -> Dict[str, Any]:
    cfg = load_config()
    base = DEFAULT_CONFIG["aiBatch"].copy()
    base.update(cfg.get("aiBatch", {}))
    return base


def get_ai_cost_config() -> Dict[str, Any]:
    cfg = load_config()
    base = DEFAULT_CONFIG["aiCost"].copy()
    user = cfg.get("aiCost", {})
    for k, v in user.items():
        if isinstance(v, dict) and k in base:
            tmp = base.get(k, {}).copy()
            tmp.update(v)
            base[k] = tmp
        else:
            base[k] = v
    return base


def get_ai_calibration_config() -> Dict[str, Any]:
    cfg = load_config()
    base = DEFAULT_CONFIG["aiCalibration"].copy()
    user = cfg.get("aiCalibration", {})
    for k, v in user.items():
        if isinstance(v, dict) and k in base:
            tmp = base.get(k, {}).copy()
            tmp.update(v)
            base[k] = tmp
        else:
            base[k] = v
    return base


def get_ai_runtime_config() -> Dict[str, Any]:
    cfg = load_config()
    base = DEFAULT_CONFIG["ai"].copy()
    user = cfg.get("ai", {})
    if isinstance(user, dict):
        for k, v in user.items():
            base[k] = v

    cpu_parallel = max(1, (os.cpu_count() or 1) * 2)
    default_parallel = min(32, cpu_parallel)
    try:
        parallel = int(base.get("parallelism") or 0)
    except Exception:
        parallel = default_parallel
    if parallel <= 0:
        parallel = default_parallel
    base["parallelism"] = max(1, parallel)

    try:
        micro = int(base.get("microbatch") or 0)
    except Exception:
        micro = DEFAULT_CONFIG["ai"]["microbatch"]
    if micro <= 0:
        micro = DEFAULT_CONFIG["ai"]["microbatch"]
    base["microbatch"] = max(1, micro)

    try:
        temp_val = float(base.get("temperature", DEFAULT_CONFIG["ai"]["temperature"]))
    except Exception:
        temp_val = DEFAULT_CONFIG["ai"]["temperature"]
    base["temperature"] = max(0.0, temp_val)

    try:
        top_p = float(base.get("topP", DEFAULT_CONFIG["ai"]["topP"]))
    except Exception:
        top_p = DEFAULT_CONFIG["ai"]["topP"]
    base["topP"] = max(0.0, min(1.0, top_p))

    try:
        per_item = int(base.get("maxOutputTokensPerItem", DEFAULT_CONFIG["ai"]["maxOutputTokensPerItem"]))
    except Exception:
        per_item = DEFAULT_CONFIG["ai"]["maxOutputTokensPerItem"]
    if per_item <= 0:
        per_item = DEFAULT_CONFIG["ai"]["maxOutputTokensPerItem"]
    base["maxOutputTokensPerItem"] = max(1, per_item)

    truncate_cfg = base.get("truncate") or {}
    if not isinstance(truncate_cfg, dict):
        truncate_cfg = {}
    trunc_title = truncate_cfg.get("title", DEFAULT_CONFIG["ai"]["truncate"]["title"])
    trunc_desc = truncate_cfg.get("description", DEFAULT_CONFIG["ai"]["truncate"]["description"])
    try:
        trunc_title = int(trunc_title)
    except Exception:
        trunc_title = DEFAULT_CONFIG["ai"]["truncate"]["title"]
    try:
        trunc_desc = int(trunc_desc)
    except Exception:
        trunc_desc = DEFAULT_CONFIG["ai"]["truncate"]["description"]
    base["truncate"] = {
        "title": max(10, trunc_title),
        "description": max(20, trunc_desc),
    }

    base["enableCache"] = bool(base.get("enableCache", True))

    return base


def is_auto_fill_ia_on_import_enabled() -> bool:
    cfg = load_config()
    try:
        return bool(cfg.get("autoFillIAOnImport", True))
    except Exception:
        return True


def include_image_in_ai() -> bool:
    cfg = load_config()
    try:
        return bool(cfg.get("includeImageInAI", True))
    except Exception:
        return True


def get_ai_image_cost_max_usd() -> float:
    cfg = load_config()
    try:
        return float(cfg.get("aiImageCostMaxUSD", 0.02))
    except Exception:
        return 0.02


SCORING_DEFAULT_WEIGHTS: Dict[str, float] = {
    "price": 1.0,
    "rating": 1.0,
    "units_sold": 1.0,
    "revenue": 1.0,
    "desire": 1.0,
    "competition": 1.0,
    "oldness": 1.0,
    "awareness": 1.0,
}


def get_weights() -> Dict[str, float]:
    """Return Winner Score weights as raw integers (0-100)."""

    from .services.config import get_winner_weights_raw  # lazy import

    return get_winner_weights_raw()


def set_weights(weights: Dict[str, float]) -> None:
    """Persist Winner Score weights (RAW)."""

    from .services.config import set_winner_weights_raw  # lazy import
    from .services import winner_score  # lazy import

    set_winner_weights_raw(weights)
    winner_score.invalidate_weights_cache()


def update_weight(key: str, value: float) -> None:
    """Update a single Winner Score weight and persist immediately."""

    from .services.config import set_winner_weights_raw, ALLOWED_FIELDS  # lazy import
    from .services import winner_score  # lazy import

    k = str(key or "").strip()
    if k.endswith("_weight"):
        k = k[:-7]
    if k not in ALLOWED_FIELDS:
        raise ValueError(f"Invalid weight key: {key}")
    set_winner_weights_raw({k: value})
    winner_score.invalidate_weights_cache()


def get_weights_version() -> int:
    cfg = load_config()
    try:
        return int(cfg.get("weightsUpdatedAt", 0))
    except Exception:
        return 0
