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


AI_MICROBATCH_DEFAULT = 8
AI_MAX_OUTPUT_TOKENS = 3000
AI_JSON_ENFORCED = True


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
        "costCapUSD": 0.25,
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
        "fallback_percentiles": {
            "desire": [0.33, 0.66],
            "competition": [0.33, 0.66],
        },
    },
    "ai": {
        "parallelism": 8,
        "microbatch": AI_MICROBATCH_DEFAULT,
        "cache_enabled": True,
        "version": 1,
        "tpm_limit": None,
        "rpm_limit": 150,
        "timeout": 45,
        "trunc_title": 180,
        "trunc_desc": 800,
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
    """Return the stored OpenAI API key if present."""

    config = load_config()
    return config.get("api_key")


def get_model() -> str:
    """Return the configured model or default to 'gpt-4o'."""

    config = load_config()
    model = config.get("model")
    if not model:
        return "gpt-4o"
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

    def _env_int(name: str) -> Optional[int]:
        raw = os.environ.get(name)
        if raw is None:
            return None
        try:
            return int(float(raw))
        except Exception:
            return None

    def _env_float(name: str) -> Optional[float]:
        raw = os.environ.get(name)
        if raw is None:
            return None
        try:
            return float(raw)
        except Exception:
            return None

    env_micro = _env_int("AI_MICROBATCH")
    if env_micro is not None:
        base["microbatch"] = env_micro

    env_parallel = _env_int("AI_PARALLELISM")
    if env_parallel is not None:
        base["parallelism"] = env_parallel

    env_timeout = _env_float("AI_TIMEOUT")
    if env_timeout is not None:
        base["timeout"] = env_timeout

    env_rpm = _env_int("AI_RPM")
    if env_rpm is not None:
        base["rpm_limit"] = env_rpm

    env_tpm = _env_int("AI_TPM")
    if env_tpm is not None:
        base["tpm_limit"] = env_tpm

    env_trunc_title = _env_int("AI_TRUNC_TITLE")
    if env_trunc_title is not None:
        base["trunc_title"] = env_trunc_title

    env_trunc_desc = _env_int("AI_TRUNC_DESC")
    if env_trunc_desc is not None:
        base["trunc_desc"] = env_trunc_desc

    cpu_parallel = max(1, (os.cpu_count() or 1) * 2)
    default_parallel = min(8, cpu_parallel)
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

    base["cache_enabled"] = bool(base.get("cache_enabled", True))
    try:
        base["version"] = int(base.get("version", DEFAULT_CONFIG["ai"]["version"]))
    except Exception:
        base["version"] = DEFAULT_CONFIG["ai"]["version"]

    tpm_limit = base.get("tpm_limit")
    if tpm_limit is None:
        base["tpm_limit"] = None
    else:
        try:
            limit_val = int(tpm_limit)
        except Exception:
            base["tpm_limit"] = None
        else:
            base["tpm_limit"] = max(0, limit_val) or None

    try:
        rpm_val = int(base.get("rpm_limit", DEFAULT_CONFIG["ai"].get("rpm_limit", 0)))
    except Exception:
        rpm_val = DEFAULT_CONFIG["ai"].get("rpm_limit", 0)
    base["rpm_limit"] = max(0, rpm_val) or None

    try:
        timeout_val = float(base.get("timeout", DEFAULT_CONFIG["ai"].get("timeout", 45)))
    except Exception:
        timeout_val = DEFAULT_CONFIG["ai"].get("timeout", 45)
    base["timeout"] = max(5.0, timeout_val)

    try:
        trunc_title = int(base.get("trunc_title", DEFAULT_CONFIG["ai"].get("trunc_title", 180)))
    except Exception:
        trunc_title = DEFAULT_CONFIG["ai"].get("trunc_title", 180)
    base["trunc_title"] = max(40, trunc_title)

    try:
        trunc_desc = int(base.get("trunc_desc", DEFAULT_CONFIG["ai"].get("trunc_desc", 800)))
    except Exception:
        trunc_desc = DEFAULT_CONFIG["ai"].get("trunc_desc", 800)
    base["trunc_desc"] = max(80, trunc_desc)

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
