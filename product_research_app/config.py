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
        "costCapUSD": None,
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
        "parallelism": 8,
        "microbatch": 32,
        "cache_enabled": True,
        "version": 1,
        "tpm_limit": None,
        "rpm_limit": None,
        "temperature": 0.0,
        "top_p": 0.0,
        "response_format": "json",
        "costCapUSD": None,
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
    """Return the API key following the documented priority order."""

    cfg = load_config()
    key = str(cfg.get("api_key") or "").strip()
    if key:
        return key
    enrich_key = os.environ.get("ENRICH_API_KEY")
    if enrich_key:
        enrich_key = enrich_key.strip()
        if enrich_key:
            return enrich_key
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        openai_key = openai_key.strip()
        if openai_key:
            return openai_key
    return None


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
    for key, value in (user or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            tmp = base.get(key, {}).copy()
            tmp.update(value)
            base[key] = tmp
        else:
            base[key] = value

    cpu_parallel = max(1, (os.cpu_count() or 1) * 2)
    default_parallel = min(8, cpu_parallel)

    def _coerce_int(value: Any, default: int) -> int:
        try:
            num = int(value)
        except Exception:
            return default
        return default if num <= 0 else num

    def _coerce_optional_int(value: Any) -> Optional[int]:
        if value in (None, "", 0, "0"):
            return None
        try:
            num = int(value)
        except Exception:
            return None
        return num if num > 0 else None

    def _coerce_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _coerce_optional_float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except Exception:
            return None

    runtime: Dict[str, Any] = {}
    runtime["model"] = str(base.get("model") or "gpt-4o-mini")
    runtime["parallelism"] = max(1, _coerce_int(base.get("parallelism"), default_parallel))
    runtime["microbatch"] = max(1, _coerce_int(base.get("microbatch"), DEFAULT_CONFIG["ai"]["microbatch"]))
    runtime["cache_enabled"] = bool(base.get("cache_enabled", True))
    runtime["version"] = _coerce_int(base.get("version", DEFAULT_CONFIG["ai"]["version"]), DEFAULT_CONFIG["ai"]["version"])
    runtime["temperature"] = _coerce_float(base.get("temperature"), 0.0)
    runtime["top_p"] = _coerce_float(base.get("top_p"), 0.0)
    runtime["response_format"] = str(base.get("response_format") or "json")
    runtime["rpm_limit"] = _coerce_optional_int(base.get("rpm_limit"))
    runtime["tpm_limit"] = _coerce_optional_int(base.get("tpm_limit"))

    cost_cap = base.get("costCapUSD")
    if cost_cap is None:
        cost_cap = cfg.get("aiCost", {}).get("costCapUSD")
    runtime["costCapUSD"] = _coerce_optional_float(cost_cap)

    return runtime


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
