"""Configuration management for Product Research Copilot.

The application stores user configuration such as the OpenAI API key and
preferred model in a JSON file (config.json) located in the application's
directory. These helpers encapsulate loading and saving this configuration
file. If the file does not exist, default values are returned.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


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
    },
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

    changed = _merge_defaults(data, DEFAULT_CONFIG)
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


def is_auto_fill_ia_on_import_enabled() -> bool:
    cfg = load_config()
    try:
        return bool(cfg.get("autoFillIAOnImport", True))
    except Exception:
        return True


SCORING_DEFAULT_WEIGHTS: Dict[str, float] = {
    "magnitud_deseo": 1.0,
    "nivel_consciencia_headroom": 1.0,
    "evidencia_demanda": 1.0,
    "tasa_conversion": 1.0,
    "ventas_por_dia": 1.0,
    "recencia_lanzamiento": 1.0,
    "competition_level_invertido": 1.0,
    "facilidad_anuncio": 1.0,
    "escalabilidad": 1.0,
    "durabilidad_recurrencia": 1.0,
}


def get_weights() -> Dict[str, float]:
    """Return the weighting factors for Winner Score variables.

    The configuration may include a ``weights`` object mapping the ten Winner
    Score variables to numeric values between 0 and 1. If weights are missing or
    invalid, defaults are used and the result is normalised so that the sum of
    all weights equals 1.
    """

    cfg = load_config()
    user_weights = cfg.get("weights", {})
    weights: Dict[str, float] = {}
    total = 0.0
    for key, default in SCORING_DEFAULT_WEIGHTS.items():
        try:
            val = float(user_weights.get(key, default))
            if val < 0:
                val = 0.0
        except Exception:
            val = default
        weights[key] = val
        total += val
    if total <= 0:
        return SCORING_DEFAULT_WEIGHTS.copy()
    return {k: v / total for k, v in weights.items()}


def set_weights(weights: Dict[str, float]) -> None:
    """Persist Winner Score weights to configuration."""

    cfg = load_config()
    cfg["weights"] = weights
    save_config(cfg)
