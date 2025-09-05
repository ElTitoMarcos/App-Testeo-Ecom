"""Configuration management for Product Research Copilot.

The application stores user configuration such as the OpenAI API key and
preferred model in a JSON file (config.json) located in the application's
directory. These helpers encapsulate loading and saving this configuration
file. If the file does not exist, default values are returned.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


CONFIG_FILE = Path(__file__).resolve().parent / "config.json"


def load_config() -> Dict[str, Any]:
    """Load configuration from disk.

    Returns a dictionary with at least the keys ``api_key`` and ``model``.
    If the file does not exist, an empty configuration is returned.
    """

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {}
            return data
        except Exception:
            return {}
    return {}


def save_config(config: Dict[str, Any]) -> None:
    """Persist configuration to disk atomically."""

    tmp_path = CONFIG_FILE.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    tmp_path.replace(CONFIG_FILE)


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


def get_weights() -> Dict[str, float]:
    """Return weighting factors for each metric.

    The configuration may include a ``weights`` object mapping metric names
    (momentum, saturation, differentiation, social_proof, margin, logistics)
    to numeric values. If a weight is missing or invalid it defaults to 1.0.

    Returns:
        A dictionary of six weights used to compute the overall score.
    """

    cfg = load_config()
    default = {
        "momentum": 1.0,
        "saturation": 1.0,
        "differentiation": 1.0,
        "social_proof": 1.0,
        "margin": 1.0,
        "logistics": 1.0,
    }
    user_weights = cfg.get("weights", {})
    weights: Dict[str, float] = {}
    for k, v in default.items():
        try:
            weights[k] = float(user_weights.get(k, v))
        except Exception:
            weights[k] = v
    return weights


def is_scoring_v2_enabled() -> bool:
    """Return whether Winner Score v2 flow is enabled.

    The configuration may contain a nested structure like::

        {"scoring": {"v2": {"enabled": true}}}

    If the key is missing or invalid the flag defaults to ``True``.
    """

    cfg = load_config()
    try:
        return bool(cfg.get("scoring", {}).get("v2", {}).get("enabled", True))
    except Exception:
        return True

