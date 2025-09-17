"""Static application settings for runtime behavior.

These settings are intentionally kept simple and rely on environment
variables for overrides so deployments can tweak automation behaviour
without touching the persistent ``config.json`` used by the UI.
"""

from __future__ import annotations

import os


def _get_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    return value in {"1", "true", "yes", "on", "y"}


def _get_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


AI_AUTO_ENABLED: bool = _get_bool("AI_AUTO_ENABLED", True)
AI_MAX_CALLS_PER_IMPORT: int = max(0, _get_int("AI_MAX_CALLS_PER_IMPORT", 4))
AI_MIN_BATCH_SIZE: int = max(1, _get_int("AI_MIN_BATCH_SIZE", 100))
AI_MAX_BATCH_SIZE: int = max(AI_MIN_BATCH_SIZE, _get_int("AI_MAX_BATCH_SIZE", 250))
AI_MAX_PARALLEL: int = max(1, _get_int("AI_MAX_PARALLEL", 1))
AI_COALESCE_MS: int = max(0, _get_int("AI_COALESCE_MS", 400))


__all__ = [
    "AI_AUTO_ENABLED",
    "AI_MAX_CALLS_PER_IMPORT",
    "AI_MIN_BATCH_SIZE",
    "AI_MAX_BATCH_SIZE",
    "AI_MAX_PARALLEL",
    "AI_COALESCE_MS",
]
