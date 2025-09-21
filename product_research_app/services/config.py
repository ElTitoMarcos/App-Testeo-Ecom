import time
from pathlib import Path
from typing import Dict, List, Tuple

from ..config import (
    load_config,
    save_config,
    DEFAULT_WINNER_ORDER,
    DEFAULT_WINNER_WEIGHTS_INT,
)

ALLOWED_FIELDS = tuple(DEFAULT_WINNER_ORDER)
DEFAULT_WEIGHTS_RAW: Dict[str, int] = DEFAULT_WINNER_WEIGHTS_INT.copy()
DEFAULT_ORDER: List[str] = list(DEFAULT_WINNER_ORDER)
DEFAULT_ENABLED: Dict[str, bool] = {k: True for k in ALLOWED_FIELDS}

# Compatibility placeholder; not used but kept for tests that monkeypatch it
DB_PATH = Path(__file__).resolve().parents[1] / "data.sqlite3"


def _coerce_weights(raw: Dict[str, object] | None) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k, v in (raw or {}).items():
        try:
            iv = int(round(float(v)))
        except Exception:
            iv = 0
        if iv < 0:
            iv = 0
        if iv > 50:
            iv = int(round(iv * 0.5))
        out[k] = max(0, min(50, iv))
    return out


def validate_order(order: List[str] | None) -> List[str]:
    if order is None:
        raise ValueError("order is required")
    if not isinstance(order, list):
        raise ValueError("order must be a list")
    cleaned: List[str] = []
    seen: set[str] = set()
    for item in order:
        if item in ALLOWED_FIELDS and item not in seen:
            cleaned.append(item)
            seen.add(item)
    missing = [k for k in ALLOWED_FIELDS if k not in seen]
    if missing or len(cleaned) != len(ALLOWED_FIELDS):
        raise ValueError("order must include each metric exactly once")
    return cleaned


def _normalize_order(order, weights: Dict[str, int]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = [k for k in (order or []) if k in weights and not (k in seen or seen.add(k))]
    out += [k for k in weights.keys() if k not in out]
    return out


def init_app_config() -> None:
    cfg = load_config()
    changed = False
    weights_in = cfg.get("winner_weights")
    if not isinstance(weights_in, dict):
        weights = DEFAULT_WEIGHTS_RAW.copy()
        cfg["winner_weights"] = weights
        changed = True
    else:
        weights = DEFAULT_WEIGHTS_RAW.copy()
        weights.update(_coerce_weights(weights_in))
        if weights != weights_in:
            cfg["winner_weights"] = weights
            changed = True

    order_in = cfg.get("winner_order")
    try:
        order = validate_order(order_in if isinstance(order_in, list) else None)
    except Exception:
        order = DEFAULT_ORDER.copy()
        changed = True
    cfg["winner_order"] = order
    cfg.setdefault("weights_order", order.copy())

    enabled = cfg.get("weights_enabled")
    if not isinstance(enabled, dict):
        cfg["weights_enabled"] = DEFAULT_ENABLED.copy()
        changed = True
    else:
        for k in DEFAULT_ENABLED.keys():
            if k not in enabled:
                enabled[k] = True
                changed = True

    if "weightsUpdatedAt" not in cfg:
        cfg["weightsUpdatedAt"] = int(time.time())
        changed = True
    if changed:
        save_config(cfg)


def _load() -> Tuple[Dict[str, int], List[str], Dict[str, bool]]:
    cfg = load_config()
    weights = cfg.get("winner_weights")
    if not isinstance(weights, dict) or not weights:
        weights = DEFAULT_WEIGHTS_RAW.copy()
    weights = _coerce_weights(weights)
    for k, v in DEFAULT_WEIGHTS_RAW.items():
        weights.setdefault(k, v)
    order = _normalize_order(cfg.get("winner_order"), weights)
    enabled = cfg.get("weights_enabled")
    if not isinstance(enabled, dict):
        enabled = DEFAULT_ENABLED.copy()
    else:
        enabled = {k: bool(enabled.get(k, True)) for k in DEFAULT_ENABLED.keys()}
    return weights, order, enabled


def update_winner_settings(
    weights_in=None,
    order_in=None,
    enabled_in=None,
) -> Tuple[Dict[str, int], List[str], Dict[str, bool]]:
    init_app_config()
    cfg = load_config()
    weights = cfg.get("winner_weights", DEFAULT_WEIGHTS_RAW.copy())
    order = cfg.get("winner_order", DEFAULT_ORDER.copy())
    enabled = cfg.get("weights_enabled", DEFAULT_ENABLED.copy())

    weights = _coerce_weights(weights)
    for k, v in DEFAULT_WEIGHTS_RAW.items():
        weights.setdefault(k, v)
    order = _normalize_order(order, weights)
    enabled = {k: bool(enabled.get(k, True)) for k in DEFAULT_ENABLED.keys()}

    if weights_in is not None:
        wi = _coerce_weights(weights_in)
        weights.update(wi)
        for k, v in DEFAULT_WEIGHTS_RAW.items():
            weights.setdefault(k, v)
    if order_in is not None:
        order = _normalize_order(order_in, weights)
    if enabled_in is not None:
        enabled = {
            k: bool(enabled_in.get(k, enabled.get(k, True)))
            for k in DEFAULT_ENABLED.keys()
        }

    cfg["winner_weights"] = weights
    cfg["winner_order"] = order
    cfg["weights_order"] = order
    cfg["weights_enabled"] = enabled
    cfg["weightsUpdatedAt"] = int(time.time())
    save_config(cfg)
    return weights, order, enabled


def get_winner_weights_raw() -> Dict[str, int]:
    weights, _, _ = _load()
    return weights


def get_winner_order_raw() -> List[str]:
    _, order, _ = _load()
    return order


def get_weights_enabled_raw() -> Dict[str, bool]:
    _, _, enabled = _load()
    return enabled


def set_winner_weights_raw(weights: Dict[str, object]) -> Dict[str, int]:
    weights, _, _ = update_winner_settings(weights_in=weights, order_in=None)
    return weights


def set_winner_order_raw(order: List[str]) -> List[str]:
    valid = validate_order(order)
    _, stored_order, _ = update_winner_settings(weights_in=None, order_in=valid)
    return stored_order


def set_weights_enabled_raw(enabled: Dict[str, object]) -> Dict[str, bool]:
    _, _, enabled = update_winner_settings(weights_in=None, order_in=None, enabled_in=enabled)
    return enabled


# Útil para logs/cálculo: pesos efectivos enteros 0..100 considerando prioridad
def compute_effective_int(weights_raw: Dict[str, int], order: List[str] | None = None) -> Dict[str, int]:
    from . import winner_score  # lazy to avoid circular import

    eff = winner_score.compute_effective_weights(weights_raw, order or list(weights_raw.keys()))
    return {k: int(round(v * 100)) for k, v in eff.items()}
