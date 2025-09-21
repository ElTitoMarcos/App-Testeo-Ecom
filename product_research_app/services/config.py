import time
from pathlib import Path
from typing import Dict, List, Tuple

from ..config import load_config, save_config, DEFAULT_WINNER_ORDER

ALLOWED_FIELDS = (
    "price",
    "rating",
    "units_sold",
    "revenue",
    "desire",
    "competition",
    "oldness",
    "awareness",
)
DEFAULT_WEIGHTS_RAW: Dict[str, int] = {k: 50 for k in ALLOWED_FIELDS}
DEFAULT_ORDER: List[str] = list(DEFAULT_WINNER_ORDER)
DEFAULT_ENABLED: Dict[str, bool] = {k: True for k in ALLOWED_FIELDS}


def get_default_winner_weights() -> Dict[str, int]:
    """Return a copy of the default weight map (0-100 scale)."""

    return DEFAULT_WEIGHTS_RAW.copy()


# Compatibility placeholder; not used but kept for tests that monkeypatch it
DB_PATH = Path(__file__).resolve().parents[1] / "data.sqlite3"


def _coerce_weights(raw: Dict[str, object] | None) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k, v in (raw or {}).items():
        try:
            iv = int(round(float(v)))
        except Exception:
            iv = 0
        out[k] = max(0, min(100, iv))
    return out


def _normalize_order(order, weights: Dict[str, int]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
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


def init_app_config() -> None:
    cfg = load_config()
    # Garantizar orden en ambas claves y persistir si faltaba
    order = cfg.get("winner_order")
    if not isinstance(order, list) or len(order) != 8:
        order = list(DEFAULT_WINNER_ORDER)
        cfg["winner_order"] = order[:]
        cfg["weights_order"] = order[:]
        save_config(cfg)  # persistir para sobrevivir reinicios

    # Si solo falta weights_order, refléjalo desde winner_order
    if not isinstance(cfg.get("weights_order"), list) or len(cfg["weights_order"]) != 8:
        cfg["weights_order"] = cfg["winner_order"][:]
        save_config(cfg)

    changed = False
    weights = cfg.get("winner_weights")
    if not isinstance(weights, dict):
        cfg["winner_weights"] = DEFAULT_WEIGHTS_RAW.copy()
        changed = True
        weights = cfg["winner_weights"]
    else:
        for k, v in DEFAULT_WEIGHTS_RAW.items():
            if k not in weights:
                weights[k] = v
                changed = True

    order = cfg.get("winner_order")
    if not isinstance(order, list):
        order = DEFAULT_ORDER.copy()
        cfg["winner_order"] = order[:]
        cfg["weights_order"] = order[:]
        changed = True
    normalized_order = _normalize_order(order, cfg["winner_weights"])
    if normalized_order != order:
        cfg["winner_order"] = normalized_order[:]
        order = normalized_order
        changed = True
    weights_order = cfg.get("weights_order")
    if not isinstance(weights_order, list) or weights_order != order:
        cfg["weights_order"] = order[:]
        changed = True

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
    order_raw = cfg.get("winner_order")
    order = _normalize_order(order_raw, weights)
    cfg_changed = False
    if order_raw != order:
        cfg["winner_order"] = order[:]
        cfg_changed = True
    weights_order = cfg.get("weights_order")
    if not isinstance(weights_order, list) or weights_order != order:
        cfg["weights_order"] = order[:]
        cfg_changed = True
    enabled = cfg.get("weights_enabled")
    if not isinstance(enabled, dict):
        enabled = DEFAULT_ENABLED.copy()
    else:
        enabled = {k: bool(enabled.get(k, True)) for k in DEFAULT_ENABLED.keys()}
    if cfg_changed:
        save_config(cfg)
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
    cfg["winner_order"] = order[:]
    cfg["weights_order"] = order[:]
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
    _, order, _ = update_winner_settings(weights_in=None, order_in=order)
    return order


def set_weights_enabled_raw(enabled: Dict[str, object]) -> Dict[str, bool]:
    _, _, enabled = update_winner_settings(weights_in=None, order_in=None, enabled_in=enabled)
    return enabled


# Útil para logs/cálculo: pesos efectivos enteros 0..100 considerando prioridad
def compute_effective_int(weights_raw: Dict[str, int], order: List[str] | None = None) -> Dict[str, int]:
    from . import winner_score  # lazy to avoid circular import

    eff = winner_score.compute_effective_weights(weights_raw, order or list(weights_raw.keys()))
    return {k: int(round(v * 100)) for k, v in eff.items()}
