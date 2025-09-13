import time
from pathlib import Path
from typing import Dict, List, Tuple

from ..config import load_config, save_config

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
DEFAULT_ORDER: List[str] = list(ALLOWED_FIELDS)

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
    out: List[str] = [k for k in (order or []) if k in weights and not (k in seen or seen.add(k))]
    out += [k for k in weights.keys() if k not in out]
    return out


def init_app_config() -> None:
    cfg = load_config()
    changed = False
    weights = cfg.get("winner_weights")
    if not isinstance(weights, dict):
        cfg["winner_weights"] = DEFAULT_WEIGHTS_RAW.copy()
        changed = True
    else:
        for k, v in DEFAULT_WEIGHTS_RAW.items():
            if k not in weights:
                weights[k] = v
                changed = True
    order = cfg.get("winner_order")
    if not isinstance(order, list):
        cfg["winner_order"] = DEFAULT_ORDER.copy()
        changed = True
    else:
        if "awareness" not in order:
            order.append("awareness")
            changed = True
    if "weightsUpdatedAt" not in cfg:
        cfg["weightsUpdatedAt"] = int(time.time())
        changed = True
    if changed:
        save_config(cfg)


def _load() -> Tuple[Dict[str, int], List[str]]:
    cfg = load_config()
    weights = cfg.get("winner_weights")
    if not isinstance(weights, dict) or not weights:
        weights = DEFAULT_WEIGHTS_RAW.copy()
    weights = _coerce_weights(weights)
    for k, v in DEFAULT_WEIGHTS_RAW.items():
        weights.setdefault(k, v)
    order = _normalize_order(cfg.get("winner_order"), weights)
    return weights, order


def update_winner_settings(weights_in=None, order_in=None) -> Tuple[Dict[str, int], List[str]]:
    init_app_config()
    cfg = load_config()
    weights = cfg.get("winner_weights", DEFAULT_WEIGHTS_RAW.copy())
    order = cfg.get("winner_order", DEFAULT_ORDER.copy())
    weights = _coerce_weights(weights)
    for k, v in DEFAULT_WEIGHTS_RAW.items():
        weights.setdefault(k, v)
    order = _normalize_order(order, weights)
    if weights_in is not None:
        wi = _coerce_weights(weights_in)
        weights.update(wi)
        for k, v in DEFAULT_WEIGHTS_RAW.items():
            weights.setdefault(k, v)
    if order_in is not None:
        order = _normalize_order(order_in, weights)
    cfg["winner_weights"] = weights
    cfg["winner_order"] = order
    cfg["weightsUpdatedAt"] = int(time.time())
    save_config(cfg)
    return weights, order


def get_winner_weights_raw() -> Dict[str, int]:
    weights, _ = _load()
    return weights


def get_winner_order_raw() -> List[str]:
    _, order = _load()
    return order


def set_winner_weights_raw(weights: Dict[str, object]) -> Dict[str, int]:
    weights, _ = update_winner_settings(weights_in=weights, order_in=None)
    return weights


def set_winner_order_raw(order: List[str]) -> List[str]:
    _, order = update_winner_settings(weights_in=None, order_in=order)
    return order


# Útil para logs/cálculo: pesos efectivos enteros 0..100 considerando prioridad
def compute_effective_int(weights_raw: Dict[str, int], order: List[str] | None = None) -> Dict[str, int]:
    from . import winner_score  # lazy to avoid circular import

    eff = winner_score.compute_effective_weights(weights_raw, order or list(weights_raw.keys()))
    return {k: int(round(v * 100)) for k, v in eff.items()}
