from __future__ import annotations

KNOWN = ['price','rating','units_sold','revenue','desire','competition','oldness','awareness']
ALIASES = {
  'income':'revenue','turnover':'revenue','sales_revenue':'revenue',
  'sold_units':'units_sold','qty_sold':'units_sold','units':'units_sold',
  'score':'rating','stars':'rating'
}

def _map_name(k: str) -> str:
    k = (k or '').strip().lower()
    return ALIASES.get(k, k)

def _to_abs_0_100(v):
    if isinstance(v, (int, float)):
        x = float(v)
        if 0 <= x <= 100:
            return int(round(x))
        if 0 <= x <= 1:
            return int(round(x * 100))
        return int(round(max(0, min(100, x))))
    if isinstance(v, str):
        s = v.strip().replace('%', '')
        try:
            x = float(s)
            if 0 <= x <= 100:
                return int(round(x))
            if 0 <= x <= 1:
                return int(round(x * 100))
            return int(round(max(0, min(100, x))))
        except Exception:
            return None
    return None

def ai_to_abs(prev_cfg: dict, ai_raw: dict) -> dict[str, int]:
    prev = {k: int(v) for k, v in (prev_cfg.get('weights') or {}).items()}
    out = {}
    for k, v in (ai_raw or {}).items():
        fk = _map_name(k)
        if fk in KNOWN:
            mv = _to_abs_0_100(v)
            if mv is not None:
                out[fk] = mv
    for f in KNOWN:
        if f not in out and f in prev:
            out[f] = prev[f]
    if not out:
        out = {f: 50 for f in KNOWN}
    return out

def is_uniform(vals: list[int]) -> bool:
    if not vals:
        return True
    mn, mx = min(vals), max(vals)
    if mx - mn <= 5:
        return True
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    if var ** 0.5 < 3:
        return True
    uniq = sorted(set(vals))
    if len(uniq) <= 2 and (uniq[-1] - uniq[0] <= 5):
        return True
    return False

def compute_final_weights(prev_cfg: dict, ai_raw: dict) -> tuple[dict[str, int], list[str], bool]:
    enabled = (prev_cfg.get('weights_enabled') or {})
    prev = {k: int(v) for k, v in (prev_cfg.get('weights') or {}).items()}
    cand = ai_to_abs(prev_cfg, ai_raw)
    enabled_vals = [cand[k] for k in KNOWN if enabled.get(k, True) and k in cand]
    fallback = is_uniform(enabled_vals)
    final_w = prev.copy()
    if not fallback:
        for k in KNOWN:
            if enabled.get(k, True) and k in cand:
                final_w[k] = cand[k]
    for k in KNOWN:
        final_w.setdefault(k, prev.get(k, 50))
    order = sorted(final_w, key=lambda k: final_w[k], reverse=True)
    return final_w, order, fallback
