import json, sqlite3, time, math
from pathlib import Path

ALLOWED_FIELDS = ("price","rating","units_sold","revenue","desire","competition","oldness")
DEFAULT_WEIGHTS_RAW = {k: 50 for k in ALLOWED_FIELDS}  # 50 = neutro
DEFAULT_ORDER = list(ALLOWED_FIELDS)

DB_PATH = Path(__file__).resolve().parents[1] / "data.sqlite3"
KEY_WEIGHTS_RAW = "winner_weights_v2_raw"
KEY_ORDER = "winner_weights_v2_order"


def _conn():
    cx = sqlite3.connect(DB_PATH)
    cx.execute("PRAGMA journal_mode=WAL;")
    return cx


def init_app_config():
    with _conn() as cx:
        cx.execute("""CREATE TABLE IF NOT EXISTS app_config (
            key TEXT PRIMARY KEY,
            json_value TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        )""")
        row = cx.execute("SELECT 1 FROM app_config WHERE key=?", (KEY_WEIGHTS_RAW,)).fetchone()
        if not row:
            cx.execute(
                "INSERT INTO app_config(key,json_value,updated_at) VALUES (?,?,?)",
                (KEY_WEIGHTS_RAW, json.dumps(DEFAULT_WEIGHTS_RAW), int(time.time()))
            )
        row = cx.execute("SELECT 1 FROM app_config WHERE key=?", (KEY_ORDER,)).fetchone()
        if not row:
            cx.execute(
                "INSERT INTO app_config(key,json_value,updated_at) VALUES (?,?,?)",
                (KEY_ORDER, json.dumps(DEFAULT_ORDER), int(time.time()))
            )
        cx.commit()


def _clamp_int01(x, lo=0, hi=100):
    try:
        v = int(float(x))
    except Exception:
        v = 50
    return max(lo, min(hi, v))


def get_winner_weights_raw() -> dict:
    init_app_config()
    with _conn() as cx:
        row = cx.execute("SELECT json_value FROM app_config WHERE key=?", (KEY_WEIGHTS_RAW,)).fetchone()
        if not row:
            return DEFAULT_WEIGHTS_RAW.copy()
        data = json.loads(row[0])
    out = {}
    for k in ALLOWED_FIELDS:
        out[k] = _clamp_int01(data.get(k, 50))
    return out


def get_winner_order_raw() -> list:
    init_app_config()
    with _conn() as cx:
        row = cx.execute("SELECT json_value FROM app_config WHERE key=?", (KEY_ORDER,)).fetchone()
        if row:
            try:
                data = json.loads(row[0])
                if isinstance(data, list):
                    cleaned = [k for k in data if k in ALLOWED_FIELDS]
                    for k in ALLOWED_FIELDS:
                        if k not in cleaned:
                            cleaned.append(k)
                    return cleaned
            except Exception:
                pass
    return list(DEFAULT_ORDER)


def set_winner_weights_raw(weights: dict) -> dict:
    current = get_winner_weights_raw()
    for k in ALLOWED_FIELDS:
        if k in weights:
            current[k] = _clamp_int01(weights[k])
    with _conn() as cx:
        prev = cx.execute(
            "SELECT updated_at FROM app_config WHERE key=?", (KEY_WEIGHTS_RAW,)
        ).fetchone()
        ts = int(time.time())
        if prev and ts <= int(prev[0]):
            ts = int(prev[0]) + 1
        cx.execute(
            "UPDATE app_config SET json_value=?, updated_at=? WHERE key=?",
            (json.dumps(current), ts, KEY_WEIGHTS_RAW)
        )
        cx.commit()
    return current


def set_winner_order_raw(order: list[str]) -> list[str]:
    current = get_winner_order_raw()
    cleaned = [k for k in order if k in ALLOWED_FIELDS]
    for k in ALLOWED_FIELDS:
        if k not in cleaned:
            cleaned.append(k)
    with _conn() as cx:
        prev = cx.execute(
            "SELECT updated_at FROM app_config WHERE key=?", (KEY_ORDER,)
        ).fetchone()
        ts = int(time.time())
        if prev and ts <= int(prev[0]):
            ts = int(prev[0]) + 1
        cx.execute(
            "UPDATE app_config SET json_value=?, updated_at=? WHERE key=?",
            (json.dumps(cleaned), ts, KEY_ORDER)
        )
        cx.commit()
    return cleaned


# Útil para logs/cálculo: pesos efectivos enteros 0..100 considerando prioridad
def compute_effective_int(weights_raw: dict, order: list[str] | None = None) -> dict:
    from . import winner_score  # lazy to avoid circular import

    eff = winner_score.compute_effective_weights(weights_raw, order or list(weights_raw.keys()))
    return {k: int(round(v * 100)) for k, v in eff.items()}

