import json, sqlite3, time, math
from pathlib import Path

ALLOWED_FIELDS = ("price","rating","units_sold","revenue","desire","competition","oldness")
DEFAULT_WEIGHTS_RAW = {k: 50 for k in ALLOWED_FIELDS}  # 50 = neutro

DB_PATH = Path(__file__).resolve().parents[1] / "data.sqlite3"
KEY_WEIGHTS_RAW = "winner_weights_v2_raw"


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
        cx.commit()


def _clamp_int01(x, lo=0, hi=100):
    try:
        v = int(float(x))
    except Exception:
        v = 50
    return max(lo, min(hi, v))


def get_winner_weights_raw() -> dict:
    with _conn() as cx:
        row = cx.execute("SELECT json_value FROM app_config WHERE key=?", (KEY_WEIGHTS_RAW,)).fetchone()
        if not row:
            return DEFAULT_WEIGHTS_RAW.copy()
        data = json.loads(row[0])
    out = {}
    for k in ALLOWED_FIELDS:
        out[k] = _clamp_int01(data.get(k, 50))
    return out


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


# Útil para logs/cálculo: normaliza a suma 100 con “largest remainder”
def compute_effective_int(weights_raw: dict) -> dict:
    vals = [max(0, float(weights_raw[k])) for k in ALLOWED_FIELDS]
    s = sum(vals)
    if s <= 0:
        return {k: (100 // len(ALLOWED_FIELDS)) for k in ALLOWED_FIELDS}
    shares = [v * 100.0 / s for v in vals]
    floors = [int(math.floor(x)) for x in shares]
    rem = 100 - sum(floors)
    # reparte los decimales sobrantes a los mayores restos
    order = sorted(range(len(shares)), key=lambda i: shares[i] - floors[i], reverse=True)
    eff = floors[:]
    for i in range(rem):
        eff[order[i]] += 1
    return {k: eff[i] for i, k in enumerate(ALLOWED_FIELDS)}

