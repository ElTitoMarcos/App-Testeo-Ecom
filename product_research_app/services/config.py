import json, sqlite3, time
from pathlib import Path

ALLOWED_FIELDS = ("price","rating","units_sold","revenue","desire","competition","oldness")
DEFAULT_WEIGHTS = {k: 50 for k in ALLOWED_FIELDS}  # todos 50 (neutro)

DB_PATH = Path(__file__).resolve().parents[1] / "data.sqlite3"

def _conn():
    return sqlite3.connect(DB_PATH)

def init_app_config():
    with _conn() as cx:
        cx.execute("""CREATE TABLE IF NOT EXISTS app_config (
            key TEXT PRIMARY KEY,
            json_value TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        )""")
        cur = cx.execute("SELECT json_value FROM app_config WHERE key='winner_weights_v2'")
        row = cur.fetchone()
        if not row:
            cx.execute(
                "INSERT INTO app_config(key,json_value,updated_at) VALUES (?,?,?)",
                ("winner_weights_v2", json.dumps(DEFAULT_WEIGHTS), int(time.time()))
            )

def get_winner_weights() -> dict:
    with _conn() as cx:
        cur = cx.execute("SELECT json_value FROM app_config WHERE key='winner_weights_v2'")
        j = cur.fetchone()[0]
    data = json.loads(j)
    out = {}
    for k in ALLOWED_FIELDS:
        try:
            v = int(float(data.get(k, DEFAULT_WEIGHTS[k])))
        except Exception:
            v = DEFAULT_WEIGHTS[k]  # 50
        out[k] = max(0, min(100, v))
    return out

def set_winner_weights(weights: dict) -> dict:
    current = get_winner_weights()
    for k in ALLOWED_FIELDS:
        if k in weights:
            try:
                v = int(float(weights[k]))
            except Exception:
                continue
            current[k] = max(0, min(100, v))
    with _conn() as cx:
        cx.execute(
            "UPDATE app_config SET json_value=?, updated_at=? WHERE key='winner_weights_v2'",
            (json.dumps(current), int(time.time()))
        )
    return current
