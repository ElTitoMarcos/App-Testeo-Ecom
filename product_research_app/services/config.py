import json, sqlite3, time
from pathlib import Path

ALLOWED_FIELDS = ("price","rating","units_sold","revenue","desire","competition","oldness")
DEFAULT_WEIGHTS = {k: 50 for k in ALLOWED_FIELDS}  # 50 = neutro

DB_PATH = Path(__file__).resolve().parents[1] / "data.sqlite3"

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
        row = cx.execute(
            "SELECT 1 FROM app_config WHERE key='winner_weights_v2'"
        ).fetchone()
        if not row:
            cx.execute(
                "INSERT INTO app_config(key,json_value,updated_at) VALUES (?,?,?)",
                ("winner_weights_v2", json.dumps(DEFAULT_WEIGHTS), int(time.time()))
            )
        cx.commit()

def _sanitize_weights(data: dict) -> dict:
    out = {}
    for k in ALLOWED_FIELDS:
        try:
            v = int(float(data.get(k, DEFAULT_WEIGHTS[k])))
        except Exception:
            v = DEFAULT_WEIGHTS[k]
        out[k] = max(0, min(100, v))
    return out

def get_winner_weights() -> dict:
    with _conn() as cx:
        row = cx.execute(
            "SELECT json_value FROM app_config WHERE key='winner_weights_v2'"
        ).fetchone()
        if not row:
            return DEFAULT_WEIGHTS.copy()
        return _sanitize_weights(json.loads(row[0]))

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
        cx.commit()
    return current
