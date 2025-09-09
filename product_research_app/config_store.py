import json, os
from typing import Dict

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
_DEFAULTS: Dict[str, str] = {
    "openai_api_key": ""
}

def _ensure_file():
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(_DEFAULTS, f, ensure_ascii=False, indent=2)

def read_config() -> Dict[str, str]:
    _ensure_file()
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    return {**_DEFAULTS, **(data or {})}

def write_config(data: Dict[str, str]) -> None:
    _ensure_file()
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_api_key() -> str:
    cfg = read_config()
    return (cfg.get("openai_api_key") or os.getenv("OPENAI_API_KEY") or "").strip()
