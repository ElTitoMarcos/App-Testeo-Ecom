from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from flask import Flask, render_template

from . import database
from .api import (
    DEFAULT_ORDER,
    DEFAULT_WEIGHTS,
    api_bp,
    recalc_winner_scores,
)

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "data.sqlite3"


def create_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(api_bp)

    _startup()

    @app.get("/")
    def index() -> str:
        return render_template("index.html")
    return app


def _startup() -> None:
    conn = database.get_connection(DB_PATH)
    database.initialize_database(conn)
    settings = database.get_settings(conn)
    if "weights" not in settings or "order" not in settings:
        database.set_setting(conn, "winner_weights", DEFAULT_WEIGHTS)
        database.set_setting(conn, "winner_order", DEFAULT_ORDER)
        settings = database.get_settings(conn)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM products")
    total = cur.fetchone()[0]
    if total:
        cur.execute("SELECT COUNT(*) FROM products WHERE winner_score IS NULL")
        missing = cur.fetchone()[0]
        if missing:
            recalc_winner_scores(
                conn,
                settings.get("weights", DEFAULT_WEIGHTS),
                settings.get("order", DEFAULT_ORDER),
            )
    conn.close()


if __name__ == "__main__":
    create_app().run()
