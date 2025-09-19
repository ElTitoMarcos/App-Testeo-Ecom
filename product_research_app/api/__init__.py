from datetime import datetime

from flask import Flask, jsonify, current_app

from product_research_app.db import get_db
from product_research_app.sse import sse_bp
from product_research_app.api.legacy_shims import legacy_bp
from product_research_app import web_app

app = Flask(__name__)
app.register_blueprint(sse_bp)
app.register_blueprint(legacy_bp)

try:  # Warm legacy services so imports continue working
    web_app.ensure_db()
    web_app.resume_incomplete_imports()
except Exception as exc:  # pragma: no cover - best effort
    app.logger.warning("legacy warmup failed: %s", exc)

@app.get("/healthz")
def healthz():
    """Return a JSON health report with a quick database connectivity test."""

    timestamp = datetime.utcnow().isoformat() + "Z"
    try:
        conn = get_db()
        cur = conn.execute("SELECT 1;")
        cur.fetchone()
    except Exception as exc:  # pragma: no cover - best effort logging
        current_app.logger.exception("healthcheck failed: %s", exc)
        return jsonify({"ok": False, "error": str(exc), "time": timestamp}), 500
    return jsonify({"ok": True, "time": timestamp}), 200

# Import API modules which attach routes to ``app``.
from . import config  # noqa: E402,F401

# Log registered routes for easier debugging in start-up logs.
for r in app.url_map.iter_rules():
    app.logger.info("ROUTE %s %s", ",".join(sorted(r.methods)), r.rule)
