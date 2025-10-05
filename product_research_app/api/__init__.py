"""Flask application factory and blueprint registration."""

from flask import Flask

app = Flask(__name__)

# Import API modules which attach routes to ``app``.
from . import config  # noqa: E402,F401
from .winner_score import winner_score_api  # noqa: E402
from ..sse import sse_bp  # noqa: E402

app.register_blueprint(winner_score_api, url_prefix="/api")
app.register_blueprint(sse_bp)


@app.get("/healthz")
def healthz():
    return {"ok": True}


# Garantizar endpoint /health si no existe previamente
try:
    has_health = False
    try:
        for r in getattr(app, "routes", []):
            if getattr(r, "path", None) == "/health":
                has_health = True
                break
    except Exception:
        pass

    try:
        if not has_health and hasattr(app, "view_functions"):
            if "health._health" in app.view_functions or "health" in app.view_functions:
                has_health = True
    except Exception:
        pass

    if not has_health:
        from product_research_app.utils.health import mount_health

        mount_health(app)
except Exception:
    pass


# Log registered routes for easier debugging in start-up logs.
for r in app.url_map.iter_rules():
    app.logger.info("ROUTE %s %s", ",".join(sorted(r.methods)), r.rule)
