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


# Garantizar utilidades básicas (healthcheck y raíz)
try:
    from product_research_app.utils.health import mount_health
    from product_research_app.utils.root_redirect import mount_root

    try:
        mount_health(app)
    except Exception:
        pass

    try:
        mount_root(app)
    except Exception:
        pass
except Exception:
    pass


# Log registered routes for easier debugging in start-up logs.
for r in app.url_map.iter_rules():
    app.logger.info("ROUTE %s %s", ",".join(sorted(r.methods)), r.rule)
