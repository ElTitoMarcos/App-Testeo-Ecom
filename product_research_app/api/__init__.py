from flask import Flask

app = Flask(__name__)

# Import API modules which attach routes to ``app`` or expose blueprints.
from . import config  # noqa: E402,F401
from .winner_score import winner_score_api  # noqa: E402

# Mount winner score blueprint under the /api namespace.
app.register_blueprint(winner_score_api, url_prefix="/api")

# Log registered routes for easier debugging in start-up logs.
for r in app.url_map.iter_rules():
    app.logger.info("ROUTE %s %s", ",".join(sorted(r.methods)), r.rule)
