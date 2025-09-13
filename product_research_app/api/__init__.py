from flask import Flask

app = Flask(__name__)

# Import API modules which attach routes to ``app``.
from . import config  # noqa: E402,F401

# Log registered routes for easier debugging in start-up logs.
for r in app.url_map.iter_rules():
    app.logger.info("ROUTE %s %s", ",".join(sorted(r.methods)), r.rule)
