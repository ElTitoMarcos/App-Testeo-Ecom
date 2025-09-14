from flask import Flask

app = Flask(__name__)

# Import API modules which attach routes to ``app``.
from . import config  # noqa: E402,F401

# Log URL map after routes are registered for easier debugging.
try:
    app.logger.info(
        "URL MAP:\n" + "\n".join(sorted(str(r) for r in app.url_map.iter_rules()))
    )
except Exception:
    pass
