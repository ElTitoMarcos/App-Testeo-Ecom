from flask import Flask

app = Flask(__name__)

from . import config  # noqa: E402,F401
from .winner_score import bp as winner_score_bp  # noqa: E402

app.register_blueprint(winner_score_bp)
