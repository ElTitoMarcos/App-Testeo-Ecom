"""Product Research Copilot package initializer."""
from flask import Flask

app = Flask(__name__)

from .api_config import bp as config_bp
app.register_blueprint(config_bp)
