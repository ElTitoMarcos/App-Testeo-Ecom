"""Product Research Copilot package initializer."""
from flask import Flask, render_template

app = Flask(__name__)


@app.get("/settings")
def settings_page() -> str:
    """Render the advanced settings page."""
    return render_template("settings.html")


from .api_config import bp as config_bp
app.register_blueprint(config_bp)
