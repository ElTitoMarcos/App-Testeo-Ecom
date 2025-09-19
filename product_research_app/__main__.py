from product_research_app.services.config import init_app_config
from product_research_app.api import app

init_app_config()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False, threaded=True, use_reloader=False)
