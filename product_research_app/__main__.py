from __future__ import annotations

from fastapi import FastAPI

from product_research_app.services.config import init_app_config
from product_research_app.web_app import router as web_router

app = FastAPI()
app.include_router(web_router)

try:  # pragma: no cover - optional legacy export
    from product_research_app.api import app as flask_app  # type: ignore
except Exception:  # pragma: no cover - Flask stack optional
    flask_app = None  # type: ignore

init_app_config()

if __name__ == "__main__":  # pragma: no cover - manual execution helper
    try:
        import uvicorn
    except Exception:
        if flask_app is not None:
            flask_app.run(
                host="127.0.0.1", port=8000, debug=False, threaded=True, use_reloader=False
            )
    else:
        uvicorn.run(app, host="127.0.0.1", port=8000)
