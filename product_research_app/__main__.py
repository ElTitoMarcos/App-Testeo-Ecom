import logging
import os

from product_research_app.services.config import init_app_config
from product_research_app.api import app
from product_research_app.utils.auto_open import open_browser_when_ready

# Silenciar 400 ruidosos (p.ej. handshakes TLS) del servidor de desarrollo
try:
    import werkzeug.serving as _serving

    class _Quiet400Handler(_serving.WSGIRequestHandler):  # type: ignore[misc]
        def log_request(self, code="-", size="-") -> None:
            try:
                if int(code) == 400:
                    return
            except Exception:
                pass
            super().log_request(code, size)

    _serving.WSGIRequestHandler = _Quiet400Handler
except Exception:
    pass

logger = logging.getLogger(__name__)


def main() -> None:
    init_app_config()

    host = os.getenv("PRAPP_HOST", "127.0.0.1")
    port_value = os.getenv("PRAPP_PORT", "8000")
    try:
        port = int(port_value)
    except (TypeError, ValueError):
        port = 8000
    browser_url = os.getenv("PRAPP_BROWSER_URL", f"http://{host}:{port}/")

    open_browser_when_ready(browser_url, host=host, port=port, timeout_s=90)

    message = f"Servidor iniciado en {browser_url.rstrip('/')}"

    try:
        logger.info(message)
        print(message)
        app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
    except Exception:
        logger.exception("Fallo fatal al iniciar el servidor")
        try:
            if os.name == "nt":
                input("\n[CRASH] Revisa logs y pulsa Enter para cerrar...")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
