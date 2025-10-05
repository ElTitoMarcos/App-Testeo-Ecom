import logging
import os

from product_research_app.services.config import init_app_config
from product_research_app.api import app
from product_research_app.utils.auto_open import open_browser_when_ready

QuietHandler = None
try:
    from werkzeug.serving import WSGIRequestHandler as _BaseRequestHandler

    class QuietHandler(_BaseRequestHandler):  # type: ignore[misc]
        """WSGIRequestHandler que silencia handshakes 400 ruidosos."""

        def log_request(self, code="-", size="-") -> None:
            try:
                if int(code) == 400:
                    return
            except Exception:
                pass
            super().log_request(code, size)

        def log_error(self, format: str, *args) -> None:  # type: ignore[override]
            try:
                message = (format % args) if args else str(format)
                if "Bad request version" in message or "Bad request syntax" in message:
                    return
            except Exception:
                pass
            super().log_error(format, *args)
except Exception:
    QuietHandler = None

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
        run_kwargs = dict(host=host, port=port, debug=False, threaded=True, use_reloader=False)
        if QuietHandler is not None:
            run_kwargs["request_handler"] = QuietHandler
        app.run(**run_kwargs)
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
