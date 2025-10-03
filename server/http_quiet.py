from http.server import BaseHTTPRequestHandler
import json
import logging
import os

# Coma-separado por env; por defecto estas rutas se silencian
DEFAULT_QUIET = "/_ai_fill/status,/_import_status"
QUIET_PATHS = tuple(
    p.strip() for p in os.getenv("PRAPP_HTTP_QUIET", DEFAULT_QUIET).split(",") if p.strip()
)


logger = logging.getLogger(__name__)


def _raw_request_bytes(handler: BaseHTTPRequestHandler) -> bytes:
    raw_line = getattr(handler, "raw_requestline", b"")
    if isinstance(raw_line, memoryview):
        raw_line = raw_line.tobytes()
    if isinstance(raw_line, (bytes, bytearray)):
        return bytes(raw_line)
    return b""


def _is_tls_client_hello(payload: bytes) -> bool:
    return bool(payload and payload[0] == 0x16 and payload.startswith(b"\x16\x03\x01"))


class QuietHandlerMixin(BaseHTTPRequestHandler):
    # Evita 501 en HEAD
    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()

    # Favicon “silencioso”
    def do_GET(self):
        if getattr(self, "path", "") == "/favicon.ico":
            # Responde 204 sin log de error
            self.send_response(204)
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            return
        # Delega al handler real si existe método real; si no, 404 estándar
        if hasattr(super(), "do_GET"):
            return super().do_GET()
        self.send_error(404, "Not Found")

    # Silenciar GETs de estado del poller
    def log_message(self, fmt, *args):
        path = getattr(self, "path", "")
        if any(path.startswith(p) for p in QUIET_PATHS) or path == "/favicon.ico":
            return
        return super().log_message(fmt, *args)

    # Ocultar “Bad request version …” de handshakes TLS erróneos
    def log_error(self, fmt, *args):
        raw_payload = _raw_request_bytes(self)
        if _is_tls_client_hello(raw_payload):
            client = getattr(self, "client_address", (None, None))
            logger.debug(
                "http_quiet: ClientHello TLS suprimido addr=%s payload_len=%d",
                client,
                len(raw_payload),
            )
            return
        # También podemos ignorar 404 de favicon
        path = getattr(self, "path", "")
        if path == "/favicon.ico":
            return
        return super().log_error(fmt, *args)
