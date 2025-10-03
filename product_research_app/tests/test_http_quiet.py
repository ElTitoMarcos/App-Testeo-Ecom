from __future__ import annotations

from typing import List, Tuple

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import http_quiet


class DummyQuietHandler(http_quiet.QuietHandlerMixin):
    def __init__(self) -> None:
        self.server_version = "HTTP"
        self.sys_version = ""
        self.protocol_version = "HTTP/1.1"
        self.requestline = ""
        self.raw_requestline = b""
        self.path = "/"
        self._records: List[Tuple[str, Tuple]] = []

    # BaseHTTPRequestHandler.log_error -> log_message
    def log_message(self, fmt: str, *args) -> None:  # type: ignore[override]
        self._records.append((fmt, args))


def test_tls_client_hello_is_silenced():
    handler = DummyQuietHandler()
    handler.requestline = "\x16\x03\x01\x02"
    handler.raw_requestline = b"\x16\x03\x01\x02"
    handler.log_error("code %d, message %s", 400, "Bad request version ('\x16\x03')")
    assert handler._records == []


def test_tls_client_hello_without_bad_request_is_silenced():
    handler = DummyQuietHandler()
    handler.raw_requestline = b"\x16\x03\x01\x00"
    handler.log_error("code %d", 400)
    assert handler._records == []


def test_regular_error_is_logged():
    handler = DummyQuietHandler()
    handler.requestline = "GET / HTTP/1.1"
    handler.log_error("code %d, message %s", 404, "File not found")
    assert handler._records == [("code %d, message %s", (404, "File not found"))]

