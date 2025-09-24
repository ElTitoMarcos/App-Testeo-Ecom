"""Entry point for the macOS desktop bundle."""

from __future__ import annotations

import os
import socket
import threading
import time
import webbrowser

from .web_app import run


def _wait_for_server(host: str, port: int, timeout: int = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False


def _open_browser_when_ready(host: str, port: int, delay: float = 1.0) -> None:
    if not _wait_for_server(host, port):
        return
    time.sleep(delay)
    try:
        webbrowser.open(f"http://{host}:{port}")
    except Exception:
        pass


def main() -> None:
    host = os.environ.get("APP_HOST", "127.0.0.1")
    port = int(os.environ.get("APP_PORT", "8000"))
    disable_browser = os.environ.get("DISABLE_AUTO_BROWSER")
    if not disable_browser:
        threading.Thread(target=_open_browser_when_ready, args=(host, port), daemon=True).start()
    run(host=host, port=port)


if __name__ == "__main__":
    main()
