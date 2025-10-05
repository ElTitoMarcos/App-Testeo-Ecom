from __future__ import annotations
import atexit
import os
import socket
import tempfile
import threading
import time
import webbrowser
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


def _can_connect(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _is_truthy(val: str | None) -> bool:
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")


def open_browser_when_ready(url: str, host: str = "127.0.0.1", port: int = 8000,
                            timeout_s: int = 60, check_interval_s: float = 0.5,
                            lock_name: str = "prapp_auto_open.lock") -> None:
    """
    Lanza un hilo que abre el navegador cuando (host,port) acepta conexiones.
    Evita aperturas duplicadas usando un lock-file en temp y detecta recargadores comunes.
    Respeta PRAPP_AUTO_OPEN (truthy por defecto).
    """
    if not _is_truthy(os.getenv("PRAPP_AUTO_OPEN", "1")):
        return

    # Evitar ejecuciones duplicadas por recargadores (heurísticas más comunes)
    if os.getenv("WERKZEUG_RUN_MAIN") == "true":
        return
    if os.getenv("RUN_MAIN") == "true":
        return

    # Lock para "abrir solo una vez" por sesión
    lock_path = os.path.join(tempfile.gettempdir(), lock_name)
    try:
        # Crear exclusión atómica
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
    except FileExistsError:
        return  # ya abierto/abriéndose en otro proceso

    def _cleanup():
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass
    atexit.register(_cleanup)

    def _http_ok(u: str) -> bool:
        try:
            req = Request(u, method="HEAD")
            with urlopen(req, timeout=2) as r:  # type: ignore[arg-type]
                status = getattr(r, "status", getattr(r, "code", 200))
                return 200 <= status < 400
        except Exception:
            try:
                req = Request(u, method="GET")
                with urlopen(req, timeout=2) as r:  # type: ignore[arg-type]
                    status = getattr(r, "status", getattr(r, "code", 200))
                    return 200 <= status < 400
            except Exception:
                return False

    def _candidate_urls() -> list[str]:
        parts = urlsplit(url)
        if parts.scheme:
            base = f"{parts.scheme}://{parts.netloc or f'{host}:{port}'}"
        else:
            base = f"http://{host}:{port}"

        candidates: list[str] = []
        if parts.path or parts.query or parts.fragment:
            candidates.append(
                urlunsplit(
                    (
                        parts.scheme or "http",
                        parts.netloc or f"{host}:{port}",
                        parts.path or "",
                        parts.query,
                        parts.fragment,
                    )
                )
            )

        redirect = os.getenv("PRAPP_ROOT_REDIRECT", "").strip()
        if redirect and redirect != "/":
            if redirect.startswith("http://") or redirect.startswith("https://"):
                candidates.append(redirect)
            else:
                if not redirect.startswith("/"):
                    redirect = "/" + redirect
                candidates.append(base + redirect)

        for path in ("/app", "/ui", "/dashboard", "/index.html", "/"):
            candidates.append(base + path)

        seen: set[str] = set()
        deduped: list[str] = []
        for cand in candidates:
            if cand not in seen:
                seen.add(cand)
                deduped.append(cand)
        return deduped

    def _worker():
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if _can_connect(host, port, timeout=0.5):
                for candidate in _candidate_urls():
                    if _http_ok(candidate):
                        try:
                            webbrowser.open(candidate, new=1, autoraise=True)
                        finally:
                            return
            time.sleep(check_interval_s)
        # Timeout → liberar lock para futuros intentos
        _cleanup()

    t = threading.Thread(target=_worker, name="AutoOpenBrowser", daemon=True)
    t.start()
