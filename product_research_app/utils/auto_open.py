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

    # Lock para "abrir solo una vez" por sesión, con TTL configurable
    lock_path = os.path.join(tempfile.gettempdir(), lock_name)
    default_ttl = 300.0
    try:
        lock_ttl_env = os.getenv("PRAPP_AUTO_OPEN_LOCK_TTL", str(default_ttl))
        lock_ttl = float(lock_ttl_env)
        if lock_ttl < 0:
            lock_ttl = 0.0
    except (TypeError, ValueError):
        lock_ttl = default_ttl

    now = time.time()
    if os.path.exists(lock_path):
        try:
            mtime = os.path.getmtime(lock_path)
        except OSError:
            mtime = None
        if mtime is None or (lock_ttl and (now - mtime) > lock_ttl):
            try:
                os.remove(lock_path)
            except OSError:
                pass
    try:
        # Crear exclusión atómica
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
    except FileExistsError:
        # Otro proceso ya lo abrió (o lock todavía vigente)
        return

    def _cleanup():
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass
        except OSError:
            pass
    atexit.register(_cleanup)

    def _http_ok(u: str) -> bool:
        def _status_ok(resp) -> bool:
            status = getattr(resp, "status", getattr(resp, "code", 200))
            if 200 <= status < 400:
                headers = getattr(resp, "headers", None)
                if headers and headers.get("X-PRAPP-WELCOME") == "1":
                    return False
                return True
            return False

        try:
            req = Request(u, method="HEAD")
            with urlopen(req, timeout=2) as r:  # type: ignore[arg-type]
                return _status_ok(r)
        except Exception:
            try:
                req = Request(u, method="GET")
                with urlopen(req, timeout=2) as r:  # type: ignore[arg-type]
                    return _status_ok(r)
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

        paths_env = os.getenv("PRAPP_AUTO_OPEN_PATHS", "").strip()
        if paths_env:
            try:
                preferred = [p.strip() for p in paths_env.split(",") if p.strip()]
            except Exception:
                preferred = []
        else:
            preferred = ["/app", "/ui", "/dashboard", "/index.html", "/"]

        for path in preferred:
            if not path.startswith("/") and not path.startswith("http"):
                path = "/" + path
            if path.startswith("http://") or path.startswith("https://"):
                candidates.append(path)
            else:
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
