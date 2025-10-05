from __future__ import annotations
import atexit
import os
import socket
import tempfile
import threading
import time
import webbrowser


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

    def _worker():
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if _can_connect(host, port, timeout=0.5):
                try:
                    webbrowser.open(url, new=1, autoraise=True)
                finally:
                    return
            time.sleep(check_interval_s)
        # Timeout → liberar lock para futuros intentos
        _cleanup()

    t = threading.Thread(target=_worker, name="AutoOpenBrowser", daemon=True)
    t.start()
