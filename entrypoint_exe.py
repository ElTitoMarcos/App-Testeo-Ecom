# entrypoint_exe.py — arranque empaquetado
import os, sys, threading, time, runpy, socket, webbrowser

def _port_is_open(host, port):
    import socket
    s = socket.socket()
    s.settimeout(0.25)
    try:
        s.connect((host, port)); s.close(); return True
    except Exception:
        s.close(); return False

def _open_browser_when_ready():
    deadline = time.time() + 30
    while time.time() < deadline:
        for p in range(8000, 8006):
            if _port_is_open("127.0.0.1", p):
                try: webbrowser.open(f"http://127.0.0.1:{p}", new=2)
                except Exception: pass
                return
        time.sleep(0.5)

def _detect_pkg():
    default_pkg = "product_research_app"
    try:
        for d in os.listdir(os.getcwd()):
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, "__init__.py")):
                return d
    except Exception:
        pass
    return default_pkg

def main():
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    threading.Thread(target=_open_browser_when_ready, daemon=True).start()
    pkgs = ["product_research_app.web_app", "product_research_app"]
    detected = _detect_pkg()
    if detected and detected not in ("product_research_app",):
        pkgs = [f"{detected}.web_app", detected] + pkgs
    last = None
    for mod in pkgs:
        try:
            runpy.run_module(mod, run_name="__main__"); return
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 0
            sys.exit(code)
        except Exception as e:
            last = e
    raise SystemExit(f"No se pudo iniciar ningún módulo válido. Último error: {last!r}")

if __name__ == "__main__":
    main()
