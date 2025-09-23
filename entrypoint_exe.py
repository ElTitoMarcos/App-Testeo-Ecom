# entrypoint_exe.py
# Arranca la app web empacada con PyInstaller.
# Orden de intento: product_research_app.web_app -> product_research_app -> autodetección + .web_app
import os, sys, threading, time, runpy, socket, webbrowser


def _port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.25)
        return s.connect_ex((host, port)) == 0


def _try_open_browser():
    # Busca un puerto disponible entre 8000–8005 y abre el navegador cuando responda
    deadline = time.time() + 30
    while time.time() < deadline:
        for p in range(8000, 8006):
            if _port_is_open("127.0.0.1", p):
                try:
                    webbrowser.open(f"http://127.0.0.1:{p}", new=2)
                except Exception:
                    pass
                return
        time.sleep(0.5)


def _find_pkg():
    # Paquete por defecto y autodetección si el nombre difiere
    default_pkg = "product_research_app"
    # Busca en el cwd un directorio con __init__.py
    try:
        for d in os.listdir(os.getcwd()):
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, "__init__.py")):
                return d
    except Exception:
        pass
    return default_pkg


def main():
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    threading.Thread(target=_try_open_browser, daemon=True).start()

    pkgs = ["product_research_app.web_app", "product_research_app"]
    detected = _find_pkg()
    if detected and detected not in ("product_research_app",):
        pkgs = [f"{detected}.web_app", detected] + pkgs  # intenta detectado primero

    last_err = None
    for mod in pkgs:
        try:
            runpy.run_module(mod, run_name="__main__")
            return
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 0
            sys.exit(code)
        except Exception as e:
            last_err = e

    raise SystemExit(f"No se pudo iniciar ningún módulo válido. Último error: {last_err!r}")


if __name__ == "__main__":
    main()
