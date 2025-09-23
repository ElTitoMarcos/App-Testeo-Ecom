# smoketest.py
# Test de humo del servidor: crea logs, lanza el entrypoint, espera readiness del URL y finaliza.
import argparse, os, sys, time, subprocess, urllib.request, urllib.error, signal, textwrap, pathlib


def tail(path, n=40):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            size = 0
            block = 1024
            chunks = []
            while end > 0 and size < n * 120:
                step = min(block, end)
                end -= step
                f.seek(end)
                data = f.read(step)
                chunks.append(data)
                size += len(data)
            return b"".join(reversed(chunks)).decode(errors="replace").splitlines()[-n:]
    except Exception:
        return []


def http_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            # Cualquier respuesta HTTP indica que el servidor está vivo, aunque sea 404
            return 100 <= getattr(r, "status", 200) < 600
    except urllib.error.HTTPError as e:
        return 100 <= e.code < 600
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser(description="Smoke test del servidor")
    p.add_argument("--entrypoint", default=os.environ.get("ENTRYPOINT", "main.py"))
    p.add_argument("--url", default=os.environ.get("TEST_URL", "http://127.0.0.1:8000"))
    p.add_argument("--timeout", type=int, default=int(os.environ.get("TEST_TIMEOUT", "60")))
    p.add_argument("--python", default=sys.executable, help="Ruta a python para lanzar el server (normalmente el del venv)")
    args = p.parse_args()

    root = pathlib.Path(__file__).resolve().parent
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    out_log = logs / "test_server.out.log"
    err_log = logs / "test_server.err.log"

    print(f"[TEST] Lanzando {args.entrypoint} con {args.python}")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    with open(out_log, "wb") as out, open(err_log, "wb") as err:
        proc = subprocess.Popen(
            [args.python, args.entrypoint],
            cwd=str(root),
            stdout=out,
            stderr=err,
            env=env,
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
        )
        try:
            deadline = time.time() + args.timeout
            ready = False
            print(f"[TEST] Esperando readiness de {args.url} (timeout {args.timeout}s)...")
            while time.time() < deadline:
                if http_ready(args.url):
                    ready = True
                    break
                if proc.poll() is not None:
                    break
                time.sleep(1)

            if not ready:
                code = proc.poll()
                print("[FAIL] El servidor no respondió a tiempo.")
                if code is not None:
                    print(f"[INFO] Proceso terminó prematuramente con código {code}.")
                tl = tail(err_log, 40)
                if tl:
                    print("\n[LOGS: últimos errores]")
                    print("\n".join(tl))
                sys.exit(2)

            print("[OK] Servidor responde. Pasó el smoke test.")
            sys.exit(0)
        finally:
            # Intentar apagar limpio
            if proc.poll() is None:
                if os.name == "nt":
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                    time.sleep(1)
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=8)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass


if __name__ == "__main__":
    main()
