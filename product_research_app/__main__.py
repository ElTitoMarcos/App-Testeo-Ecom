# product_research_app/__main__.py
import os, sys, importlib

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))

def _import_app():
    """
    Intenta obtener la instancia de aplicación desde product_research_app.web_app
    en el siguiente orden:
    1) app (Flask o FastAPI)
    2) create_app() -> app
    """
    try:
        mod = importlib.import_module("product_research_app.web_app")
    except Exception as e:
        print(f"[ERROR] No se pudo importar product_research_app.web_app: {e}", file=sys.stderr)
        raise

    # 1) Atributo app
    app = getattr(mod, "app", None)
    if app is not None:
        return app, mod

    # 2) Fábrica
    create_app = getattr(mod, "create_app", None)
    if callable(create_app):
        app = create_app()
        return app, mod

    raise RuntimeError("No se encontró 'app' ni 'create_app()' en product_research_app.web_app")

def _is_fastapi_app(app):
    try:
        from fastapi import FastAPI  # type: ignore
        return isinstance(app, FastAPI)
    except Exception:
        return False

def _ensure_flask_health(app):
    # Si es Flask y no hay /health, añadirlo
    try:
        from flask import Flask
        if isinstance(app, Flask):
            has_health = any(r.rule == "/health" for r in app.url_map.iter_rules())
            if not has_health:
                @app.get("/health")  # type: ignore[attr-defined]
                def _health():
                    return {"status": "ok"}
    except Exception:
        pass

def main():
    app, mod = _import_app()
    print(f"SERVER: starting on http://{HOST}:{PORT}", flush=True)

    if _is_fastapi_app(app):
        # FastAPI -> Uvicorn
        try:
            import uvicorn
        except ImportError:
            print("[ERROR] Falta 'uvicorn'. Añádelo a requirements.txt", file=sys.stderr)
            sys.exit(1)
        uvicorn.run(app, host=HOST, port=PORT, log_level="info", reload=False)
    else:
        # Flask (u otro WSGI compatible que tenga .run similar)
        _ensure_flask_health(app)
        # Flask.run bloquea el proceso
        try:
            app.run(host=HOST, port=PORT, debug=False, use_reloader=False)  # type: ignore[attr-defined]
        except TypeError:
            # Fallback muy defensivo por si es otra WSGI con firma distinta
            app.run(host=HOST, port=PORT)  # type: ignore[attr-defined]

if __name__ == "__main__":
    main()
