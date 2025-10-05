from __future__ import annotations

import os


def _welcome_html() -> str:
    return """<!doctype html>
    <meta charset=\"utf-8\">
    <title>Ecom Testing App</title>
    <h1>Ecom Testing App</h1>
    <p>El servidor está corriendo.</p>
    <ul>
      <li><a href=\"/health\">/health</a></li>
      <li><a href=\"/app\">/app</a> (si existe)</li>
    </ul>
    """


def mount_root(app) -> bool:
    """Ensure `/` exists, redirecting if configured."""

    try:
        for route in getattr(app, "routes", []):
            if getattr(route, "path", None) == "/":
                return True
    except Exception:
        pass

    try:
        for rule in getattr(getattr(app, "url_map", None), "iter_rules", lambda: [])():
            if getattr(rule, "rule", None) == "/":
                return True
    except Exception:
        pass

    def _has_path_fastapi(a, path: str) -> bool:
        try:
            for route in getattr(a, "routes", []):
                if getattr(route, "path", None) == path:
                    return True
        except Exception:
            pass
        return False

    def _has_path_flask(a, path: str) -> bool:
        try:
            url_map = getattr(a, "url_map", None)
            if url_map is not None:
                for rule in url_map.iter_rules():
                    if getattr(rule, "rule", None) == path:
                        return True
        except Exception:
            pass
        return False

    def _detect_ui_path(a) -> str | None:
        for path in ("/app", "/ui"):
            if _has_path_fastapi(a, path) or _has_path_flask(a, path):
                return path
        return None

    try:
        from fastapi import APIRouter
        from fastapi.responses import HTMLResponse, RedirectResponse

        router = APIRouter()

        @router.get("/", name="root.index")
        def _index():  # type: ignore[return-type]
            target = os.getenv("PRAPP_ROOT_REDIRECT", "").strip()
            if not target:
                auto = _detect_ui_path(app)
                if auto:
                    target = auto
            if target and target != "/":
                return RedirectResponse(url=target, status_code=302)
            return HTMLResponse(
                content=_welcome_html(),
                status_code=200,
                headers={"X-PRAPP-WELCOME": "1"},
            )

        app.include_router(router)
        return True
    except Exception:
        pass

    try:
        from flask import Blueprint, Response, redirect

        blueprint = Blueprint("root", __name__)

        @blueprint.route("/")
        def index():  # type: ignore[return-type]
            target = os.getenv("PRAPP_ROOT_REDIRECT", "").strip()
            if not target:
                auto = _detect_ui_path(app)
                if auto:
                    target = auto
            if target and target != "/":
                return redirect(target, code=302)
            response = Response(_welcome_html(), mimetype="text/html; charset=utf-8")
            try:
                response.headers["X-PRAPP-WELCOME"] = "1"
            except Exception:
                pass
            return response

        app.register_blueprint(blueprint)
        return True
    except Exception:
        pass

    return False
