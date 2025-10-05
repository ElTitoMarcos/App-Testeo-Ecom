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

    try:
        from fastapi import APIRouter
        from fastapi.responses import HTMLResponse, RedirectResponse

        router = APIRouter()

        @router.get("/", name="root.index")
        def _index():  # type: ignore[return-type]
            target = os.getenv("PRAPP_ROOT_REDIRECT", "").strip()
            if target and target != "/":
                return RedirectResponse(url=target, status_code=302)
            return HTMLResponse(content=_welcome_html(), status_code=200)

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
            if target and target != "/":
                return redirect(target, code=302)
            return Response(_welcome_html(), mimetype="text/html; charset=utf-8")

        app.register_blueprint(blueprint)
        return True
    except Exception:
        pass

    return False
