def mount_health(app):
    """
    Incluye /health en FastAPI o Flask. Devuelve True si se montó.
    """
    try:
        # FastAPI
        from fastapi import APIRouter
        router = APIRouter()

        @router.get("/health")
        def _health():
            return {"status": "ok"}

        app.include_router(router)
        return True
    except Exception:
        pass

    try:
        # Flask
        from flask import Blueprint, jsonify
        bp = Blueprint("health", __name__)

        @bp.route("/health")
        def _health():
            return jsonify(status="ok")

        app.register_blueprint(bp)
        return True
    except Exception:
        pass

    return False
