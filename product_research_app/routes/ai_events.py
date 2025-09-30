"""AI progress streaming endpoints."""

from __future__ import annotations

from typing import Any, Callable, Dict

from flask import Blueprint, Response, jsonify, stream_with_context

from ..utils.event_broker import broker


ai_events_bp = Blueprint("ai_events", __name__)

_last_progress: Dict[str, Any] = {"progress": 0.0, "status": "idle", "label": "Listo"}


def _sse_headers() -> Dict[str, str]:
    return {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Access-Control-Allow-Origin": "*",
    }


@ai_events_bp.route("/events/ai")
def events_ai() -> Response:
    generator = broker.subscribe_sync()

    @stream_with_context
    def event_stream():
        for chunk in generator:
            yield chunk

    return Response(event_stream(), headers=_sse_headers())


@ai_events_bp.route("/api/ai/progress")
def get_ai_progress() -> Response:
    return jsonify(_last_progress)


def _update_progress_for_legacy(progress: float, status: str) -> None:
    pct = max(0.0, min(float(progress), 1.0))
    _last_progress["progress"] = pct
    normalized_status = "running" if str(status).startswith("running") else status
    _last_progress["status"] = normalized_status
    _last_progress["label"] = "IA Generando..." if normalized_status == "running" else "Listo"


def legacy_progress_updater() -> Callable[[float, str], None]:
    def _callback(pct: float, message: str) -> None:
        status_msg = message or "running"
        _update_progress_for_legacy(pct, status_msg)

    return _callback


# Event conventions consumed by the UI:
# - ai.progress {progress: 0..1}
# - ai.done     {progress: 1.0, reload: true}
# - products.updated / products.reload


__all__ = [
    "ai_events_bp",
    "legacy_progress_updater",
]

