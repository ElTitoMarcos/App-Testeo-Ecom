"""Server-Sent Events helpers and blueprint."""

from __future__ import annotations

import json
import logging
import queue
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any

from flask import Blueprint, Response, stream_with_context

from .settings import SSE_ENABLED

logger = logging.getLogger(__name__)

sse_bp = Blueprint("sse", __name__)
_clients: set[queue.Queue[str]] = set()
_clients_lock = threading.Lock()


def _headers() -> dict[str, str]:
    return {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Access-Control-Allow-Origin": "*",
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date, Path)):
        return str(value)
    if isinstance(value, set):
        return list(value)
    try:
        return str(value)
    except Exception:  # pragma: no cover - defensive fallback
        return repr(value)


def _format_message(event: str | None, payload: dict[str, Any]) -> str:
    try:
        data = json.dumps(payload, separators=(",", ":"), default=_json_default)
    except TypeError:  # pragma: no cover - defensive fallback
        logger.exception("Failed to encode SSE payload")
        return ""
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {data}\n\n"


def _broadcast(message: str) -> None:
    if not message:
        return
    dead: list[queue.Queue[str]] = []
    with _clients_lock:
        targets = list(_clients)
    for q in targets:
        try:
            q.put_nowait(message)
        except queue.Full:
            dead.append(q)
    if dead:
        with _clients_lock:
            for q in dead:
                _clients.discard(q)


def publish_progress(payload: dict[str, Any]) -> None:
    """Broadcast a JSON payload to all connected SSE clients."""

    if not SSE_ENABLED:
        return
    msg = _format_message(None, payload)
    _broadcast(msg)


def publish_event(event: str, payload: dict[str, Any]) -> None:
    if not SSE_ENABLED:
        return
    msg = _format_message(event, payload)
    _broadcast(msg)


@sse_bp.route("/events")
def events() -> Response:
    if not SSE_ENABLED:
        return Response("", status=204)
    client_queue: queue.Queue[str] = queue.Queue(maxsize=1000)
    with _clients_lock:
        _clients.add(client_queue)

    def gen():
        try:
            while True:
                try:
                    msg = client_queue.get(timeout=10)
                except queue.Empty:
                    yield ":keepalive\n\n"
                else:
                    yield msg
        finally:
            with _clients_lock:
                _clients.discard(client_queue)

    return Response(stream_with_context(gen()), headers=_headers())
