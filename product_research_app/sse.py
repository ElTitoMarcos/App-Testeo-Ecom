"""Server-Sent Events helpers and blueprint."""

from __future__ import annotations

import json
import queue
from typing import Any, Dict, Iterable, Iterator, Optional

from flask import Blueprint, Response, stream_with_context

from . import progress_events

sse_bp = Blueprint("sse", __name__)

_HEADERS = {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
    "Access-Control-Allow-Origin": "*",
}


def _event_stream(subscriber: Any) -> Iterator[str]:
    """Yield events from a subscriber queue with keepalive comments."""

    timeout = getattr(progress_events, "KEEPALIVE_INTERVAL", 10.0)
    q: "queue.Queue[Dict[str, Any]]" = getattr(subscriber, "queue", None)
    if q is None:
        maxsize = getattr(progress_events, "QUEUE_SIZE", 0) or 0
        q = queue.Queue(maxsize=maxsize or 0)
    while True:
        try:
            event = q.get(timeout=timeout)
        except queue.Empty:
            yield ":keepalive\n\n"
            continue
        try:
            payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            continue
        yield f"data: {payload}\n\n"


@sse_bp.route("/events")
def events() -> Response:
    """Stream progress events to connected clients using SSE."""

    subscriber = progress_events.subscribe()

    def generate() -> Iterable[str]:
        try:
            yield from _event_stream(subscriber)
        finally:
            progress_events.unsubscribe(subscriber)

    return Response(stream_with_context(generate()), headers=_HEADERS)


def publish_progress(
    job_id_or_payload: Any,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Publish a progress event to all subscribers.

    Supports both ``publish_progress(payload_dict)`` and
    ``publish_progress(job_id, payload_dict)`` call styles for backwards
    compatibility with existing modules.
    """

    if payload is None:
        if job_id_or_payload is None:
            return
        if not isinstance(job_id_or_payload, dict):
            raise TypeError("publish_progress requires a dict payload when called with a single argument")
        event = dict(job_id_or_payload)
        job_id = event.get("job_id")
    else:
        job_id = job_id_or_payload
        event = dict(payload or {})
        if job_id is not None:
            event.setdefault("job_id", job_id)
    progress_events.publish_progress(job_id, event)


__all__ = ["sse_bp", "publish_progress"]
