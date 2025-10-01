"""Flask application factory and blueprint registration."""

import json

from flask import Flask, Response, stream_with_context

app = Flask(__name__)

# Import API modules which attach routes to ``app``.
from . import config  # noqa: E402,F401
from .winner_score import winner_score_api  # noqa: E402
from ..sse import sse_bp  # noqa: E402
from ..services.ai_progress import (
    get_progress_payload,
    subscribe_progress_queue,
    unsubscribe_progress_queue,
)

app.register_blueprint(winner_score_api, url_prefix="/api")
app.register_blueprint(sse_bp)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.route("/api/ai/events")
def ai_events() -> Response:
    def stream() -> Response:
        queue = subscribe_progress_queue()
        try:
            yield "event: hello\ndata: {}\n\n"
            while True:
                event = queue.get()
                if not isinstance(event, dict):
                    continue
                event_type = event.get("type", "progress")
                data = event.get("data", {})
                try:
                    payload = json.dumps(data)
                except Exception:
                    payload = "{}"
                yield f"event: {event_type}\ndata: {payload}\n\n"
                if event_type in {"done", "error"}:
                    break
        finally:
            unsubscribe_progress_queue(queue)

    return Response(
        stream_with_context(stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/ai/progress")
def ai_progress() -> dict:
    return get_progress_payload()


# Log registered routes for easier debugging in start-up logs.
for r in app.url_map.iter_rules():
    app.logger.info("ROUTE %s %s", ",".join(sorted(r.methods)), r.rule)
