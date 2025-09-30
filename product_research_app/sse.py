import asyncio
import json
from typing import Any, AsyncIterator, Iterable, Iterator


class SSEBroker:
    def __init__(self) -> None:
        self._subs: set[asyncio.Queue[str]] = set()

    async def subscribe(self) -> AsyncIterator[str]:
        queue: asyncio.Queue[str] = asyncio.Queue()
        self._subs.add(queue)
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    # mantén la conexión viva para proxies
                    yield ": keep-alive\n\n"
        finally:
            self._subs.discard(queue)

    def subscribe_sync(self) -> Iterator[str]:
        policy = asyncio.get_event_loop_policy()
        try:
            prev_loop = policy.get_event_loop()
        except RuntimeError:
            prev_loop = None
        loop = policy.new_event_loop()
        policy.set_event_loop(loop)
        queue: asyncio.Queue[str] = asyncio.Queue()
        policy.set_event_loop(prev_loop)
        self._subs.add(queue)

        def generator() -> Iterator[str]:
            try:
                while True:
                    try:
                        msg = loop.run_until_complete(asyncio.wait_for(queue.get(), timeout=15))
                        yield f"data: {msg}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
            finally:
                self._subs.discard(queue)
                loop.close()
                if prev_loop is not None:
                    policy.set_event_loop(prev_loop)

        return generator()

    def publish(self, event: str, data: dict[str, Any]) -> None:
        payload = json.dumps({"event": event, "data": data}, ensure_ascii=False)
        for queue in list(self._subs):
            queue.put_nowait(payload)


def publish_progress(payload: dict[str, Any]) -> None:
    event = str(payload.get("event") or "message")
    data = {k: v for k, v in payload.items() if k != "event"}
    broker.publish(event, data)


def publish_many(event: str, payloads: Iterable[dict[str, Any]]) -> None:
    for data in payloads:
        broker.publish(event, data)


broker = SSEBroker()

try:  # pragma: no cover - Flask compat optional
    from flask import Blueprint, Response, stream_with_context  # type: ignore
except Exception:  # pragma: no cover - Flask not installed in FastAPI path
    sse_bp = None  # type: ignore
else:  # pragma: no cover - legacy shim
    sse_bp = Blueprint("sse", __name__)

    @sse_bp.route("/events/ai")
    def events_ai() -> Response:
        headers = {
            "Cache-Control": "no-cache, no-transform",
            "Content-Type": "text/event-stream",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        stream = broker.subscribe_sync()
        return Response(stream_with_context(stream), headers=headers, mimetype="text/event-stream")
