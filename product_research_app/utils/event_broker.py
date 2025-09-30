"""Simple asynchronous event broker for SSE streaming."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import AsyncIterator, Dict, Iterable, Iterator, List


def _json_dumps(payload: Dict) -> str:
    """Serialize payload defensively for SSE."""

    try:
        return json.dumps(payload, separators=(",", ":"))
    except TypeError:
        safe: Dict[str, object] = {}
        for key, value in payload.items():
            try:
                json.dumps(value)
            except TypeError:
                safe[key] = str(value)
            else:
                safe[key] = value  # type: ignore[assignment]
        return json.dumps(safe, separators=(",", ":"))


class AsyncEventBroker:
    """Very small async-friendly broker for Server-Sent Events."""

    def __init__(self) -> None:
        self._queues: List[asyncio.Queue] = []
        self._lock = asyncio.Lock()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    async def _register_queue(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._queues.append(q)
        return q

    async def _remove_queue(self, q: asyncio.Queue) -> None:
        async with self._lock:
            if q in self._queues:
                self._queues.remove(q)

    async def subscribe(self) -> AsyncIterator[str]:
        q = await self._register_queue()
        try:
            while True:
                msg = await q.get()
                yield f"data: {_json_dumps(msg)}\n\n"
        finally:
            await self._remove_queue(q)

    def subscribe_sync(self) -> Iterator[str]:
        fut = asyncio.run_coroutine_threadsafe(self._register_queue(), self._loop)
        q = fut.result()

        def generator() -> Iterable[str]:
            try:
                while True:
                    msg_fut = asyncio.run_coroutine_threadsafe(q.get(), self._loop)
                    msg = msg_fut.result()
                    yield f"data: {_json_dumps(msg)}\n\n"
            finally:
                asyncio.run_coroutine_threadsafe(self._remove_queue(q), self._loop).result()

        return iter(generator())

    async def _publish(self, msg: Dict) -> None:
        payload = dict(msg)
        payload.setdefault("ts", time.time())
        async with self._lock:
            targets = list(self._queues)
        for q in targets:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                continue

    def publish(self, msg: Dict) -> None:
        asyncio.run_coroutine_threadsafe(self._publish(msg), self._loop)


# Module level singleton for convenience.
broker = AsyncEventBroker()

