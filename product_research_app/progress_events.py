"""Simple in-process event bus for streaming progress updates via SSE."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)


KEEPALIVE_INTERVAL = 10.0
QUEUE_SIZE = 256


@dataclass(eq=False)
class _Subscriber:
    queue: "queue.Queue[Dict[str, Any]]"
    created_at: float
    closed: bool = False

    def push(self, payload: Dict[str, Any]) -> None:
        if self.closed:
            return
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(payload)
            except queue.Full:
                logger.debug("progress-bus queue saturated; dropping event")

    def close(self) -> None:
        self.closed = True


class _ProgressBus:
    def __init__(self) -> None:
        self._subscribers: set[_Subscriber] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> _Subscriber:
        sub = _Subscriber(queue.Queue(maxsize=QUEUE_SIZE), time.time())
        with self._lock:
            self._subscribers.add(sub)
        return sub

    def unsubscribe(self, subscriber: _Subscriber) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)
        subscriber.close()

    def publish(self, event: Dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("timestamp", time.time())
        # ensure JSON serialisable to avoid breaking SSE loop later on
        try:
            json.dumps(payload)
        except TypeError:
            payload = json.loads(json.dumps(payload, default=str))
        with self._lock:
            subscribers: Iterable[_Subscriber] = tuple(self._subscribers)
        stale: list[_Subscriber] = []
        for sub in subscribers:
            if sub.closed:
                stale.append(sub)
                continue
            sub.push(payload)
        if stale:
            with self._lock:
                for sub in stale:
                    self._subscribers.discard(sub)


_BUS = _ProgressBus()


def publish_progress(job_id: Any, payload: Dict[str, Any]) -> None:
    """Publish a progress event for the unified SSE stream."""

    if payload is None:
        return
    event: Dict[str, Any] = dict(payload)
    if job_id is not None:
        event.setdefault("job_id", job_id)
    operation = event.get("operation")
    if not operation:
        logger.debug("progress event skipped (missing operation): %s", event)
        return
    _BUS.publish(event)


def subscribe() -> _Subscriber:
    return _BUS.subscribe()


def unsubscribe(subscriber: _Subscriber) -> None:
    _BUS.unsubscribe(subscriber)


__all__ = ["publish_progress", "subscribe", "unsubscribe", "KEEPALIVE_INTERVAL"]
