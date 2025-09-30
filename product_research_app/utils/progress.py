"""Utilities to track progress of multi-step pipelines."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

ProgressCallback = Callable[[float, str], None]


def _env_float(key: str, default: float) -> float:
    """Obtain a float value from environment variables with fallback."""

    try:
        return float(os.getenv(key, str(default)))
    except Exception:
        return default


@dataclass
class ProgressTracker:
    total: int
    on_progress: Optional[ProgressCallback] = None
    _done: int = field(default=0, init=False)

    def step(self, inc: int = 1, message: str = "") -> float:
        self._done += max(0, inc)
        pct = 0.0 if self.total <= 0 else self._done / max(self.total, 1)
        pct = min(max(pct, 0.0), 1.0)
        if self.on_progress:
            self.on_progress(pct, message)
        return pct

    def update_absolute(self, value: int, message: str = "") -> float:
        if value < 0:
            value = 0
        prev_done = self._done
        self._done = max(self._done, value)
        pct = 0.0 if self.total <= 0 else self._done / max(self.total, 1)
        pct = min(max(pct, 0.0), 1.0)
        if self.on_progress and (self._done != prev_done or message):
            self.on_progress(pct, message)
        return pct

    def force_100(self, message: str = "done") -> float:
        self._done = max(self._done, self.total if self.total > 0 else 1)
        if self.on_progress:
            self.on_progress(1.0, message)
        return 1.0


class TimeProgressLoop:
    """Smooth progress driver based on estimated per-item duration."""

    def __init__(self, n_items: int, on_progress: Optional[ProgressCallback] = None):
        self.n_items = max(int(n_items or 0), 0)
        self.on_progress = on_progress
        self._tick = _env_float("PRAPP_AI_PROGRESS_TICK", 0.4)
        base = _env_float("PRAPP_AI_ITEM_SEC_BASE", 3.0)
        offset = _env_float("PRAPP_AI_ITEM_SEC_OFFSET", -0.2)
        per_item = max(base + offset, 0.05)
        self._total_time = max(self.n_items * per_item, 0.2)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._thread: Optional[threading.Thread] = None
        self._start = 0.0

    async def _run(self) -> None:
        self._start = time.monotonic()
        self._running = True
        cap = 0.98
        while self._running:
            elapsed = max(time.monotonic() - self._start, 0.0)
            pct = 0.0 if self._total_time <= 0 else min(elapsed / self._total_time, cap)
            if self.on_progress:
                try:
                    self.on_progress(pct, "running-time")
                except Exception:
                    pass
            await asyncio.sleep(self._tick)

    def start(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._task = None
            self._thread = threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True)
            self._thread.start()
        else:
            self._task = loop.create_task(self._run())
            self._thread = None

    def stop_and_force_100(self) -> None:
        self._running = False
        if self.on_progress:
            try:
                self.on_progress(1.0, "done")
            except Exception:
                pass
        if self._task and not self._task.done():
            self._task.cancel()

