"""Utilities to track progress of multi-step pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

ProgressCallback = Callable[[float, str], None]


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

