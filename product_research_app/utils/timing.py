from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator


class TimingTracker:
    """Helper to accumulate elapsed seconds per named phase.

    The tracker is designed so import stages can record timings for expensive
    operations (``parse_xlsx``, ``drop_indexes``, ``bulk_insert``,
    ``rebuild_indexes``) and expose them through the status callback.  Each
    phase can be measured multiple times; elapsed seconds are accumulated.
    """

    def __init__(self) -> None:
        self._durations: Dict[str, float] = {}

    @contextmanager
    def measure(self, phase: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._durations[phase] = self._durations.get(phase, 0.0) + elapsed

    def add(self, phase: str, seconds: float) -> None:
        """Add ``seconds`` to the accumulated time for ``phase``."""

        if seconds is None:
            return
        self._durations[phase] = self._durations.get(phase, 0.0) + float(seconds)

    def snapshot(self) -> Dict[str, float]:
        """Return a copy of the collected timings rounded to microseconds."""

        return {name: round(total, 6) for name, total in self._durations.items()}


__all__ = ["TimingTracker"]
