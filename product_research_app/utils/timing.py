"""Timing utilities for coarse-grained profiling."""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Dict, Iterator

_LOGGER = logging.getLogger("product_research_app.timing")


@contextlib.contextmanager
def phase(name: str) -> Iterator[Dict[str, int]]:
    """Measure the duration of a logical phase.

    The context manager logs the start and end of the phase and exposes the
    measured duration in milliseconds via the yielded mapping.  The mapping is
    mutated in-place when the context exits so callers can inspect ``["ms"]``
    afterwards.
    """

    start = time.perf_counter()
    payload: Dict[str, int] = {"name": name, "ms": 0}
    _LOGGER.info("phase_start %s", name)
    try:
        yield payload
    finally:
        elapsed_ms = int(round((time.perf_counter() - start) * 1000))
        payload["ms"] = elapsed_ms
        _LOGGER.info("phase_end %s %sms", name, elapsed_ms)
