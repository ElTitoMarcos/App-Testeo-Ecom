"""Compatibility layer exposing AI column helpers with progress callbacks."""

from __future__ import annotations

from typing import Any, Callable, Optional

from product_research_app.services.ai_columns import *  # noqa: F401,F403
from product_research_app.services.ai_columns import (  # noqa: F401
    ProgressCallback as _ServiceProgressCallback,
    run_ai_fill_job as _service_run_ai_fill_job,
)

ProgressCb = Optional[Callable[[int, int, str], None]]


def run_ai_fill_job(*args: Any, on_progress: ProgressCb = None, **kwargs: Any) -> Any:
    """Proxy run_ai_fill_job allowing ``on_progress`` override."""
    if on_progress is not None and "on_progress" in kwargs:
        kwargs["on_progress"] = on_progress
    elif "on_progress" not in kwargs:
        kwargs["on_progress"] = on_progress
    return _service_run_ai_fill_job(*args, **kwargs)


ProgressCallback = _ServiceProgressCallback
