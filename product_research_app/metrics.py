"""Lightweight helpers to accumulate usage metrics for AI calls."""

from __future__ import annotations

import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)

_Number = Union[int, float]

_usage_totals = {"prompt_tokens": 0, "completion_tokens": 0}


def add_usage(prompt_tokens: Optional[_Number], completion_tokens: Optional[_Number]) -> None:
    """Accumulate token usage for observability dashboards."""

    try:
        if prompt_tokens is not None:
            _usage_totals["prompt_tokens"] += max(0, int(prompt_tokens))
        if completion_tokens is not None:
            _usage_totals["completion_tokens"] += max(0, int(completion_tokens))
    except Exception:  # pragma: no cover - defensive logging only
        logger.debug("metrics.add_usage failed", exc_info=True)
        return

    logger.debug(
        "metrics.add_usage: prompt=%s completion=%s totals=%s",
        prompt_tokens,
        completion_tokens,
        dict(_usage_totals),
    )


def get_totals() -> dict:
    """Return the current aggregated usage totals."""

    return dict(_usage_totals)
