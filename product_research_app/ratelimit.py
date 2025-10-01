"""Backward compatible façade for the new async rate limiter."""

from __future__ import annotations

from product_research_app.utils.rate_limiter import (
    async_decorrelated_jitter_sleep,
    decorrelated_jitter_sleep,
    limiter as _limiter,
)


def get_async_limiter():
    """Return the shared async rate limiter instance."""

    return _limiter


__all__ = ["get_async_limiter", "async_decorrelated_jitter_sleep", "decorrelated_jitter_sleep"]
