"""Utility helpers for the Product Research app."""

from __future__ import annotations

import re

FLAME = "\U0001F525"
_FLAMES_RE = re.compile(rf"{FLAME}+$")


def sanitize_product_name(name: str | None):
    """Remove trailing flame emojis from product names."""

    if not isinstance(name, str):
        return name
    return _FLAMES_RE.sub("", name).strip()


__all__ = ["sanitize_product_name"]
