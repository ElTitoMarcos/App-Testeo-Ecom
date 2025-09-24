"""Utility helpers for the Product Research app."""

from __future__ import annotations

import re
from typing import Optional

FLAME = "\U0001F525"
_FLAMES_RE = re.compile(rf"{FLAME}+$")


def sanitize_product_name(name: Optional[str]):
    """Remove trailing flame emojis from product names."""

    if not isinstance(name, str):
        return name
    return _FLAMES_RE.sub("", name).strip()


__all__ = ["sanitize_product_name"]
