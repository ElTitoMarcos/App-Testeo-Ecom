"""Utilities to compute stable signatures for products.

The signature is used to de‑duplicate AI calls and cache their results.  It
combines a subset of product identifying attributes (name, brand, ASIN and
URL) after normalising case and whitespace.  The output is a SHA‑1 hex digest
which is deterministic for the same logical product.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional

_SIG_NORMALIZE_RE = re.compile(r"\s+")


def normalize_sig_part(value: Optional[str]) -> str:
    """Normalise a signature component.

    Strings are lowercased, leading/trailing whitespace removed and internal
    whitespace collapsed to a single space.  ``None`` values are converted to
    empty strings so they do not influence the hash.
    """

    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    return _SIG_NORMALIZE_RE.sub(" ", text)


def compute_sig_hash(
    name: str,
    brand: Optional[str] = None,
    asin: Optional[str] = None,
    url: Optional[str] = None,
) -> str:
    """Return a SHA‑1 signature for the provided fields."""

    payload = "|".join(
        normalize_sig_part(part) for part in (name, brand, asin, url)
    )
    if not payload:
        return ""
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


__all__ = ["compute_sig_hash", "normalize_sig_part"]
