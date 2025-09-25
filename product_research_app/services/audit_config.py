"""Configuration for post-import audit of critical product fields."""

from __future__ import annotations

from typing import Any, Mapping

REQUIRED_FIELDS: dict[str, dict[str, Any]] = {
    "desire": {"via": "DESIRE", "type": "text", "min_len": 280},
    "desire_primary": {"via": "DESIRE", "type": "enum"},
    "desire_magnitude": {"via": "DESIRE", "type": "object"},
    "awareness_level": {"via": "DESIRE", "type": "enum"},
    "competition_level": {"via": "DESIRE", "type": "enum"},
    "ai_desire_label": {"via": "DERIVED", "type": "text", "from": "desire_primary"},
}


def should_fill(field: str, row: Mapping[str, Any]) -> bool:
    """Return ``True`` if the audit should attempt to fill ``field``."""

    value = row.get(field)
    if field == "desire":
        text = str(value or "").strip()
        if not text:
            return True
        return len(text) < REQUIRED_FIELDS[field].get("min_len", 0)
    if field == "ai_desire_label":
        return value in (None, "")
    if REQUIRED_FIELDS.get(field, {}).get("type") in {"enum", "object"}:
        return value in (None, "")
    return value in (None, "")


__all__ = ["REQUIRED_FIELDS", "should_fill"]
