"""Utilities for parsing product titles.

The :func:`analyze_titles` function accepts a list of dictionaries with at
least a ``title`` key.  It normalizes each title, tokenizes it while removing
English and Spanish stopwords, extracts useful signals, and flags potential
risks such as SEO bloat or trademark issues.
"""
from __future__ import annotations

import re
from typing import List, Dict, Any

# Minimal stopword lists to avoid external dependencies
EN_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "with", "on",
    "in", "at", "by", "from", "is", "are", "this", "that", "it", "as",
    "be", "has", "have", "new"
}
ES_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o",
    "de", "para", "con", "en", "por", "es", "son", "este", "esta",
    "estos", "estas", "lo", "al", "del", "nuevo"
}

GENERIC_ADJECTIVES = {
    "good", "great", "amazing", "awesome", "nice", "cool", "best",
    "quality", "premium", "perfect", "excellent", "ideal", "top"
}
MATERIALS = {
    "stainless", "steel", "silicone", "glass", "plastic", "rubber",
    "wood", "leather", "cotton", "nylon", "aluminum", "metal"
}
CLAIMS = {
    "waterproof", "portable", "leakproof", "leak-proof", "eco", "eco-friendly",
    "rechargeable", "wireless", "durable", "lightweight", "heavy", "duty",
    "breathable", "smart", "flexible", "foldable", "compact", "premium"
}
BRANDS = {
    "iphone", "samsung", "apple", "playstation", "nintendo", "lego",
    "disney", "harry", "marvel"
}
BRAND_OK_WORDS = {"official", "authentic", "licensed"}

SIZE_PATTERN = re.compile(
    r"\b\d+\s*(?:pack|pcs|pieces|x|oz|ml|l|g|kg|lb|cm|mm|in(?:ch(?:es)?)?|ft|pairs?)\b"
)
COMPAT_PATTERN = re.compile(r"for\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,3})")


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    tokens = [t for t in tokens if t not in EN_STOPWORDS and t not in ES_STOPWORDS]
    return tokens


def analyze_titles(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyze a list of product title entries.

    Returns each entry annotated with normalized tokens, extracted signals and
    risk flags.
    """
    results: List[Dict[str, Any]] = []
    for entry in items:
        title = entry.get("title")
        if not title:
            continue
        lower = title.lower()
        normalized = _normalize(title)
        tokens = _tokenize(title)

        sizes = SIZE_PATTERN.findall(lower)
        compatibility = COMPAT_PATTERN.findall(lower)
        materials = [t for t in tokens if t in MATERIALS]
        claims = [t for t in tokens if t in CLAIMS]

        signals = {}
        if sizes:
            signals["sizes"] = sizes
        if compatibility:
            signals["compatibility"] = compatibility
        if materials:
            signals["materials"] = materials
        if claims:
            signals["claims"] = claims

        seo_bloat = len(title) > 120
        genericity = False
        if not signals and tokens and all(t in GENERIC_ADJECTIVES for t in tokens):
            genericity = True
        brand_hits = [b for b in BRANDS if b in lower]
        ip_risk = bool(brand_hits) and not any(w in lower for w in BRAND_OK_WORDS)

        risks = {
            "seo_bloat": seo_bloat,
            "genericity": genericity,
            "ip_risk": ip_risk,
        }

        analysis = {
            "normalized": normalized,
            "tokens": tokens,
            "signals": signals,
            "risks": risks,
        }
        new_entry = dict(entry)
        new_entry["analysis"] = analysis
        results.append(new_entry)
    return results
