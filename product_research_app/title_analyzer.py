"""Utilities for parsing product titles.

The :func:`analyze_titles` function accepts a list of dictionaries with at
least a ``title`` key.  It normalizes each title, tokenizes it while removing
English and Spanish stopwords, extracts useful signals, and flags potential
risks such as SEO bloat or trademark issues.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Dict, Any, Mapping, Optional, Tuple

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
GENERIC_NOUNS = {"kitchen", "set", "kit", "product", "item"}
MATERIALS = {
    "stainless", "steel", "silicone", "glass", "plastic", "rubber",
    "wood", "leather", "cotton", "nylon", "aluminum", "metal"
}
CLAIMS = {
    "waterproof", "portable", "leakproof", "leak-proof", "eco", "eco-friendly",
    "rechargeable", "wireless", "durable", "lightweight", "heavy", "duty",
    "breathable", "smart", "flexible", "foldable", "compact"
}
BRANDS = {
    "iphone", "samsung", "apple", "playstation", "nintendo", "lego",
    "disney", "harry", "marvel"
}
BRAND_OK_WORDS = {"official", "authentic", "licensed"}

DEFAULT_WEIGHTS = {
    "w1": 0.25,  # claim_strength
    "w2": 0.25,  # value_signals
    "w3": 0.2,   # targeting
    "w4": 0.15,  # genericity
    "w5": 0.1,   # seo_bloat
    "w6": 0.2,   # ip_risk
}

SIZE_PATTERN = re.compile(
    r"\b\d+\s*(?:pack|pcs|pieces|x|oz|ml|l|g|kg|lb|cm|mm|in(?:ch(?:es)?)?|ft|pairs?)\b"
)
COMPAT_PATTERN = re.compile(
    r"for\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,3})", re.IGNORECASE
)


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    tokens = [t for t in tokens if t not in EN_STOPWORDS and t not in ES_STOPWORDS]
    return tokens


def _compute_quantiles(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Return approximate 33%% and 66%% quantiles for ``values``."""
    if not values:
        return None, None
    values = sorted(values)
    n = len(values)
    q1_idx = max(0, min(n - 1, int(n * 0.33)))
    q2_idx = max(0, min(n - 1, int(n * 0.66)))
    return values[q1_idx], values[q2_idx]


def _price_bucket(price: Optional[float], q1: Optional[float], q2: Optional[float]) -> Optional[str]:
    if price is None or q1 is None or q2 is None:
        return None
    if price <= q1:
        return "low"
    if price <= q2:
        return "mid"
    return "high"


def _generate_summary(
    signals: Dict[str, Any],
    risks: Dict[str, bool],
    tokens: List[str],
    claims: List[str],
    repeats: List[str],
    price: Optional[float],
    bucket: Optional[str],
) -> str:
    """Craft the Spanish summary string for a product."""

    comentarios = []
    if claims:
        comentarios.append("claims: " + ", ".join(sorted(set(claims))))
    if signals.get("value"):
        comentarios.append("tamaños: " + ", ".join(signals["value"]))
    if signals.get("compat"):
        comentarios.append("compatibilidad: " + ", ".join(signals["compat"]))
    comentarios_str = "; ".join(comentarios) if comentarios else "sin señales claras"

    pros = []
    if claims:
        pros.append("valor percibido")
    if signals.get("materials"):
        pros.append("señales de calidad")
    if signals.get("compat"):
        pros.append("target claro")
    if signals.get("value"):
        pros.append("resuelve dudas de tamaño")
    pros_str = "; ".join(pros) if pros else "ninguno"

    contras = []
    if risks.get("seo_bloat"):
        contras.append("título largo (SEO bloat)")
    if risks.get("genericity"):
        contras.append("claims genéricos")
    if risks.get("ip_risk"):
        contras.append("riesgo de marca")
    if not any(signals.values()):
        contras.append("falta diferenciación")
    contras_str = "; ".join(contras) if contras else "sin contras visibles"

    if any(signals.values()):
        compet = "Mejor que la competencia: incluye señales diferenciadoras."
    else:
        compet = "Peor que la competencia: título genérico."

    repeats_str = ", ".join(repeats) if repeats else "ninguna"
    claims_str = ", ".join(sorted(set(claims))) if claims else "ninguno"

    if price is None:
        price_comment = "Precio no disponible"
    elif bucket:
        price_comment = f"Precio {price} ({bucket})"
    else:
        price_comment = f"Precio {price}"

    summary = (
        f"Comentarios: {comentarios_str}. Pros: {pros_str}. Contras: {contras_str}. "
        f"{compet} Palabras repetidas: {repeats_str}. Claims: {claims_str}. {price_comment}."
    )
    return summary


def analyze_titles(
    items: List[Dict[str, Any]], weights: Optional[Mapping[str, float]] = None
) -> List[Dict[str, Any]]:
    """Analyze a list of product title entries.

    Each output item follows the schema required for the Title Analyzer with
    ``signals`` (value, claims, materials, compat), ``flags`` for risks, a
    Spanish ``summary`` including ``price_bucket`` and a numeric ``titleScore``.
    Optional ``product_id``, ``price`` and ``rating`` fields from the input are
    preserved when possible.  Weights can be overridden via ``weights``.
    """
    results: List[Dict[str, Any]] = []
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items() if k in w})

    # Pre-compute price quantiles
    prices: List[Optional[float]] = []
    for entry in items:
        price_val = None
        if entry.get("price") is not None:
            try:
                price_val = float(str(entry["price"]).replace(",", "."))
            except Exception:
                price_val = None
        prices.append(price_val)
    q1, q2 = _compute_quantiles([p for p in prices if p is not None])

    for entry, price_val in zip(items, prices):
        title = entry.get("title")
        if not title:
            continue
        lower = title.lower()
        tokens = _tokenize(title)

        sizes = SIZE_PATTERN.findall(lower)
        compatibility = ["for " + m.strip() for m in COMPAT_PATTERN.findall(title)]
        materials = [t for t in tokens if t in MATERIALS]
        claims = [t for t in tokens if t in CLAIMS]

        signals: Dict[str, Any] = {
            "value": sizes or [],
            "claims": claims or [],
            "materials": materials or [],
            "compat": compatibility or [],
        }

        seo_bloat = len(title) > 120
        genericity = False
        if not any(signals.values()) and tokens and all(
            t in GENERIC_ADJECTIVES or t in GENERIC_NOUNS for t in tokens
        ):
            genericity = True
        brand_hits = [b for b in BRANDS if b in lower]
        ip_risk = bool(brand_hits) and not any(w in lower for w in BRAND_OK_WORDS)

        risks = {
            "seo_bloat": seo_bloat,
            "genericity": genericity,
            "ip_risk": ip_risk,
        }

        claim_strength = len(set(claims))
        value_signals = len(sizes)
        targeting = 1 if compatibility else 0
        genericity_int = 1 if genericity else 0
        seo_bloat_int = 1 if seo_bloat else 0
        ip_risk_int = 1 if ip_risk else 0
        title_score = (
            claim_strength * w["w1"]
            + value_signals * w["w2"]
            + targeting * w["w3"]
            - genericity_int * w["w4"]
            - seo_bloat_int * w["w5"]
            - ip_risk_int * w["w6"]
        )

        repeats = [t for t, c in Counter(tokens).items() if c > 1]
        bucket = _price_bucket(price_val, q1, q2)
        summary_text = _generate_summary(
            signals, risks, tokens, claims, repeats, price_val, bucket
        )

        rating_val = None
        if entry.get("rating") is not None:
            try:
                rating_val = float(str(entry["rating"]).replace(",", "."))
            except Exception:
                rating_val = None

        result_item = {
            "product_id": entry.get("product_id") or entry.get("id"),
            "title": title,
            "price": price_val,
            "rating": rating_val,
            "signals": signals,
            "flags": risks,
            "summary": {"text": summary_text, "price_bucket": bucket},
            "titleScore": title_score,
        }
        results.append(result_item)
    return results
