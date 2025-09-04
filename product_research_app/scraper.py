"""
Simple scraping utilities.

This module provides basic functions to extract product information from a given
URL.  It does not interface with paid services; instead it relies on publicly
available web pages and heuristics to infer product attributes such as the
title, description and price.  The scraping is intentionally lightweight and
best effort—real world websites often vary in structure.  The goal is to
assist manual data entry rather than fully automate scraping.
"""

from typing import Any, Dict, Optional
import re

import requests
from bs4 import BeautifulSoup


def scrape_product_from_url(url: str) -> Optional[Dict[str, Any]]:
    """Fetch and parse a web page to infer product details.

    This helper attempts to extract a title, meta description and price from the
    supplied URL.  It is not guaranteed to work for all e‑commerce sites, but
    may save time when adding products manually.  If the request fails or
    parsing yields no meaningful result, ``None`` is returned.

    Args:
        url: The URL of a product or article.

    Returns:
        A dictionary with keys ``name``, ``description``, and ``price`` if
        successful; otherwise ``None``.
    """
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return None
    except Exception:
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    # Title
    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    # Meta description
    meta_desc = soup.find("meta", attrs={"name": "description"}) or soup.find(
        "meta", attrs={"property": "og:description"}
    )
    description = None
    if meta_desc and meta_desc.get("content"):
        description = meta_desc.get("content").strip()
    # Price extraction: search for currency patterns
    price = None
    text = soup.get_text(separator=" ")
    # Look for patterns like €12.34 or $12.34 or 12,34 €
    price_patterns = [
        r"€\s*(\d+[\.,]\d+)",
        r"\$(\d+[\.,]\d+)",
        r"(\d+[\.,]\d+)\s*€",
        r"(\d+[\.,]\d+)\s*USD",
    ]
    for pat in price_patterns:
        match = re.search(pat, text)
        if match:
            price_str = match.group(1).replace(",", ".")
            try:
                price = float(price_str)
                break
            except ValueError:
                continue
    if not title and not description:
        return None
    return {
        "name": title or "",
        "description": description,
        "price": price,
    }