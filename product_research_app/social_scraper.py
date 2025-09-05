"""Utilities to gather user comments from social platforms.

These helpers retrieve public discussions about a product from Reddit and
YouTube using their public APIs.  The functions are intentionally simple and
avoid authentication where possible.  For YouTube an API key is required.

The module also provides basic text analysis utilities to extract frequent
keywords and to summarise opinions using an OpenAI model.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from . import gpt, config

# Small set of common stop words in English and Spanish to ignore when counting
# keywords.  This list is not exhaustive but helps surface relevant terms.
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "you", "are", "los", "las",
    "para", "con", "por", "una", "que", "este", "esa", "pero", "como",
    "las", "los", "del", "sus", "más", "por", "sin", "una", "un", "al",
}

# Basic lists of positive and negative words for crude sentiment estimation
POSITIVE_WORDS = {
    "bueno",
    "excelente",
    "genial",
    "love",
    "great",
    "fantastic",
    "buen",
    "recomendado",
    "amazing",
}
NEGATIVE_WORDS = {
    "malo",
    "terrible",
    "bad",
    "worst",
    "defectuoso",
    "poor",
    "horrible",
    "problema",
    "issue",
}

# Regex to detect mentions of price in comments
PRICE_RE = re.compile(
    r"(?:\$|€|eur|usd|dólar|dollar|euros?)\s?\d|\d+\s?(?:usd|dólares|dollars|euros?|€)",
    re.IGNORECASE,
)


def fetch_reddit_posts(query: str, limit: int = 10) -> Dict[str, List[str]]:
    """Return basic posts related to ``query`` from Reddit.

    This function uses Reddit's public JSON search endpoint.  It collects the
    post titles and selftext which often contain user opinions about a product.
    If the request fails an empty result is returned.
    """
    headers = {"User-Agent": "ProductResearchBot/0.1"}
    params = {"q": query, "limit": limit, "sort": "relevance", "t": "all"}
    try:
        resp = requests.get(
            "https://www.reddit.com/search.json", params=params, headers=headers, timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {"comments": [], "post_count": 0}
    comments: List[str] = []
    children = data.get("data", {}).get("children", [])
    for child in children:
        post = child.get("data", {})
        title = post.get("title")
        text = post.get("selftext")
        if title:
            comments.append(title)
        if text:
            comments.append(text)
    return {"comments": comments, "post_count": len(children)}


def fetch_youtube_comments(
    api_key: str,
    query: str,
    max_videos: int = 3,
    max_comments: int = 20,
) -> Dict[str, List[str]]:
    """Return comments from YouTube videos related to ``query``.

    The YouTube Data API is queried first to find videos matching the search
    term and then to retrieve top level comments for each video.  If the API
    key is missing or requests fail an empty result is returned.
    """
    if not api_key:
        return {"comments": [], "video_count": 0}
    try:
        search_params = {
            "key": api_key,
            "q": query,
            "part": "id",
            "type": "video",
            "maxResults": max_videos,
        }
        search_resp = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params=search_params,
            timeout=10,
        )
        search_resp.raise_for_status()
        items = search_resp.json().get("items", [])
        video_ids = [i.get("id", {}).get("videoId") for i in items if i.get("id")]
        comments: List[str] = []
        for vid in video_ids:
            if not vid:
                continue
            thread_params = {
                "key": api_key,
                "part": "snippet",
                "videoId": vid,
                "textFormat": "plainText",
                "maxResults": max_comments,
            }
            ct_resp = requests.get(
                "https://www.googleapis.com/youtube/v3/commentThreads",
                params=thread_params,
                timeout=10,
            )
            if ct_resp.status_code != 200:
                continue
            for item in ct_resp.json().get("items", []):
                snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
                text = snippet.get("textDisplay")
                if text:
                    comments.append(text)
        return {"comments": comments, "video_count": len(video_ids)}
    except Exception:
        return {"comments": [], "video_count": 0}


def fetch_web_reviews(query: str, limit: int = 5) -> Dict[str, List[str]]:
    """Retrieve snippets from general web search results for ``query``.

    DuckDuckGo's HTML endpoint is used as it does not require an API key.  The
    function collects short descriptions from search results and, when
    possible, fetches the linked pages to extract paragraph text.  It is a
    best-effort approach and may return an empty list if requests fail.
    """
    comments: List[str] = []
    sources: List[str] = []
    try:
        params = {"q": f"{query} review", "kl": "es-es"}
        resp = requests.get("https://duckduckgo.com/html/", params=params, timeout=10)
        if resp.status_code != 200:
            return {"comments": [], "sources": []}
        soup = BeautifulSoup(resp.text, "html.parser")
        results = soup.select("div.result")[:limit]
        for r in results:
            link = r.find("a", href=True)
            snippet = r.find("a", class_="result__snippet") or r.find("div", class_="result__snippet")
            if snippet:
                comments.append(snippet.get_text(" "))
            if link and link["href"]:
                sources.append(link["href"])
                try:
                    page = requests.get(link["href"], timeout=10)
                    if page.status_code == 200:
                        psoup = BeautifulSoup(page.text, "html.parser")
                        for p in psoup.find_all("p")[:3]:
                            text = p.get_text(" ")
                            if text:
                                comments.append(text)
                except Exception:
                    continue
    except Exception:
        return {"comments": [], "sources": []}
    return {"comments": comments, "sources": sources}


def fetch_amazon_reviews(query: str, max_reviews: int = 20) -> Dict[str, List[str]]:
    """Retrieve review snippets from Amazon for ``query``.

    This is a best-effort scraper that searches Amazon for the product and then
    fetches the first product's review page.  If any request fails an empty
    result is returned.  Amazon's markup changes frequently so this may not
    always succeed.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        search = requests.get("https://www.amazon.com/s", params={"k": query}, headers=headers, timeout=10)
        if search.status_code != 200:
            return {"comments": [], "review_count": 0}
        soup = BeautifulSoup(search.text, "html.parser")
        first = soup.select_one("div.s-result-item h2 a.a-link-normal")
        if not first or not first.get("href"):
            return {"comments": [], "review_count": 0}
        href = first["href"]
        m = re.search(r"/dp/([A-Z0-9]{10})", href)
        if not m:
            return {"comments": [], "review_count": 0}
        asin = m.group(1)
        rev_resp = requests.get(
            f"https://www.amazon.com/product-reviews/{asin}", headers=headers, timeout=10
        )
        if rev_resp.status_code != 200:
            return {"comments": [], "review_count": 0}
        rsoup = BeautifulSoup(rev_resp.text, "html.parser")
        comments: List[str] = []
        for rev in rsoup.select("div.review"):
            text = rev.select_one("span.review-text-content")
            if text:
                comments.append(text.get_text(" ", strip=True))
            if len(comments) >= max_reviews:
                break
        return {"comments": comments, "review_count": len(comments)}
    except Exception:
        return {"comments": [], "review_count": 0}


def extract_keywords(texts: List[str], top_n: int = 10) -> List[str]:
    """Extract the most common keywords from ``texts``.

    Words shorter than three characters or present in ``STOPWORDS`` are ignored.
    """
    counter: Counter[str] = Counter()
    for text in texts:
        for word in re.findall(r"[\wáéíóúñü]+", text.lower()):
            if len(word) < 3 or word in STOPWORDS:
                continue
            counter[word] += 1
    return [w for w, _ in counter.most_common(top_n)]


def extract_price_comments(texts: List[str]) -> List[str]:
    """Return comments that mention prices or currency symbols."""
    return [t for t in texts if PRICE_RE.search(t)]


def summarize_comments(
    comments: List[str],
    *,
    keywords: Optional[List[str]] = None,
    repeated: Optional[List[str]] = None,
    price_comments: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Use OpenAI to summarise comments into structured insights and a summary.

    The OpenAI API key and model are read from the application's configuration.
    If the API key is missing or the call fails, a fallback summary is used.
    """
    api_key = config.get_api_key()
    model = config.get_model()
    if not comments:
        return {"pros": [], "contras": [], "productos_relacionados": [], "summary": ""}
    if not api_key:
        return _basic_summary(
            comments, keywords=keywords, repeated=repeated, price_comments=price_comments
        )
    joined = "\n".join(comments[:100])
    prompt = (
        "Analiza los siguientes comentarios de usuarios sobre un producto. "
        "Resume los pros y contras más mencionados, comenta qué hace mejor y peor que la competencia, "
        "menciona referencias al precio y otros productos alternativos. "
        "Responde únicamente con un objeto JSON con las claves 'pros', 'contras', "
        "'productos_relacionados' y 'summary'.\n\n"
        f"Palabras clave frecuentes: {', '.join(keywords or [])}\n"
        f"Comentarios repetidos: {', '.join(repeated or [])}\n"
        f"Comentarios sobre precio: {', '.join(price_comments or [])}\n\n"
        f"Comentarios:\n{joined}"
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        resp = gpt.call_openai_chat(api_key, model, messages, temperature=0)
        content = resp["choices"][0]["message"]["content"].strip()
        data = json.loads(content)
        if isinstance(data, dict):
            return {
                "pros": data.get("pros", []) or [],
                "contras": data.get("contras", []) or [],
                "productos_relacionados": data.get("productos_relacionados", []) or [],
                "summary": data.get("summary", ""),
            }
    except Exception:
        return _basic_summary(
            comments, keywords=keywords, repeated=repeated, price_comments=price_comments
        )


def _basic_summary(
    comments: List[str],
    *,
    keywords: Optional[List[str]] = None,
    repeated: Optional[List[str]] = None,
    price_comments: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Provide a naive pros/cons summary without external API calls."""
    pros: List[str] = []
    cons: List[str] = []
    for c in comments:
        lc = c.lower()
        if any(p in lc for p in POSITIVE_WORDS):
            pros.append(c)
        if any(n in lc for n in NEGATIVE_WORDS):
            cons.append(c)
    parts: List[str] = []
    if pros:
        parts.append("Pros destacados: " + ", ".join(pros[:3]))
    if cons:
        parts.append("Contras frecuentes: " + ", ".join(cons[:3]))
    if repeated:
        parts.append("Comentarios repetidos: " + ", ".join(repeated[:3]))
    if keywords:
        parts.append("Palabras clave: " + ", ".join(keywords[:5]))
    if price_comments:
        parts.append("Menciones sobre precio: " + "; ".join(price_comments[:2]))
    summary = ". ".join(parts)
    return {
        "pros": pros[:5],
        "contras": cons[:5],
        "productos_relacionados": [],
        "summary": summary,
    }


def find_repeated_comments(comments: List[str], min_count: int = 2) -> List[str]:
    """Return comments that appear at least ``min_count`` times."""
    counter: Counter[str] = Counter(c.strip().lower() for c in comments)
    return [c for c, cnt in counter.items() if cnt >= min_count]


__all__ = [
    "fetch_reddit_posts",
    "fetch_youtube_comments",
    "fetch_web_reviews",
    "fetch_amazon_reviews",
    "extract_keywords",
    "extract_price_comments",
    "summarize_comments",
    "find_repeated_comments",
]
