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
from typing import Dict, List

import requests

from . import gpt, config

# Small set of common stop words in English and Spanish to ignore when counting
# keywords.  This list is not exhaustive but helps surface relevant terms.
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "you", "are", "los", "las",
    "para", "con", "por", "una", "que", "este", "esa", "pero", "como",
    "las", "los", "del", "sus", "más", "por", "sin", "una", "un", "al",
}


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


def summarize_comments(comments: List[str]) -> Dict[str, List[str]]:
    """Use OpenAI to summarise comments into pros, cons and related products.

    The OpenAI API key and model are read from the application's configuration.
    If the API key is missing or the call fails, empty lists are returned.
    """
    api_key = config.get_api_key()
    model = config.get_model()
    if not api_key or not comments:
        return {"pros": [], "contras": [], "productos_relacionados": []}
    joined = "\n".join(comments[:100])
    prompt = (
        "Analiza los siguientes comentarios de usuarios sobre un producto. "
        "Resume los pros y contras más mencionados y enumera otros productos que se mencionan como alternativas. "
        "Responde únicamente con un objeto JSON con las claves 'pros', 'contras' y 'productos_relacionados'.\n\n"
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
            }
    except Exception:
        pass
    return {"pros": [], "contras": [], "productos_relacionados": []}


__all__ = [
    "fetch_reddit_posts",
    "fetch_youtube_comments",
    "extract_keywords",
    "summarize_comments",
]
