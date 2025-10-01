"""Utilities for configuring the OpenAI client used across the backend."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Iterable, Sequence, Union

try:  # pragma: no cover - optional dependency
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - if tiktoken is unavailable we fall back
    tiktoken = None  # type: ignore

from product_research_app.config import (
    AI_CONTEXT_WINDOW,
    AI_MAX_OUTPUT_TOKENS,
    AI_BATCH_TOKEN_BUDGET,
    OPENAI_MODEL,
)

logger = logging.getLogger(__name__)


def get_model_name() -> str:
    """Return the default model name resolved from the environment."""

    return OPENAI_MODEL


def get_model_caps() -> dict[str, int]:
    """Expose context window, output and batching budgets for downstream usage."""

    return {
        "context_window": AI_CONTEXT_WINDOW,
        "max_output_tokens": AI_MAX_OUTPUT_TOKENS,
        "batch_token_budget": AI_BATCH_TOKEN_BUDGET,
    }


def get_openai_base_url() -> str:
    """Return the base URL for the OpenAI-compatible endpoint."""

    return os.getenv("PRAPP_OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))


@lru_cache(maxsize=1)
def _encoding() -> "tiktoken.Encoding | None":
    if not tiktoken:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - fallback when model tables missing
        logger.debug("tiktoken encoding resolution failed", exc_info=True)
        return None


def _ensure_iterable(text_or_list: Union[str, Sequence[str], Iterable[str]]) -> Iterable[str]:
    if isinstance(text_or_list, str):
        return [text_or_list]
    if isinstance(text_or_list, Iterable):
        return text_or_list
    return [str(text_or_list)]


def estimate_tokens(text_or_list: Union[str, Sequence[str], Iterable[str]]) -> int:
    """Estimate token usage for prompts or completions.

    We prefer :mod:`tiktoken` for accuracy, but gracefully fall back to a
    character-length heuristic (≈4 chars/token) whenever the optional dependency
    is not installed or fails to provide an encoding.
    """

    texts = list(_ensure_iterable(text_or_list))
    if not texts:
        return 0

    encoding = _encoding()
    if encoding is not None:
        try:
            return sum(len(encoding.encode(part or "")) for part in texts)
        except Exception:  # pragma: no cover - encoding errors are unlikely
            logger.debug("tiktoken estimation failed; using heuristic", exc_info=True)

    total_chars = sum(len(part or "") for part in texts)
    if total_chars <= 0:
        return 0
    # 4 chars ≈ 1 token for GPT-style BPEs; ensure minimum of 1 per non-empty part
    approx = max(1, total_chars // 4)
    non_empty = sum(1 for part in texts if part)
    return max(approx, non_empty)


# Sanity compile command for modified modules (see PR instructions)
# python -m compileall \
#   product_research_app/config.py \
#   product_research_app/services/ai_client.py \
#   product_research_app/utils/rate_limiter.py \
#   product_research_app/ratelimit.py \
#   product_research_app/gpt.py \
#   product_research_app/services/ai_columns.py \
#   product_research_app/product_enrichment.py \
#   product_research_app/web_app.py \
#   product_research_app/main.py
