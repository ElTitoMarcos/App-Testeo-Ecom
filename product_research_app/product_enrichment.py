"""Asynchronous product enrichment pipeline.

This module coordinates AI-driven enrichment of imported catalog items.  It
fetches pending items from the database, applies caching and triage heuristics,
and fans out concurrent HTTP requests using ``httpx``.  Results are persisted
back into the database with detailed metrics so the web layer can expose
progress and observability endpoints.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import httpx

from . import config, database
from .db import get_db
from .sse import increment_enrich_batches, publish_progress

logger = logging.getLogger(__name__)


def _emit_enrich_progress(job_id: int, **payload: Any) -> None:
    """Emit an enrichment progress event via SSE."""

    publish_progress({"event": "enrich", "job_id": job_id, **payload})


DEFAULT_BATCH_SIZE = int(os.getenv("ENRICH_BATCH_SIZE", "20"))
DEFAULT_CONCURRENCY = int(os.getenv("ENRICH_CONCURRENCY", "12"))
TARGET_INPUT_TOKENS = int(os.getenv("ENRICH_TARGET_INPUT_TOKENS", "6000"))
MIN_BATCH_SIZE = int(os.getenv("ENRICH_MIN_BATCH_SIZE", "10"))
MAX_BATCH_SIZE = int(os.getenv("ENRICH_MAX_BATCH_SIZE", "30"))
MIN_CONCURRENCY = int(os.getenv("ENRICH_MIN_CONCURRENCY", "8"))
MAX_CONCURRENCY = int(os.getenv("ENRICH_MAX_CONCURRENCY", "16"))
MAX_RETRIES = int(os.getenv("ENRICH_MAX_RETRIES", "5"))
CACHE_MAX_AGE_DAYS = int(os.getenv("ENRICH_CACHE_TTL_DAYS", "30"))
DEFAULT_MAX_REQUESTS = int(os.getenv("ENRICH_MAX_REQUESTS", "0"))
DEFAULT_MAX_COST_CENTS = float(os.getenv("ENRICH_MAX_COST_CENTS", "0"))
COST_PER_1K_INPUT_CENTS = float(os.getenv("ENRICH_COST_PER_1K_INPUT_CENTS", "15"))
DEFAULT_MODE_EXHAUSTIVO = os.getenv("ENRICH_MODE_EXHAUSTIVO", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}
AI_TIMEOUT = float(os.getenv("ENRICH_TIMEOUT_SECONDS", "30"))
AI_URL = os.getenv("ENRICH_API_URL", "https://api.openai.com/v1/chat/completions")
AI_MODEL_ENV = os.getenv("ENRICH_MODEL")
TRIAGE_RATING = float(os.getenv("ENRICH_TRIAGE_RATING_LT", "3"))
TRIAGE_UNITS = int(os.getenv("ENRICH_TRIAGE_UNITS_LT", "50"))
TRIAGE_MAX_AGE = int(os.getenv("ENRICH_TRIAGE_MAX_AGE_DAYS", "540"))

SYSTEM_PROMPT = (
    "Eres un analista de marketing. Evalúa cada producto y responde EXCLUSIVAMENTE "
    "con JSON válido siguiendo este esquema: {\"results\": [{\"id\": int, \"desire\": int, "
    "\"awareness\": int, \"reason\": string<=120 chars}]}."
)

DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d")

TRI_LABELS = {
    "low": "Low",
    "medio": "Medium",
    "medium": "Medium",
    "high": "High",
    "alto": "High",
    "bajo": "Low",
}

AWARENESS_LABELS = {
    "unaware": "Unaware",
    "problem-aware": "Problem-Aware",
    "problema": "Problem-Aware",
    "solution-aware": "Solution-Aware",
    "solucion": "Solution-Aware",
    "product-aware": "Product-Aware",
    "producto": "Product-Aware",
    "most aware": "Most Aware",
}


def _norm_tri_label(value: Any) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip().lower()
    return TRI_LABELS.get(text)


def _norm_awareness_label(value: Any) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip().lower()
    return AWARENESS_LABELS.get(text)


def _parse_winner_score(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        score = int(round(float(value)))
    except Exception:
        return None
    return max(0, min(100, score))


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def clamp_score(value: Any, *, low: int = 0, high: int = 100) -> int:
    try:
        num = int(float(value))
    except (TypeError, ValueError):
        num = low
    return clamp(num, low, high)


def parse_job_config(raw: Any) -> Dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return {}
    return {}


def ensure_enrich_config(
    config_payload: Optional[Dict[str, Any]]
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    base = dict(config_payload or {})
    raw_enrich = base.get("enrich") if isinstance(base.get("enrich"), dict) else {}
    if not isinstance(raw_enrich, dict):
        raw_enrich = {}
    defaults = {
        "batch_size": DEFAULT_BATCH_SIZE,
        "concurrency": DEFAULT_CONCURRENCY,
        "target_input_tokens_per_req": TARGET_INPUT_TOKENS,
        "max_requests": DEFAULT_MAX_REQUESTS,
        "max_cost_cents": DEFAULT_MAX_COST_CENTS,
        "cost_per_1k_input_cents": COST_PER_1K_INPUT_CENTS,
        "mode_exhaustivo": DEFAULT_MODE_EXHAUSTIVO,
        "cache_ttl_days": CACHE_MAX_AGE_DAYS,
    }
    triage_defaults = {
        "enabled": True,
        "rating_threshold": TRIAGE_RATING,
        "units_sold_threshold": TRIAGE_UNITS,
        "max_age_days": TRIAGE_MAX_AGE,
    }
    triage_cfg = dict(triage_defaults)
    if isinstance(raw_enrich.get("triage"), dict):
        triage_cfg.update(raw_enrich["triage"])
    merged = {**defaults, **{k: v for k, v in raw_enrich.items() if k != "triage"}}
    merged["triage"] = triage_cfg
    base["enrich"] = merged
    return base, merged


def determine_model() -> str:
    if AI_MODEL_ENV:
        return AI_MODEL_ENV
    try:
        model = config.get_model()
        if model:
            return str(model)
    except Exception:
        pass
    return "gpt-4o-mini"


def resolve_api_key() -> Optional[str]:
    key = os.getenv("ENRICH_API_KEY")
    if key:
        return key
    try:
        return config.get_api_key()
    except Exception:
        return None


def build_prompt(items: Sequence["PendingItem"]) -> str:
    payload = []
    for item in items:
        entry = {k: v for k, v in item.payload.items() if v not in (None, "", [], {}, ())}
        entry["id"] = item.item_id
        payload.append(entry)
    instructions = (
        "Analiza los productos y asigna desire (0-100) y awareness (0-100) como enteros. "
        "Incluye una razón breve (<=120 caracteres) sobre el estado del producto. "
        "Devuelve únicamente JSON siguiendo el esquema indicado, sin texto adicional ni comentarios."
    )
    return instructions + "\n\n" + json.dumps({"items": payload}, ensure_ascii=False, indent=2)


def build_request(items: Sequence["PendingItem"], model: str) -> Dict[str, Any]:
    prompt = build_prompt(items)
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }


def estimate_tokens(payload: Any) -> int:
    try:
        text = json.dumps(payload, ensure_ascii=False)
    except TypeError:
        text = str(payload)
    return max(1, len(text) // 4)


def _minimal_json_repair(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    if stripped[0] not in "{[":
        start_candidates = [i for i, ch in enumerate(stripped) if ch in "{"]
        start = start_candidates[0] if start_candidates else 0
    else:
        start = 0
    end = stripped.rfind("}")
    if end == -1:
        end = len(stripped) - 1
    return stripped[start : end + 1]


def _retry_after_seconds(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        try:
            dt = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        now = datetime.utcnow().replace(tzinfo=dt.tzinfo)
        delta = dt - now
        return max(delta.total_seconds(), 0.0)


async def call_ai(
    payload: Dict[str, Any],
    *,
    client: httpx.AsyncClient,
    api_key: Optional[str],
    logger: logging.Logger,
) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    attempt = 0
    backoff = 1.0
    last_error: Optional[Exception] = None
    while attempt < MAX_RETRIES:
        try:
            response = await client.post(AI_URL, json=payload, headers=headers)
            if response.status_code == 429:
                retry = _retry_after_seconds(response.headers.get("Retry-After")) or backoff
                logger.warning("AI 429 throttled; sleeping %.2fs", retry)
                await asyncio.sleep(retry)
                backoff = min(backoff * 2, 30.0)
                attempt += 1
                continue
            response.raise_for_status()
            text = response.text.strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                repaired = _minimal_json_repair(text)
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    last_error = exc
                    logger.warning("AI JSON parse failed (attempt %d)", attempt + 1)
        except httpx.HTTPError as exc:
            last_error = exc
            logger.warning("AI request error (attempt %d): %s", attempt + 1, exc)
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 30.0)
        attempt += 1
    raise RuntimeError(f"AI request failed after {MAX_RETRIES} attempts: {last_error}")


def normalize_results(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("La respuesta AI no contiene 'results'")
    normalised: List[Dict[str, Any]] = []
    for entry in results:
        if not isinstance(entry, dict):
            continue
        try:
            item_id = int(entry.get("id"))
        except (TypeError, ValueError):
            continue
        desire_raw = entry.get("desire")
        desire = clamp_score(desire_raw) if desire_raw is not None else None
        awareness_raw = entry.get("awareness")
        awareness = clamp_score(awareness_raw) if awareness_raw is not None else None
        reason = str(entry.get("reason") or "").strip()
        if len(reason) > 120:
            reason = reason[:117].rstrip() + "..."
        desire_mag = _norm_tri_label(
            entry.get("desire_magnitude") or entry.get("desireMagnitude")
        )
        competition = _norm_tri_label(
            entry.get("competition_level") or entry.get("competitionLevel")
        )
        awareness_level = _norm_awareness_label(
            entry.get("awareness_level") or entry.get("awarenessLevel")
        )
        if awareness_level is None:
            awareness_level = _norm_awareness_label(entry.get("awareness_label"))
        winner_score = _parse_winner_score(entry.get("winner_score") or entry.get("winnerScore"))
        normalised.append(
            {
                "id": item_id,
                "desire": desire,
                "awareness": awareness,
                "reason": reason,
                "source": entry.get("source") or "ai",
                "desire_magnitude": desire_mag,
                "awareness_level": awareness_level,
                "competition_level": competition,
                "winner_score": winner_score,
            }
        )
    return normalised


def _parse_date(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _extract_number(data: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key not in data:
            continue
        value = data.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            s = str(value).replace("%", "").replace(",", ".")
            try:
                return float(s)
            except ValueError:
                continue
    return None


@dataclass
class PendingItem:
    item_id: int
    sig_hash: str
    payload: Dict[str, Any]
    raw: Dict[str, Any]
    tokens_estimate: int
    low_priority: bool = False


@dataclass
class EnrichmentPipeline:
    conn: Any
    job_id: int
    config: Dict[str, Any]
    logger: logging.Logger = field(default_factory=lambda: logger)

    high_priority: deque[PendingItem] = field(init=False, default_factory=deque)
    low_priority: deque[PendingItem] = field(init=False, default_factory=deque)
    cache_hits: int = 0
    triage_skipped: int = 0
    enriched_success: int = 0
    enriched_failed: int = 0
    cache_updates: list[Dict[str, Any]] = field(init=False, default_factory=list)
    processed: int = 0
    requests: int = 0
    tokens_sent: int = 0
    cost_cents: float = 0.0
    batches: int = 0
    total_duration_ms: float = 0.0
    ai_items: int = 0
    ai_tokens: int = 0
    avg_tokens_per_item: Optional[float] = None
    budget_paused: bool = False
    total_items: int = 0

    def __post_init__(self) -> None:
        self.batch_size = clamp(int(self.config.get("batch_size", DEFAULT_BATCH_SIZE)), MIN_BATCH_SIZE, MAX_BATCH_SIZE)
        self.concurrency = clamp(int(self.config.get("concurrency", DEFAULT_CONCURRENCY)), MIN_CONCURRENCY, MAX_CONCURRENCY)
        self.target_tokens = int(self.config.get("target_input_tokens_per_req", TARGET_INPUT_TOKENS))
        self.max_requests = int(self.config.get("max_requests") or 0)
        self.max_cost_cents = float(self.config.get("max_cost_cents") or 0.0)
        self.cost_per_1k = float(self.config.get("cost_per_1k_input_cents") or COST_PER_1K_INPUT_CENTS)
        self.mode_exhaustivo = bool(self.config.get("mode_exhaustivo", False))
        triage_cfg = self.config.get("triage") or {}
        self.triage_enabled = bool(triage_cfg.get("enabled", True))
        self.triage_rating = float(triage_cfg.get("rating_threshold", TRIAGE_RATING))
        self.triage_units = int(triage_cfg.get("units_sold_threshold", TRIAGE_UNITS))
        self.triage_max_age = int(triage_cfg.get("max_age_days", TRIAGE_MAX_AGE))
        self.cache_ttl_days = int(self.config.get("cache_ttl_days", CACHE_MAX_AGE_DAYS))
        self.model = str(self.config.get("model") or determine_model())
        self.start_time = time.perf_counter()
        self.started_iso = datetime.utcnow().isoformat()
        self.lock = asyncio.Lock()

    @property
    def remaining(self) -> int:
        return len(self.high_priority) + len(self.low_priority)

    def effective_config(self) -> Dict[str, Any]:
        cfg = dict(self.config)
        cfg.update(
            {
                "batch_size": self.batch_size,
                "concurrency": self.concurrency,
                "target_input_tokens_per_req": self.target_tokens,
                "max_requests": self.max_requests,
                "max_cost_cents": self.max_cost_cents,
                "cost_per_1k_input_cents": self.cost_per_1k,
                "mode_exhaustivo": self.mode_exhaustivo,
                "cache_ttl_days": self.cache_ttl_days,
                "model": self.model,
            }
        )
        triage = dict(cfg.get("triage") or {})
        triage.update(
            {
                "enabled": self.triage_enabled,
                "rating_threshold": self.triage_rating,
                "units_sold_threshold": self.triage_units,
                "max_age_days": self.triage_max_age,
            }
        )
        cfg["triage"] = triage
        return cfg

    def snapshot_metrics(self) -> Dict[str, Any]:
        total_wall = time.perf_counter() - self.start_time
        per_item = self.ai_tokens / self.ai_items if self.ai_items else None
        overall_throughput = self.processed / total_wall if total_wall > 0 else None
        return {
            "requests": self.requests,
            "batches": self.batches,
            "tokens": self.tokens_sent,
            "cost_cents": round(self.cost_cents, 6),
            "processed": self.processed,
            "cache_hits": self.cache_hits,
            "triage_skipped": self.triage_skipped,
            "total_duration_ms": self.total_duration_ms,
            "avg_tokens_per_item": per_item,
            "concurrency": self.concurrency,
            "batch_size": self.batch_size,
            "target_tokens": self.target_tokens,
            "budget_paused": self.budget_paused,
            "overall_throughput": overall_throughput,
            "started_at": self.started_iso,
            "model": self.model,
            "total": self.total_items,
        }

    def _emit_metrics(self, *, status: Optional[str] = None, **extra: Any) -> None:
        metrics = extra.pop("metrics", None)
        if metrics is None:
            metrics = self.snapshot_metrics()
        payload = {
            "phase": "enrich",
            "status": status or ("paused_by_budget" if self.budget_paused else "enriching"),
            "metrics": metrics,
            "remaining": self.remaining,
            "cache_hits": self.cache_hits,
            "triage_skipped": self.triage_skipped,
        }
        payload.update(extra)
        _emit_enrich_progress(self.job_id, **payload)

    def _emit_enrichment_updates(self, updates: Sequence[Dict[str, Any]]) -> None:
        if not updates:
            return
        sorted_updates = sorted(
            updates,
            key=lambda row: ((row.get("enrichment_updated_at") or ""), row.get("id") or 0),
        )
        max_updated: Optional[str] = None
        for row in sorted_updates:
            stamp = row.get("enrichment_updated_at")
            if stamp and (max_updated is None or str(stamp) > max_updated):
                max_updated = str(stamp)
        for idx in range(0, len(sorted_updates), 100):
            chunk = sorted_updates[idx : idx + 100]
            prep_start = time.perf_counter()
            payload_updates = [
                {
                    "id": entry.get("id"),
                    "desire": entry.get("desire"),
                    "desire_magnitude": entry.get("desire_magnitude"),
                    "awareness_level": entry.get("awareness_level"),
                    "competition_level": entry.get("competition_level"),
                    "winner_score": entry.get("winner_score"),
                }
                for entry in chunk
            ]
            payload = {
                "type": "enrich.batch",
                "updates": payload_updates,
                "count": len(payload_updates),
            }
            message = json.dumps(payload, separators=(",", ":"))
            size_bytes = len(message.encode("utf-8"))
            publish_progress(payload)
            increment_enrich_batches()
            prep_ms = (time.perf_counter() - prep_start) * 1000.0
            self.logger.info(
                "SSE enrich.batch job=%s count=%d bytes=%d prep_ms=%.2f",
                self.job_id,
                len(payload_updates),
                size_bytes,
                prep_ms,
            )
        if max_updated:
            database.set_stream_cursor(self.conn, "enrich_since", max_updated, commit=True)

    def prepare(self) -> None:
        pending = database.list_items_by_state(self.conn, self.job_id, "pending_enrich")
        self.total_items = len(pending)
        if not pending:
            return
        self.cache_updates = []
        sig_hashes = [row["sig_hash"] for row in pending]
        cache_rows = database.get_enrichment_cache(
            self.conn, sig_hashes, max_age_days=self.cache_ttl_days
        )
        for row in pending:
            raw_data: Dict[str, Any] = {}
            if row["raw"]:
                try:
                    raw_data = json.loads(row["raw"])
                except Exception:
                    raw_data = {}
            product = database.get_product_by_sig_hash(self.conn, row["sig_hash"])
            payload = self._build_payload(row["id"], raw_data, product)
            tokens_estimate = estimate_tokens(payload)
            low_priority = self._is_low_priority(raw_data, product)
            item = PendingItem(
                item_id=row["id"],
                sig_hash=row["sig_hash"],
                payload=payload,
                raw=raw_data,
                tokens_estimate=tokens_estimate,
                low_priority=low_priority,
            )
            cache_entry = cache_rows.get(row["sig_hash"])
            if cache_entry:
                self._apply_cache_hit(item, cache_entry)
            else:
                if low_priority and self.triage_enabled:
                    self.low_priority.append(item)
                else:
                    self.high_priority.append(item)
        self.conn.commit()
        if self.cache_updates:
            self._emit_enrichment_updates(self.cache_updates)
            self.cache_updates.clear()
        if self.cache_hits:
            metrics = self.snapshot_metrics()
            database.update_enrichment_metrics(self.conn, self.job_id, metrics)
            self._emit_metrics(metrics=metrics)
        self.logger.info(
            "enrich job=%s queued high=%d low=%d cache_hits=%d",
            self.job_id,
            len(self.high_priority),
            len(self.low_priority),
            self.cache_hits,
        )

    def _apply_cache_hit(self, item: PendingItem, cache_row: Any) -> None:
        desire = cache_row["desire"]
        awareness = cache_row["awareness"]
        reason = cache_row["reason"]
        source = cache_row["source"] or "cache"
        update = database.update_product_enrichment(
            self.conn,
            item.sig_hash,
            int(desire) if desire is not None else None,
            int(awareness) if awareness is not None else None,
            reason,
            source=source,
        )
        if update:
            self.cache_updates.append(update)
        database.mark_item_enriched(
            self.conn,
            item.item_id,
            {
                "id": item.item_id,
                "desire": desire,
                "awareness": awareness,
                "reason": reason,
                "source": source,
            },
        )
        self.cache_hits += 1
        self.enriched_success += 1
        self.processed += 1
        self.logger.info(
            "enrich job=%s cache hit item=%s sig=%s",
            self.job_id,
            item.item_id,
            item.sig_hash,
        )

    def _build_payload(
        self, item_id: int, raw: Dict[str, Any], product: Optional[Any]
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for key, value in raw.items():
            if value is None:
                continue
            if isinstance(value, (int, float, bool)):
                payload[key] = value
            elif isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    payload[key] = stripped[:500]
        extra: Dict[str, Any] = {}
        if product is not None:
            for field_name in [
                "name",
                "brand",
                "category",
                "price",
                "currency",
                "import_date",
                "date_range",
                "source",
            ]:
                value = product[field_name]
                if value and field_name not in payload:
                    payload[field_name] = value
            if product["description"] and "description" not in payload:
                payload["description"] = product["description"][:800]
            if product["extra"]:
                try:
                    extra = json.loads(product["extra"])
                except Exception:
                    extra = {}
        for key in ["rating", "units_sold", "revenue"]:
            if key in extra and key not in payload:
                payload[key] = extra[key]
        payload["item_id"] = item_id
        return payload

    def _is_low_priority(self, raw: Dict[str, Any], product: Optional[Any]) -> bool:
        if not self.triage_enabled:
            return False
        extra: Dict[str, Any] = {}
        if product is not None and product["extra"]:
            try:
                extra = json.loads(product["extra"])
            except Exception:
                extra = {}
        rating = _extract_number(raw, ["rating", "valoracion", "stars"])
        if rating is None:
            rating = _extract_number(extra, ["rating"])
        if rating is not None and rating < self.triage_rating:
            return True
        units = _extract_number(raw, ["units_sold", "ventas", "sold"])
        if units is None:
            units = _extract_number(extra, ["units_sold", "units"])
        if units is not None and units < self.triage_units:
            return True
        age = self._compute_age_days(raw, product)
        if age is not None and age > self.triage_max_age:
            return True
        return False

    def _compute_age_days(self, raw: Dict[str, Any], product: Optional[Any]) -> Optional[int]:
        candidates = [
            raw.get("launch_date"),
            raw.get("release_date"),
            raw.get("first_seen"),
            raw.get("date"),
        ]
        if product is not None:
            candidates.extend([product["import_date"], product["date_range"]])
        for candidate in candidates:
            parsed = _parse_date(candidate)
            if parsed:
                return (datetime.utcnow() - parsed).days
        return None

    def _determine_batch_size(self, queue: deque[PendingItem]) -> int:
        if not queue:
            return 0
        if self.ai_items and self.ai_tokens:
            avg = self.ai_tokens / max(self.ai_items, 1)
            size = int(self.target_tokens / avg) if avg > 0 else self.batch_size
        else:
            size = self.batch_size
        size = clamp(size, MIN_BATCH_SIZE, MAX_BATCH_SIZE)
        size = min(size, len(queue))
        if size <= 0:
            size = min(len(queue), MAX_BATCH_SIZE)
        return max(1, size)

    def _budget_would_exceed(self, estimated_tokens: int) -> bool:
        if self.max_requests and self.requests >= self.max_requests:
            self.logger.info(
                "enrich job=%s reached max_requests=%d",
                self.job_id,
                self.max_requests,
            )
            self.budget_paused = True
            return True
        projected_cost = self.cost_cents + (estimated_tokens / 1000.0) * self.cost_per_1k
        if self.max_cost_cents and projected_cost > self.max_cost_cents:
            self.logger.info(
                "enrich job=%s paused by budget projected=%.2f max=%.2f",
                self.job_id,
                projected_cost,
                self.max_cost_cents,
            )
            self.budget_paused = True
            return True
        return False

    def _skip_low_priority_pending(self) -> None:
        if not self.low_priority:
            return
        skipped = list(self.low_priority)
        self.low_priority.clear()
        for item in skipped:
            database.mark_item_failed(self.conn, item.item_id, error="skipped_by_triage")
            self.triage_skipped += 1
            self.processed += 1
            self.logger.info(
                "enrich job=%s item=%s skipped by triage",
                self.job_id,
                item.item_id,
            )
        self.conn.commit()

    async def dequeue_batch(self) -> Optional[List[PendingItem]]:
        async with self.lock:
            if self.budget_paused:
                return None
            if not self.high_priority and self.low_priority and not self.mode_exhaustivo:
                self._skip_low_priority_pending()
                metrics = self.snapshot_metrics()
                database.update_enrichment_metrics(self.conn, self.job_id, metrics)
                self._emit_metrics(metrics=metrics)
                return None
            queue = self.high_priority if self.high_priority else self.low_priority
            if not queue:
                return None
            batch_size = self._determine_batch_size(queue)
            items = list(itertools.islice(queue, 0, batch_size))
            estimated_tokens = sum(max(item.tokens_estimate, 1) for item in items)
            if self._budget_would_exceed(estimated_tokens):
                metrics = self.snapshot_metrics()
                database.update_enrichment_metrics(self.conn, self.job_id, metrics)
                self._emit_metrics(status="paused_by_budget", metrics=metrics)
                return None
            for _ in range(len(items)):
                queue.popleft()
            return items

    async def worker_loop(self, client: httpx.AsyncClient, api_key: Optional[str]) -> None:
        while True:
            batch = await self.dequeue_batch()
            if not batch:
                return
            await self.handle_batch(client, api_key, batch)

    async def handle_batch(
        self, client: httpx.AsyncClient, api_key: Optional[str], items: List[PendingItem]
    ) -> None:
        payload = build_request(items, self.model)
        estimated_tokens = estimate_tokens(payload)
        start = time.perf_counter()
        try:
            response = await call_ai(payload, client=client, api_key=api_key, logger=self.logger)
            normalised = normalize_results(response)
        except Exception as exc:
            self.logger.exception("enrich job=%s batch error: %s", self.job_id, exc)
            failed = await asyncio.to_thread(self._mark_batch_failed_sync, items, str(exc))
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.enriched_failed += failed
            self._update_metrics_after_batch(len(items), 0, failed, estimated_tokens, elapsed_ms)
            metrics = self.snapshot_metrics()
            database.update_enrichment_metrics(self.conn, self.job_id, metrics)
            self._emit_metrics(metrics=metrics)
            return
        mapping = {entry["id"]: entry for entry in normalised}
        enriched, failed, updates = await asyncio.to_thread(
            self._apply_results_sync, items, mapping
        )
        self.enriched_success += enriched
        self.enriched_failed += failed
        if updates:
            self._emit_enrichment_updates(updates)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._update_metrics_after_batch(len(items), enriched, failed, estimated_tokens, elapsed_ms)
        throughput = (enriched + failed) / ((elapsed_ms / 1000.0) or 1.0)
        self.logger.info(
            "enrich job=%s batch=%d size=%d enriched=%d failed=%d ms=%.1f throughput=%.2f tokens=%d cost=%.4f",
            self.job_id,
            self.batches,
            len(items),
            enriched,
            failed,
            elapsed_ms,
            throughput,
            estimated_tokens,
            self.cost_cents,
        )
        metrics = self.snapshot_metrics()
        database.update_enrichment_metrics(self.conn, self.job_id, metrics)
        self._emit_metrics(metrics=metrics)

    def _apply_results_sync(
        self, items: List[PendingItem], mapping: Dict[int, Dict[str, Any]]
    ) -> tuple[int, int, List[Dict[str, Any]]]:
        enriched = 0
        failed = 0
        updates: List[Dict[str, Any]] = []
        for item in items:
            result = mapping.get(item.item_id)
            if result:
                update = database.update_product_enrichment(
                    self.conn,
                    item.sig_hash,
                    result.get("desire"),
                    result.get("awareness"),
                    result.get("reason"),
                    desire_magnitude=result.get("desire_magnitude"),
                    awareness_level=result.get("awareness_level"),
                    competition_level=result.get("competition_level"),
                    winner_score=result.get("winner_score"),
                    source=result.get("source", "ai"),
                )
                if update:
                    updates.append(update)
                database.mark_item_enriched(self.conn, item.item_id, result)
                database.upsert_enrichment_cache(
                    self.conn,
                    item.sig_hash,
                    result.get("desire"),
                    result.get("awareness"),
                    result.get("reason"),
                    source=result.get("source", "ai"),
                )
                enriched += 1
            else:
                database.mark_item_failed(
                    self.conn, item.item_id, error="missing_result"
                )
                failed += 1
        self.conn.commit()
        return enriched, failed, updates

    def _mark_batch_failed_sync(
        self, items: List[PendingItem], error: str
    ) -> int:
        for item in items:
            database.mark_item_failed(self.conn, item.item_id, error=error)
        self.conn.commit()
        return len(items)

    def _update_metrics_after_batch(
        self,
        total_items: int,
        enriched: int,
        failed: int,
        estimated_tokens: int,
        elapsed_ms: float,
    ) -> None:
        self.requests += 1
        self.batches += 1
        self.tokens_sent += estimated_tokens
        self.cost_cents += (estimated_tokens / 1000.0) * self.cost_per_1k
        self.total_duration_ms += elapsed_ms
        self.ai_items += total_items
        self.ai_tokens += estimated_tokens
        if self.ai_items:
            self.avg_tokens_per_item = self.ai_tokens / self.ai_items
        self.processed += enriched + failed


async def run_job(job_id: int, *, logger: logging.Logger = logger) -> None:
    conn = get_db()
    job = database.get_import_job(conn, job_id)
    if job is None:
        logger.error("Enrichment job %s not found", job_id)
        return
    config_data = parse_job_config(job["config"])
    full_config, enrich_cfg = ensure_enrich_config(config_data)
    pipeline = EnrichmentPipeline(conn, job_id, enrich_cfg, logger=logger)
    full_config["enrich"] = pipeline.effective_config()
    database.update_import_job_progress(
        conn,
        job_id,
        phase="enrich",
        status="enriching",
        config=full_config,
    )
    _emit_enrich_progress(job_id, phase="enrich", status="enriching", config=full_config)
    pipeline.prepare()
    metrics = pipeline.snapshot_metrics()
    database.update_enrichment_metrics(conn, job_id, metrics)
    pipeline._emit_metrics(metrics=metrics)
    if pipeline.remaining == 0:
        status = job["status"] if job["status"] in {"done", "paused_by_budget"} else "done"
        database.update_import_job_progress(
            conn,
            job_id,
            phase="enrich",
            status=status,
        )
        pipeline._emit_metrics(status=status, metrics=metrics)
        publish_progress(
            {
                "type": "enrich.done",
                "enriched": pipeline.enriched_success,
                "failed": pipeline.enriched_failed,
            }
        )
        return
    api_key = resolve_api_key()
    if not api_key:
        logger.warning("Enrichment job %s starting without API key", job_id)
    logger.info(
        "enrich job=%s starting concurrency=%d batch_size=%d target_tokens=%d",
        job_id,
        pipeline.concurrency,
        pipeline.batch_size,
        pipeline.target_tokens,
    )
    try:
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            worker_count = max(1, min(pipeline.concurrency, pipeline.remaining or pipeline.concurrency))
            tasks = [
                asyncio.create_task(pipeline.worker_loop(client, api_key))
                for _ in range(worker_count)
            ]
            await asyncio.gather(*tasks)
        metrics = pipeline.snapshot_metrics()
        database.update_enrichment_metrics(conn, job_id, metrics)
        if pipeline.budget_paused:
            database.update_import_job_progress(
                conn,
                job_id,
                phase="enrich",
                status="paused_by_budget",
            )
            pipeline._emit_metrics(status="paused_by_budget", metrics=metrics)
        else:
            database.update_import_job_progress(
                conn,
                job_id,
                phase="enrich",
                status="done",
            )
            pipeline._emit_metrics(status="done", metrics=metrics)
    except Exception as exc:
        logger.exception("Enrichment job %s crashed", job_id)
        database.update_import_job_progress(
            conn,
            job_id,
            phase="enrich",
            status="error",
            error=str(exc),
        )
        _emit_enrich_progress(job_id, phase="enrich", status="error", error=str(exc))
        raise
    finally:
        publish_progress(
            {
                "type": "enrich.done",
                "enriched": pipeline.enriched_success,
                "failed": pipeline.enriched_failed,
            }
        )


def run_job_sync(job_id: int) -> None:
    asyncio.run(run_job(job_id))
