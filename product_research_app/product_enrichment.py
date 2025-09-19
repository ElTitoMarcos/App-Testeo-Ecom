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
from .progress_events import publish_progress
from .student_model import StudentModelManager, build_feature_sample as build_student_sample
from .similarity_engine import SimilarityEngine, SimilarityMatch

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

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
STUDENT_ENABLED_DEFAULT = _env_flag("ENRICH_STUDENT_ENABLED", True)
STUDENT_CONFIDENCE_DEFAULT = float(os.getenv("ENRICH_STUDENT_CONFIDENCE", "0.65"))
STUDENT_MODEL_DIR = os.getenv("ENRICH_STUDENT_MODEL_DIR")
SIMILARITY_ENABLED_DEFAULT = _env_flag("ENRICH_SIMILARITY_ENABLED", True)
SIMILARITY_THRESHOLD_DEFAULT = float(os.getenv("ENRICH_SIMILARITY_THRESHOLD", "0.88"))
SIMILARITY_MAX_ENTRIES_DEFAULT = int(os.getenv("ENRICH_SIMILARITY_MAX_ENTRIES", "5000"))

SYSTEM_PROMPT = (
    "Eres un analista de marketing. Evalúa cada producto y responde EXCLUSIVAMENTE "
    "con JSON válido siguiendo este esquema: {\"results\": [{\"id\": int, \"desire\": int, "
    "\"awareness\": int, \"reason\": string<=120 chars}]}."
)

DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d")


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
    student_defaults = {
        "enabled": STUDENT_ENABLED_DEFAULT,
        "confidence_threshold": STUDENT_CONFIDENCE_DEFAULT,
        "model_dir": STUDENT_MODEL_DIR,
    }
    student_cfg = dict(student_defaults)
    if isinstance(raw_enrich.get("student"), dict):
        student_cfg.update(raw_enrich["student"])
    similarity_defaults = {
        "enabled": SIMILARITY_ENABLED_DEFAULT,
        "threshold": SIMILARITY_THRESHOLD_DEFAULT,
        "max_entries": SIMILARITY_MAX_ENTRIES_DEFAULT,
    }
    similarity_cfg = dict(similarity_defaults)
    if isinstance(raw_enrich.get("similarity"), dict):
        similarity_cfg.update(raw_enrich["similarity"])
    merged = {
        **defaults,
        **{
            k: v
            for k, v in raw_enrich.items()
            if k not in {"triage", "student", "similarity"}
        },
    }
    merged["triage"] = triage_cfg
    merged["student"] = student_cfg
    merged["similarity"] = similarity_cfg
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
        desire = clamp_score(entry.get("desire"))
        awareness = clamp_score(entry.get("awareness"))
        reason = str(entry.get("reason") or "").strip()
        if len(reason) > 120:
            reason = reason[:117].rstrip() + "..."
        normalised.append(
            {
                "id": item_id,
                "desire": desire,
                "awareness": awareness,
                "reason": reason,
                "source": entry.get("source") or "ai",
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
    features: Dict[str, Any] = field(default_factory=dict)
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
    total_enriched: int = 0
    total_failed: int = 0
    last_emit: float = field(init=False, default=0.0)
    student_manager: Optional[StudentModelManager] = field(init=False, default=None)
    student_enabled: bool = field(init=False, default=False)
    student_attempts: int = 0
    student_predictions: int = 0
    student_low_confidence: int = 0
    similarity_engine: Optional[SimilarityEngine] = field(init=False, default=None)
    similarity_enabled: bool = field(init=False, default=False)
    similarity_attempts: int = 0
    similarity_matches: int = 0
    local_predictions: int = 0
    similarity_threshold: float = 0.0
    student_confidence: float = 0.0

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
        student_cfg = self.config.get("student") or {}
        self.student_confidence = float(
            student_cfg.get("confidence_threshold", STUDENT_CONFIDENCE_DEFAULT)
        )
        self.student_enabled = bool(student_cfg.get("enabled", False))
        self.student_manager = None
        if self.student_enabled:
            self.student_manager = StudentModelManager(
                enabled=True,
                confidence_threshold=self.student_confidence,
                model_dir=student_cfg.get("model_dir") or STUDENT_MODEL_DIR,
                logger=self.logger,
            )
            if not self.student_manager.is_ready:
                self.logger.info("Student model not ready; disabling local predictions")
                self.student_enabled = False
                self.student_manager = None
        similarity_cfg = self.config.get("similarity") or {}
        self.similarity_threshold = float(
            similarity_cfg.get("threshold", SIMILARITY_THRESHOLD_DEFAULT)
        )
        self.similarity_enabled = bool(similarity_cfg.get("enabled", False))
        self.similarity_engine = None
        if self.similarity_enabled:
            self.similarity_engine = SimilarityEngine(
                enabled=True,
                threshold=self.similarity_threshold,
                max_entries=int(
                    similarity_cfg.get("max_entries", SIMILARITY_MAX_ENTRIES_DEFAULT)
                ),
                logger=self.logger,
            )
            if not self.similarity_engine.enabled:
                self.similarity_enabled = False
                self.similarity_engine = None
                
        self.start_time = time.perf_counter()
        self.started_iso = datetime.utcnow().isoformat()
        self.lock = asyncio.Lock()
        self.last_emit = time.perf_counter()

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
        student_cfg = dict(cfg.get("student") or {})
        student_cfg.update(
            {
                "enabled": self.student_enabled and self.student_manager is not None,
                "confidence_threshold": self.student_confidence,
                "model_dir": student_cfg.get("model_dir") or STUDENT_MODEL_DIR,
            }
        )
        cfg["student"] = student_cfg
        similarity_cfg = dict(cfg.get("similarity") or {})
        similarity_cfg.update(
            {
                "enabled": self.similarity_enabled and self.similarity_engine is not None,
                "threshold": self.similarity_threshold,
                "max_entries": similarity_cfg.get("max_entries")
                or SIMILARITY_MAX_ENTRIES_DEFAULT,
            }
        )
        cfg["similarity"] = similarity_cfg
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
            "enriched": self.total_enriched,
            "failed": self.total_failed,
            "student_attempts": self.student_attempts,
            "student_predictions": self.student_predictions,
            "student_low_confidence": self.student_low_confidence,
            "student_confidence_threshold": self.student_confidence,
            "similarity_attempts": self.similarity_attempts,
            "similarity_matches": self.similarity_matches,
            "similarity_threshold": self.similarity_threshold,
            "local_predictions": self.local_predictions,
        }

    def _estimate_eta_ms(self) -> Optional[int]:
        if not self.total_items:
            return 0
        remaining = max(self.total_items - self.processed, 0)
        if remaining <= 0:
            return 0
        if self.processed <= 0 or self.total_duration_ms <= 0:
            return None
        avg_ms = self.total_duration_ms / max(self.processed, 1)
        if avg_ms <= 0:
            return None
        return int(max(0, round(avg_ms * remaining)))

    def _emit_progress(
        self,
        *,
        message: Optional[str] = None,
        status: Optional[str] = None,
    ) -> None:
        try:
            total = max(self.total_items, 0)
            processed = max(self.processed, 0)
            status_lower = str(status).lower() if status else None
            if total > 0:
                percent = int(round((processed / max(total, 1)) * 100))
            elif status_lower in {"done", "completed", "error"}:
                percent = 100
            else:
                percent = 0
            percent = max(0, min(100, percent))
            queued = max(total - processed, 0) if total else 0
            payload: Dict[str, Any] = {
                "operation": "enrich",
                "phase": "enrich",
                "percent": percent,
                "imported": processed,
                "processed": processed,
                "total": total,
                "enriched": self.total_enriched,
                "failed": self.total_failed,
                "queued": queued,
                "cache_hits": self.cache_hits,
                "triage_skipped": self.triage_skipped,
                "student_predictions": self.student_predictions,
                "similarity_matches": self.similarity_matches,
                "ai_requests": self.requests,
            }
            if status:
                payload["status"] = status
            elif self.budget_paused:
                payload["status"] = "paused_by_budget"
            if message:
                payload["message"] = message
            eta_ms = self._estimate_eta_ms()
            if eta_ms is not None:
                payload["eta_ms"] = eta_ms
            publish_progress(self.job_id, payload)
            self.last_emit = time.perf_counter()
        except Exception:
            self.logger.exception("Failed to publish enrichment progress")

    def prepare(self) -> None:
        pending = database.list_items_by_state(self.conn, self.job_id, "pending_enrich")
        self.total_items = len(pending)
        if not pending:
            return
        if self.similarity_engine and self.similarity_enabled:
            try:
                base_rows = database.get_enriched_items_for_similarity(
                    self.conn,
                    getattr(self.similarity_engine, "max_entries", SIMILARITY_MAX_ENTRIES_DEFAULT),
                )
                self.similarity_engine.prepare(base_rows)
            except Exception:
                self.logger.exception("Failed to prepare similarity index")
                
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
            features = build_student_sample(raw_data, product)
            item = PendingItem(
                item_id=row["id"],
                sig_hash=row["sig_hash"],
                payload=payload,
                raw=raw_data,
                tokens_estimate=tokens_estimate,
                features=features,
                low_priority=low_priority,
            )
            cache_entry = cache_rows.get(row["sig_hash"])
            if cache_entry:
                self._apply_cache_hit(item, cache_entry)
            else:
                attempted_student = False
                if (
                    self.student_manager
                    and self.student_enabled
                    and bool(str(features.get("text") or "").strip())
                ):
                    self.student_attempts += 1
                    predicted = self.student_manager.predict(features, sig_hash=item.sig_hash)
                    attempted_student = True
                    if predicted:
                        self.student_predictions += 1
                        self._apply_local_enrichment(item, predicted)
                        continue
                if attempted_student:
                    self.student_low_confidence += 1
                if (
                    self.similarity_engine
                    and self.similarity_enabled
                    and bool(str(features.get("text") or "").strip())
                ):
                    self.similarity_attempts += 1
                    match = self.similarity_engine.match(features, sig_hash=item.sig_hash)
                    if match and (match.desire is not None or match.awareness is not None):
                        result = self._result_from_similarity(match)
                        self.similarity_matches += 1
                        self._apply_local_enrichment(item, result)
                        continue

                if low_priority and self.triage_enabled:
                    self.low_priority.append(item)
                else:
                    self.high_priority.append(item)
        self.conn.commit()
        database.update_enrichment_metrics(self.conn, self.job_id, self.snapshot_metrics())
        self.logger.info(
            "enrich job=%s queued high=%d low=%d cache_hits=%d student=%d similarity=%d local=%d",
            self.job_id,
            len(self.high_priority),
            len(self.low_priority),
            self.cache_hits,
            self.student_predictions,
            self.similarity_matches,
            self.local_predictions,
        )

    def _apply_cache_hit(self, item: PendingItem, cache_row: Any) -> None:
        desire = cache_row["desire"]
        awareness = cache_row["awareness"]
        reason = cache_row["reason"]
        source = cache_row["source"] or "cache"
        database.update_product_enrichment(
            self.conn,
            item.sig_hash,
            int(desire) if desire is not None else None,
            int(awareness) if awareness is not None else None,
            reason,
            source=source,
        )
        stored_result = {
            "id": item.item_id,
            "desire": int(desire) if desire is not None else None,
            "awareness": int(awareness) if awareness is not None else None,
            "reason": reason,
            "source": source,
        }
        database.mark_item_enriched(self.conn, item.item_id, stored_result)
        self.cache_hits += 1
        self.processed += 1
        self.total_enriched += 1
        self.logger.info(
            "enrich job=%s cache hit item=%s sig=%s",
            self.job_id,
            item.item_id,
            item.sig_hash,
        )
        self._register_similarity(item, stored_result)

    def _register_similarity(self, item: PendingItem, result: Dict[str, Any]) -> None:
        if not self.similarity_engine or not self.similarity_enabled:
            return
        try:
            payload = {
                "desire": result.get("desire"),
                "awareness": result.get("awareness"),
                "reason": result.get("reason"),
                "source": result.get("source"),
            }
            self.similarity_engine.register(item.features, payload, sig_hash=item.sig_hash)
        except Exception:
            self.logger.exception(
                "Failed to register similarity vector for item=%s", item.item_id
            )

    def _result_from_similarity(self, match: SimilarityMatch) -> Dict[str, Any]:
        def _adjust(value: Optional[int]) -> Optional[int]:
            if value is None:
                return None
            delta = 1 if match.score >= 0.95 else 2
            if match.score < 0.85:
                delta = 3
            if value >= 50:
                return clamp(value - delta, 0, 100)
            return clamp(value + delta, 0, 100)

        reason = (match.reason or "Hereda de similar")[:120]
        return {
            "desire": _adjust(match.desire),
            "awareness": _adjust(match.awareness),
            "reason": reason,
            "source": match.source,
            "similarity": match.score,
        }

    def _apply_local_enrichment(self, item: PendingItem, result: Dict[str, Any]) -> None:
        desire = result.get("desire")
        awareness = result.get("awareness")
        reason = str(result.get("reason") or "").strip()
        if len(reason) > 120:
            reason = reason[:117].rstrip() + "..."
        source = result.get("source") or "student"
        database.update_product_enrichment(
            self.conn,
            item.sig_hash,
            int(desire) if desire is not None else None,
            int(awareness) if awareness is not None else None,
            reason or None,
            source=source,
        )
        stored = dict(result)
        stored.update({"id": item.item_id, "reason": reason, "source": source})
        database.mark_item_enriched(self.conn, item.item_id, stored)
        database.upsert_enrichment_cache(
            self.conn,
            item.sig_hash,
            int(desire) if desire is not None else None,
            int(awareness) if awareness is not None else None,
            reason or None,
            source=source,
        )
        self.processed += 1
        self.total_enriched += 1
        self.local_predictions += 1
        self.logger.info(
            "enrich job=%s local source=%s item=%s sig=%s",
            self.job_id,
            source,
            item.item_id,
            item.sig_hash,
        )
        self._register_similarity(item, stored)

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
            self._emit_progress(message="Máximo de peticiones alcanzado")
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
            self._emit_progress(message="Pausado por presupuesto")
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
        self.total_failed += len(skipped)
        self.conn.commit()
        if skipped:
            self._emit_progress(message="Triaged low priority items")

    async def dequeue_batch(self) -> Optional[List[PendingItem]]:
        async with self.lock:
            if self.budget_paused:
                return None
            if not self.high_priority and self.low_priority and not self.mode_exhaustivo:
                self._skip_low_priority_pending()
                database.update_enrichment_metrics(
                    self.conn, self.job_id, self.snapshot_metrics()
                )
                return None
            queue = self.high_priority if self.high_priority else self.low_priority
            if not queue:
                return None
            batch_size = self._determine_batch_size(queue)
            items = list(itertools.islice(queue, 0, batch_size))
            estimated_tokens = sum(max(item.tokens_estimate, 1) for item in items)
            if self._budget_would_exceed(estimated_tokens):
                database.update_enrichment_metrics(
                    self.conn, self.job_id, self.snapshot_metrics()
                )
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
            self._update_metrics_after_batch(len(items), 0, failed, estimated_tokens, elapsed_ms)
            database.update_enrichment_metrics(self.conn, self.job_id, self.snapshot_metrics())
            self._emit_progress(message=f"Lote con error ({failed} items)")
            return
        mapping = {entry["id"]: entry for entry in normalised}
        enriched, failed, applied_results = await asyncio.to_thread(
            self._apply_results_sync, items, mapping
        )
        for pending_item, result in applied_results:
            self._register_similarity(pending_item, result)
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
        database.update_enrichment_metrics(self.conn, self.job_id, self.snapshot_metrics())
        self._emit_progress(
            message=f"Lote completado ({self.processed}/{self.total_items})",
        )

    def _apply_results_sync(
        self, items: List[PendingItem], mapping: Dict[int, Dict[str, Any]]
    ) -> tuple[int, int, List[tuple[PendingItem, Dict[str, Any]]]]:
        enriched = 0
        failed = 0
        applied: List[tuple[PendingItem, Dict[str, Any]]] = []
        for item in items:
            result = mapping.get(item.item_id)
            if result:
                source = result.get("source", "ai")
                reason = result.get("reason")
                database.update_product_enrichment(
                    self.conn,
                    item.sig_hash,
                    result.get("desire"),
                    result.get("awareness"),
                    reason,
                    source=source,
                )
                database.mark_item_enriched(self.conn, item.item_id, result)
                database.upsert_enrichment_cache(
                    self.conn,
                    item.sig_hash,
                    result.get("desire"),
                    result.get("awareness"),
                    reason,
                    source=source,
                )
                enriched += 1
                applied.append(
                    (
                        item,
                        {
                            "desire": result.get("desire"),
                            "awareness": result.get("awareness"),
                            "reason": reason,
                            "source": source,
                        },
                    )
                )
            else:
                database.mark_item_failed(
                    self.conn, item.item_id, error="missing_result"
                )
                failed += 1
        self.conn.commit()
        return enriched, failed, applied

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
        self.total_enriched += enriched
        self.total_failed += failed

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
    pipeline.prepare()
    database.update_enrichment_metrics(conn, job_id, pipeline.snapshot_metrics())
    pipeline._emit_progress(message="Cola preparada")
    if pipeline.remaining == 0:
        status = job["status"] if job["status"] in {"done", "paused_by_budget"} else "done"
        database.update_import_job_progress(
            conn,
            job_id,
            phase="enrich",
            status=status,
        )
        pipeline._emit_progress(message="Enriquecimiento completado", status=status)
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
        database.update_enrichment_metrics(conn, job_id, pipeline.snapshot_metrics())
        if pipeline.budget_paused:
            database.update_import_job_progress(
                conn,
                job_id,
                phase="enrich",
                status="paused_by_budget",
            )
            pipeline._emit_progress(
                message="Enriquecimiento pausado por presupuesto",
                status="paused_by_budget",
            )
        else:
            database.update_import_job_progress(
                conn,
                job_id,
                phase="enrich",
                status="done",
            )
            pipeline._emit_progress(
                message="Enriquecimiento completado",
                status="done",
            )
    except Exception as exc:
        logger.exception("Enrichment job %s crashed", job_id)
        database.update_import_job_progress(
            conn,
            job_id,
            phase="enrich",
            status="error",
            error=str(exc),
        )
        pipeline._emit_progress(message=str(exc), status="error")
        raise


def run_job_sync(job_id: int) -> None:
    asyncio.run(run_job(job_id))
