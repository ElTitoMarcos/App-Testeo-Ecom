"""Utilities to guard GPT usage with batching, rate limiting and caching."""

from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import json
import logging
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .. import database

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parents[1]
DB_PATH = APP_DIR / "data.sqlite3"

JsonLike = Dict[str, Any]
CallResult = Any


@dataclass
class _PendingRequest:
    """Internal representation of a batch waiting to be executed."""

    items: List[JsonLike]
    future: concurrent.futures.Future
    call_fn: Callable[[Sequence[JsonLike]], CallResult]
    created_at: float


@dataclass
class _BatchOutcome:
    """Result of a batch execution."""

    success: bool
    result: Optional[CallResult] = None
    error: Optional[str] = None
    note: Optional[str] = None
    attempts: int = 0


class GPTGuard:
    """Orchestrate GPT calls honouring batching, rate limits and budgets."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.max_parallel = max(1, int(config.get("max_parallel", 1) or 1))
        self.max_calls_per_import = int(config.get("max_calls_per_import", 0) or 0)
        self.min_batch = max(1, int(config.get("min_batch", 1) or 1))
        max_batch = int(config.get("max_batch", self.min_batch) or self.min_batch)
        self.max_batch = max(self.min_batch, max_batch)
        self.coalesce_ms = max(0, int(config.get("coalesce_ms", 0) or 0))
        self._coalesce_window = self.coalesce_ms / 1000.0

        self._lock = threading.Lock()
        self._queues: Dict[str, Deque[_PendingRequest]] = {}
        self._conditions: Dict[str, threading.Condition] = {}
        self._workers: Dict[str, threading.Thread] = {}
        self._shutdown = False

        self._parallel_cond = threading.Condition(self._lock)
        self._current_parallel_limit = self.max_parallel
        self._inflight_calls = 0
        self._allocated_calls = 0
        self._total_attempts = 0

    def submit(
        self,
        task_type: str,
        items: Sequence[Mapping[str, Any]],
        call_fn: Callable[[Sequence[JsonLike]], CallResult],
    ) -> Dict[str, Any]:
        """Submit items for a GPT-powered task, respecting quotas and batching."""

        if not task_type or not isinstance(task_type, str):
            raise ValueError("task_type must be a non-empty string")
        if not callable(call_fn):
            raise TypeError("call_fn must be callable")

        normalized_items = _normalize_items(items)
        submitted = len(normalized_items)
        if not normalized_items:
            return {
                "task_type": task_type,
                "submitted": submitted,
                "calls": 0,
                "processed": 0,
                "skipped": 0,
                "skipped_items": [],
                "results": [],
                "errors": [],
                "notes": [],
            }

        requests: List[_PendingRequest] = []
        skipped_items: List[JsonLike] = []

        for chunk in _chunk_items(normalized_items, self.max_batch):
            request = _PendingRequest(
                items=list(chunk),
                future=concurrent.futures.Future(),
                call_fn=call_fn,
                created_at=time.monotonic(),
            )
            with self._lock:
                condition = self._conditions.get(task_type)
                if condition is None:
                    condition = threading.Condition(self._lock)
                    self._conditions[task_type] = condition
                queue = self._queues.setdefault(task_type, deque())
                if self.max_calls_per_import > 0 and self._allocated_calls >= self.max_calls_per_import:
                    skipped_items.extend(_clone_items(chunk))
                    continue
                if self.max_calls_per_import > 0:
                    self._allocated_calls += 1
                queue.append(request)
                condition.notify_all()
                if task_type not in self._workers:
                    worker = threading.Thread(
                        target=self._worker_loop,
                        args=(task_type,),
                        name=f"gpt-guard-{task_type}",
                        daemon=True,
                    )
                    self._workers[task_type] = worker
                    worker.start()
            requests.append(request)

        outcomes: List[Dict[str, Any]] = []
        errors: List[str] = []
        processed = 0
        notes: List[str] = []

        for req in requests:
            try:
                outcome = req.future.result()
            except Exception as exc:  # pragma: no cover - defensive
                message = self._format_error(exc)
                outcome = {
                    "success": False,
                    "result": None,
                    "error": message,
                    "note": None,
                    "items": _clone_items(req.items),
                    "batch_items": _clone_items(req.items),
                    "attempts": 0,
                }
            if outcome.get("success"):
                processed += len(req.items)
            else:
                error_msg = outcome.get("error")
                if error_msg:
                    errors.append(error_msg)
            note = outcome.get("note")
            if note:
                notes.append(str(note))
            outcomes.append(outcome)

        if skipped_items:
            notes.append("budget_exhausted")

        summary = {
            "task_type": task_type,
            "submitted": submitted,
            "calls": len(outcomes),
            "processed": processed,
            "skipped": len(skipped_items),
            "skipped_items": skipped_items,
            "results": outcomes,
            "errors": errors,
            "notes": _dedupe(notes),
        }
        return summary

    def _worker_loop(self, task_type: str) -> None:
        condition = self._conditions.get(task_type)
        if condition is None:
            condition = threading.Condition(self._lock)
            self._conditions[task_type] = condition
        queue = self._queues.setdefault(task_type, deque())

        while True:
            with condition:
                while not queue and not self._shutdown:
                    condition.wait()
                if self._shutdown and not queue:
                    return
                batch = self._collect_ready_batch_locked(task_type, condition)
                if batch is None:
                    continue
            call_fn, batch_items, requests = batch
            try:
                outcome = self._execute_batch(task_type, call_fn, batch_items)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("GPTGuard batch execution crashed: task=%s error=%s", task_type, exc)
                outcome = _BatchOutcome(
                    success=False,
                    error=self._format_error(exc),
                    note=None,
                    attempts=0,
                )

            for req in requests:
                result_payload = {
                    "success": outcome.success,
                    "result": outcome.result,
                    "error": outcome.error,
                    "note": outcome.note,
                    "items": _clone_items(req.items),
                    "batch_items": _clone_items(batch_items),
                    "attempts": outcome.attempts,
                }
                try:
                    req.future.set_result(result_payload)
                except Exception:  # pragma: no cover - defensive
                    logger.exception("Setting GPTGuard future result failed: task=%s", task_type)

    def _collect_ready_batch_locked(
        self,
        task_type: str,
        condition: threading.Condition,
    ) -> Optional[Tuple[Callable[[Sequence[JsonLike]], CallResult], List[JsonLike], List[_PendingRequest]]]:
        queue = self._queues.get(task_type)
        if not queue:
            return None
        try:
            request = queue.popleft()
        except IndexError:
            return None

        requests = [request]
        items = _clone_items(request.items)
        call_fn = request.call_fn
        deadline = request.created_at + self._coalesce_window

        while True:
            if len(items) >= self.max_batch:
                break
            now = time.monotonic()
            queue = self._queues.get(task_type)
            if len(items) >= self.min_batch and (not queue or self._coalesce_window <= 0 or now >= deadline):
                break
            if not queue:
                if self._coalesce_window <= 0:
                    break
                remaining = max(0.0, deadline - now)
                if remaining <= 0:
                    break
                condition.wait(remaining)
                queue = self._queues.get(task_type)
                continue
            next_req = queue[0]
            if next_req.call_fn is not call_fn:
                if len(items) >= self.min_batch or now >= deadline or self._coalesce_window <= 0:
                    break
                remaining = max(0.0, deadline - now)
                if remaining <= 0:
                    break
                condition.wait(remaining)
                continue
            prospective_size = len(items) + len(next_req.items)
            if prospective_size > self.max_batch:
                if len(items) >= self.min_batch or now >= deadline:
                    break
                remaining = max(0.0, deadline - now)
                if remaining <= 0:
                    break
                condition.wait(remaining)
                continue
            requests.append(queue.popleft())
            items.extend(_clone_items(next_req.items))

        return call_fn, items, requests

    def _execute_batch(
        self,
        task_type: str,
        call_fn: Callable[[Sequence[JsonLike]], CallResult],
        items: Sequence[JsonLike],
    ) -> _BatchOutcome:
        attempts = 0
        note: Optional[str] = None
        last_error: Optional[BaseException] = None

        while attempts < 2:
            attempts += 1
            try:
                with self._acquire_slot():
                    result = call_fn(items)
                return _BatchOutcome(success=True, result=result, note=note, attempts=attempts)
            except Exception as exc:  # pragma: no cover - network guarded
                last_error = exc
                status = self._extract_status_code(exc)
                if attempts == 1 and self._is_rate_limit(exc, status):
                    note = "rate_limited"
                    self._reduce_parallel_due_to_rate_limit()
                    delay = self._retry_after_seconds(exc)
                    logger.warning(
                        "GPTGuard rate limit encountered on %s; retrying after %.2fs", task_type, delay
                    )
                    self._sleep(delay)
                    continue
                if attempts == 1 and status is not None and 500 <= status < 600:
                    note = f"server_error_{status}"
                    delay = self._server_retry_delay()
                    logger.warning(
                        "GPTGuard server error %s on %s; retrying after %.2fs", status, task_type, delay
                    )
                    self._sleep(delay)
                    continue
                break

        error_message = self._format_error(last_error)
        return _BatchOutcome(success=False, error=error_message, note=note, attempts=attempts)

    @contextlib.contextmanager
    def _acquire_slot(self):
        with self._parallel_cond:
            while self._inflight_calls >= self._current_parallel_limit and not self._shutdown:
                self._parallel_cond.wait()
            self._inflight_calls += 1
            self._total_attempts += 1
        try:
            yield
        finally:
            with self._parallel_cond:
                self._inflight_calls = max(0, self._inflight_calls - 1)
                self._parallel_cond.notify_all()

    def _reduce_parallel_due_to_rate_limit(self) -> None:
        with self._parallel_cond:
            if self._current_parallel_limit != 1:
                logger.info(
                    "Reducing GPT parallelism to 1 due to rate limiting (was %s)",
                    self._current_parallel_limit,
                )
                self._current_parallel_limit = 1
                self._parallel_cond.notify_all()

    def _retry_after_seconds(self, exc: BaseException) -> float:
        for attr in ("retry_after", "retry_after_ms"):
            value = getattr(exc, attr, None)
            if value is None:
                continue
            parsed = self._parse_retry_after(value, is_ms=attr.endswith("_ms"))
            if parsed is not None:
                return parsed
        response = getattr(exc, "response", None)
        if response is not None:
            headers = getattr(response, "headers", {}) or {}
            retry_after = headers.get("Retry-After") or headers.get("retry-after")
            if retry_after:
                parsed = self._parse_retry_after(retry_after)
                if parsed is not None:
                    return parsed
        return 10.0

    def _server_retry_delay(self) -> float:
        return random.uniform(1.0, 3.0)

    def _sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)

    @staticmethod
    def _format_error(exc: BaseException) -> str:
        message = str(exc)
        if message:
            return message
        return exc.__class__.__name__

    @staticmethod
    def _is_rate_limit(exc: BaseException, status: Optional[int]) -> bool:
        if status == 429:
            return True
        name = exc.__class__.__name__.lower()
        message = str(exc).lower()
        return any(
            token in message or token in name
            for token in ("rate limit", "ratelimit", "too many requests", "429")
        )

    @staticmethod
    def _extract_status_code(exc: BaseException) -> Optional[int]:
        for attr in ("status", "status_code", "http_status"):
            value = getattr(exc, attr, None)
            if isinstance(value, int):
                return value
        response = getattr(exc, "response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
            if isinstance(status, int):
                return status
        return None

    @staticmethod
    def _parse_retry_after(value: Any, *, is_ms: bool = False) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            seconds = float(value)
            if is_ms:
                seconds /= 1000.0
            return max(0.0, seconds)
        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return None
            try:
                seconds = float(candidate)
                if is_ms:
                    seconds /= 1000.0
                return max(0.0, seconds)
            except ValueError:
                try:
                    dt = parsedate_to_datetime(candidate)
                except Exception:
                    return None
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                return max(0.0, (dt - now).total_seconds())
        return None


def hash_key_for_item(task_type: str, item: Mapping[str, Any]) -> str:
    """Return a stable hash key for caching GPT results."""

    normalized_task = (task_type or "").strip().lower()
    normalized_payload = _normalize_for_hash(item)
    serialized = json.dumps(normalized_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha1(f"{normalized_task}|{serialized}".encode("utf-8")).hexdigest()
    return digest


def ai_cache_get(task_type: str, key: str) -> Optional[Dict[str, Any]]:
    """Fetch a cached payload for ``task_type`` if present."""

    if not task_type or not key:
        return None
    conn = database.get_connection(DB_PATH)
    try:
        database.initialize_database(conn)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT payload_json, model_version, created_at
            FROM ai_cache
            WHERE task_type=? AND cache_key=?
            """,
            (task_type, key),
        )
        row = cur.fetchone()
    finally:
        try:
            conn.close()
        except Exception:  # pragma: no cover - defensive
            pass
    if row is None:
        return None
    payload_json = row["payload_json"] if isinstance(row, MutableMapping) else row[0]
    model_version = row["model_version"] if isinstance(row, MutableMapping) else row[1]
    created_at = row["created_at"] if isinstance(row, MutableMapping) else row[2]
    try:
        payload = json.loads(payload_json) if payload_json else None
    except Exception:
        payload = None
    return {
        "payload": payload,
        "payload_json": payload_json,
        "model_version": model_version,
        "created_at": created_at,
    }


def ai_cache_set(
    task_type: str,
    key: str,
    payload_json: Any,
    model_version: str,
    ttl_days: int = 180,
) -> None:
    """Persist a payload in the cache with optional TTL pruning."""

    if not task_type or not key:
        raise ValueError("task_type and key are required for caching")
    if not isinstance(payload_json, str):
        payload_str = json.dumps(payload_json, ensure_ascii=False)
    else:
        payload_str = payload_json

    now = datetime.utcnow()
    created_at = now.isoformat()
    cutoff: Optional[str] = None
    if ttl_days and ttl_days > 0:
        cutoff = (now - timedelta(days=int(ttl_days))).isoformat()

    conn = database.get_connection(DB_PATH)
    try:
        database.initialize_database(conn)
        cur = conn.cursor()
        if cutoff is not None:
            cur.execute("DELETE FROM ai_cache WHERE created_at < ?", (cutoff,))
        cur.execute(
            """
            INSERT INTO ai_cache (task_type, cache_key, payload_json, model_version, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(task_type, cache_key) DO UPDATE SET
                payload_json=excluded.payload_json,
                model_version=excluded.model_version,
                created_at=excluded.created_at
            """,
            (task_type, key, payload_str, model_version, created_at),
        )
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:  # pragma: no cover - defensive
            pass


def _normalize_items(items: Sequence[Mapping[str, Any]]) -> List[JsonLike]:
    normalized: List[JsonLike] = []
    for item in items or []:
        try:
            mapping = dict(item)
        except Exception:
            if isinstance(item, Mapping):
                mapping = dict(item.items())
            else:
                logger.debug("Skipping non-mapping item in GPTGuard submit: %r", item)
                continue
        normalized.append(mapping)
    return normalized


def _clone_items(items: Sequence[Mapping[str, Any]]) -> List[JsonLike]:
    return [dict(item) for item in items]


def _chunk_items(items: Sequence[JsonLike], size: int) -> Iterable[Sequence[JsonLike]]:
    if size <= 0:
        size = 1
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _normalize_for_hash(item: Mapping[str, Any]) -> Any:
    if isinstance(item, Mapping):
        return {str(key): _normalize_for_hash(value) for key, value in sorted(item.items())}
    if isinstance(item, str):
        return " ".join(item.strip().split())
    if isinstance(item, Sequence) and not isinstance(item, (bytes, bytearray, str)):
        return [_normalize_for_hash(elem) for elem in item]
    return item


def _dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


__all__ = [
    "GPTGuard",
    "hash_key_for_item",
    "ai_cache_get",
    "ai_cache_set",
]
