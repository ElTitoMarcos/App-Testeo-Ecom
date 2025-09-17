"""Background runner for post-import AI tasks.

This module drains the ``ai_task_queue`` table in batches, executing the
configured AI tasks while keeping concurrency under control.  The execution
prioritises cheap operations (``desire`` first, then ``imputacion`` and finally
``winner_score``) and reports progress back to interested listeners so the
frontend can reflect real-time status updates.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .. import config, database, gpt, settings
from ..services import aggregates as aggregates_service
from ..services import config as winner_config
from ..services import winner_score
from ..utils.db import row_to_dict, rget
from . import gpt_orchestrator
from .gpt_guard import GPTGuard, ai_cache_get, ai_cache_set, hash_key_for_item

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parents[1]
DB_PATH = APP_DIR / "data.sqlite3"

_TASK_ORDER: Sequence[str] = ("desire", "imputacion", "winner_score")
_GPT_TASKS = {"desire", "imputacion"}
_MAX_ATTEMPTS = 3  # first run + 2 retries
_GPT_CALL_SEMAPHORE = threading.Semaphore(3)

_ProgressCallback = Callable[[str, str, Mapping[str, int]], None]
_PROGRESS_CALLBACKS: Dict[str, _ProgressCallback] = {}
_PROGRESS_LOCK = threading.Lock()

_AI_STATUS: Dict[str, Dict[str, Dict[str, int]]] = {}
_AI_STATUS_LOCK = threading.Lock()


@dataclass
class _BatchResult:
    task_type: str
    processed: Dict[str, int] = field(default_factory=dict)
    failed: Dict[str, int] = field(default_factory=dict)
    errors: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class _RunnerContext:
    api_key: Optional[str]
    model: Optional[str]
    include_image: bool
    max_attempts: int
    max_calls_per_import: int = 0
    calls_used: Dict[str, int] = field(default_factory=dict)
    winner_weights_ready: bool = False
    winner_weights_lock: threading.Lock = field(default_factory=threading.Lock)

    def remaining_calls(self, import_id: str) -> int:
        if self.max_calls_per_import <= 0 or not import_id:
            return 1_000_000
        used = self.calls_used.get(import_id, 0)
        return max(self.max_calls_per_import - used, 0)

    def consume_call(self, import_id: str, count: int = 1) -> bool:
        if not import_id or self.max_calls_per_import <= 0:
            return True
        remaining = self.remaining_calls(import_id)
        if remaining < count:
            return False
        self.calls_used[import_id] = self.calls_used.get(import_id, 0) + count
        return True

def register_progress_callback(import_task_id: str, callback: Optional[_ProgressCallback]) -> None:
    """Register or remove a progress callback for an import."""

    key = str(import_task_id or "")
    with _PROGRESS_LOCK:
        if callback is None:
            _PROGRESS_CALLBACKS.pop(key, None)
        else:
            _PROGRESS_CALLBACKS[key] = callback


def unregister_progress_callback(import_task_id: str) -> None:
    register_progress_callback(import_task_id, None)


def _notify_progress(import_task_id: str, task_type: str, payload: Mapping[str, int]) -> None:
    with _PROGRESS_LOCK:
        callback = _PROGRESS_CALLBACKS.get(str(import_task_id or ""))
    if callback is None:
        return
    try:
        callback(str(import_task_id or ""), task_type, dict(payload))
    except Exception:  # pragma: no cover - defensive
        logger.exception("Progress callback failed: import=%s", import_task_id)


def _empty_status() -> Dict[str, Dict[str, int]]:
    return {
        name: {"requested": 0, "processed": 0, "failed": 0, "skipped": 0}
        for name in _TASK_ORDER
    }


def _status_entry(task_id: str, task_type: str) -> Dict[str, int]:
    if task_type not in _TASK_ORDER:
        raise ValueError(f"Unknown task type {task_type!r}")
    with _AI_STATUS_LOCK:
        status = _AI_STATUS.setdefault(task_id, _empty_status())
        entry = status.setdefault(task_type, {"requested": 0, "processed": 0, "failed": 0, "skipped": 0})
        snapshot = dict(entry)
    return snapshot


def _set_requested(task_id: str, task_type: str, count: int) -> None:
    with _AI_STATUS_LOCK:
        status = _AI_STATUS.setdefault(task_id, _empty_status())
        entry = status.setdefault(task_type, {"requested": 0, "processed": 0, "failed": 0, "skipped": 0})
        entry["requested"] = max(int(entry.get("requested", 0)), int(count))
        snapshot = dict(entry)
    _notify_progress(task_id, task_type, snapshot)


def _increment_counts(
    task_id: str,
    task_type: str,
    *,
    processed: int = 0,
    failed: int = 0,
    skipped: int = 0,
) -> None:
    with _AI_STATUS_LOCK:
        status = _AI_STATUS.setdefault(task_id, _empty_status())
        entry = status.setdefault(task_type, {"requested": 0, "processed": 0, "failed": 0, "skipped": 0})
        entry["processed"] = int(entry.get("processed", 0)) + int(processed)
        entry["failed"] = int(entry.get("failed", 0)) + int(failed)
        entry["skipped"] = int(entry.get("skipped", 0)) + int(skipped)
        snapshot = dict(entry)
    _notify_progress(task_id, task_type, snapshot)


def _status_snapshot(task_id: str) -> Dict[str, Dict[str, int]]:
    with _AI_STATUS_LOCK:
        status = _AI_STATUS.get(task_id)
        if status is None:
            return _empty_status()
        return {task: dict(values) for task, values in status.items()}


def run_auto(tasks: set[str], *, batch_size: int = 200, max_parallel: int = 3) -> Dict[str, Dict[str, object]]:
    """Drain ``ai_task_queue`` executing the requested task types."""

    ordered_tasks = [name for name in _TASK_ORDER if name in tasks]
    if not ordered_tasks:
        return {}

    try:
        batch_size = int(batch_size)
    except Exception:
        batch_size = settings.AI_MAX_BATCH_SIZE
    min_batch = max(1, settings.AI_MIN_BATCH_SIZE)
    max_batch = max(min_batch, settings.AI_MAX_BATCH_SIZE)
    batch_size = max(min_batch, min(batch_size, max_batch))

    try:
        max_parallel = int(max_parallel)
    except Exception:
        max_parallel = settings.AI_MAX_PARALLEL
    max_parallel = max(1, min(max_parallel, settings.AI_MAX_PARALLEL))

    api_key = config.get_api_key() or None
    model = config.get_model() or "gpt-4o-mini"
    include_image = config.include_image_in_ai()

    context = _RunnerContext(
        api_key=api_key,
        model=model,
        include_image=include_image,
        max_attempts=_MAX_ATTEMPTS,
        max_calls_per_import=settings.AI_MAX_CALLS_PER_IMPORT,
    )

    progress: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: {
            name: {"requested": 0, "processed": 0, "failed": 0, "skipped": 0}
            for name in _TASK_ORDER
        }
    )
    errors: Dict[str, List[str]] = defaultdict(list)
    seen_task_ids: set[int] = set()

    conn = database.get_connection(DB_PATH)
    database.initialize_database(conn)

    try:
        for task_type in ordered_tasks:
            while True:
                pending_rows = database.fetch_pending_ai_tasks(
                    conn,
                    task_types=[task_type],
                    limit=batch_size * max_parallel,
                )
                if not pending_rows:
                    break

                if task_type in _GPT_TASKS and (not context.api_key or not context.model):
                    logger.warning(
                        "AI runner skipping %s tasks due to missing API configuration",
                        task_type,
                    )

                groups: Dict[str, List[Mapping[str, object]]] = {}
                for row in pending_rows:
                    row_dict = dict(row)
                    import_id = str(row_dict.get("import_task_id") or "")
                    entry = progress[import_id][task_type]
                    task_id = int(row_dict["id"])
                    if task_id not in seen_task_ids:
                        seen_task_ids.add(task_id)
                        entry["requested"] += 1
                    groups.setdefault(import_id, []).append(row_dict)

                batches: List[List[Mapping[str, object]]] = []
                process_task_ids: List[int] = []
                skipped_task_ids: List[int] = []
                skipped_imports: set[str] = set()

                for import_id, rows_for_import in groups.items():
                    chunks = [
                        rows_for_import[i : i + batch_size]
                        for i in range(0, len(rows_for_import), batch_size)
                    ]
                    if not chunks:
                        continue
                    if context.max_calls_per_import > 0 and import_id:
                        remaining_calls = context.remaining_calls(import_id)
                        allowed_count = min(len(chunks), remaining_calls)
                    else:
                        allowed_count = len(chunks)
                    for idx, chunk in enumerate(chunks):
                        if idx < allowed_count and context.consume_call(import_id):
                            batches.append(chunk)
                            process_task_ids.extend(int(row["id"]) for row in chunk)
                        else:
                            skipped_task_ids.extend(int(row["id"]) for row in chunk)
                            progress[import_id][task_type]["skipped"] += len(chunk)
                            skipped_imports.add(import_id)

                if skipped_task_ids:
                    database.skip_ai_tasks(conn, skipped_task_ids, "budget_exhausted")
                    for import_id in skipped_imports:
                        _notify_progress(import_id, task_type, progress[import_id][task_type])

                if not batches:
                    continue

                if process_task_ids:
                    database.mark_ai_tasks_in_progress(conn, process_task_ids)

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
                    futures = [executor.submit(_process_batch, task_type, batch, context) for batch in batches]
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        for import_id, value in result.processed.items():
                            progress[import_id][task_type]["processed"] += int(value)
                        for import_id, value in result.failed.items():
                            progress[import_id][task_type]["failed"] += int(value)
                        for import_id, msgs in result.errors.items():
                            if msgs:
                                errors[import_id].extend(msgs)
                        touched = set(result.processed.keys()) | set(result.failed.keys())
                        for import_id in touched:
                            _notify_progress(import_id, task_type, progress[import_id][task_type])
    finally:
        try:
            conn.close()
        except Exception:
            pass

    summary: Dict[str, Dict[str, object]] = {}
    for import_id, task_map in progress.items():
        has_activity = any(
            entry["requested"]
            or entry["processed"]
            or entry["failed"]
            or entry.get("skipped")
            for entry in task_map.values()
        )
        if not has_activity:
            continue
        summary[import_id] = {
            "tasks": {name: dict(vals) for name, vals in task_map.items()},
            "errors": list(errors.get(import_id, [])),
        }
    return summary


def _process_batch(task_type: str, rows: Sequence[Mapping[str, object]], context: _RunnerContext) -> _BatchResult:
    if task_type in _GPT_TASKS:
        return _process_columns_batch(task_type, rows, context)
    if task_type == "winner_score":
        return _process_winner_batch(rows, context)
    logger.warning("Unknown AI task type encountered: %s", task_type)
    return _BatchResult(task_type=task_type)


def _process_columns_batch(task_type: str, rows: Sequence[Mapping[str, object]], context: _RunnerContext) -> _BatchResult:
    result = _BatchResult(task_type=task_type)
    conn = database.get_connection(DB_PATH)
    try:
        product_ids = [int(row["product_id"]) for row in rows if row.get("product_id") is not None]
        if not product_ids:
            return _record_batch_failure(
                conn,
                rows,
                result,
                reason="missing_products",
                allow_retry=False,
                context=context,
            )

        products = database.get_products_by_ids(conn, product_ids)
        if not products:
            return _record_batch_failure(
                conn,
                rows,
                result,
                reason="missing_products",
                allow_retry=False,
                context=context,
            )

        items = _build_product_payloads(products, include_image=context.include_image)
        if not items:
            return _record_batch_failure(
                conn,
                rows,
                result,
                reason="missing_payload",
                allow_retry=False,
                context=context,
            )

        if not context.api_key or not context.model:
            return _record_batch_failure(
                conn,
                rows,
                result,
                reason="openai_unavailable",
                allow_retry=False,
                context=context,
            )

        try:
            with _GPT_CALL_SEMAPHORE:
                ok_map, _, _, _ = gpt.generate_batch_columns(context.api_key, context.model, items)
        except gpt.InvalidJSONError:
            logger.warning("GPT returned invalid JSON for task=%s batch=%s", task_type, [row.get("id") for row in rows])
            return _record_batch_failure(
                conn,
                rows,
                result,
                reason="invalid_json",
                allow_retry=True,
                context=context,
            )
        except Exception as exc:  # pragma: no cover - network guarded
            message = str(exc) or exc.__class__.__name__
            logger.error("GPT batch failed task=%s error=%s", task_type, message)
            return _record_batch_failure(
                conn,
                rows,
                result,
                reason=message,
                allow_retry=True,
                context=context,
            )

        updated = False
        missing_rows: List[Mapping[str, object]] = []
        for row in rows:
            pid = int(row.get("product_id"))
            import_id = str(row.get("import_task_id") or "")
            entry = ok_map.get(str(pid))
            if entry:
                updates = {
                    "desire": entry.get("desire"),
                    "desire_magnitude": entry.get("desire_magnitude"),
                    "awareness_level": entry.get("awareness_level"),
                    "competition_level": entry.get("competition_level"),
                    "ai_columns_completed_at": datetime.utcnow().isoformat(),
                }
                clean_updates = {k: v for k, v in updates.items() if v not in (None, "")}
                if clean_updates:
                    database.update_product(conn, pid, **clean_updates)
                    updated = True
                result.processed[import_id] = result.processed.get(import_id, 0) + 1
            else:
                missing_rows.append(row)

        if updated:
            conn.commit()

        if missing_rows:
            result = _record_batch_failure(
                conn,
                missing_rows,
                result,
                reason="missing_result",
                allow_retry=True,
                context=context,
            )

        missing_ids = {int(r.get("id")) for r in missing_rows}
        completed_ids = [int(row.get("id")) for row in rows if int(row.get("id")) not in missing_ids]
        if completed_ids:
            database.complete_ai_tasks(conn, completed_ids)
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return result


def _process_winner_batch(rows: Sequence[Mapping[str, object]], context: _RunnerContext) -> _BatchResult:
    result = _BatchResult(task_type="winner_score")
    conn = database.get_connection(DB_PATH)
    try:
        product_ids = [int(row["product_id"]) for row in rows if row.get("product_id") is not None]
        if not product_ids:
            return _record_batch_failure(
                conn,
                rows,
                result,
                reason="missing_products",
                allow_retry=False,
                context=context,
            )
        weights = _ensure_winner_weights(conn, product_ids, context)
        try:
            winner_score.generate_winner_scores(conn, product_ids=product_ids, weights=weights)
            task_ids = [int(row["id"]) for row in rows]
            database.complete_ai_tasks(conn, task_ids)
            for row in rows:
                import_id = str(row.get("import_task_id") or "")
                result.processed[import_id] = result.processed.get(import_id, 0) + 1
        except Exception as exc:
            message = str(exc) or "winner_score_error"
            result = _record_batch_failure(
                conn,
                rows,
                result,
                reason=message,
                allow_retry=True,
                context=context,
            )
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return result

def _ensure_winner_weights(
    conn,
    product_ids: Sequence[int],
    context: _RunnerContext,
) -> Optional[Dict[str, int]]:
    """Ensure Winner Score weights are ready, optionally auto-adjusting them."""

    with context.winner_weights_lock:
        if context.winner_weights_ready:
            return winner_config.get_winner_weights_raw()

        cfg = config.load_config()
        auto_adjust = bool(cfg.get("auto_adjust_weights", True))
        existing = cfg.get("winner_weights")
        has_existing = isinstance(existing, dict) and bool(existing)

        if has_existing and not auto_adjust:
            context.winner_weights_ready = True
            return winner_config.get_winner_weights_raw()

        if not product_ids:
            context.winner_weights_ready = True
            return winner_config.get_winner_weights_raw()

        if not auto_adjust:
            context.winner_weights_ready = True
            return winner_config.get_winner_weights_raw()

        if not context.api_key:
            logger.info("Skipping winner weight auto-adjust: missing API credentials")
            context.winner_weights_ready = True
            return winner_config.get_winner_weights_raw()

        aggregates = aggregates_service.compute_dataset_aggregates(
            conn, scope_ids=product_ids
        )
        if not aggregates.get("total_products"):
            context.winner_weights_ready = True
            return winner_config.get_winner_weights_raw()

        try:
            with _GPT_CALL_SEMAPHORE:
                suggestion = gpt.recommend_weights_from_aggregates(
                    context.api_key,
                    "gpt-4o",
                    aggregates,
                )
        except Exception as exc:  # pragma: no cover - network guarded
            logger.warning("Winner weight auto-adjust failed: %s", exc)
            context.winner_weights_ready = True
            return winner_config.get_winner_weights_raw()

        weights = suggestion.get("weights") if isinstance(suggestion, dict) else None
        order = suggestion.get("order") if isinstance(suggestion, dict) else None
        if not isinstance(weights, dict) or not weights:
            logger.info("Winner weight suggestion missing payload; keeping existing weights")
            context.winner_weights_ready = True
            return winner_config.get_winner_weights_raw()

        try:
            weights_final, _, _ = winner_config.update_winner_settings(
                weights_in=weights,
                order_in=order,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Persisting winner weight suggestion failed: %s", exc)
            context.winner_weights_ready = True
            return winner_config.get_winner_weights_raw()

        context.winner_weights_ready = True
        return weights_final

def _record_batch_failure(
    conn,
    rows: Sequence[Mapping[str, object]],
    result: _BatchResult,
    *,
    reason: str,
    allow_retry: bool,
    context: _RunnerContext,
) -> _BatchResult:
    task_ids_requeue: List[int] = []
    task_ids_fail: List[int] = []
    for row in rows:
        task_id = int(row.get("id"))
        import_id = str(row.get("import_task_id") or "")
        result.failed[import_id] = result.failed.get(import_id, 0) + 1
        if allow_retry and _should_retry(row, context.max_attempts):
            task_ids_requeue.append(task_id)
        else:
            task_ids_fail.append(task_id)
            result.errors.setdefault(import_id, []).append(reason)
    if task_ids_requeue:
        database.requeue_ai_tasks(conn, task_ids_requeue)
    if task_ids_fail:
        database.fail_ai_tasks(conn, task_ids_fail, reason[:512])
    return result


def _should_retry(row: Mapping[str, object], max_attempts: int) -> bool:
    try:
        attempts = int(row.get("attempts") or 0)
    except Exception:
        attempts = 0
    return (attempts + 1) < max_attempts


def _build_product_payloads(products: Sequence[Mapping[str, object]], *, include_image: bool) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    for prod in products:
        product = row_to_dict(prod)
        try:
            extra = json.loads(rget(product, "extra") or "{}")
        except Exception:
            extra = {}
        item = {
            "id": rget(product, "id"),
            "name": rget(product, "name"),
            "category": rget(product, "category"),
            "price": rget(product, "price"),
            "rating": extra.get("rating"),
            "units_sold": extra.get("units_sold"),
            "revenue": extra.get("revenue"),
            "conversion_rate": extra.get("conversion_rate"),
            "launch_date": extra.get("launch_date"),
            "date_range": rget(product, "date_range") or extra.get("date_range"),
            "image_url": rget(product, "image_url") or extra.get("image_url"),
        }
        if not include_image:
            item.pop("image_url", None)
        items.append(item)
    return items


def _prepare_product_ids(product_ids: Sequence[int]) -> List[int]:
    seen: set[int] = set()
    ordered: List[int] = []
    for raw in product_ids or []:
        try:
            num = int(raw)
        except Exception:
            continue
        if num in seen:
            continue
        seen.add(num)
        ordered.append(num)
    return ordered


def _load_products(conn, product_ids: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    rows = database.get_products_by_ids(conn, product_ids)
    result: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        product = row_to_dict(row)
        extra_raw = product.get("extra")
        if isinstance(extra_raw, dict):
            extra = dict(extra_raw)
        elif isinstance(extra_raw, str) and extra_raw.strip():
            try:
                extra = json.loads(extra_raw)
            except Exception:
                extra = {}
        else:
            extra = {}
        product["_extra"] = extra
        pid = int(product.get("id"))
        result[pid] = product
    return result


def _update_extra_json(conn, product: Dict[str, Any], product_id: int, updates: Mapping[str, Any]) -> bool:
    if not updates:
        return False
    extra = product.get("_extra")
    if not isinstance(extra, dict):
        extra = {}
    changed = False
    for key, value in updates.items():
        if value is None:
            continue
        if extra.get(key) == value:
            continue
        extra[key] = value
        changed = True
    if changed:
        conn.execute("UPDATE products SET extra = json(?) WHERE id = ?", (json.dumps(extra), product_id))
        product["_extra"] = extra
    return changed


def _normalize_desire_text(text: str) -> str:
    lines = []
    for raw_line in str(text or "").replace("\r\n", "\n").split("\n"):
        clean = " ".join(raw_line.strip().split())
        if not clean:
            continue
        lines.append(clean[:90])
        if len(lines) >= 3:
            break
    return "\n".join(lines)


def _looks_like_desire_summary(text: str) -> bool:
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    if not lines:
        return False
    if not (2 <= len(lines) <= 3):
        return False
    return all(len(ln) <= 90 for ln in lines)


def _load_queue_index(conn, task_id: str, product_ids: Sequence[int]) -> Dict[Tuple[str, int], int]:
    if not task_id or not product_ids:
        return {}
    placeholders = ",".join("?" for _ in product_ids)
    params: List[Any] = [task_id, *product_ids]
    cur = conn.execute(
        f"SELECT id, task_type, product_id FROM ai_task_queue WHERE import_task_id=? AND product_id IN ({placeholders})",
        params,
    )
    index: Dict[Tuple[str, int], int] = {}
    for row in cur.fetchall():
        try:
            task_type = str(row["task_type"])
            product_id = int(row["product_id"])
            index[(task_type, product_id)] = int(row["id"])
        except Exception:
            continue
    return index


class _QueueActions:
    def __init__(self, index: Mapping[Tuple[str, int], int]):
        self._index = dict(index)
        self._touched: set[int] = set()
        self.completed: List[int] = []
        self.skipped: Dict[str, List[int]] = defaultdict(list)
        self.failed: Dict[str, List[int]] = defaultdict(list)

    def _resolve(self, task_type: str, product_id: int) -> Optional[int]:
        queue_id = self._index.get((task_type, product_id))
        if queue_id is None or queue_id in self._touched:
            return None
        self._touched.add(queue_id)
        return queue_id

    def complete(self, task_type: str, product_id: int) -> None:
        queue_id = self._resolve(task_type, product_id)
        if queue_id is not None:
            self.completed.append(queue_id)

    def skip(self, task_type: str, product_id: int, note: str) -> None:
        queue_id = self._resolve(task_type, product_id)
        if queue_id is not None:
            self.skipped[note or "skipped"].append(queue_id)

    def fail(self, task_type: str, product_id: int, reason: str) -> None:
        queue_id = self._resolve(task_type, product_id)
        if queue_id is not None:
            self.failed[reason or "error"].append(queue_id)

    def flush(self, conn) -> None:
        if self.completed:
            database.complete_ai_tasks(conn, self.completed)
        for note, ids in self.skipped.items():
            database.skip_ai_tasks(conn, ids, note[:255])
        for reason, ids in self.failed.items():
            database.fail_ai_tasks(conn, ids, reason[:255])


def _stringify_desire_text(value: Any) -> str:
    if isinstance(value, list):
        lines = [str(line).strip() for line in value if str(line).strip()]
        return "\n".join(lines)
    if isinstance(value, str):
        return value.strip()
    return ""


def _parse_desire_result_payload(payload: Any) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    items: Dict[str, Dict[str, Any]] = {}
    notes: List[str] = []
    if not isinstance(payload, Mapping):
        return items, notes

    raw_items = payload.get("items")
    if isinstance(raw_items, list):
        for entry in raw_items:
            if not isinstance(entry, Mapping):
                continue
            pid = entry.get("id")
            if pid is None:
                continue
            text = _stringify_desire_text(entry.get("normalized_text"))
            if not text:
                text = _stringify_desire_text(entry.get("text"))
            keywords_raw = entry.get("keywords")
            if isinstance(keywords_raw, list):
                keywords = [str(kw).strip() for kw in keywords_raw if str(kw).strip()]
            else:
                keywords = []
            items[str(pid)] = {"normalized_text": text, "keywords": keywords}
    else:
        for key, value in payload.items():
            if key == "notes":
                continue
            if not isinstance(key, str) or not isinstance(value, Mapping):
                continue
            text = _stringify_desire_text(value.get("normalized_text") or value.get("text"))
            keywords_raw = value.get("keywords")
            if isinstance(keywords_raw, list):
                keywords = [str(kw).strip() for kw in keywords_raw if str(kw).strip()]
            else:
                keywords = []
            items[key] = {"normalized_text": text, "keywords": keywords}

    notes_field = payload.get("notes")
    if isinstance(notes_field, list):
        notes.extend(str(note).strip() for note in notes_field if str(note).strip())
    elif isinstance(notes_field, Mapping):
        for key, message in notes_field.items():
            msg = str(message).strip()
            if not msg:
                continue
            if key not in (None, ""):
                notes.append(f"{key}: {msg}")
            else:
                notes.append(msg)
    elif isinstance(notes_field, str) and notes_field.strip():
        notes.append(notes_field.strip())

    return items, list(dict.fromkeys(notes))


def _parse_imputacion_result_payload(payload: Any) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    items: Dict[str, Dict[str, Any]] = {}
    notes: List[str] = []
    if not isinstance(payload, Mapping):
        return items, notes

    raw_items = payload.get("items")
    if isinstance(raw_items, list):
        for entry in raw_items:
            if not isinstance(entry, Mapping):
                continue
            pid = entry.get("id")
            if pid is None:
                continue
            items[str(pid)] = {
                "review_count": entry.get("review_count"),
                "image_count": entry.get("image_count"),
            }
    else:
        for key, value in payload.items():
            if key == "notes":
                continue
            if not isinstance(key, str) or not isinstance(value, Mapping):
                continue
            items[key] = {
                "review_count": value.get("review_count"),
                "image_count": value.get("image_count"),
            }

    notes_field = payload.get("notes")
    if isinstance(notes_field, Mapping):
        for key, message in notes_field.items():
            msg = str(message).strip()
            if not msg:
                continue
            if key not in (None, ""):
                notes.append(f"{key}: {msg}")
            else:
                notes.append(msg)
    elif isinstance(notes_field, list):
        notes.extend(str(note).strip() for note in notes_field if str(note).strip())
    elif isinstance(notes_field, str) and notes_field.strip():
        notes.append(notes_field.strip())

    return items, list(dict.fromkeys(notes))


def _apply_desire_payload(
    conn,
    products: Mapping[int, Dict[str, Any]],
    product_id: int,
    text: Optional[str],
    keywords: Optional[Sequence[str]] = None,
) -> Tuple[str, List[str]]:
    product = products.get(product_id)
    if product is None:
        return "", []
    normalized = _normalize_desire_text(text or "")
    stored_keywords = [str(kw).strip() for kw in (keywords or []) if str(kw).strip()]
    if normalized:
        database.update_product(
            conn,
            product_id,
            desire=normalized,
            ai_columns_completed_at=datetime.utcnow().isoformat(),
        )
        product["desire"] = normalized
    if stored_keywords:
        _update_extra_json(conn, product, product_id, {"desire_keywords": stored_keywords})
    return normalized, stored_keywords


def _coerce_non_negative_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        num = int(float(value))
    except Exception:
        return None
    return max(0, num)


def _apply_imputacion_payload(
    conn,
    products: Mapping[int, Dict[str, Any]],
    product_id: int,
    payload: Mapping[str, Any],
) -> None:
    product = products.get(product_id)
    if product is None:
        return
    updates: Dict[str, Any] = {}
    review = _coerce_non_negative_int(payload.get("review_count"))
    images = _coerce_non_negative_int(payload.get("image_count"))
    if review is not None:
        updates["review_count"] = review
    if images is not None:
        updates["image_count"] = images
    if updates:
        _update_extra_json(conn, product, product_id, updates)


def _weights_expired(cfg: Mapping[str, Any]) -> bool:
    if not isinstance(cfg, Mapping):
        return True
    ts = cfg.get("weightsUpdatedAt")
    if ts in (None, ""):
        return True
    try:
        updated = datetime.utcfromtimestamp(float(ts))
    except Exception:
        return True
    return datetime.utcnow() - updated > timedelta(days=14)


def _build_desire_tasks(
    products: Mapping[int, Dict[str, Any]],
    queue_index: Mapping[Tuple[str, int], int],
) -> Dict[str, Any]:
    pending: List[Dict[str, Any]] = []
    cache_hits: List[Dict[str, Any]] = []
    local: List[Dict[str, Any]] = []
    requested = 0
    for product_id, product in products.items():
        queue_id = queue_index.get(("desire", product_id))
        requested += 1
        existing = (product.get("desire") or "").strip()
        if existing:
            if _looks_like_desire_summary(existing):
                normalized = _normalize_desire_text(existing)
                local.append(
                    {
                        "product_id": product_id,
                        "text": normalized,
                        "keywords": product.get("_extra", {}).get("desire_keywords"),
                        "queue_id": queue_id,
                        "needs_update": normalized != existing,
                    }
                )
            else:
                local.append(
                    {
                        "product_id": product_id,
                        "text": existing,
                        "keywords": product.get("_extra", {}).get("desire_keywords"),
                        "queue_id": queue_id,
                        "needs_update": False,
                    }
                )
            continue
        payload = {
            "id": product_id,
            "title": product.get("name") or product.get("title") or product.get("_extra", {}).get("title"),
            "name": product.get("name") or product.get("title") or "",
            "description": product.get("description")
            or product.get("_extra", {}).get("description")
            or "",
            "existing_desire": existing,
        }
        cache_key = hash_key_for_item("desire", payload)
        cached = ai_cache_get("desire", cache_key)
        if cached and isinstance(cached.get("payload"), Mapping):
            cache_hits.append(
                {
                    "product_id": product_id,
                    "payload": cached["payload"],
                    "cache_key": cache_key,
                    "queue_id": queue_id,
                }
            )
        else:
            pending.append(
                {
                    "product_id": product_id,
                    "payload": payload,
                    "cache_key": cache_key,
                    "queue_id": queue_id,
                }
            )
    return {"pending": pending, "cache": cache_hits, "local": local, "requested": requested}


def run_post_import_auto(task_id: str, product_ids: Sequence[int]) -> Dict[str, Any]:
    """Execute post-import automation for the given products using GPTGuard."""

    task_id_str = str(task_id or "")
    product_list = _prepare_product_ids(product_ids)
    notes: List[str] = []
    errors: List[str] = []

    gpt_orchestrator.start_import(task_id_str)

    guard = GPTGuard(
        {
            "max_parallel": settings.AI_MAX_PARALLEL,
            "max_calls_per_import": settings.AI_MAX_CALLS_PER_IMPORT,
            "min_batch": settings.AI_MIN_BATCH_SIZE,
            "max_batch": settings.AI_MAX_BATCH_SIZE,
            "coalesce_ms": settings.AI_COALESCE_MS,
        }
    )

    api_key = config.get_api_key()
    model = config.get_model()
    imputacion_enabled = config.is_imputacion_via_ia_enabled()

    conn = database.get_connection(DB_PATH)
    database.initialize_database(conn)

    try:
        products = _load_products(conn, product_list)
        existing_ids = [pid for pid in product_list if pid in products]
        missing_ids = [pid for pid in product_list if pid not in products]
        queue_index = _load_queue_index(conn, task_id_str, product_list)
        queue_actions = _QueueActions(queue_index)

        if not products and product_list:
            for name in ("desire", "imputacion", "winner_score"):
                _set_requested(task_id_str, name, len(product_list))
                _increment_counts(task_id_str, name, failed=len(product_list))
            errors.append("no_products_found")
            queue_actions.flush(conn)
            conn.commit()
            gpt_orchestrator.flush_import_metrics(task_id_str)
            return {
                "task_id": task_id_str,
                "tasks": _status_snapshot(task_id_str),
                "errors": errors,
                "notes": notes,
                "product_ids": product_list,
            }

        # Desire processing
        desire_work = _build_desire_tasks(products, queue_index)
        gpt_orchestrator.record_cache_saved(task_id_str, len(desire_work["cache"]))
        _set_requested(task_id_str, "desire", desire_work["requested"] + len(missing_ids))
        for entry in desire_work["local"]:
            pid = int(entry["product_id"])
            text = entry.get("text")
            keywords = entry.get("keywords")
            if entry.get("needs_update") and text:
                _apply_desire_payload(conn, products, pid, text, keywords or [])
            _increment_counts(task_id_str, "desire", processed=1)
            queue_actions.complete("desire", pid)

        for entry in desire_work["cache"]:
            pid = int(entry["product_id"])
            payload = entry.get("payload") if isinstance(entry.get("payload"), Mapping) else {}
            text_value = payload.get("normalized_text")
            if not text_value:
                text_value = payload.get("text")
            text = _stringify_desire_text(text_value)
            keywords_raw = payload.get("keywords")
            if isinstance(keywords_raw, list):
                keywords = [str(kw).strip() for kw in keywords_raw if str(kw).strip()]
            else:
                keywords = []
            normalized, stored_keywords = _apply_desire_payload(conn, products, pid, text, keywords)
            cache_key = entry.get("cache_key")
            if cache_key:
                ai_cache_set(
                    "desire",
                    cache_key,
                    {"normalized_text": normalized, "keywords": stored_keywords},
                    f"{model}:desire" if model else "desire",
                )
            _increment_counts(task_id_str, "desire", processed=1)
            queue_actions.complete("desire", pid)

        pending_desire = desire_work["pending"]
        if pending_desire:
            if not api_key or not model:
                for item in pending_desire:
                    pid = int(item["product_id"])
                    _increment_counts(task_id_str, "desire", skipped=1)
                    queue_actions.skip("desire", pid, "openai_unavailable")
                errors.append("desire:openai_unavailable")
            else:
                summary = guard.submit(
                    "desire",
                    [item["payload"] for item in pending_desire],
                    lambda batch: gpt.orchestrate_desire_summary(api_key, model, batch),
                    import_id=task_id_str,
                )
                for note in summary.get("notes", []):
                    if note:
                        notes.append(str(note))
                pending_map = {str(item["payload"].get("id")): item for item in pending_desire}
                pending_map.update({str(item["product_id"]): item for item in pending_desire})
                skipped_ids: set[int] = set()
                for skipped in summary.get("skipped_items", []):
                    pid_raw = skipped.get("id")
                    try:
                        pid_int = int(pid_raw)
                    except Exception:
                        continue
                    skipped_ids.add(pid_int)
                    _increment_counts(task_id_str, "desire", skipped=1)
                    queue_actions.skip("desire", pid_int, "budget_exhausted")
                for outcome in summary.get("results", []):
                    batch_items = outcome.get("items") or []
                    if outcome.get("success"):
                        result_map, result_notes = _parse_desire_result_payload(outcome.get("result"))
                        for note in result_notes:
                            if note:
                                notes.append(str(note))
                        for batch_item in batch_items:
                            pid_raw = batch_item.get("id")
                            try:
                                pid_int = int(pid_raw)
                            except Exception:
                                continue
                            if pid_int in skipped_ids:
                                continue
                            meta = pending_map.get(str(pid_raw)) or pending_map.get(str(pid_int))
                            entry = None
                            if isinstance(result_map, Mapping):
                                entry = result_map.get(str(pid_raw)) or result_map.get(str(pid_int))
                            if isinstance(entry, Mapping):
                                text = _stringify_desire_text(entry.get("normalized_text") or entry.get("text"))
                                keywords_raw = entry.get("keywords")
                                if isinstance(keywords_raw, list):
                                    keywords = [str(kw).strip() for kw in keywords_raw if str(kw).strip()]
                                else:
                                    keywords = []
                                normalized, stored_keywords = _apply_desire_payload(conn, products, pid_int, text, keywords)
                                if meta and meta.get("cache_key"):
                                    ai_cache_set(
                                        "desire",
                                        meta["cache_key"],
                                        {"normalized_text": normalized, "keywords": stored_keywords},
                                        f"{model}:desire",
                                    )
                                _increment_counts(task_id_str, "desire", processed=1)
                                queue_actions.complete("desire", pid_int)
                            else:
                                _increment_counts(task_id_str, "desire", failed=1)
                                queue_actions.fail("desire", pid_int, "missing_result")
                                errors.append(f"desire:{pid_int}:missing_result")
                    else:
                        error_message = outcome.get("error") or "batch_failed"
                        for batch_item in batch_items:
                            pid_raw = batch_item.get("id")
                            try:
                                pid_int = int(pid_raw)
                            except Exception:
                                continue
                            if pid_int in skipped_ids:
                                continue
                    _increment_counts(task_id_str, "desire", failed=1)
                    queue_actions.fail("desire", pid_int, error_message)
                    errors.append(f"desire:{pid_int}:{error_message}")

        for pid in missing_ids:
            _increment_counts(task_id_str, "desire", failed=1)
            queue_actions.fail("desire", int(pid), "missing_product")

        # Imputacion processing
        imputacion_work = _build_imputacion_tasks(products, queue_index)
        gpt_orchestrator.record_cache_saved(task_id_str, len(imputacion_work["cache"]))
        _set_requested(task_id_str, "imputacion", imputacion_work["requested"] + len(missing_ids))
        if imputacion_enabled:
            for entry in imputacion_work["local"]:
                pid = int(entry["product_id"])
                _increment_counts(task_id_str, "imputacion", processed=1)
                queue_actions.complete("imputacion", pid)

            for entry in imputacion_work["cache"]:
                pid = int(entry["product_id"])
                payload = entry.get("payload") if isinstance(entry.get("payload"), Mapping) else {}
                _apply_imputacion_payload(conn, products, pid, payload)
                cache_key = entry.get("cache_key")
                if cache_key:
                    ai_cache_set(
                        "imputacion",
                        cache_key,
                        payload,
                        f"{model}:imputacion" if model else "imputacion",
                    )
                _increment_counts(task_id_str, "imputacion", processed=1)
                queue_actions.complete("imputacion", pid)

            pending_imputacion = imputacion_work["pending"]
            if pending_imputacion:
                if not api_key or not model:
                    for item in pending_imputacion:
                        pid = int(item["product_id"])
                        _increment_counts(task_id_str, "imputacion", skipped=1)
                        queue_actions.skip("imputacion", pid, "openai_unavailable")
                    errors.append("imputacion:openai_unavailable")
                else:
                    summary = guard.submit(
                        "imputacion",
                        [item["payload"] for item in pending_imputacion],
                        lambda batch: gpt.orchestrate_imputation(api_key, model, batch),
                        import_id=task_id_str,
                    )
                    for note in summary.get("notes", []):
                        if note:
                            notes.append(str(note))
                    pending_map = {str(item["payload"].get("id")): item for item in pending_imputacion}
                    pending_map.update({str(item["product_id"]): item for item in pending_imputacion})
                    skipped_ids: set[int] = set()
                    for skipped in summary.get("skipped_items", []):
                        pid_raw = skipped.get("id")
                        try:
                            pid_int = int(pid_raw)
                        except Exception:
                            continue
                        skipped_ids.add(pid_int)
                        _increment_counts(task_id_str, "imputacion", skipped=1)
                        queue_actions.skip("imputacion", pid_int, "budget_exhausted")
                    for outcome in summary.get("results", []):
                        batch_items = outcome.get("items") or []
                        if outcome.get("success"):
                            result_map, result_notes = _parse_imputacion_result_payload(outcome.get("result"))
                            for note in result_notes:
                                if note:
                                    notes.append(str(note))
                            for batch_item in batch_items:
                                pid_raw = batch_item.get("id")
                                try:
                                    pid_int = int(pid_raw)
                                except Exception:
                                    continue
                                if pid_int in skipped_ids:
                                    continue
                                entry = None
                                if isinstance(result_map, Mapping):
                                    entry = result_map.get(str(pid_raw)) or result_map.get(str(pid_int))
                                if isinstance(entry, Mapping):
                                    _apply_imputacion_payload(conn, products, pid_int, entry)
                                    meta = pending_map.get(str(pid_raw)) or pending_map.get(str(pid_int))
                                    if meta and meta.get("cache_key"):
                                        ai_cache_set(
                                            "imputacion",
                                            meta["cache_key"],
                                            entry,
                                            f"{model}:imputacion",
                                        )
                                    _increment_counts(task_id_str, "imputacion", processed=1)
                                    queue_actions.complete("imputacion", pid_int)
                                else:
                                    _increment_counts(task_id_str, "imputacion", failed=1)
                                    queue_actions.fail("imputacion", pid_int, "missing_result")
                                    errors.append(f"imputacion:{pid_int}:missing_result")
                        else:
                            error_message = outcome.get("error") or "batch_failed"
                            for batch_item in batch_items:
                                pid_raw = batch_item.get("id")
                                try:
                                    pid_int = int(pid_raw)
                                except Exception:
                                    continue
                                if pid_int in skipped_ids:
                                    continue
                            _increment_counts(task_id_str, "imputacion", failed=1)
                            queue_actions.fail("imputacion", pid_int, error_message)
                            errors.append(f"imputacion:{pid_int}:{error_message}")
        else:
            for pid in existing_ids:
                _increment_counts(task_id_str, "imputacion", skipped=1)
                queue_actions.skip("imputacion", pid, "config_disabled")

        for pid in missing_ids:
            _increment_counts(task_id_str, "imputacion", failed=1)
            queue_actions.fail("imputacion", int(pid), "missing_product")

        # Winner score processing
        _set_requested(task_id_str, "winner_score", len(product_list))
        cfg = config.load_config()
        weights = cfg.get("winner_weights") if isinstance(cfg.get("winner_weights"), Mapping) else None
        if not weights or _weights_expired(cfg):
            if api_key and model:
                try:
                    aggregates = aggregates_service.compute_dataset_aggregates(conn, scope_ids=product_list)
                    suggestion = gpt.recommend_weights_from_aggregates(api_key, model, aggregates)
                    weights_in = suggestion.get("weights") if isinstance(suggestion, Mapping) else None
                    order_in = suggestion.get("order") if isinstance(suggestion, Mapping) else None
                    if weights_in:
                        winner_config.update_winner_settings(weights_in=weights_in, order_in=order_in)
                        notes.append("winner_weights_refreshed")
                        cfg = config.load_config()
                    else:
                        errors.append("winner_score:weights_missing")
                except Exception as exc:
                    errors.append(f"winner_score:weights:{exc}")
            else:
                errors.append("winner_score:weights_openai_unavailable")

        weights_raw = winner_config.get_winner_weights_raw()
        try:
            result = winner_score.generate_winner_scores(conn, product_ids=existing_ids, weights=weights_raw)
            processed = int(result.get("processed", 0) or 0)
            if processed:
                _increment_counts(task_id_str, "winner_score", processed=processed)
            remaining = max(0, len(existing_ids) - processed)
            if remaining:
                _increment_counts(task_id_str, "winner_score", failed=remaining)
            for pid in existing_ids:
                queue_actions.complete("winner_score", pid)
        except Exception as exc:
            errors.append(f"winner_score:{exc}")
            for pid in existing_ids:
                _increment_counts(task_id_str, "winner_score", failed=1)
                queue_actions.fail("winner_score", pid, "winner_score_failed")

        for pid in missing_ids:
            _increment_counts(task_id_str, "winner_score", failed=1)
            queue_actions.fail("winner_score", int(pid), "missing_product")

        queue_actions.flush(conn)
        conn.commit()

        return {
            "task_id": task_id_str,
            "tasks": _status_snapshot(task_id_str),
            "errors": list(dict.fromkeys(errors)),
            "notes": list(dict.fromkeys(notes)),
            "product_ids": product_list,
        }
    finally:
        gpt_orchestrator.flush_import_metrics(task_id_str)
        try:
            conn.close()
        except Exception:  # pragma: no cover - defensive
            pass


def _build_imputacion_tasks(
    products: Mapping[int, Dict[str, Any]],
    queue_index: Mapping[Tuple[str, int], int],
) -> Dict[str, Any]:
    pending: List[Dict[str, Any]] = []
    cache_hits: List[Dict[str, Any]] = []
    local: List[Dict[str, Any]] = []
    requested = 0
    for product_id, product in products.items():
        queue_id = queue_index.get(("imputacion", product_id))
        requested += 1
        extra = product.get("_extra") if isinstance(product.get("_extra"), dict) else {}
        if extra and (extra.get("review_count") is not None or extra.get("image_count") is not None):
            local.append(
                {
                    "product_id": product_id,
                    "queue_id": queue_id,
                }
            )
            continue
        payload = {
            "id": product_id,
            "title": product.get("name") or product.get("title") or extra.get("title"),
            "description": product.get("description") or extra.get("description") or "",
            "category": product.get("category") or extra.get("category") or "",
        }
        cache_key = hash_key_for_item("imputacion", payload)
        cached = ai_cache_get("imputacion", cache_key)
        if cached and isinstance(cached.get("payload"), Mapping):
            cache_hits.append(
                {
                    "product_id": product_id,
                    "payload": cached["payload"],
                    "cache_key": cache_key,
                    "queue_id": queue_id,
                }
            )
        else:
            pending.append(
                {
                    "product_id": product_id,
                    "payload": payload,
                    "cache_key": cache_key,
                    "queue_id": queue_id,
                }
            )
    return {"pending": pending, "cache": cache_hits, "local": local, "requested": requested}
