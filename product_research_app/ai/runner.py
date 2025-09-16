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
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from .. import config, database, gpt
from ..services import winner_score
from ..utils.db import row_to_dict, rget

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


def run_auto(tasks: set[str], *, batch_size: int = 200, max_parallel: int = 3) -> Dict[str, Dict[str, object]]:
    """Drain ``ai_task_queue`` executing the requested task types."""

    ordered_tasks = [name for name in _TASK_ORDER if name in tasks]
    if not ordered_tasks:
        return {}

    try:
        batch_size = int(batch_size)
    except Exception:
        batch_size = 200
    batch_size = max(1, min(batch_size, 200))

    try:
        max_parallel = int(max_parallel)
    except Exception:
        max_parallel = 3
    max_parallel = max(1, min(max_parallel, 8))

    api_key = config.get_api_key() or None
    model = config.get_model() or "gpt-4o-mini"
    include_image = config.include_image_in_ai()

    context = _RunnerContext(api_key=api_key, model=model, include_image=include_image, max_attempts=_MAX_ATTEMPTS)

    progress: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: {name: {"requested": 0, "processed": 0, "failed": 0} for name in _TASK_ORDER})
    errors: Dict[str, List[str]] = defaultdict(list)
    seen_task_ids: set[int] = set()

    conn = database.get_connection(DB_PATH)
    database.initialize_database(conn)

    try:
        for task_type in ordered_tasks:
            while True:
                pending = database.fetch_pending_ai_tasks(
                    conn,
                    task_types=[task_type],
                    limit=batch_size * max_parallel,
                )
                if not pending:
                    break

                if task_type in _GPT_TASKS and (not context.api_key or not context.model):
                    logger.warning("AI runner skipping %s tasks due to missing API configuration", task_type)

                task_ids = [int(row["id"]) for row in pending]
                database.mark_ai_tasks_in_progress(conn, task_ids)

                batches: List[List[Mapping[str, object]]] = []
                current: List[Mapping[str, object]] = []
                for row in pending:
                    row_dict = dict(row)
                    import_id = str(row_dict.get("import_task_id") or "")
                    entry = progress[import_id][task_type]
                    task_id = int(row_dict["id"])
                    if task_id not in seen_task_ids:
                        seen_task_ids.add(task_id)
                        entry["requested"] += 1
                    current.append(row_dict)
                    if len(current) >= batch_size:
                        batches.append(current)
                        current = []
                if current:
                    batches.append(current)

                if not batches:
                    continue

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
            entry["requested"] or entry["processed"] or entry["failed"]
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
        try:
            winner_score.generate_winner_scores(conn, product_ids=product_ids)
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
