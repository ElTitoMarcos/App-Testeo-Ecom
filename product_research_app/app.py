from __future__ import annotations

import csv
import io
import logging
import threading
import time
from itertools import islice
from typing import Any, Dict, Iterable, Iterator, List

from flask import Flask, request

from product_research_app.ai.queue import dequeue_batch, enqueue_post_import
from product_research_app.db import get_db
from product_research_app.services import ai_columns
from product_research_app.services.importer_fast import fast_import_adaptive
from product_research_app.utils.timing import phase


app = Flask(__name__)


IMPORT_STATUS: Dict[str, Dict[str, Any]] = {}
_IMPORT_LOCK = threading.Lock()


logger = logging.getLogger(__name__)


_MAX_IDS_FOR_DEDUPE = 200_000

_POST_IMPORT_TASK_ALIASES = {
    "desire": "desire_summarize",
    "desire_summarize": "desire_summarize",
    "imputacion": "imputacion_campos",
    "imputacion_campos": "imputacion_campos",
}

_POST_IMPORT_CANONICAL_TO_ALIAS = {
    "desire_summarize": "desire",
    "imputacion_campos": "imputacion",
}

_STATE_ALIASES = {
    "": "PENDING",
    "PENDING": "PENDING",
    "QUEUED": "PENDING",
    "RUNNING": "RUNNING",
    "IN_PROGRESS": "RUNNING",
    "DONE": "DONE",
    "SUCCESS": "DONE",
    "COMPLETED": "DONE",
    "ERROR": "ERROR",
    "FAILED": "ERROR",
}

def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}


def _canonical_task_name(value: Any) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    return _POST_IMPORT_TASK_ALIASES.get(key)


def _normalize_post_import_tasks(values: Iterable[Any]) -> list[str]:
    seen: list[str] = []
    for value in values:
        canonical = _canonical_task_name(value)
        if canonical and canonical not in seen:
            seen.append(canonical)
    return seen


def _normalize_product_ids(values: Iterable[Any]) -> list[int]:
    seen: set[int] = set()
    normalized: list[int] = []
    for value in values:
        try:
            num = int(value)
        except Exception:
            continue
        if num <= 0 or num in seen:
            continue
        seen.add(num)
        normalized.append(num)
    return normalized

def _chunked(iterable: Iterable[int], size: int) -> Iterator[list[int]]:
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            break
        yield chunk


def _analyze_csv_bytes(csv_bytes: bytes) -> tuple[int, int, set[int]]:
    column_count = 0
    row_count = 0
    id_candidates: set[int] = set()

    if not csv_bytes:
        return column_count, row_count, id_candidates

    text_stream = io.StringIO(csv_bytes.decode("utf-8", errors="ignore"))
    reader = csv.DictReader(text_stream)
    if reader.fieldnames:
        column_count = len(reader.fieldnames)

    for row in reader:
        row_count += 1
        raw_id = row.get("id") or row.get("ID")
        if raw_id in (None, ""):
            continue
        if len(id_candidates) >= _MAX_IDS_FOR_DEDUPE:
            continue
        try:
            candidate = int(str(raw_id).strip())
        except Exception:
            continue
        id_candidates.add(candidate)

    return column_count, row_count, id_candidates


def _count_existing_ids(candidates: set[int]) -> int:
    if not candidates:
        return 0

    db = get_db()
    total = 0
    for chunk in _chunked(sorted(candidates), 900):
        placeholders = ",".join("?" for _ in chunk)
        query = f"SELECT COUNT(*) FROM products WHERE id IN ({placeholders})"
        try:
            row = db.execute(query, tuple(chunk)).fetchone()
        except Exception:
            continue
        if row and row[0] is not None:
            total += int(row[0])
    return total

def _enqueue_post_import_tasks(
    task_id: str, product_ids: Iterable[int], task_types: Iterable[str]
) -> Dict[str, int]:
    ids = _normalize_product_ids(product_ids)
    tasks = _normalize_post_import_tasks(task_types)
    if not ids or not tasks:
        logger.debug(
            "import_job[%s] post_import_queue skipped ids=%s tasks=%s",
            task_id,
            len(ids),
            tasks,
        )
        return {}

    counts: Dict[str, int] = {}
    for task_type in tasks:
        inserted = enqueue_post_import(task_type, ids)
        counts[task_type] = inserted
        logger.info(
            "import_job[%s] post_import_queue task=%s inserted=%s total_ids=%s",
            task_id,
            _POST_IMPORT_CANONICAL_TO_ALIAS.get(task_type, task_type),
            inserted,
            len(ids),
        )
    return counts

def _round_ms(delta: float) -> int:
    return max(int(round(delta * 1000)), 0)


def _baseline_status(task_id: str) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "state": "PENDING",
        "processed": 0,
        "total": 0,
        "error": None,
        "eta_ms": 0,
        "phases": [],
        "file_size_bytes": 0,
        "row_count": 0,
        "column_count": 0,
        "total_ms": 0,
        "post_import_ready": False,
        "post_import_tasks": [],
        "started_at": None,
        "finished_at": None,
        "last_progress_at": None,
        "_started_monotonic": None,
    }


def _normalize_state_name(value: Any) -> str:
    key = str(value or "").strip().upper()
    return _STATE_ALIASES.get(key, key if key in _STATE_ALIASES.values() else "PENDING")


def _compute_eta_ms(status: Dict[str, Any]) -> int:
    state = status.get("state")
    if state != "RUNNING":
        return 0
    try:
        total = int(status.get("total", 0) or 0)
        processed = int(status.get("processed", 0) or 0)
    except Exception:
        return 0
    if total <= 0 or processed <= 0 or processed >= total:
        return 0
    start = status.get("_started_monotonic")
    if not start:
        return 0
    elapsed = max(time.perf_counter() - float(start), 0.0)
    if elapsed <= 0:
        return 0
    remaining = max(total - processed, 0)
    if remaining <= 0:
        return 0
    estimate = (elapsed / processed) * remaining
    if estimate <= 0:
        return 0
    try:
        return max(int(round(estimate * 1000)), 0)
    except Exception:
        return 0


def _update_status(task_id: str, **updates: Any) -> Dict[str, Any]:
    now = time.time()
    with _IMPORT_LOCK:
        status = IMPORT_STATUS.setdefault(task_id, _baseline_status(task_id))
        status["task_id"] = task_id

        if "state" in updates:
            updates["state"] = _normalize_state_name(updates["state"])

        if "done" in updates and "processed" not in updates:
            updates["processed"] = updates.pop("done")
        else:
            updates.pop("done", None)

        if "imported" in updates and "processed" not in updates:
            updates["processed"] = updates.pop("imported")
        else:
            updates.pop("imported", None)

        updates.pop("status", None)
        updates.pop("stage", None)

        if "total" in updates:
            try:
                updates["total"] = max(
                    int(updates["total"]), int(status.get("total", 0) or 0)
                )
            except Exception:
                updates.pop("total", None)

        if "processed" in updates:
            try:
                updates["processed"] = max(
                    int(updates["processed"]), int(status.get("processed", 0) or 0)
                )
            except Exception:
                updates.pop("processed", None)
            else:
                updates.setdefault("last_progress_at", now)

        if "optimizing" in updates:
            updates["optimizing"] = bool(updates["optimizing"])

        for key in ("row_count", "column_count", "file_size_bytes", "total_ms", "t_optimize"):
            if key in updates:
                try:
                    updates[key] = int(updates[key])
                except Exception:
                    updates.pop(key, None)

        if "phases" in updates:
            try:
                normalized = []
                for item in updates["phases"] or []:
                    if isinstance(item, dict):
                        name = str(item.get("name", ""))
                        ms_val = item.get("ms", 0)
                    elif isinstance(item, (list, tuple)) and item:
                        name = str(item[0])
                        ms_val = item[1] if len(item) > 1 else 0
                    else:
                        continue
                    try:
                        ms_int = int(ms_val)
                    except Exception:
                        continue
                    normalized.append({"name": name, "ms": ms_int})
                updates["phases"] = normalized
            except Exception:
                updates.pop("phases", None)

        if "post_import_ready" in updates:
            updates["post_import_ready"] = bool(updates["post_import_ready"])

        if "post_import_tasks" in updates:
            try:
                updates["post_import_tasks"] = _normalize_post_import_tasks(
                    updates["post_import_tasks"] or []
                )
            except Exception:
                updates["post_import_tasks"] = []

        new_state = updates.get("state") or status.get("state", "PENDING")
        if new_state == "RUNNING" and not status.get("_started_monotonic"):
            status["_started_monotonic"] = time.perf_counter()
        if new_state == "RUNNING" and not status.get("started_at"):
            status["started_at"] = now
        if new_state in {"DONE", "ERROR"} and not updates.get("finished_at"):
            updates["finished_at"] = now

        if new_state == "RUNNING":
            updates.setdefault("started_at", status.get("started_at") or now)
            updates["_started_monotonic"] = status.get("_started_monotonic") or time.perf_counter()
        elif new_state in {"DONE", "ERROR"}:
            updates.setdefault("_started_monotonic", status.get("_started_monotonic"))
            updates.setdefault("eta_ms", 0)
        status.update(updates)

        if status.get("total", 0) < status.get("processed", 0):
            status["total"] = status.get("processed", 0)

        state_after = status.get("state", "PENDING")
        if state_after == "RUNNING" and not status.get("_started_monotonic"):
            status["_started_monotonic"] = time.perf_counter()
        status["eta_ms"] = _compute_eta_ms(status)

        return {k: v for k, v in status.items() if not str(k).startswith("_")}


def _get_status(task_id: str) -> Dict[str, Any] | None:
    with _IMPORT_LOCK:
        data = IMPORT_STATUS.get(task_id)
        if not data:
            return None
        return {k: v for k, v in data.items() if not str(k).startswith("_")}


@app.post("/upload")
def upload():
    file = request.files.get("file")
    if file is None:
        return {"error": "missing_file"}, 400

    task_id = str(int(time.time() * 1000))
    raw_tasks = request.form.getlist("post_import_tasks")
    if not raw_tasks:
        if _truthy(request.form.get("post_import_desire")):
            raw_tasks.append("desire")
        if _truthy(request.form.get("post_import_imputacion")):
            raw_tasks.append("imputacion")
    post_import_tasks = tuple(_normalize_post_import_tasks(raw_tasks))

    _update_status(
        task_id,
        filename=file.filename or None,
        post_import_tasks=list(post_import_tasks),
        post_import_ready=False,
    )

    phase_records: list[Dict[str, int]] = []

    def record_phase(info: Dict[str, Any]) -> None:
        name = str(info.get("name", ""))
        try:
            ms_val = int(info.get("ms", 0))
        except Exception:
            ms_val = 0
        phase_records.append({"name": name, "ms": ms_val})
        _update_status(task_id, phases=[dict(item) for item in phase_records])

    total_start = time.perf_counter()
    csv_bytes = b""
    read_phase: Dict[str, Any] | None = None
    try:
        with phase("read_file") as ph:
            read_phase = ph
            csv_bytes = file.read()
    finally:
        if read_phase is not None:
            record_phase(read_phase)

    file_size = len(csv_bytes or b"")
    _update_status(task_id, file_size_bytes=file_size)

    def run():
        _update_status(task_id, state="RUNNING")
        row_count_source = 0
        column_count = 0
        existing_ids_count = 0
        rows_imported = 0
        id_candidates: set[int] = set()
        optimize = None
        product_ids: List[int] = []
        post_ready = False
        post_counts: Dict[str, int] = {}

        def cb(**payload):
            _update_status(task_id, **payload)

        try:
            parse_phase: Dict[str, Any] | None = None
            try:
                with phase("parse_csv") as ph:
                    parse_phase = ph
                    column_count, row_count_source, id_candidates = _analyze_csv_bytes(csv_bytes)
            finally:
                if parse_phase is not None:
                    record_phase(parse_phase)
            _update_status(
                task_id,
                row_count=row_count_source,
                column_count=column_count,
            )
            logger.info(
                "import_job[%s] parse_csv rows=%d columns=%d id_candidates=%d",
                task_id,
                row_count_source,
                column_count,
                len(id_candidates),
            )

            dedupe_phase: Dict[str, Any] | None = None
            try:
                with phase("dedupe_prepare") as ph:
                    dedupe_phase = ph
                    existing_ids_count = _count_existing_ids(id_candidates)
            finally:
                if dedupe_phase is not None:
                    record_phase(dedupe_phase)
            logger.info(
                "import_job[%s] dedupe_prepare existing_ids=%d",
                task_id,
                existing_ids_count,
            )
            id_candidates.clear()

            db_phase: Dict[str, Any] | None = None
            try:
                with phase("db_bulk_insert") as ph:
                    db_phase = ph
                    optimize = fast_import_adaptive(
                        csv_bytes,
                        status_cb=cb,
                        phase_recorder=record_phase,
                    )
            finally:
                if db_phase is not None:
                    record_phase(db_phase)

            rows_imported = int(getattr(optimize, "rows_imported", 0) or 0)
            product_ids = _normalize_product_ids(
                getattr(optimize, "product_ids", [])
            )
            snapshot = _get_status(task_id) or {}
            prev_processed = int(snapshot.get("processed", 0) or 0)
            prev_total = int(snapshot.get("total", 0) or 0)
            processed_val = max(prev_processed, rows_imported)
            expected_total = row_count_source or processed_val
            total_val = max(prev_total, expected_total, processed_val)
            _update_status(
                task_id,
                processed=processed_val,
                total=total_val,
                row_count=row_count_source,
                column_count=column_count,
            )
            _update_status(task_id, state="DONE")

            post_phase: Dict[str, Any] | None = None
            try:
                with phase("post_import_queue") as ph:
                    post_phase = ph
                    post_counts = _enqueue_post_import_tasks(
                        task_id, product_ids, post_import_tasks
                    )
                    post_ready = bool(product_ids and post_import_tasks)
                    _update_status(
                        task_id,
                        post_import_ready=post_ready,
                        post_import_tasks=list(post_import_tasks),
                    )
            finally:
                if post_phase is not None:
                    record_phase(post_phase)

            post_phase: Dict[str, Any] | None = None
            try:
                with phase("post_import_queue") as ph:
                    post_phase = ph
                    post_counts = _enqueue_post_import_tasks(
                        task_id, product_ids, post_import_tasks
                    )
                    post_ready = bool(product_ids and post_import_tasks)
                    _update_status(
                        task_id,
                        post_import_ready=post_ready,
                        post_import_tasks=list(post_import_tasks),
                    )
            finally:
                if post_phase is not None:
                    record_phase(post_phase)

            def do_opt():
                t0 = time.time()
                try:
                    _update_status(task_id, optimizing=True)
                    if callable(optimize):
                        optimize()
                except Exception as exc:
                    _update_status(
                        task_id,
                        optimizing=False,
                        t_optimize=_round_ms(time.time() - t0),
                        error=str(exc),
                    )
                else:
                    _update_status(
                        task_id,
                        optimizing=False,
                        t_optimize=_round_ms(time.time() - t0),
                    )

            if callable(optimize):
                threading.Thread(target=do_opt, daemon=True).start()

        except Exception as exc:
            logger.exception("import_job[%s] failed", task_id)
            _update_status(task_id, state="ERROR", error=str(exc))
        finally:
            total_elapsed_ms = int(round((time.perf_counter() - total_start) * 1000))
            _update_status(
                task_id,
                total_ms=total_elapsed_ms,
                file_size_bytes=file_size,
                row_count=row_count_source,
                column_count=column_count,
                phases=[dict(item) for item in phase_records],
                post_import_ready=post_ready,
                post_import_tasks=list(post_import_tasks),
            )
            logger.info(
                "import_job[%s] summary rows=%d columns=%d file_size=%dB existing_ids=%d imported=%d total_ms=%d post_tasks=%s post_counts=%s",
                task_id,
                row_count_source,
                column_count,
                file_size,
                existing_ids_count,
                rows_imported,
                total_elapsed_ms,
                list(post_import_tasks),
                post_counts,
            )

    threading.Thread(target=run, daemon=True).start()
    return {"task_id": task_id}, 202


@app.get("/_import_status")
def import_status():
    task_id = request.args.get("task_id", "")
    if not task_id:
        return {
            "task_id": "",
            "state": "PENDING",
            "processed": 0,
            "total": 0,
            "phases": [],
            "eta_ms": 0,
            "error": None,
            "post_import_ready": False,
            "post_import_tasks": [],
        }, 200

    status = _get_status(task_id)
    if status is None:
        return {
            "task_id": task_id,
            "state": "PENDING",
            "processed": 0,
            "total": 0,
            "phases": [],
            "eta_ms": 0,
            "error": None,
            "post_import_ready": False,
            "post_import_tasks": [],
        }, 200

    response: Dict[str, Any] = {
        "task_id": task_id,
        "state": status.get("state", "PENDING"),
        "processed": int(status.get("processed", 0) or 0),
        "total": int(status.get("total", 0) or 0),
        "phases": status.get("phases", []),
        "eta_ms": int(status.get("eta_ms", 0) or 0),
        "error": status.get("error"),
        "post_import_ready": bool(status.get("post_import_ready")),
        "post_import_tasks": status.get("post_import_tasks", []),
    }

    for numeric_key in ("file_size_bytes", "row_count", "column_count", "total_ms"):
        if numeric_key in status and status[numeric_key] is not None:
            try:
                response[numeric_key] = int(status[numeric_key])
            except Exception:
                response[numeric_key] = 0

    for optional_key in ("filename", "started_at", "finished_at"):
        if optional_key in status and status[optional_key] is not None:
            response[optional_key] = status[optional_key]

    return response


@app.post("/api/ai/run_post_import")
def run_post_import_tasks():
    payload = request.get_json(silent=True) or {}
    raw_tasks = payload.get("tasks") or []
    if not isinstance(raw_tasks, (list, tuple, set)):
        raw_tasks = [raw_tasks]
    normalized = _normalize_post_import_tasks(raw_tasks)
    if not normalized:
        return {"error": "invalid_tasks"}, 400

    try:
        limit_val = int(payload.get("limit", 200))
    except Exception:
        limit_val = 200
    limit_val = max(1, min(limit_val, 1000))

    alias_map = {
        task_type: _POST_IMPORT_CANONICAL_TO_ALIAS.get(task_type, task_type)
        for task_type in normalized
    }

    drained: Dict[str, List[int]] = {}
    drained_counts: Dict[str, int] = {}
    for task_type in normalized:
        ids = dequeue_batch(task_type, limit_val)
        drained[task_type] = ids
        drained_counts[task_type] = len(ids)

    union_ids = sorted({pid for ids in drained.values() for pid in ids})
    ai_result: Dict[str, Any] = {}
    pending_ids: list[int] = []

    if union_ids:
        try:
            ai_result = ai_columns.fill_ai_columns(union_ids)
        except Exception as exc:
            for task_type, ids in drained.items():
                if ids:
                    enqueue_post_import(task_type, ids)
            logger.exception(
                "run_post_import failed tasks=%s limit=%s", normalized, limit_val
            )
            return {"error": str(exc)}, 500

        pending_ids = _normalize_product_ids(ai_result.get("pending_ids") or [])
        if pending_ids:
            for task_type in normalized:
                enqueue_post_import(task_type, pending_ids)
    else:
        ai_result = {
            "ok": {},
            "ko": {},
            "counts": {
                "n_importados": 0,
                "n_para_ia": 0,
                "n_procesados": 0,
                "n_omitidos_por_valor_existente": 0,
                "n_reintentados": 0,
                "n_error_definitivo": 0,
                "truncated": False,
                "cost_estimated_usd": 0.0,
            },
            "pending_ids": [],
        }

    processed_ids = set(
        _normalize_product_ids((ai_result.get("ok") or {}).keys())
    )
    failed_ids = set(
        _normalize_product_ids((ai_result.get("ko") or {}).keys())
    )

    results: Dict[str, Dict[str, int]] = {}
    for task_type in normalized:
        alias = alias_map[task_type]
        ids = drained.get(task_type, [])
        processed = sum(1 for pid in ids if pid in processed_ids)
        failed = sum(1 for pid in ids if pid in failed_ids and pid not in processed_ids)
        pending = max(len(ids) - processed - failed, 0)
        results[alias] = {
            "requested": len(ids),
            "processed": processed,
            "failed": failed,
            "pending": pending,
        }

    remaining: Dict[str, int] = {}
    has_more = False
    conn = get_db()
    for task_type in normalized:
        alias = alias_map[task_type]
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM ai_task_queue WHERE task_type = ?",
                (task_type,),
            ).fetchone()
            count = int(row[0]) if row and row[0] is not None else 0
        except Exception:
            count = 0
        remaining[alias] = count
        if count:
            has_more = True
    if pending_ids:
        has_more = True

    logger.info(
        "run_post_import tasks=%s limit=%s drained=%s processed=%s has_more=%s pending_ids=%s",
        normalized,
        limit_val,
        drained_counts,
        ai_result.get("counts", {}),
        has_more,
        len(pending_ids),
    )

    return {
        "ok": True,
        "results": results,
        "details": ai_result,
        "has_more": has_more,
        "remaining": remaining,
    }


@app.post("/api/ai/run_post_import")
def run_post_import_tasks():
    payload = request.get_json(silent=True) or {}
    raw_tasks = payload.get("tasks") or []
    if not isinstance(raw_tasks, (list, tuple, set)):
        raw_tasks = [raw_tasks]
    normalized = _normalize_post_import_tasks(raw_tasks)
    if not normalized:
        return {"error": "invalid_tasks"}, 400

    try:
        limit_val = int(payload.get("limit", 200))
    except Exception:
        limit_val = 200
    limit_val = max(1, min(limit_val, 1000))

    alias_map = {
        task_type: _POST_IMPORT_CANONICAL_TO_ALIAS.get(task_type, task_type)
        for task_type in normalized
    }

    drained: Dict[str, List[int]] = {}
    drained_counts: Dict[str, int] = {}
    for task_type in normalized:
        ids = dequeue_batch(task_type, limit_val)
        drained[task_type] = ids
        drained_counts[task_type] = len(ids)

    union_ids = sorted({pid for ids in drained.values() for pid in ids})
    ai_result: Dict[str, Any] = {}
    pending_ids: list[int] = []

    if union_ids:
        try:
            ai_result = ai_columns.fill_ai_columns(union_ids)
        except Exception as exc:
            for task_type, ids in drained.items():
                if ids:
                    enqueue_post_import(task_type, ids)
            logger.exception(
                "run_post_import failed tasks=%s limit=%s", normalized, limit_val
            )
            return {"error": str(exc)}, 500

        pending_ids = _normalize_product_ids(ai_result.get("pending_ids") or [])
        if pending_ids:
            for task_type in normalized:
                enqueue_post_import(task_type, pending_ids)
    else:
        ai_result = {
            "ok": {},
            "ko": {},
            "counts": {
                "n_importados": 0,
                "n_para_ia": 0,
                "n_procesados": 0,
                "n_omitidos_por_valor_existente": 0,
                "n_reintentados": 0,
                "n_error_definitivo": 0,
                "truncated": False,
                "cost_estimated_usd": 0.0,
            },
            "pending_ids": [],
        }

    processed_ids = set(
        _normalize_product_ids((ai_result.get("ok") or {}).keys())
    )
    failed_ids = set(
        _normalize_product_ids((ai_result.get("ko") or {}).keys())
    )

    results: Dict[str, Dict[str, int]] = {}
    for task_type in normalized:
        alias = alias_map[task_type]
        ids = drained.get(task_type, [])
        processed = sum(1 for pid in ids if pid in processed_ids)
        failed = sum(1 for pid in ids if pid in failed_ids and pid not in processed_ids)
        pending = max(len(ids) - processed - failed, 0)
        results[alias] = {
            "requested": len(ids),
            "processed": processed,
            "failed": failed,
            "pending": pending,
        }

    remaining: Dict[str, int] = {}
    has_more = False
    conn = get_db()
    for task_type in normalized:
        alias = alias_map[task_type]
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM ai_task_queue WHERE task_type = ?",
                (task_type,),
            ).fetchone()
            count = int(row[0]) if row and row[0] is not None else 0
        except Exception:
            count = 0
        remaining[alias] = count
        if count:
            has_more = True
    if pending_ids:
        has_more = True

    logger.info(
        "run_post_import tasks=%s limit=%s drained=%s processed=%s has_more=%s pending_ids=%s",
        normalized,
        limit_val,
        drained_counts,
        ai_result.get("counts", {}),
        has_more,
        len(pending_ids),
    )

    return {
        "ok": True,
        "results": results,
        "details": ai_result,
        "has_more": has_more,
        "remaining": remaining,
    }


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, threaded=True, use_reloader=False)
