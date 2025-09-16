from __future__ import annotations

import csv
import io
import logging
import threading
import time
from itertools import islice
from typing import Any, Dict, Iterable, Iterator

from flask import Flask, request

from product_research_app.db import get_db
from product_research_app.services.importer_fast import fast_import_adaptive
from product_research_app.utils.timing import phase


app = Flask(__name__)


IMPORT_STATUS: Dict[str, Dict[str, Any]] = {}
_IMPORT_LOCK = threading.Lock()


logger = logging.getLogger(__name__)


_MAX_IDS_FOR_DEDUPE = 200_000


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


def _enqueue_post_import_tasks(task_id: str, rows_imported: int) -> int:
    """Placeholder for post-import asynchronous work."""

    logger.debug(
        "import_job[%s] post_import_queue noop rows_imported=%s",
        task_id,
        rows_imported,
    )
    return 0


def _round_ms(delta: float) -> int:
    return max(int(round(delta * 1000)), 0)


def _baseline_status(task_id: str) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "state": "queued",
        "status": "queued",
        "stage": "queued",
        "done": 0,
        "total": 0,
        "imported": 0,
        "error": None,
        "optimizing": False,
        "t_parse": 0,
        "t_staging": 0,
        "t_upsert": 0,
        "t_commit": 0,
        "t_optimize": 0,
        "file_size_bytes": 0,
        "row_count": 0,
        "column_count": 0,
        "total_ms": 0,
        "phases": [],
    }


def _update_status(task_id: str, **updates: Any) -> Dict[str, Any]:
    with _IMPORT_LOCK:
        status = IMPORT_STATUS.setdefault(task_id, _baseline_status(task_id))
        if "state" in updates and "status" not in updates:
            updates["status"] = updates["state"]

        if "done" in updates:
            try:
                updates["done"] = max(int(updates["done"]), int(status.get("done", 0) or 0))
            except Exception:
                updates.pop("done", None)
        if "total" in updates:
            try:
                updates["total"] = max(
                    int(updates["total"]), int(status.get("total", 0) or 0)
                )
            except Exception:
                updates.pop("total", None)
        if "imported" in updates:
            try:
                updates["imported"] = max(
                    int(updates["imported"]), int(status.get("imported", 0) or 0)
                )
            except Exception:
                updates.pop("imported", None)

        for key in ("t_parse", "t_staging", "t_upsert", "t_commit", "t_optimize"):
            if key in updates:
                try:
                    updates[key] = int(updates[key])
                except Exception:
                    updates.pop(key, None)

        for key in ("row_count", "column_count", "file_size_bytes", "total_ms"):
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

        status.update(updates)
        if status.get("total", 0) < status.get("done", 0):
            status["total"] = status.get("done", 0)
        return dict(status)


def _get_status(task_id: str) -> Dict[str, Any] | None:
    with _IMPORT_LOCK:
        data = IMPORT_STATUS.get(task_id)
        return dict(data) if data else None


@app.post("/upload")
def upload():
    file = request.files.get("file")
    if file is None:
        return {"error": "missing_file"}, 400

    task_id = str(int(time.time() * 1000))
    _update_status(task_id, filename=file.filename or None)

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
        _update_status(task_id, state="running", stage="running")
        row_count_source = 0
        column_count = 0
        existing_ids_count = 0
        rows_imported = 0
        id_candidates: set[int] = set()
        optimize = None

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
                    optimize = fast_import_adaptive(csv_bytes, status_cb=cb)
            finally:
                if db_phase is not None:
                    record_phase(db_phase)

            rows_imported = int(getattr(optimize, "rows_imported", 0) or 0)
            snapshot = _get_status(task_id) or {}
            done_val = max(int(snapshot.get("done", 0) or 0), rows_imported)
            total_val = max(int(snapshot.get("total", 0) or 0), done_val)
            _update_status(
                task_id,
                done=done_val,
                total=total_val,
                imported=rows_imported,
                row_count=row_count_source,
                column_count=column_count,
            )
            _update_status(task_id, state="done")

            post_phase: Dict[str, Any] | None = None
            try:
                with phase("post_import_queue") as ph:
                    post_phase = ph
                    _enqueue_post_import_tasks(task_id, rows_imported)
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
            _update_status(task_id, state="error", error=str(exc))
        finally:
            total_elapsed_ms = int(round((time.perf_counter() - total_start) * 1000))
            _update_status(
                task_id,
                total_ms=total_elapsed_ms,
                file_size_bytes=file_size,
                row_count=row_count_source,
                column_count=column_count,
                phases=[dict(item) for item in phase_records],
            )
            logger.info(
                "import_job[%s] summary rows=%d columns=%d file_size=%dB existing_ids=%d imported=%d total_ms=%d",
                task_id,
                row_count_source,
                column_count,
                file_size,
                existing_ids_count,
                rows_imported,
                total_elapsed_ms,
            )

    threading.Thread(target=run, daemon=True).start()
    return {"task_id": task_id}, 202


@app.get("/_import_status")
def import_status():
    task_id = request.args.get("task_id", "")
    if not task_id:
        return {
            "task_id": "",
            "state": "unknown",
            "status": "unknown",
            "done": 0,
            "total": 0,
            "imported": 0,
            "error": None,
            "optimizing": False,
            "file_size_bytes": 0,
            "row_count": 0,
            "column_count": 0,
            "total_ms": 0,
            "phases": [],
        }, 200

    status = _get_status(task_id)
    if status is None:
        return {
            "task_id": task_id,
            "state": "unknown",
            "status": "unknown",
            "done": 0,
            "total": 0,
            "imported": 0,
            "error": None,
            "optimizing": False,
            "file_size_bytes": 0,
            "row_count": 0,
            "column_count": 0,
            "total_ms": 0,
            "phases": [],
        }, 200

    status.setdefault("task_id", task_id)
    status.setdefault("status", status.get("state"))
    return status


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, threaded=True, use_reloader=False)
