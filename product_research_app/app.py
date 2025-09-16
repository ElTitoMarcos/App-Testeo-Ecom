from __future__ import annotations

import threading
import time
from typing import Any, Dict

from flask import Flask, request

from product_research_app.services.importer_fast import fast_import_adaptive


app = Flask(__name__)


IMPORT_STATUS: Dict[str, Dict[str, Any]] = {}
_IMPORT_LOCK = threading.Lock()


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

    csv_bytes = file.read()
    task_id = str(int(time.time() * 1000))
    _update_status(task_id, filename=file.filename or None)

    def run():
        _update_status(task_id, state="running", stage="running")
        try:
            def cb(**payload):
                _update_status(task_id, **payload)

            optimize = fast_import_adaptive(csv_bytes, status_cb=cb)
            rows_imported = int(getattr(optimize, "rows_imported", 0) or 0)
            snapshot = _get_status(task_id) or {}
            done_val = max(int(snapshot.get("done", 0) or 0), rows_imported)
            total_val = max(int(snapshot.get("total", 0) or 0), done_val)
            _update_status(task_id, done=done_val, total=total_val, imported=rows_imported)
            _update_status(task_id, state="done")

            def do_opt():
                t0 = time.time()
                try:
                    _update_status(task_id, optimizing=True)
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

            threading.Thread(target=do_opt, daemon=True).start()

        except Exception as exc:
            _update_status(task_id, state="error", error=str(exc))

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
        }, 200

    status.setdefault("task_id", task_id)
    status.setdefault("status", status.get("state"))
    return status


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, threaded=True, use_reloader=False)
