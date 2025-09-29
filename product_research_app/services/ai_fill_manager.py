from __future__ import annotations

import math
import sqlite3
import threading
import time
import uuid
from collections import defaultdict, deque
from statistics import median
from typing import Deque, Dict, List, Optional, Sequence

from . import ai_columns


DEFAULT_MS_PER_ITEM = 2500.0
HISTORY_LIMIT = 25


class AIFillJobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, object]] = {}
        self._history: Dict[int, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    # ---- Public API -------------------------------------------------
    def start_job(self, product_ids: Sequence[int]) -> Dict[str, object]:
        job_id = uuid.uuid4().hex
        total = int(len(product_ids))
        now_ms = self._now_ms()
        bucket = self._bucket_for_total(total)
        eta_ms = int(math.ceil(total * self._estimate_ms_per_item(bucket))) if total else 0

        state: Dict[str, object] = {
            "job_id": job_id,
            "status": "running" if total else "done",
            "total": total,
            "processed": 0,
            "remaining": total,
            "pct": 0.0 if total else 100.0,
            "eta_ms": eta_ms,
            "started_at": now_ms,
            "updated_at": now_ms,
            "cancel_flag": False,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = state

        if total == 0:
            return dict(state)

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, list(product_ids), bucket),
            daemon=True,
        )
        thread.start()
        return dict(state)

    def get_job(self, job_id: str) -> Optional[Dict[str, object]]:
        with self._lock:
            state = self._jobs.get(job_id)
            return dict(state) if state else None

    def cancel_job(self, job_id: str) -> bool:
        with self._lock:
            state = self._jobs.get(job_id)
            if not state:
                return False
            state["cancel_flag"] = True
            if state.get("status") in {"running", "pending"}:
                state["status"] = "canceling"
            state["updated_at"] = self._now_ms()
        return True

    # ---- Internal helpers ------------------------------------------
    def _now_ms(self) -> float:
        return time.time() * 1000.0

    def _bucket_for_total(self, total: int) -> int:
        if total <= 0:
            return 0
        if total <= 10:
            return 10
        if total <= 25:
            return 25
        if total <= 50:
            return 50
        if total <= 100:
            return 100
        if total <= 250:
            return 250
        return 500

    def _estimate_ms_per_item(self, bucket: int) -> float:
        if bucket <= 0:
            return DEFAULT_MS_PER_ITEM
        history = self._history.get(bucket)
        if history:
            try:
                return float(median(history))
            except Exception:
                return DEFAULT_MS_PER_ITEM
        return DEFAULT_MS_PER_ITEM

    def _record_duration(self, bucket: int, total: int, duration_ms: float) -> None:
        if bucket <= 0 or total <= 0 or duration_ms <= 0:
            return
        per_item = duration_ms / max(total, 1)
        with self._lock:
            history = self._history[bucket]
            history.append(per_item)
            while len(history) > HISTORY_LIMIT:
                history.popleft()

    def _update_state(
        self,
        job_id: str,
        *,
        processed: Optional[int] = None,
        status: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            state = self._jobs.get(job_id)
            if not state:
                return
            total = int(state.get("total", 0))
            if processed is not None:
                proc = max(0, min(total, int(processed)))
                state["processed"] = proc
                state["remaining"] = max(total - proc, 0)
                pct = 100.0 if total == 0 else (proc / max(total, 1)) * 100.0
                state["pct"] = round(pct, 2)
            if status is not None:
                state["status"] = status
                if status in {"done", "canceled", "error"}:
                    state["eta_ms"] = 0
            if error is not None:
                state["error"] = error
            state["updated_at"] = self._now_ms()

    def _run_job(self, job_id: str, product_ids: List[int], bucket: int) -> None:
        start_ms = self._now_ms()
        total = len(product_ids)

        def status_cb(**payload: object) -> None:
            done = int(payload.get("done", 0) or 0)  # type: ignore[arg-type]
            total_from_cb = int(payload.get("total", total) or total)  # type: ignore[arg-type]
            self._update_state(job_id, processed=min(done, total_from_cb))

        def cancel_checker() -> bool:
            with self._lock:
                state = self._jobs.get(job_id)
                return bool(state and state.get("cancel_flag"))

        self._update_state(job_id, status="running")

        try:
            result = ai_columns.run_ai_fill_job(
                job_id=None,
                product_ids=product_ids,
                status_cb=status_cb,
                cancel_checker=cancel_checker,
            )
            counts = result.get("counts", {}) if isinstance(result, dict) else {}
            done = int(counts.get("ok", 0) + counts.get("cached", 0))
            error = result.get("error") if isinstance(result, dict) else None
            if cancel_checker():
                final_status = "canceled"
                error = None
            elif error == "cancelled":
                final_status = "canceled"
                error = None
            elif error:
                final_status = "error"
            else:
                final_status = "done"
            self._update_state(job_id, processed=done, status=final_status, error=error)
            if final_status == "done":
                duration_ms = self._now_ms() - start_ms
                self._record_duration(bucket, total, duration_ms)
        except Exception as exc:  # pragma: no cover - defensive
            self._update_state(job_id, status="error", error=str(exc))


def select_candidates(conn: sqlite3.Connection, limit: int) -> List[int]:
    cur = conn.cursor()
    if limit <= 0:
        limit = 1
    cur.execute(
        """
        SELECT id
        FROM products
        WHERE ai_columns_completed_at IS NULL
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    return [int(row[0]) for row in rows]


MANAGER = AIFillJobManager()

