from __future__ import annotations

import math
import sqlite3
import threading
import time
import uuid
from collections import defaultdict, deque
from statistics import median
from typing import Deque, Dict, List, Optional, Sequence

from ..sse import publish_ai_event
from . import ai_columns


DEFAULT_MS_PER_ITEM = 2500.0
HISTORY_LIMIT = 25


class AIFillJobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, object]] = {}
        self._history: Dict[int, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._last_job_id: Optional[str] = None

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
            self._last_job_id = job_id

        snapshot = dict(state)
        if total == 0:
            self._emit_state(job_id, snapshot, event_type="ai.done")
            publish_ai_event({"type": "products.reload"})
            return snapshot

        self._emit_state(job_id, snapshot)

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, list(product_ids), bucket),
            daemon=True,
        )
        thread.start()
        return snapshot

    def get_job(self, job_id: str) -> Optional[Dict[str, object]]:
        with self._lock:
            state = self._jobs.get(job_id)
            return dict(state) if state else None

    def get_last_job(self) -> Optional[Dict[str, object]]:
        with self._lock:
            if not self._last_job_id:
                return None
            state = self._jobs.get(self._last_job_id)
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

    def _emit_state(
        self,
        job_id: str,
        state: Dict[str, object],
        *,
        event_type: str = "ai.progress",
    ) -> None:
        if not state:
            return
        total = int(state.get("total", 0) or 0)
        processed = int(state.get("processed", 0) or 0)
        remaining = int(state.get("remaining", total - processed) or 0)
        pct_val = float(state.get("pct", 0.0) or 0.0)
        status_val = str(state.get("status", "") or "").lower()
        if total > 0:
            progress = processed / max(total, 1)
        else:
            progress = 1.0 if status_val in {"done"} else 0.0
        if progress <= 0 and pct_val:
            progress = pct_val / 100.0
        progress = max(0.0, min(1.0, progress))
        payload: Dict[str, object] = {
            "type": event_type,
            "job_id": job_id,
            "status": state.get("status"),
            "total": total,
            "processed": processed,
            "remaining": max(0, remaining),
            "progress": round(progress, 4),
            "percent": round(progress * 100.0, 2),
            "eta_ms": int(state.get("eta_ms", 0) or 0),
            "updated_at": state.get("updated_at"),
        }
        error_val = state.get("error")
        if error_val:
            payload["error"] = str(error_val)
        publish_ai_event(payload)

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
            prev_status = str(state.get("status", "") or "")
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
            snapshot = dict(state)
            new_status = str(snapshot.get("status", "") or "")

        self._emit_state(job_id, snapshot)

        final_states = {"done", "canceled", "error"}
        if new_status in final_states and new_status != prev_status:
            if new_status == "error":
                self._emit_state(job_id, snapshot, event_type="ai.error")
            else:
                self._emit_state(job_id, snapshot, event_type="ai.done")
                if new_status == "done":
                    publish_ai_event({"type": "products.reload"})


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

