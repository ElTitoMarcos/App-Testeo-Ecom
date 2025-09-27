from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Phase:
    name: str
    weight: float
    total: int = 0
    done: int = 0


@dataclass
class JobProgress:
    job_id: str
    created_ts: float = field(default_factory=time.time)
    phases: Dict[str, Phase] = field(default_factory=dict)
    message: str = "starting"
    error: Optional[str] = None
    finished: bool = False

    def percent(self) -> int:
        if not self.phases:
            return 0
        acc = 0.0
        for ph in self.phases.values():
            part = 0.0 if ph.total <= 0 else (ph.done / max(1, ph.total))
            acc += ph.weight * max(0.0, min(1.0, part))
        pct = int(round(acc * 100))
        if self.finished and pct < 100 and not self.error:
            return 100
        return min(100, max(0, pct))


class ProgressRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: Dict[str, JobProgress] = {}

    def create(self, job_id: str, phases: Dict[str, Phase]) -> JobProgress:
        with self._lock:
            jp = JobProgress(job_id=job_id, phases=phases)
            self._jobs[job_id] = jp
            return jp

    def get(self, job_id: str) -> Optional[JobProgress]:
        with self._lock:
            return self._jobs.get(job_id)

    def update_phase(
        self,
        job_id: str,
        phase: str,
        *,
        done_delta: int = 0,
        total: Optional[int] = None,
        message: Optional[str] = None,
    ) -> None:
        with self._lock:
            jp = self._jobs.get(job_id)
            if not jp:
                return
            ph = jp.phases.get(phase)
            if not ph:
                return
            if total is not None:
                ph.total = total
            if done_delta:
                ph.done += done_delta
                if ph.done > ph.total and ph.total > 0:
                    ph.done = ph.total
            if message:
                jp.message = message

    def set_finished(
        self,
        job_id: str,
        *,
        error: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        with self._lock:
            jp = self._jobs.get(job_id)
            if not jp:
                return
            jp.error = error
            if message:
                jp.message = message
            jp.finished = True


registry = ProgressRegistry()


def default_phases() -> Dict[str, Phase]:
    return {
        "import": Phase(name="import", weight=0.20, total=0, done=0),
        "ai_fill": Phase(name="ai_fill", weight=0.75, total=0, done=0),
        "post": Phase(name="post", weight=0.05, total=1, done=0),
    }

