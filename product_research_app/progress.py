from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Iterable, Tuple
import threading, time

IA_FIELDS: Tuple[str, ...] = (
    "desire",
    "desire_magnitude",
    "awareness_level",
    "competition_level",
)

@dataclass
class Phase:
    name: str
    weight: float     # suma = 1.0
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
        acc = 0.0
        for ph in self.phases.values():
            frac = (ph.done / ph.total) if ph.total > 0 else 0.0
            if frac < 0: frac = 0.0
            if frac > 1: frac = 1.0
            acc += ph.weight * frac
        pct = int(round(acc * 100))
        if self.finished and not self.error:
            return 100
        return max(0, min(100, pct))

class ProgressRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: Dict[str, JobProgress] = {}

    def create(self, job_id: str, import_total: int = 0) -> JobProgress:
        with self._lock:
            jp = JobProgress(job_id=job_id, phases={
                "import": Phase("import", 0.30, total=import_total, done=0),
                "ai":     Phase("ai",     0.70, total=0,             done=0),
            })
            self._jobs[job_id] = jp
            return jp

    def get(self, job_id: str) -> Optional[JobProgress]:
        with self._lock:
            return self._jobs.get(job_id)

    def set_message(self, job_id: str, msg: str):
        with self._lock:
            jp = self._jobs.get(job_id)
            if jp: jp.message = msg

    def set_finished(self, job_id: str, error: Optional[str] = None, msg: Optional[str] = None):
        with self._lock:
            jp = self._jobs.get(job_id)
            if not jp: return
            jp.error = error
            if msg: jp.message = msg
            jp.finished = True

    def set_import_total(self, job_id: str, total: int):
        with self._lock:
            jp = self._jobs.get(job_id)
            if jp and "import" in jp.phases:
                jp.phases["import"].total = max(0, int(total))

    def inc_import_done(self, job_id: str, delta: int = 1):
        with self._lock:
            jp = self._jobs.get(job_id)
            if jp and "import" in jp.phases:
                ph = jp.phases["import"]; ph.done = min(ph.total or ph.done+delta, ph.done + delta)

    def set_ai_total_cells(self, job_id: str, total_cells: int):
        with self._lock:
            jp = self._jobs.get(job_id)
            if jp and "ai" in jp.phases:
                jp.phases["ai"].total = max(0, int(total_cells))
                jp.phases["ai"].done = min(jp.phases["ai"].done, jp.phases["ai"].total)

    def inc_ai_done_cells(self, job_id: str, delta_cells: int):
        if delta_cells <= 0: return
        with self._lock:
            jp = self._jobs.get(job_id)
            if jp and "ai" in jp.phases:
                ph = jp.phases["ai"]
                ph.done = min(ph.total or ph.done+delta_cells, ph.done + delta_cells)

# Registro global
registry = ProgressRegistry()

def count_missing_ai_cells(products_iter: Iterable[dict]) -> int:
    """
    Devuelve el número de celdas IA faltantes en los productos dados.
    Cuenta 1 por cada campo de IA ausente.
    """
    missing = 0
    for p in products_iter:
        for k in IA_FIELDS:
            if p.get(k) in (None, "", []):
                missing += 1
    return missing
