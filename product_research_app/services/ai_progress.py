from dataclasses import dataclass
import time, uuid

@dataclass
class AiJobState:
    job_id: str = ""
    status: str = "idle"        # idle | running | done | error
    total: int = 0
    processed: int = 0
    max_progress: float = 0.0   # nunca decrece dentro del mismo job
    started_at: float = 0.0

state = AiJobState()

def start_job(total: int) -> str:
    state.job_id = uuid.uuid4().hex[:12]
    state.status = "running"
    state.total = max(1, int(total or 1))
    state.processed = 0
    state.max_progress = 0.0
    state.started_at = time.time()
    return state.job_id

def bump_processed(n: int = 1) -> float:
    state.processed = min(state.total, state.processed + int(n))
    p = state.processed / float(state.total)
    if p > state.max_progress:
        state.max_progress = p
    return state.max_progress

def finish_job() -> str:
    state.status = "done"
    state.max_progress = 1.0
    state.processed = state.total
    return state.job_id

def get_progress_payload() -> dict:
    # Garantiza progreso monótono por job
    return {
        "job_id": state.job_id,
        "status": state.status,
        "progress": state.max_progress,  # 0.0..1.0
    }
