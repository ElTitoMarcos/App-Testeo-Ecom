import logging
import time

from product_research_app.ai import gpt_orchestrator


def test_metrics_logging(caplog):
    import_id = "import-123"
    gpt_orchestrator.start_import(import_id)
    gpt_orchestrator.record_cache_saved(import_id, 5)
    gpt_orchestrator.record_call(import_id, 10)
    gpt_orchestrator.record_call(import_id, 0)

    with caplog.at_level(logging.INFO, logger=gpt_orchestrator.logger.name):
        gpt_orchestrator.flush_import_metrics(import_id)

    messages = [msg for msg in caplog.messages if "calls_total" in msg]
    assert messages
    assert "calls_total=2" in messages[-1]
    assert "calls_saved_by_cache=5" in messages[-1]


def test_handle_retry_after_reduces_parallel(monkeypatch):
    limiter = gpt_orchestrator._GlobalLimiter()
    monkeypatch.setattr(gpt_orchestrator, "_GLOBAL_LIMITER", limiter)

    before_next = limiter._rate._next_time
    start = time.monotonic()

    gpt_orchestrator.handle_retry_after(2.5)

    assert limiter._current_limit == 1
    assert limiter._restore_at >= start + 59.0
    assert limiter._rate._next_time >= max(before_next, start + 2.5)

