import asyncio
import sqlite3

import pytest

from product_research_app import database, product_enrichment


@pytest.fixture()
def memory_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    database.initialize_database(conn)
    yield conn
    conn.close()


def capture_events(monkeypatch):
    events = []

    def _capture(job_id, payload):
        events.append((job_id, dict(payload)))

    monkeypatch.setattr(product_enrichment, "publish_progress", _capture)
    return events


def test_emit_progress_handles_zero_totals(memory_db, monkeypatch):
    events = capture_events(monkeypatch)
    pipeline = product_enrichment.EnrichmentPipeline(memory_db, job_id=7, config={})
    pipeline.total_items = 0
    pipeline.processed = 0
    pipeline._emit_progress(message="Sin elementos")
    assert events[-1][1]["percent"] == 0
    assert events[-1][1]["total"] == 0
    pipeline._emit_progress(status="done", message="Terminado")
    assert events[-1][1]["percent"] == 100
    assert events[-1][1]["status"] == "done"


def test_emit_progress_for_cache_only_run(memory_db, monkeypatch):
    events = capture_events(monkeypatch)
    pipeline = product_enrichment.EnrichmentPipeline(memory_db, job_id=5, config={})
    pipeline.total_items = 3
    pipeline.processed = 3
    pipeline.total_enriched = 3
    pipeline.cache_hits = 3
    pipeline._emit_progress(message="Cache completado")
    payload = events[-1][1]
    assert payload["percent"] == 100
    assert payload["enriched"] == 3
    assert payload["cache_hits"] == 3


def test_budget_pause_emits_progress(memory_db, monkeypatch):
    events = capture_events(monkeypatch)
    pipeline = product_enrichment.EnrichmentPipeline(
        memory_db,
        job_id=9,
        config={"max_cost_cents": 1, "cost_per_1k_input_cents": 2},
    )
    pipeline.total_items = 5
    pipeline.cost_cents = 0.5
    triggered = pipeline._budget_would_exceed(estimated_tokens=400)
    assert triggered is True
    assert pipeline.budget_paused is True
    payload = events[-1][1]
    assert payload["status"] == "paused_by_budget"
    assert payload["percent"] == 0


def test_worker_loop_emits_progress_with_concurrency(memory_db, monkeypatch):
    events = capture_events(monkeypatch)
    monkeypatch.setattr(product_enrichment, "MIN_BATCH_SIZE", 1)
    monkeypatch.setattr(product_enrichment, "MAX_BATCH_SIZE", 10)
    pipeline = product_enrichment.EnrichmentPipeline(
        memory_db,
        job_id=11,
        config={"batch_size": 2, "concurrency": 4},
    )
    items = [
        product_enrichment.PendingItem(
            item_id=i,
            sig_hash=f"sig-{i}",
            payload={"item_id": i},
            raw={},
            tokens_estimate=1,
        )
        for i in range(1, 5)
    ]
    pipeline.high_priority.extend(items)
    pipeline.total_items = len(items)

    async def fake_handle_batch(self, client, api_key, batch):
        await asyncio.sleep(0)
        self._update_metrics_after_batch(len(batch), len(batch), 0, len(batch) * 10, 50)
        self._emit_progress(message=f"Procesado {self.processed}/{self.total_items}")

    monkeypatch.setattr(product_enrichment.EnrichmentPipeline, "handle_batch", fake_handle_batch)

    async def run_workers():
        client = object()
        workers = [
            asyncio.create_task(pipeline.worker_loop(client, None))
            for _ in range(2)
        ]
        await asyncio.gather(*workers)

    asyncio.run(run_workers())

    percents = [evt[1]["percent"] for evt in events]
    assert percents, "No progress events captured"
    assert percents[-1] == 100
    assert pipeline.processed == pipeline.total_items == 4
    assert pipeline.batches == 2
    assert pipeline.requests == 2
