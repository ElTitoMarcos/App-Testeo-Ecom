import asyncio
import logging
import re

import httpx

from product_research_app.services import ai_columns


def test_build_payload_template_replaces_placeholders():
    batch = [
        {
            "id": 123,
            "title": "Widget",
            "description": "Descripción extensa",
            "category": "Gadgets",
            "brand": "Acme",
            "price": 19.99,
            "rating": 4.7,
            "units_sold": 42,
            "revenue": 1234.56,
            "oldness": 0.3,
        }
    ]
    weights = {"desire": 0.8, "winner_score": 0.6}

    payload = ai_columns._build_payload(batch, weights)
    messages = payload["messages"]
    assert len(messages) >= 2
    user_content = messages[1]["content"]

    assert '{"results":' in user_content
    assert "0-100" in user_content
    assert not re.findall(r"\{[A-Za-z_][A-Za-z0-9_]*\}", user_content)
    assert ai_columns.RNG_SENTINEL not in user_content


def test_run_batches_retries_on_timeout(monkeypatch, caplog):
    call_count = 0

    async def fake_score_batch(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise httpx.ReadTimeout("boom")

    monkeypatch.setattr(
        ai_columns, "_score_batch_with_backfill", fake_score_batch, raising=True
    )

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", DummyAsyncClient)

    batches = [[{"id": "1", "title": "Item"}]]
    logger = logging.getLogger("test-timeout")

    async def run():
        return await ai_columns._run_batches_parallel(batches, None, None, logger)

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(run())

    assert result == 0
    assert call_count == ai_columns.MAX_RETRIES + 1
    timeout_logs = [
        record for record in caplog.records if "timeout" in record.getMessage().lower()
    ]
    assert timeout_logs
