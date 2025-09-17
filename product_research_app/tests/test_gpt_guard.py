import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from product_research_app import database
from product_research_app.ai import gpt_guard
from product_research_app.ai.gpt_guard import GPTGuard, ai_cache_get, ai_cache_set, hash_key_for_item


def test_hash_key_for_item_stable():
    item = {"title": "  Test Product  ", "description": "Great", "features": ["a", "b"]}
    key_one = hash_key_for_item("Desire", item)
    key_two = hash_key_for_item(
        "desire",
        {"description": "Great", "features": ["a", "b"], "title": "Test Product"},
    )
    assert key_one == key_two
    key_three = hash_key_for_item("desire", {"title": "Test Product", "description": "Different"})
    assert key_three != key_one


def test_ai_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(gpt_guard, "DB_PATH", tmp_path / "data.sqlite3")
    conn = database.get_connection(gpt_guard.DB_PATH)
    database.initialize_database(conn)
    conn.close()

    payload = {"foo": "bar"}
    ai_cache_set("desire", "sample", payload, "v1", ttl_days=90)
    cached = ai_cache_get("desire", "sample")
    assert cached is not None
    assert cached["payload"] == payload
    assert cached["model_version"] == "v1"

    stale = datetime.utcnow() - timedelta(days=200)
    conn = database.get_connection(gpt_guard.DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO ai_cache (task_type, cache_key, payload_json, model_version, created_at) VALUES (?, ?, ?, ?, ?)",
        ("desire", "old", json.dumps({"x": 1}), "v0", stale.isoformat()),
    )
    conn.commit()
    conn.close()

    ai_cache_set("desire", "fresh", {"bar": "baz"}, "v2", ttl_days=30)
    conn = database.get_connection(gpt_guard.DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM ai_cache WHERE cache_key='old'")
    assert cur.fetchone()[0] == 0
    conn.close()


def test_gpt_guard_enforces_budget():
    guard = GPTGuard(
        {
            "max_parallel": 2,
            "max_calls_per_import": 1,
            "min_batch": 1,
            "max_batch": 2,
            "coalesce_ms": 0,
        }
    )

    calls = []

    def fake_call(batch):
        calls.append([item.get("id") for item in batch])
        return {"ids": [item.get("id") for item in batch]}

    items = [{"id": 1}, {"id": 2}, {"id": 3}]
    summary = guard.submit("desire", items, fake_call)
    assert summary["processed"] == 2
    assert summary["skipped"] == 1
    assert summary["skipped_items"] == [{"id": 3}]
    assert summary["notes"] == ["budget_exhausted"]
    assert summary["errors"] == []
    assert len(calls) == 1


def test_gpt_guard_handles_rate_limit(monkeypatch):
    guard = GPTGuard(
        {
            "max_parallel": 2,
            "max_calls_per_import": 5,
            "min_batch": 1,
            "max_batch": 5,
            "coalesce_ms": 0,
        }
    )

    guard._sleep = lambda _: None  # avoid waiting in tests

    attempts = {"count": 0}

    class RateLimitError(Exception):
        status_code = 429
        retry_after = 0

    def fake_call(batch):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RateLimitError("rate limit")
        return {"attempt": attempts["count"], "ids": [item.get("id") for item in batch]}

    summary = guard.submit("desire", [{"id": 1}, {"id": 2}], fake_call)
    assert summary["processed"] == 2
    assert summary["errors"] == []
    assert "rate_limited" in summary["notes"]
    assert attempts["count"] == 2
    assert guard._current_parallel_limit == 1


def test_gpt_guard_server_error_records_note(monkeypatch):
    guard = GPTGuard(
        {
            "max_parallel": 1,
            "max_calls_per_import": 5,
            "min_batch": 1,
            "max_batch": 5,
            "coalesce_ms": 0,
        }
    )

    guard._sleep = lambda _: None
    guard._server_retry_delay = lambda: 0

    class ServerError(Exception):
        def __init__(self, status):
            super().__init__("server error")
            self.status_code = status

    def failing_call(batch):
        raise ServerError(503)

    summary = guard.submit("desire", [{"id": 42}], failing_call)
    assert summary["processed"] == 0
    assert len(summary["errors"]) == 1
    assert "server_error_503" in summary["notes"]
    assert summary["results"][0]["success"] is False
    assert summary["results"][0]["attempts"] == 2
