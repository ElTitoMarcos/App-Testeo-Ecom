import importlib
import logging
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from product_research_app import ratelimit


class _FakeTime:
    def __init__(self, start: float):
        self.current = start
        self.total_slept = 0.0

    def monotonic(self) -> float:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.total_slept += seconds
        self.current += seconds


@pytest.mark.parametrize("over_capacity_factor", [2, 3])
def test_reserve_handles_requests_above_capacity(monkeypatch, caplog, request, over_capacity_factor):
    monkeypatch.setenv("PRAPP_OPENAI_TPM", "120")
    monkeypatch.setenv("PRAPP_OPENAI_RPM", "120")
    monkeypatch.setenv("PRAPP_OPENAI_HEADROOM", "1.0")
    monkeypatch.setenv("PRAPP_OPENAI_MAX_CONCURRENCY", "1")

    importlib.reload(ratelimit)
    request.addfinalizer(lambda: importlib.reload(ratelimit))

    start_monotonic = ratelimit.time.monotonic()
    fake_time = _FakeTime(start_monotonic)
    monkeypatch.setattr(ratelimit, "time", fake_time)
    ratelimit._tokens_bucket.last = fake_time.current
    ratelimit._tokens_bucket.tokens = ratelimit._tokens_bucket.capacity
    ratelimit._requests_bucket.last = fake_time.current
    ratelimit._requests_bucket.tokens = ratelimit._requests_bucket.capacity

    caplog.set_level(logging.WARNING, logger=ratelimit.logger.name)

    tokens_needed = ratelimit._EFF_TPM * over_capacity_factor
    with ratelimit.reserve(tokens_needed):
        pass

    assert fake_time.total_slept > 0
    assert any(
        "exceeds bucket capacity" in record.getMessage() for record in caplog.records
    )
