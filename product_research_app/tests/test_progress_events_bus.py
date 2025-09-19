import queue

import pytest

from product_research_app import progress_events


def _drain(q: "queue.Queue") -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


def test_publish_progress_reaches_all_subscribers():
    sub_a = progress_events.subscribe()
    sub_b = progress_events.subscribe()
    try:
        _drain(sub_a.queue)
        _drain(sub_b.queue)
        progress_events.publish_progress("job-1", {"operation": "import", "percent": 42})
        evt_a = sub_a.queue.get(timeout=0.5)
        evt_b = sub_b.queue.get(timeout=0.5)
    finally:
        progress_events.unsubscribe(sub_a)
        progress_events.unsubscribe(sub_b)
    assert evt_a["job_id"] == "job-1"
    assert evt_b["operation"] == "import"
    assert evt_b["percent"] == 42


def test_unsubscribe_stops_event_delivery():
    sub = progress_events.subscribe()
    try:
        _drain(sub.queue)
        progress_events.publish_progress("job-2", {"operation": "enrich", "percent": 5})
        first = sub.queue.get(timeout=0.5)
        assert first["percent"] == 5
    finally:
        progress_events.unsubscribe(sub)
    progress_events.publish_progress("job-2", {"operation": "enrich", "percent": 10})
    with pytest.raises(queue.Empty):
        sub.queue.get_nowait()
