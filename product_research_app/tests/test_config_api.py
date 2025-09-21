import json
from pathlib import Path

import pytest

from product_research_app import config
from product_research_app.api import app as flask_app
from product_research_app.services import winner_score as ws


def _setup_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_file = tmp_path / "config.json"
    monkeypatch.setattr(config, "CONFIG_FILE", cfg_file)
    ws.invalidate_weights_cache()


def test_default_order_applied(tmp_path, monkeypatch):
    _setup_config(tmp_path, monkeypatch)
    with flask_app.test_client() as client:
        resp = client.get("/api/config/winner-weights")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["order"] == config.DEFAULT_WINNER_ORDER
        assert data["version"] == "v2"
        for key in config.DEFAULT_WINNER_WEIGHTS_INT:
            assert 0 <= data["weights"][key] <= 50


def test_patch_and_get_roundtrip(tmp_path, monkeypatch):
    _setup_config(tmp_path, monkeypatch)
    order = config.DEFAULT_WINNER_ORDER[:]
    order[0], order[1] = order[1], order[0]
    payload = {"weights": {"price": 18}, "order": order}
    with flask_app.test_client() as client:
        patch = client.patch(
            "/api/config/winner-weights",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert patch.status_code == 200
        body = patch.get_json()
        assert body["weights"]["price"] == 18
        assert body["order"] == order
        get_resp = client.get("/api/config/winner-weights")
        data = get_resp.get_json()
        assert data["weights"]["price"] == 18
        assert data["order"] == order


def test_ai_endpoint_persists(tmp_path, monkeypatch):
    from product_research_app.api import config as api_config

    _setup_config(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "get_api_key", lambda: "test-key")
    fake_result = {
        "weights": {k: 14 for k in config.DEFAULT_WINNER_ORDER},
        "order": config.DEFAULT_WINNER_ORDER[::-1],
        "justification": "test",
    }
    monkeypatch.setattr(api_config.gpt, "recommend_winner_weights", lambda *args, **kwargs: fake_result)
    monkeypatch.setattr(ws, "recompute_scores_for_all_products", lambda scope="all": 0)
    payload = {
        "features": list(config.DEFAULT_WINNER_ORDER),
        "target": "revenue",
        "data_sample": [{**{k: 1.0 for k in config.DEFAULT_WINNER_ORDER}, "target": 1.0}],
    }
    with flask_app.test_client() as client:
        resp = client.post(
            "/api/config/winner-weights/ai?can_reorder=true",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["persisted"] is True
        assert body["order"] == fake_result["order"]
        assert body["winner_weights"]["revenue"] == 14
