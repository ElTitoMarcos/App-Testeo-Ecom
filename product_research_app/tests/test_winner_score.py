import sqlite3
import pytest
import sys
from pathlib import Path
import math

sys.path.append(str(Path(__file__).resolve().parents[2]))
from product_research_app import config, database, gpt
from product_research_app.utils.db import row_to_dict, rget
from product_research_app.services import winner_score as ws
from datetime import date, timedelta

def test_insert_score_normalizes_to_int():
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    database.initialize_database(conn)
    pid = database.insert_product(
        conn,
        name='Prod',
        description='',
        category='',
        price=0.0,
        currency=None,
        image_url='',
        source='',
        extra={},
    )
    database.insert_score(
        conn,
        product_id=pid,
        model='winner_score',
        total_score=0,
        momentum=0,
        saturation=0,
        differentiation=0,
        social_proof=0,
        margin=0,
        logistics=0,
        summary='',
        explanations={},
        winner_score=123,
    )
    score = database.get_scores_for_product(conn, pid)[0]
    assert score['winner_score'] == 100


def test_weights_persist(tmp_path, monkeypatch):
    cfg_file = tmp_path / 'config.json'
    monkeypatch.setattr(config, 'CONFIG_FILE', cfg_file)
    from product_research_app.services import config as cfg_service
    monkeypatch.setattr(cfg_service, 'DB_PATH', tmp_path / 'data.sqlite3')
    cfg_service.init_app_config()
    config.set_weights({'price': 2.0, 'rating': 1.0})
    w = config.get_weights()
    assert w['price'] > w['rating']

def test_row_to_dict_and_rget_sqlite_row():
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.execute('CREATE TABLE t (id INTEGER, name TEXT)')
    conn.execute('INSERT INTO t VALUES (1, "x")')
    row = conn.execute('SELECT * FROM t').fetchone()
    d = row_to_dict(row)
    assert d == {"id": 1, "name": "x"}
    assert rget(row, "name") == "x"
    assert rget(row, "missing", 7) == 7


def test_oldness_weight_direction():
    today = date.today()
    prod_old = {"first_seen": (today - timedelta(days=200)).isoformat()}
    prod_new = {"first_seen": (today - timedelta(days=10)).isoformat()}
    ws.prepare_oldness_bounds([prod_old, prod_new])

    res_old = ws.compute_winner_score_v2(prod_old, {"oldness": 100}, oldness_dir=1)
    res_new = ws.compute_winner_score_v2(prod_new, {"oldness": 100}, oldness_dir=1)
    assert res_old["score"] > res_new["score"]

    res_old_rev = ws.compute_winner_score_v2(prod_old, {"oldness": 100}, oldness_dir=-1)
    res_new_rev = ws.compute_winner_score_v2(prod_new, {"oldness": 100}, oldness_dir=-1)
    assert res_new_rev["score"] > res_old_rev["score"]

    res_old_neu = ws.compute_winner_score_v2(prod_old, {"oldness": 0}, oldness_dir=0)
    res_new_neu = ws.compute_winner_score_v2(prod_new, {"oldness": 0}, oldness_dir=0)
    assert res_old_neu["score"] == res_new_neu["score"] == 0

    eff = ws.compute_winner_score_v2(prod_old, {"oldness": 100}, oldness_dir=1)
    assert eff["effective_weights"]["oldness"] == 1.0
    for k in ws.ALLOWED_FIELDS:
        if k != "oldness":
            assert eff["effective_weights"][k] == 0.0


def test_order_no_longer_affects_score():
    prod = {"price": 10.0, "rating": 5.0}
    ws.prepare_oldness_bounds([])
    weights = {"price": 50, "rating": 50}
    res_price_first = ws.compute_winner_score_v2(prod, weights, order=["price", "rating"])
    res_rating_first = ws.compute_winner_score_v2(prod, weights, order=["rating", "price"])
    assert res_price_first["score"] == res_rating_first["score"]


def test_awareness_weight_impacts_score():
    ws.prepare_oldness_bounds([])
    prod_low = {"awareness_level": "unaware"}
    prod_high = {"awareness_level": "most aware"}
    hi = ws.compute_winner_score_v2(prod_high, {"awareness": 100})
    lo = ws.compute_winner_score_v2(prod_low, {"awareness": 100})
    assert hi["score"] > lo["score"]
    hi0 = ws.compute_winner_score_v2(prod_high, {"awareness": 0})
    lo0 = ws.compute_winner_score_v2(prod_low, {"awareness": 0})
    assert hi0["score"] == lo0["score"]

def test_recommend_winner_weights_includes_awareness(monkeypatch):
    # simulate GPT returning weights for price and awareness
    def fake_call(api_key, model, messages):
        return {"choices": [{"message": {"content": '{"weights": {"price": 1, "awareness": 3}}'}}]}

    monkeypatch.setattr(gpt, "call_openai_chat", fake_call)
    samples = [{"price": 10.0, "awareness": 0.75, "target": 5.0}]
    res = gpt.recommend_winner_weights("k", "m", samples, "target")
    weights = res["weights"]
    assert set(weights) == {"price", "awareness"}
    assert weights["awareness"] == 3


def test_awareness_priority_and_closeness():
    order = ws.awareness_priority_order_from_weight(44)
    assert order == [2, 1, 3, 0, 4]
    prod = {"awareness_level": "solution aware"}
    val = ws.awareness_feature_value(prod, 44)
    idx = ws.awareness_stage_index_from_product(prod)
    expected = ws.awareness_closeness_from_weight(44, idx)
    assert math.isclose(val, expected)


def test_compute_effective_weights_includes_awareness():
    eff = ws.compute_effective_weights({"awareness": 10}, [])
    assert "awareness" in eff and eff["awareness"] > 0


def test_disabled_weight_excluded_from_score():
    ws.prepare_oldness_bounds([])
    prod = {"price": 10.0, "rating": 5.0}
    weights = {"price": 50, "rating": 50}
    enabled = {"price": False, "rating": True}
    res = ws.compute_winner_score_v2(prod, weights, order=["price", "rating"], enabled=enabled)
    assert "price" in res.get("disabled_fields", [])
    assert res["effective_weights"]["price"] == 0.0


def test_to_int_weights_clamps_and_orders():
    raw = {
        "price": 72,
        "rating": 88,
        "units_sold": 65,
        "revenue": 50,
        "desire": 40,
        "competition": 30,
        "oldness": 10,
        "awareness": 20,
    }
    ints, order, rng = ws.to_int_weights_0_100(raw, {})
    assert ints == raw
    assert order == [
        "rating",
        "price",
        "units_sold",
        "revenue",
        "desire",
        "competition",
        "awareness",
        "oldness",
    ]
    assert sum(ints.values()) == 375
    assert rng == "0_100"


def test_effective_weights_and_order_from_ints():
    ws.prepare_oldness_bounds([])
    raw = {
        "price": 72,
        "rating": 88,
        "units_sold": 65,
        "revenue": 50,
        "desire": 40,
        "competition": 30,
        "oldness": 10,
        "awareness": 20,
    }
    res = ws.compute_winner_score_v2({}, raw)
    assert res["effective_weights"]["rating"] == pytest.approx(0.88)
    assert res["effective_weights"]["oldness"] == pytest.approx(0.10)
    assert res["order"] == [
        "rating",
        "price",
        "units_sold",
        "revenue",
        "desire",
        "competition",
        "awareness",
        "oldness",
    ]


def test_to_int_weights_missing_metric_warns(caplog):
    raw = {
        "price": 72,
        "rating": 88,
        "units_sold": 65,
        "desire": 40,
        "competition": 30,
        "oldness": 10,
        "awareness": 20,
    }
    with caplog.at_level("WARNING"):
        ints, order, rng = ws.to_int_weights_0_100(raw, {})
    assert ints["revenue"] == 0
    assert any("missing weight for revenue" in rec.message for rec in caplog.records)
    assert set(ints.keys()) == set(ws.ALLOWED_FIELDS)
    assert rng == "0_100"


def test_to_int_weights_scales_from_zero_to_one():
    raw = {
        "price": 0.72,
        "rating": 0.88,
        "units_sold": 0.65,
        "revenue": 0.50,
        "desire": 0.40,
        "competition": 0.30,
        "oldness": 0.10,
        "awareness": 0.20,
    }
    ints, order, rng = ws.to_int_weights_0_100(raw, {})
    assert ints["price"] == 72
    assert ints["revenue"] == 50
    assert order[0] == "rating"
    assert rng == "0_1"
