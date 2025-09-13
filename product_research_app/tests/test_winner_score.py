import sqlite3
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from product_research_app import config, database
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

    res_old = ws.compute_winner_score_v2(prod_old, {"oldness": 100})
    res_new = ws.compute_winner_score_v2(prod_new, {"oldness": 100})
    assert res_old["score"] > res_new["score"]

    res_old_rev = ws.compute_winner_score_v2(prod_old, {"oldness": 0})
    res_new_rev = ws.compute_winner_score_v2(prod_new, {"oldness": 0})
    assert res_new_rev["score"] > res_old_rev["score"]

    res_old_neu = ws.compute_winner_score_v2(prod_old, {"oldness": 50})
    res_new_neu = ws.compute_winner_score_v2(prod_new, {"oldness": 50})
    assert res_old_neu["score"] == res_new_neu["score"] == 0
