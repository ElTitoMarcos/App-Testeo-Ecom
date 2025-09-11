import sqlite3
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from product_research_app import config, database
from product_research_app.utils.db import row_to_dict, rget

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
    from product_research_app.services import winner_score
    monkeypatch.setattr(winner_score, 'WINNER_WEIGHTS_FILE', tmp_path / 'winner_weights.json')
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
