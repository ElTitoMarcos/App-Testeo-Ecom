import io
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parents[2]))

from product_research_app import web_app, database, config
from product_research_app.utils.db import row_to_dict

def setup_env(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "DB_PATH", tmp_path / "data.sqlite3")
    monkeypatch.setattr(web_app, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(web_app, "LOG_PATH", tmp_path / "logs" / "app.log")
    web_app.LOG_DIR.mkdir(exist_ok=True)
    for h in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler(web_app.LOG_PATH, encoding="utf-8")],
        force=True,
    )
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    return web_app.ensure_db()

def make_xlsx(path: Path, rows: List[List[object]]):
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.append([
        "name",
        "price",
        "rating",
        "units_sold",
        "revenue",
        "review_count",
        "image_count",
        "shipping_days",
        "profit_margin",
        "desire",
        "competition",
    ])
    for r in rows:
        ws.append(r)
    wb.save(path)

def test_app_startup(tmp_path, monkeypatch):
    conn = setup_env(tmp_path, monkeypatch)
    assert web_app.LOG_PATH.exists()
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='scores'")
    assert cur.fetchone() is not None

def test_import_generates_scores(tmp_path, monkeypatch):
    conn = setup_env(tmp_path, monkeypatch)
    xlsx = tmp_path / "products.xlsx"
    make_xlsx(
        xlsx,
        [
            ["Prod1", 10, 4.5, 100, 1000, 50, 3, 5, 0.3, "High", "Low"],
            ["Prod2", 20, 3.0, 50, 500, 10, 2, 7, 0.2, "Medium", "Medium"],
        ],
    )
    job_id = database.create_import_job(conn, str(xlsx))
    web_app._process_import_job(job_id, xlsx, "products.xlsx")
    products = [row_to_dict(r) for r in database.list_products(conn)]
    assert len(products) == 2
    for p in products:
        assert 0 <= p.get("winner_score", 0) <= 100

def test_scoring_v2_generate_cases(tmp_path, monkeypatch):
    conn = setup_env(tmp_path, monkeypatch)
    monkeypatch.setattr(web_app, "ensure_db", lambda: conn)
    pid_a = database.insert_product(
        conn,
        name="A",
        description="",
        category="",
        price=None,
        currency=None,
        image_url="",
        source="",
        extra={},
        product_id=1,
    )
    pid_b = database.insert_product(
        conn,
        name="B",
        description="",
        category="",
        price=20.0,
        currency=None,
        image_url="",
        source="",
        extra={"rating": 4.0},
        product_id=2,
    )
    body = json.dumps({"ids": [pid_a, pid_b]})
    class Dummy:
        def __init__(self, body):
            self.headers = {"Content-Length": str(len(body))}
            self.rfile = io.BytesIO(body.encode("utf-8"))
            self.wfile = io.BytesIO()
        def _set_json(self, code=200):
            self.status = code

    handler = Dummy(body)
    web_app.RequestHandler.handle_scoring_v2_generate(handler)
    resp = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert resp["success"] is True
    assert resp["updated"] == 1
    details = {d["id"]: d for d in resp["details"]}
    da = details[pid_a]
    db = details[pid_b]
    assert da["score"] is None and da["fallback"] is True and da["used"] == 0
    assert db["fallback"] is False and db["used"] > 0 and 0 <= db["score"] <= 100
    prod_b = database.get_product(conn, pid_b)
    assert prod_b["winner_score"] == db["score"]

    body2 = json.dumps({"ids": [pid_b]})
    handler2 = Dummy(body2)
    web_app.RequestHandler.handle_scoring_v2_generate(handler2)
    resp2 = json.loads(handler2.wfile.getvalue().decode("utf-8"))
    assert resp2["success"] is True
    assert resp2["updated"] == 0
    assert resp2["skipped"] == 1


def test_scoring_v2_generate_all_when_no_ids(tmp_path, monkeypatch):
    conn = setup_env(tmp_path, monkeypatch)
    monkeypatch.setattr(web_app, "ensure_db", lambda: conn)
    pid_a = database.insert_product(
        conn,
        name="A",
        description="",
        category="",
        price=None,
        currency=None,
        image_url="",
        source="",
        extra={"rating": 4.0},
        product_id=1,
    )
    pid_b = database.insert_product(
        conn,
        name="B",
        description="",
        category="",
        price=None,
        currency=None,
        image_url="",
        source="",
        extra={},
        product_id=2,
    )

    body = json.dumps({})

    class Dummy:
        def __init__(self, body):
            self.headers = {"Content-Length": str(len(body))}
            self.rfile = io.BytesIO(body.encode("utf-8"))
            self.wfile = io.BytesIO()

        def _set_json(self, code=200):
            self.status = code

    handler = Dummy(body)
    web_app.RequestHandler.handle_scoring_v2_generate(handler)
    resp = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert resp["received"] == 2

def test_products_endpoint_serializes_rows(tmp_path, monkeypatch):
    conn = setup_env(tmp_path, monkeypatch)
    monkeypatch.setattr(web_app, "ensure_db", lambda: conn)
    pid = database.insert_product(
        conn,
        name="X",
        description="",
        category="",
        price=0.0,
        currency=None,
        image_url="",
        source="",
        extra={},
        product_id=1,
    )
    conn.execute("UPDATE products SET winner_score=42 WHERE id=?", (pid,))
    conn.commit()

    class Dummy:
        def __init__(self):
            self.path = "/products"
            self.headers = {}
            self.wfile = io.BytesIO()

        def _set_json(self, code=200):
            self.status = code

        def send_error(self, code, msg=None):
            raise AssertionError(f"error {code}")

        def safe_write(self, func):
            try:
                func()
                return True
            except Exception:
                return False

    handler = Dummy()
    web_app.RequestHandler.do_GET(handler)
    resp = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert isinstance(resp, list) and resp and resp[0]["id"] == pid
    assert resp[0]["winner_score"] == 42


def test_patch_winner_weights_persists(tmp_path, monkeypatch):
    setup_env(tmp_path, monkeypatch)

    body = json.dumps({"key": "rating_weight", "value": 0.25})

    class Dummy:
        def __init__(self, body):
            self.path = "/api/config/winner-weights"
            self.headers = {"Content-Length": str(len(body))}
            self.rfile = io.BytesIO(body.encode("utf-8"))
            self.wfile = io.BytesIO()

        def _set_json(self, code=200):
            self.status = code

    handler = Dummy(body)
    web_app.RequestHandler.do_PATCH(handler)
    resp = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert resp.get("status") == "ok"
    cfg = config.load_config()
    assert cfg.get("weights", {}).get("rating") == 0.25

def test_get_endpoints_return_json(tmp_path, monkeypatch):
    setup_env(tmp_path, monkeypatch)
    from http.server import HTTPServer
    import threading, urllib.request, json, time

    server = HTTPServer(("127.0.0.1", 0), web_app.RequestHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    try:
        time.sleep(0.1)
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/config")
        assert resp.status == 200
        cfg = json.loads(resp.read().decode("utf-8"))
        weights = cfg.get("weights", {})
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert set(weights.keys()) == set(config.SCORING_DEFAULT_WEIGHTS.keys())
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/lists")
        assert resp.status == 200
        json.loads(resp.read().decode("utf-8"))
    finally:
        server.shutdown()
        thread.join()


def test_weight_changes_persist_without_score_reset(tmp_path, monkeypatch):
    conn = setup_env(tmp_path, monkeypatch)
    monkeypatch.setattr(web_app, "ensure_db", lambda: conn)
    pid = database.insert_product(
        conn,
        name="A",
        description="",
        category="",
        price=None,
        currency=None,
        image_url="",
        source="",
        extra={},
        product_id=1,
    )
    conn.execute("UPDATE products SET winner_score=55 WHERE id=?", (pid,))
    conn.commit()

    def patch_weight(key, value):
        body = json.dumps({"key": key, "value": value})
        class Dummy:
            def __init__(self, body):
                self.path = "/api/config/winner-weights"
                self.headers = {"Content-Length": str(len(body))}
                self.rfile = io.BytesIO(body.encode("utf-8"))
                self.wfile = io.BytesIO()

            def _set_json(self, code=200):
                self.status = code

        h = Dummy(body)
        web_app.RequestHandler.do_PATCH(h)
        resp = json.loads(h.wfile.getvalue().decode("utf-8"))
        assert resp.get("status") == "ok"

    patch_weight("rating_weight", 0.2)
    patch_weight("price_weight", 0.3)
    patch_weight("units_sold_weight", 0.5)

    cfg = config.load_config()
    weights = cfg.get("weights", {})
    assert weights.get("rating") == 0.2
    assert weights.get("price") == 0.3
    assert weights.get("units_sold") == 0.5
    prod = database.get_product(conn, pid)
    assert prod["winner_score"] == 55


def test_import_new_batch_preserves_existing_scores(tmp_path, monkeypatch):
    conn = setup_env(tmp_path, monkeypatch)
    monkeypatch.setattr(web_app, "ensure_db", lambda: conn)
    pid = database.insert_product(
        conn,
        name="Old", description="", category="", price=None, currency=None,
        image_url="", source="", extra={}, product_id=1,
    )
    conn.execute("UPDATE products SET winner_score=10 WHERE id=?", (pid,))
    conn.commit()

    xlsx = tmp_path / "batch.xlsx"
    make_xlsx(xlsx, [["New", 10, 4.5, 100, 1000, 50, 3, 5, 0.3, "High", "Low"]])
    job_id = database.create_import_job(conn, str(xlsx))
    web_app._process_import_job(job_id, xlsx, "batch.xlsx")

    prod_old = database.get_product(conn, pid)
    assert prod_old["winner_score"] == 10
    cur = conn.execute("SELECT id, winner_score FROM products WHERE id != ?", (pid,))
    rows = cur.fetchall()
    assert rows
    for _id, score in rows:
        assert score is not None and score > 0


def test_recalculate_selected_and_all(tmp_path, monkeypatch):
    conn = setup_env(tmp_path, monkeypatch)
    monkeypatch.setattr(web_app, "ensure_db", lambda: conn)
    ids = []
    for i in range(8):
        pid = database.insert_product(
            conn,
            name=f"P{i}", description="", category="", price=None, currency=None,
            image_url="", source="", extra={"rating": 4.5}, product_id=i + 1,
        )
        conn.execute("UPDATE products SET winner_score=1 WHERE id=?", (pid,))
        ids.append(pid)
    conn.commit()

    subset = ids[:5]
    body = json.dumps({"product_ids": subset})
    class Dummy:
        def __init__(self, body):
            self.headers = {"Content-Length": str(len(body))}
            self.rfile = io.BytesIO(body.encode("utf-8"))
            self.wfile = io.BytesIO()

        def _set_json(self, code=200):
            self.status = code

    h = Dummy(body)
    web_app.RequestHandler.handle_scoring_v2_generate(h)
    for pid in subset:
        assert database.get_product(conn, pid)["winner_score"] != 1
    for pid in ids[5:]:
        assert database.get_product(conn, pid)["winner_score"] == 1

    body2 = json.dumps({})
    h2 = Dummy(body2)
    web_app.RequestHandler.handle_scoring_v2_generate(h2)
    for pid in ids:
        assert database.get_product(conn, pid)["winner_score"] != 1
