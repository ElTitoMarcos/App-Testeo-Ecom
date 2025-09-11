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
        "magnitud_deseo",
        "nivel_consciencia_headroom",
        "evidencia_demanda",
        "tasa_conversion",
        "ventas_por_dia",
        "recencia_lanzamiento",
        "competition_level_invertido",
        "facilidad_anuncio",
        "escalabilidad",
        "durabilidad_recurrencia",
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
            ["Prod1", "high", "unaware", 100, 0.5, 10, 30, "low", "high", "high", "consumible"],
            ["Prod2", "medium", "solution", 50, 0.3, 5, 60, "medium", "med", "med", "intermedio"],
        ],
    )
    job_id = database.create_import_job(conn, str(xlsx))
    web_app._process_import_job(job_id, xlsx, "products.xlsx")
    products = [row_to_dict(r) for r in database.list_products(conn)]
    assert len(products) == 2
    for p in products:
        score = database.get_scores_for_product(conn, p["id"])[0]
        assert 0 <= score["winner_score"] <= 100

def test_scoring_v2_generate_cases(tmp_path, monkeypatch):
    conn = setup_env(tmp_path, monkeypatch)
    monkeypatch.setattr(web_app, "ensure_db", lambda: conn)
    pid_a = database.insert_product(
        conn,
        name="A",
        description="",
        category="",
        price=0.0,
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
        price=0.0,
        currency=None,
        image_url="",
        source="",
        extra={},
        product_id=2,
    )
    database.update_product(conn, pid_b, magnitud_deseo=80, tasa_conversion=0.5)

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
    assert resp["updated"] == 2
    details = {d["id"]: d for d in resp["details"]}
    da = details[pid_a]
    db = details[pid_b]
    assert da["score"] == 50 and da["fallback"] is True and da["used"] == 0
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
