import io
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parents[2]))

from product_research_app import web_app, database, config

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
    products = [dict(r) for r in database.list_products(conn)]
    assert len(products) == 2
    for p in products:
        score = database.get_scores_for_product(conn, p["id"])[0]
        assert 0 <= score["winner_score"] <= 100

def test_generate_endpoint_updates_scores(tmp_path, monkeypatch):
    conn = setup_env(tmp_path, monkeypatch)
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
        product_id=10,
    )
    database.update_product(
        conn,
        pid,
        magnitud_deseo="high",
        nivel_consciencia_headroom="unaware",
        evidencia_demanda=100,
        tasa_conversion=0.5,
        ventas_por_dia=10,
        recencia_lanzamiento=30,
        competition_level_invertido="low",
        facilidad_anuncio="high",
        escalabilidad="high",
        durabilidad_recurrencia="consumible",
    )
    body = json.dumps({"ids": [pid]})
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
    assert resp["updated"] == 1
    score = database.get_scores_for_product(conn, pid)[0]
    assert 0 <= score["winner_score"] <= 100
