import threading
from http.server import HTTPServer

import requests
from openpyxl import Workbook

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from product_research_app import database
import product_research_app.web_app as web_app


def test_import_date_range(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    web_app.DB_PATH = db_path
    conn = web_app.ensure_db()

    wb = Workbook()
    ws = wb.active
    ws.append(["Name", "Rango de Fechas"])
    ws.append(["Prod", "2020-2021"])
    xlsx_path = tmp_path / "sample.xlsx"
    wb.save(xlsx_path)

    job_id = database.create_import_job(conn, str(xlsx_path))
    web_app._process_import_job(job_id, xlsx_path, "sample.xlsx")

    rows = database.list_products(conn)
    assert rows and rows[0]["date_range"] == "2020-2021"

    server = HTTPServer(("localhost", 0), web_app.RequestHandler)
    port = server.server_port
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        resp = requests.get(f"http://localhost:{port}/products")
        data = resp.json()
        assert data and data[0]["date_range"] == "2020-2021"
        assert data[0]["extras"]["Date Range"] == "2020-2021"
    finally:
        server.shutdown()
        t.join()

