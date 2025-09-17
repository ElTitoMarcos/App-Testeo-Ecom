import io
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import List
from urllib.parse import urlparse

sys.path.append(str(Path(__file__).resolve().parents[2]))

from product_research_app import db, web_app, database, config, gpt
from product_research_app.ai import runner
from product_research_app.services import importer_fast, importer_unified, winner_score
from product_research_app.services import config as cfg_service
from product_research_app.utils.db import row_to_dict

def setup_env(tmp_path, monkeypatch):
    db.close_db()
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
    monkeypatch.setattr(cfg_service, "DB_PATH", tmp_path / "data.sqlite3")
    monkeypatch.setattr(runner, "DB_PATH", tmp_path / "data.sqlite3")
    cfg_service.init_app_config()
    conn = web_app.ensure_db()

    def _fake_get_db(path="", write=False):
        return conn

    monkeypatch.setattr(db, "get_db", _fake_get_db)
    monkeypatch.setattr(importer_fast, "get_db", _fake_get_db)

    def _fake_import_xlsx(bytes_data, *, source, status_cb):
        from openpyxl import load_workbook

        status_cb(stage="parse_xlsx", done=0, total=0)
        wb = load_workbook(io.BytesIO(bytes_data), read_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            status_cb(stage="parse_xlsx", done=0, total=0)
            return []
        headers = [str(h) for h in rows[0]]
        records = []
        for idx, row in enumerate(rows[1:], start=1):
            rec = {headers[i]: row[i] for i in range(len(headers))}
            records.append(rec)
            if idx % 500 == 0:
                status_cb(stage="parse_xlsx", done=idx, total=len(rows) - 1)
        status_cb(stage="parse_xlsx", done=len(records), total=len(records))
        return records

    monkeypatch.setattr(importer_unified, "import_xlsx", _fake_import_xlsx)
    return conn

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
        "date_range",
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

def test_import_unified_inserts_rows(tmp_path, monkeypatch):
    conn = setup_env(tmp_path, monkeypatch)
    xlsx = tmp_path / "products.xlsx"
    make_xlsx(
        xlsx,
        [
            ["Prod1", 10, 4.5, 100, 1000, "2024-01-01~2024-02-01", 3, 5, 0.3, "High", "Low"],
            ["Prod2", 20, 3.0, 50, 500, "2024-03-01~2024-04-01", 2, 7, 0.2, "Medium", "Medium"],
        ],
    )
    importer_unified.run_import(xlsx.read_bytes(), "products.xlsx", status_cb=lambda **_: None)
    products = [row_to_dict(r) for r in database.list_products(conn)]
    assert len(products) == 2
    for p in products:
        assert p.get("winner_score") == 0
        assert p.get("source") == "products.xlsx"

def test_enqueue_post_import_tasks_dedupes(tmp_path, monkeypatch):
    conn = setup_env(tmp_path, monkeypatch)
    ids = [
        database.insert_product(
            conn,
            name=f"Prod{idx}",
            description="",
            category="",
            price=None,
            currency=None,
            image_url="",
            source="",
            extra={},
            product_id=idx,
        )
        for idx in range(1, 4)
    ]
    summary = web_app._enqueue_post_import_tasks(
        "task-xyz",
        ids + ids[:1],
        ["desire", "imputacion", "desire", "winner_score", "ignored"],
    )
    assert set(summary.keys()) == {"desire", "imputacion", "winner_score"}
    cur = conn.cursor()
    cur.execute("SELECT task_type, COUNT(*) FROM ai_task_queue GROUP BY task_type")
    rows = {row[0]: row[1] for row in cur.fetchall()}
    assert rows.get("desire") == len(ids)
    assert rows.get("imputacion") == len(ids)
    assert rows.get("winner_score") == len(ids)


def test_handle_ai_run_post_import(tmp_path, monkeypatch):
    conn = setup_env(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "get_api_key", lambda: "sk-test")
    monkeypatch.setattr(config, "get_model", lambda: "gpt-test")

    weight_prompts: list = []

    pid_a = database.insert_product(
        conn,
        name="ProdA",
        description="",
        category="Home",
        price=10.0,
        currency="USD",
        image_url="",
        source="",
        extra={"rating": 4.2, "units_sold": 120, "revenue": 5000, "conversion_rate": 0.12},
        product_id=1,
    )
    pid_b = database.insert_product(
        conn,
        name="ProdB",
        description="",
        category="Kitchen",
        price=20.0,
        currency="USD",
        image_url="",
        source="",
        extra={"rating": 3.5, "units_sold": 80, "revenue": 3000, "conversion_rate": 0.08},
        product_id=2,
    )
    database.enqueue_ai_tasks(conn, "desire", [pid_a, pid_b], import_task_id="task")
    database.enqueue_ai_tasks(conn, "imputacion", [pid_a, pid_b], import_task_id="task")

    def fake_generate_batch_columns(api_key, model, items):
        ok = {
            str(item["id"]): {
                "desire": f"Desire {item['id']}",
                "desire_magnitude": "High",
                "awareness_level": "Most Aware",
                "competition_level": "Low",
            }
            for item in items
        }
        return ok, {}, {"total_tokens": 123}, 0.5

    monkeypatch.setattr(gpt, "generate_batch_columns", fake_generate_batch_columns)

    body = json.dumps({"limit": 10}).encode("utf-8")

    class Dummy:
        def __init__(self, payload: bytes):
            self.headers = {"Content-Length": str(len(payload))}
            self.rfile = io.BytesIO(payload)
            self.wfile = io.BytesIO()
            self.path = "/api/ai/run_post_import"

        def _set_json(self, code=200):
            self.status = code

    handler = Dummy(body)
    web_app.RequestHandler.handle_ai_run_post_import(handler)
    resp = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert resp["ok"] is True
    assert resp["processed"] == 4
    assert resp["completed"] == 4
    assert resp["failed"] == 0
    assert resp["pending_left"] == 0
    assert set(resp.get("product_ids", [])) == {pid_a, pid_b}

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM ai_task_queue WHERE state='done'")
    assert cur.fetchone()[0] == 4
    prod_a = row_to_dict(database.get_product(conn, pid_a))
    assert prod_a.get("desire") == f"Desire {pid_a}"
    assert prod_a.get("desire_magnitude") == "High"


def test_run_post_import_auto(tmp_path, monkeypatch):
    conn = setup_env(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "get_api_key", lambda: "sk-test")
    monkeypatch.setattr(config, "get_model", lambda: "gpt-test")

    pid_a = database.insert_product(
        conn,
        name="ProdA",
        description="",
        category="Home",
        price=10.0,
        currency="USD",
        image_url="",
        source="",
        extra={"rating": 4.2},
        product_id=1,
    )
    pid_b = database.insert_product(
        conn,
        name="ProdB",
        description="",
        category="Kitchen",
        price=20.0,
        currency="USD",
        image_url="",
        source="",
        extra={"rating": 3.5},
        product_id=2,
    )

    def fake_desire_orchestrator(api_key, model, batch):
        return {
            "items": [
                {
                    "id": str(item["id"]),
                    "normalized_text": [
                        f"Auto Desire {item['id']}",
                        f"Linea {item['id']}",
                    ],
                    "keywords": [f"k{item['id']}", "growth"],
                }
                for item in batch
            ]
        }

    def fake_imputacion_orchestrator(api_key, model, batch):
        return {
            "items": [
                {
                    "id": str(item["id"]),
                    "review_count": 10 * int(item["id"]),
                    "image_count": 3,
                }
                for item in batch
            ]
        }

    winner_calls = []
    weight_prompts: list = []

    def fake_generate_winner_scores(conn_arg, product_ids=None, weights=None, debug=False):
        ids = list(product_ids or [])
        winner_calls.append(ids)
        assert weights is not None
        assert set(weights) == set(winner_score.ALLOWED_FIELDS)
        return {"processed": len(ids), "updated": len(ids)}

    monkeypatch.setattr(gpt, "orchestrate_desire_summary", fake_desire_orchestrator)
    monkeypatch.setattr(gpt, "orchestrate_imputation", fake_imputacion_orchestrator)
    monkeypatch.setattr(
        gpt,
        "recommend_weights_from_aggregates",
        lambda api_key, model, aggregates: (
            weight_prompts.append(aggregates)
            or {
                "weights": {k: 10 for k in winner_score.ALLOWED_FIELDS},
                "order": list(winner_score.ALLOWED_FIELDS),
                "notes": ["ok"],
                "version": "B.v2",
            }
        ),
    )
    monkeypatch.setattr(winner_score, "generate_winner_scores", fake_generate_winner_scores)

    task_id = "task-auto"
    web_app._update_import_status(
        task_id,
        state="RUNNING",
        stage="ai_post",
        post_import_ready=True,
        ai_progress=web_app._empty_ai_progress(),
    )

    web_app._run_post_import_auto(task_id, [pid_a, pid_b])

    status = web_app._get_import_status(task_id)
    assert status["state"] == "DONE"
    assert status["stage"] == "done"
    assert status["post_import_ready"] is False
    progress = status["ai_progress"]
    assert progress["desire"]["processed"] == 2
    assert progress["imputacion"]["processed"] == 2
    assert progress["winner_score"]["processed"] == 2
    assert weight_prompts and weight_prompts[0]["total_products"] == 2

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM ai_task_queue WHERE state='done'")
    assert cur.fetchone()[0] == 6

    prod_a = row_to_dict(database.get_product(conn, pid_a))
    assert prod_a.get("desire") == f"Auto Desire {pid_a}\nLinea {pid_a}"
    extra_a = json.loads(prod_a.get("extra") or "{}")
    assert extra_a.get("desire_keywords") == [f"k{pid_a}", "growth"]
    assert extra_a.get("review_count") == 10 * pid_a
    assert extra_a.get("image_count") == 3
    assert winner_calls and set(winner_calls[0]) == {pid_a, pid_b}

def test_runner_retries_batches(tmp_path, monkeypatch):
    conn = setup_env(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "get_api_key", lambda: "sk-test")
    monkeypatch.setattr(config, "get_model", lambda: "gpt-test")

    pid = database.insert_product(
        conn,
        name="RetryProd",
        description="",
        category="Home",
        price=15.0,
        currency="USD",
        image_url="",
        source="",
        extra={},
        product_id=1,
    )

    import_id = "task-retry"
    database.enqueue_ai_tasks(conn, "desire", [pid], import_task_id=import_id)

    calls = {"count": 0}

    def flaky_generate(api_key, model, items):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("boom")
        return {
            str(items[0]["id"]): {
                "desire": "Recovered",
                "desire_magnitude": "High",
                "awareness_level": "Most Aware",
                "competition_level": "Low",
            }
        }, {}, {}, 0.1

    monkeypatch.setattr(gpt, "generate_batch_columns", flaky_generate)

    summary = runner.run_auto({"desire"}, batch_size=1, max_parallel=1)
    assert calls["count"] >= 2
    assert import_id in summary
    desire_progress = summary[import_id]["tasks"]["desire"]
    assert desire_progress["processed"] == 1
    assert desire_progress["failed"] == 1
    assert not summary[import_id]["errors"]


def test_runner_respects_call_budget(tmp_path, monkeypatch):
    conn = setup_env(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "get_api_key", lambda: "sk-test")
    monkeypatch.setattr(config, "get_model", lambda: "gpt-test")
    monkeypatch.setattr(runner.settings, "AI_MAX_CALLS_PER_IMPORT", 1)
    monkeypatch.setattr(runner.settings, "AI_MIN_BATCH_SIZE", 1)
    monkeypatch.setattr(runner.settings, "AI_MAX_BATCH_SIZE", 1)
    monkeypatch.setattr(runner.settings, "AI_MAX_PARALLEL", 1)

    product_ids = [
        database.insert_product(
            conn,
            name=f"BudgetProd{i}",
            description="",
            category="Home",
            price=10.0 + i,
            currency="USD",
            image_url="",
            source="",
            extra={},
            product_id=i,
        )
        for i in range(1, 4)
    ]

    import_id = "task-budget"
    database.enqueue_ai_tasks(conn, "desire", product_ids, import_task_id=import_id)

    calls = {"count": 0}

    def fake_generate(api_key, model, items):
        calls["count"] += 1
        ok = {
            str(item["id"]): {
                "desire": f"Budget Desire {item['id']}",
                "desire_magnitude": "High",
                "awareness_level": "Most Aware",
                "competition_level": "Low",
            }
            for item in items
        }
        return ok, {}, {}, 0.1

    monkeypatch.setattr(gpt, "generate_batch_columns", fake_generate)

    summary = runner.run_auto({"desire"}, batch_size=1, max_parallel=1)

    assert calls["count"] == 1
    assert import_id in summary
    desire_summary = summary[import_id]["tasks"]["desire"]
    assert desire_summary["processed"] == 1
    assert desire_summary["skipped"] == len(product_ids) - 1

    cur = conn.cursor()
    cur.execute(
        "SELECT state, note FROM ai_task_queue WHERE import_task_id=? ORDER BY id",
        (import_id,),
    )
    rows = cur.fetchall()
    states = [row[0] for row in rows]
    notes = [row[1] for row in rows]
    assert states.count("done") == 1
    assert states.count("skipped") == len(product_ids) - 1
    assert all(note == "budget_exhausted" for note in notes if note)


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
            self.path = "/api/winner-score/generate"
        def _set_json(self, code=200):
            self.status = code

    handler = Dummy(body)
    web_app.RequestHandler.handle_scoring_v2_generate(handler)
    resp = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert resp["ok"] is True
    assert resp["processed"] == 2
    assert resp["updated"] == 2
    assert resp["weights_all"]
    assert resp["weights_eff"]
    prod_a = database.get_product(conn, pid_a)
    prod_b = database.get_product(conn, pid_b)
    assert 0 <= prod_a["winner_score"] <= 100
    assert 0 <= prod_b["winner_score"] <= 100

    body2 = json.dumps({"ids": [pid_b]})
    handler2 = Dummy(body2)
    web_app.RequestHandler.handle_scoring_v2_generate(handler2)
    resp2 = json.loads(handler2.wfile.getvalue().decode("utf-8"))
    assert resp2["ok"] is True
    assert resp2["processed"] == 1
    assert resp2["updated"] == 0
    assert resp2["weights_all"]
    assert resp2["weights_eff"]


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
            self.path = "/api/winner-score/generate"

        def _set_json(self, code=200):
            self.status = code

    handler = Dummy(body)
    web_app.RequestHandler.handle_scoring_v2_generate(handler)
    resp = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert resp["processed"] == 2
    assert resp["weights_all"]
    assert resp["weights_eff"]

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


def test_desire_serialization_and_logging(tmp_path, monkeypatch):
    conn = setup_env(tmp_path, monkeypatch)
    monkeypatch.setattr(web_app, "ensure_db", lambda: conn)
    pid1 = database.insert_product(
        conn,
        name="P1",
        description="",
        category="",
        price=1.0,
        currency=None,
        image_url="",
        source="",
        desire="Top",
        extra={},
        product_id=1,
    )
    pid2 = database.insert_product(
        conn,
        name="P2",
        description="",
        category="",
        price=2.0,
        currency=None,
        image_url="",
        source="",
        extra={"desire": "Extra"},
        product_id=2,
    )
    pid3 = database.insert_product(
        conn,
        name="P3",
        description="",
        category="",
        price=3.0,
        currency=None,
        image_url="",
        source="",
        extra={},
        product_id=3,
    )

    class Dummy:
        def __init__(self, path):
            self.path = path
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

    handler = Dummy("/products")
    web_app.RequestHandler.do_GET(handler)
    resp = json.loads(handler.wfile.getvalue().decode("utf-8"))
    p1 = next(p for p in resp if p["id"] == pid1)
    p2 = next(p for p in resp if p["id"] == pid2)
    p3 = next(p for p in resp if p["id"] == pid3)
    assert p1["desire"] == "Top"
    assert isinstance(p1["price"], (int, float))
    assert p2["desire"] == "Extra"
    assert p2["extras"].get("desire") == "Extra"
    assert p3["desire"] is None
    log_text = web_app.LOG_PATH.read_text()
    assert f"desire_missing=true" in log_text and f"product={pid3}" in log_text

    lid = database.create_list(conn, "L")
    database.add_product_to_list(conn, lid, pid1)
    handler_list = Dummy(f"/list/{lid}")
    web_app.RequestHandler.do_GET(handler_list)
    resp_list = json.loads(handler_list.wfile.getvalue().decode("utf-8"))
    assert resp_list and resp_list[0]["desire"] == "Top"


def test_patch_product_desire(tmp_path, monkeypatch):
    conn = setup_env(tmp_path, monkeypatch)
    monkeypatch.setattr(web_app, "ensure_db", lambda: conn)
    pid = database.insert_product(
        conn,
        name="P1",
        description="",
        category="",
        price=1.0,
        currency=None,
        image_url="",
        source="",
        desire="old",
    )
    body = json.dumps({"desire": "nuevo texto", "unknown": "x"})

    class Dummy:
        def __init__(self, path, body):
            self.path = path
            self.headers = {"Content-Length": str(len(body))}
            self.rfile = io.BytesIO(body.encode("utf-8"))
            self.wfile = io.BytesIO()

        def _set_json(self, code=200):
            self.status = code

    handler = Dummy(f"/api/products/{pid}", body)
    web_app.RequestHandler.do_PATCH(handler)
    resp = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 200
    assert resp["desire"] == "nuevo texto"
    assert database.get_product(conn, pid)["desire"] == "nuevo texto"

    handler_nf = Dummy("/api/products/9999", body)
    web_app.RequestHandler.do_PATCH(handler_nf)
    assert handler_nf.status == 404
    log_text = web_app.LOG_PATH.read_text()
    assert "PATCH not found product_id=9999" in log_text

def test_patch_winner_weights_persists(tmp_path, monkeypatch):
    setup_env(tmp_path, monkeypatch)

    body = json.dumps({"weights": {"rating": 25}, "order": ["rating", "price"]})

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
    assert resp["weights"]["rating"] == 25
    assert resp["order"][0] == "rating"
    from product_research_app.services.config import get_winner_weights_raw, get_winner_order_raw
    data = get_winner_weights_raw()
    order = get_winner_order_raw()
    assert data.get("rating") == 25
    assert order[0] == "rating"

def test_config_oldness_preference_roundtrip(tmp_path, monkeypatch):
    setup_env(tmp_path, monkeypatch)

    class DummyGet:
        def __init__(self):
            self.path = "/config"
            self.headers = {}
            self.rfile = io.BytesIO()
            self.wfile = io.BytesIO()

        def _set_json(self, code=200):
            self.status = code

    g = DummyGet()
    web_app.RequestHandler.do_GET(g)
    resp = json.loads(g.wfile.getvalue().decode("utf-8"))
    assert resp.get("oldness_preference") == "newer"

    body = json.dumps({"oldness_preference": "older"})

    class DummyPost:
        def __init__(self, body):
            self.path = "/setconfig"
            self.headers = {"Content-Length": str(len(body))}
            self.rfile = io.BytesIO(body.encode("utf-8"))
            self.wfile = io.BytesIO()

        def _set_json(self, code=200):
            self.status = code

    p = DummyPost(body)
    web_app.RequestHandler.handle_setconfig(p)
    assert p.status == 200
    assert config.load_config().get("oldness_preference") == "older"

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
        assert all(0 <= v <= 100 for v in weights.values())
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
        body = json.dumps({"weights": {key: value}})
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
        assert resp["weights"][key] == value

    patch_weight("rating", 20)
    patch_weight("price", 30)
    patch_weight("units_sold", 50)
    from product_research_app.services.config import get_winner_weights_raw

    weights = get_winner_weights_raw()
    assert weights.get("rating") == 20
    assert weights.get("price") == 30
    assert weights.get("units_sold") == 50
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
    importer_unified.run_import(xlsx.read_bytes(), "batch.xlsx", status_cb=lambda **_: None)

    prod_old = database.get_product(conn, pid)
    assert prod_old["winner_score"] == 10
    cur = conn.execute("SELECT id, winner_score FROM products WHERE id != ?", (pid,))
    rows = cur.fetchall()
    assert rows
    for _id, score in rows:
        assert score == 0


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
            self.path = "/api/winner-score/generate"

        def _set_json(self, code=200):
            self.status = code

    h = Dummy(body)
    web_app.RequestHandler.handle_scoring_v2_generate(h)
    json.loads(h.wfile.getvalue().decode("utf-8"))
    for pid in subset:
        assert database.get_product(conn, pid)["winner_score"] != 1
    for pid in ids[5:]:
        assert database.get_product(conn, pid)["winner_score"] == 1

    body2 = json.dumps({})
    h2 = Dummy(body2)
    web_app.RequestHandler.handle_scoring_v2_generate(h2)
    json.loads(h2.wfile.getvalue().decode("utf-8"))
    for pid in ids:
        assert database.get_product(conn, pid)["winner_score"] != 1


def test_weights_hash_changes_after_patch(tmp_path, monkeypatch):
    conn = setup_env(tmp_path, monkeypatch)
    monkeypatch.setattr(web_app, "ensure_db", lambda: conn)

    pid = database.insert_product(
        conn,
        name="P0",
        description="",
        category="",
        price=10.0,
        currency=None,
        image_url="",
        source="",
        extra={"rating": 4.5},
        product_id=1,
    )
    conn.execute("UPDATE products SET winner_score=1 WHERE id=?", (pid,))
    conn.commit()

    body = json.dumps({})

    class Dummy:
        def __init__(self, body):
            self.headers = {"Content-Length": str(len(body))}
            self.rfile = io.BytesIO(body.encode("utf-8"))
            self.wfile = io.BytesIO()
            self.path = "/api/winner-score/generate"

        def _set_json(self, code=200):
            self.status = code

    h1 = Dummy(body)
    web_app.RequestHandler.handle_scoring_v2_generate(h1)
    resp1 = json.loads(h1.wfile.getvalue().decode("utf-8"))
    hash1 = resp1["weights_all"]
    eff1 = resp1["weights_eff"]
    ver1 = config.get_weights_version()

    body_patch = json.dumps({"weights": {"rating": 20}})

    class Patcher:
        def __init__(self, body):
            self.path = "/api/config/winner-weights"
            self.headers = {"Content-Length": str(len(body))}
            self.rfile = io.BytesIO(body.encode("utf-8"))
            self.wfile = io.BytesIO()

        def _set_json(self, code=200):
            self.status = code

    p = Patcher(body_patch)
    web_app.RequestHandler.do_PATCH(p)

    h2 = Dummy(body)
    web_app.RequestHandler.handle_scoring_v2_generate(h2)
    resp2 = json.loads(h2.wfile.getvalue().decode("utf-8"))
    assert resp2["weights_all"] != hash1
    assert resp2["weights_eff"] != eff1
    assert config.get_weights_version() > ver1

def test_logging_and_explain_endpoint(tmp_path, monkeypatch):
    conn = setup_env(tmp_path, monkeypatch)
    monkeypatch.setattr(web_app, "ensure_db", lambda: conn)
    web_app.DEBUG = True

    config.update_weight("oldness", 2.0)

    pid = database.insert_product(
        conn,
        name="L",
        description="",
        category="",
        price=None,
        currency=None,
        image_url="",
        source="",
        extra={"rating": 4.0},
        product_id=1,
    )

    body = json.dumps({"ids": [pid]})

    class DummyGen:
        def __init__(self, body):
            self.headers = {"Content-Length": str(len(body))}
            self.rfile = io.BytesIO(body.encode("utf-8"))
            self.wfile = io.BytesIO()
            self.path = "/api/winner-score/generate"

        def _set_json(self, code=200):
            self.status = code

    handler = DummyGen(body)
    web_app.RequestHandler.handle_scoring_v2_generate(handler)

    log_text = web_app.LOG_PATH.read_text()
    assert "oldness" in log_text
    assert "effective_weights={" in log_text
    assert "order=" in log_text

    parsed = urlparse(f"/api/winner-score/explain?ids={pid}")

    class DummyExplain:
        def __init__(self):
            self.wfile = io.BytesIO()

        def _set_json(self, code=200):
            self.status = code

    handler2 = DummyExplain()
    web_app.RequestHandler.handle_winner_score_explain(handler2, parsed)
    resp = json.loads(handler2.wfile.getvalue().decode("utf-8"))
    info = resp[str(pid)]
    assert "rating" in info["present"]
    assert "oldness" in info["missing"]
    eff = info["effective_weights"]
    assert set(eff.keys()) == set(winner_score.ALLOWED_FIELDS)
    assert abs(sum(eff.values()) - 1.0) < 1e-2


def test_weights_eff_stable_when_touching_missing_metric(tmp_path, monkeypatch):
    conn = setup_env(tmp_path, monkeypatch)
    monkeypatch.setattr(web_app, "ensure_db", lambda: conn)

    pid = database.insert_product(
        conn,
        name="MM",
        description="",
        category="",
        price=None,
        currency=None,
        image_url="",
        source="",
        extra={"rating": 4.5},
        product_id=1,
    )
    conn.commit()

    config.update_weight("rating", 0.0)

    body = json.dumps({"product_ids": [pid]})

    class Dummy:
        def __init__(self, body, path="/api/winner-score/generate?debug=1"):
            self.path = path
            self.headers = {"Content-Length": str(len(body))}
            self.rfile = io.BytesIO(body.encode("utf-8"))
            self.wfile = io.BytesIO()

        def _set_json(self, code=200):
            self.status = code

    h1 = Dummy(body)
    web_app.RequestHandler.handle_scoring_v2_generate(h1)
    resp1 = json.loads(h1.wfile.getvalue().decode("utf-8"))
    hash_all1 = resp1["weights_all"]
    hash_eff1 = resp1["weights_eff"]
    sum1 = resp1["diag"]["sum_filtered"]
    assert sum1 > 0.0

    body_patch = json.dumps({"weights": {"units_sold": 20}})
    class Patcher:
        def __init__(self, body):
            self.path = "/api/config/winner-weights"
            self.headers = {"Content-Length": str(len(body))}
            self.rfile = io.BytesIO(body.encode("utf-8"))
            self.wfile = io.BytesIO()

        def _set_json(self, code=200):
            self.status = code

    p = Patcher(body_patch)
    web_app.RequestHandler.do_PATCH(p)

    h2 = Dummy(body)
    web_app.RequestHandler.handle_scoring_v2_generate(h2)
    resp2 = json.loads(h2.wfile.getvalue().decode("utf-8"))
    assert resp2["weights_all"] != hash_all1
    assert resp2["weights_eff"] != hash_eff1
    assert resp2["diag"]["sum_filtered"] > sum1
