import logging
from datetime import datetime, timedelta
from pathlib import Path

from product_research_app import web_app, database, config
from product_research_app.services import trends_service
from product_research_app.services import config as cfg_service


def setup_env(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "DB_PATH", tmp_path / "data.sqlite3")
    monkeypatch.setattr(web_app, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(web_app, "LOG_PATH", tmp_path / "logs" / "app.log")
    web_app.LOG_DIR.mkdir(exist_ok=True)
    for h in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(h)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(web_app.LOG_PATH, encoding="utf-8")], force=True)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg_service, "DB_PATH", tmp_path / "data.sqlite3")
    cfg_service.init_app_config()
    return web_app.ensure_db()


def test_trends_no_data(tmp_path, monkeypatch):
    setup_env(tmp_path, monkeypatch)
    monkeypatch.setattr(trends_service, "DB_PATH", tmp_path / "data.sqlite3")
    start = datetime.utcnow() - timedelta(days=1)
    end = datetime.utcnow()
    res = trends_service.get_trends_summary(start, end)
    assert res["categories"] == []
    assert res["timeseries"] == []
    assert res["totals"]["units"] == 0
    assert res["totals"]["revenue"] == 0


def test_trends_with_data_delta(tmp_path, monkeypatch):
    conn = setup_env(tmp_path, monkeypatch)
    monkeypatch.setattr(trends_service, "DB_PATH", tmp_path / "data.sqlite3")
    database.insert_product(
        conn,
        name="P1",
        description="",
        category="Cat/Sub",
        price=10.0,
        currency=None,
        image_url="",
        source="",
        extra={"units_sold": 2, "revenue": 20.0, "rating": 4.0},
    )
    start = datetime.utcnow() - timedelta(days=1)
    end = datetime.utcnow() + timedelta(days=1)
    res = trends_service.get_trends_summary(start, end)
    assert res["categories"]
    cat = res["categories"][0]
    assert "delta_revenue_pct" in cat
    assert cat["delta_revenue_pct"] == 0
    assert res["timeseries"]
    totals = res["totals"]
    assert "delta_revenue_pct" in totals
    assert "delta_units_pct" in totals
