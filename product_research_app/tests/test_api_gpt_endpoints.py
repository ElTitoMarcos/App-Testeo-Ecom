import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from product_research_app.api import app
from product_research_app.api import gpt_endpoints


def test_consulta_endpoint_filters_products_and_uses_template(monkeypatch):
    client = app.test_client()
    captured = {}

    def fake_run_task(task, *, prompt_text, json_payload, model_hint=None, system_prompt=None):
        captured["task"] = task
        captured["prompt_text"] = prompt_text
        captured["json_payload"] = json_payload
        captured["system_prompt"] = system_prompt
        return {
            "ok": True,
            "text": "Análisis",
            "data": {"refs": [{"id": "1"}], "prompt_version": "v1"},
            "warnings": [],
            "meta": {"chunks": 1},
            "model": "gpt-4o-mini",
        }

    monkeypatch.setattr(gpt_endpoints.gpt_orchestrator, "run_task", fake_run_task)

    body = {
        "prompt_text": "   Hola ",
        "context": {
            "group_id": "g1",
            "time_window": "ultima_semana",
            "products": [
                {
                    "id": 1,
                    "price": 10,
                    "title": "Prod1",
                    "email": "user@example.com",
                    "group_id": "g1",
                },
                {
                    "id": 2,
                    "price": 20,
                    "title": "Prod2",
                    "group_id": "g2",
                },
                {
                    "id": "3",
                    "price": 30,
                    "title": "Prod3",
                    "description": "Algo",
                    "groupId": "g1",
                },
            ],
            "visible_ids": [1, None, "3"],
        },
        "params": {"tone": "casual"},
    }

    response = client.post("/api/gpt/consulta", data=json.dumps(body), content_type="application/json")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["text"] == "Análisis"
    assert payload["data"]["prompt_version"] == "v1"

    assert captured["task"] == "consulta"
    assert captured["prompt_text"] == "Hola"
    sanitized = captured["json_payload"]["products"]
    assert sanitized == [
        {"id": "1", "price": 10, "title": "Prod1"},
        {"id": "3", "price": 30, "title": "Prod3", "description": "Algo"},
    ]
    assert captured["json_payload"]["group_id"] == "g1"
    assert captured["json_payload"]["visible_ids"] == ["1", "3"]
    assert captured["json_payload"]["params"] == {"tone": "casual"}
    expected_system = gpt_endpoints._get_system_prompt("consulta")
    assert captured["system_prompt"] == expected_system


def test_pesos_endpoint_uses_default_prompt(monkeypatch):
    client = app.test_client()
    captured = {}

    def fake_run_task(task, *, prompt_text, json_payload, model_hint=None, system_prompt=None):
        captured["task"] = task
        captured["prompt_text"] = prompt_text
        captured["json_payload"] = json_payload
        return {
            "ok": True,
            "text": "Resumen",
            "data": {"prompt_version": "v1"},
            "warnings": [],
            "meta": {"chunks": 1},
            "model": "gpt-4o",
        }

    monkeypatch.setattr(gpt_endpoints.gpt_orchestrator, "run_task", fake_run_task)

    body = {
        "prompt_text": "",
        "context": {
            "group_id": None,
            "time_window": None,
            "products": [
                {"id": "A", "price": 12, "store": "Shop", "title": "Prod 1"},
                {"id": "B", "price": 22, "store": "Shop", "title": "Prod 2", "extra": "ignore"},
            ],
        },
        "params": {},
    }

    response = client.post("/api/gpt/pesos", json=body)
    assert response.status_code == 200
    assert captured["task"] == "pesos"
    assert captured["prompt_text"] == gpt_endpoints._DEFAULT_PROMPTS["pesos"]
    payload_products = captured["json_payload"]["products"]
    assert set(payload_products.keys()) == {"aggregates", "sample_titles"}
    aggregates = payload_products["aggregates"]
    assert aggregates["total_products"] == 2
    price_stats = aggregates["metrics"]["price"]
    assert price_stats["min"] == 12.0
    assert price_stats["max"] == 22.0
    assert price_stats["coverage"] == 1.0
    assert payload_products["sample_titles"] == ["Prod 1", "Prod 2"]


def test_imputacion_endpoint_shapes_imputed_payload(monkeypatch):
    client = app.test_client()
    captured = {}

    def fake_run_task(task, *, prompt_text, json_payload, model_hint=None, system_prompt=None):
        captured["task"] = task
        captured["json_payload"] = json_payload
        return {
            "ok": True,
            "text": "Resumen",
            "data": {
                "prompt_version": "v2",
                "results": {
                    "42": {
                        "review_count": {"value": "10"},
                        "image_count": -3,
                        "profit_margin": 0.35,
                        "otro": 99,
                    },
                    "ghost": {"review_count": 20},
                },
            },
            "warnings": [],
            "meta": {"chunks": 1},
            "model": "gpt-4o-mini",
        }

    monkeypatch.setattr(gpt_endpoints.gpt_orchestrator, "run_task", fake_run_task)

    body = {
        "prompt_text": "",
        "context": {
            "products": [
                {"id": 42, "title": "Producto"},
            ],
        },
    }

    response = client.post("/api/gpt/imputacion", json=body)
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["text"] == "Resumen"
    data = payload["data"]
    assert data["prompt_version"] == "v2"
    imputed = data["imputed"]
    assert "42" in imputed and "ghost" not in imputed
    fields = imputed["42"]
    assert fields["review_count"]["value"] == 10
    assert fields["image_count"]["value"] == 0
    assert fields["profit_margin"]["value"] == 0.35
    assert fields["profit_margin"]["confidence"] == "low_confidence"
    warnings = payload["warnings"]
    assert any("Producto 42: image_count ajustado a 0" in w for w in warnings)
    assert any("Producto ghost fuera del contexto" in w for w in warnings)
    assert captured["task"] == "imputacion"
    assert captured["json_payload"]["products"] == [{"id": "42", "title": "Producto"}]


def test_invalid_body_returns_400():
    client = app.test_client()
    response = client.post("/api/gpt/consulta", data="not-json", content_type="application/json")
    assert response.status_code == 400
    payload = response.get_json()
    assert payload["ok"] is False
    assert "error" in payload
