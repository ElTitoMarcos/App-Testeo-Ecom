import json

from product_research_app.ai import gpt_orchestrator


def _context_from_prompt(prompt: str) -> dict:
    marker = "### CONTEXTO JSON\n"
    if marker not in prompt:
        return {}
    after = prompt.split(marker, 1)[1]
    split_marker = "\n\n### INSTRUCCIONES DE FORMATO"
    if split_marker in after:
        json_part = after.split(split_marker, 1)[0]
    else:
        json_part = after
    return json.loads(json_part)


def test_consulta_chunking_and_refs(monkeypatch):
    responses = [
        {
            "content": "Resumen 1\n```json\n{\"refs\":[{\"id\":\"1\",\"category\":\"A\"}],\"prompt_version\":\"v1\"}\n```",
            "usage": 120,
        },
        {
            "content": "Resumen 2\n```json\n{\"refs\":[{\"id\":\"2\",\"category\":\"A\"}],\"prompt_version\":\"v2\"}\n```",
            "usage": 130,
        },
    ]
    captured_contexts = []

    def fake_call(model, prompt, api_key, timeout, system_prompt):
        context = _context_from_prompt(prompt)
        captured_contexts.append(context)
        return responses.pop(0)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MAX_ITEMS", "2")
    monkeypatch.setattr(gpt_orchestrator, "_call_openai", fake_call)

    payload = {"products": [{"id": "1"}, {"id": "2"}, {"id": "3"}]}
    result = gpt_orchestrator.run_task("consulta", prompt_text="Analiza", json_payload=payload)

    assert result["ok"] is True
    assert result["meta"]["calls"] == 2
    assert len(captured_contexts) == 2
    assert len(captured_contexts[0]["products"]) == 2
    assert len(captured_contexts[1]["products"]) == 1
    refs = result["data"]["refs"]
    assert {ref["id"] for ref in refs} == {"1", "2"}
    assert result["data"]["prompt_version"] == "v2"


def test_pesos_uses_aggregated_summary(monkeypatch):
    captured = {}

    def fake_call(model, prompt, api_key, timeout, system_prompt):
        context = _context_from_prompt(prompt)
        captured["context"] = context
        return {
            "content": "Hecho\n```json\n{\"weights\":{\"score\":0.7},\"prompt_version\":\"2024-01\"}\n```",
            "usage": 88,
        }

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MAX_ITEMS", "5")
    monkeypatch.setattr(gpt_orchestrator, "_call_openai", fake_call)

    payload = {
        "products": [
            {"id": "p1", "price": 10, "metrics": {"score": 1}, "title": "Alpha"},
            {"id": "p2", "price": 15, "metrics": {"score": 3}, "title": "Beta"},
            {"id": "p3", "price": 20, "metrics": {"score": 2}, "title": "Gamma"},
        ]
    }
    result = gpt_orchestrator.run_task("pesos", prompt_text="Calibra", json_payload=payload)

    context = captured["context"]
    assert "products" not in context
    aggregates = context["aggregates"]
    metrics = aggregates["metrics"]
    assert "score" in metrics and "price" in metrics
    score_stats = metrics["score"]
    assert score_stats["min"] == 1.0
    assert score_stats["max"] == 3.0
    assert score_stats["top_ids"][0] == "p2"
    assert score_stats["bottom_ids"][0] == "p1"
    assert aggregates["total_products"] == 3
    assert context.get("sample_titles") == ["Alpha", "Beta", "Gamma"]
    assert result["ok"] is True
    assert result["data"]["weights"]["score"] == 0.7
    assert result["data"]["prompt_version"] == "2024-01"


def test_desire_batches_into_mapping(monkeypatch):
    responses = [
        {
            "content": "Bloque1\n```json\n{\"results\":{\"a\":{\"desire\":\"Alta\"}},\"prompt_version\":\"v1\"}\n```",
            "usage": None,
        },
        {
            "content": "Bloque2\n```json\n{\"results\":{\"b\":{\"desire\":\"Media\"},\"c\":{\"desire\":\"Baja\"}},\"prompt_version\":\"v2\"}\n```",
            "usage": None,
        },
    ]
    contexts = []

    def fake_call(model, prompt, api_key, timeout, system_prompt):
        context = _context_from_prompt(prompt)
        contexts.append(context)
        return responses.pop(0)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MAX_ITEMS", "2")
    monkeypatch.setattr(gpt_orchestrator, "_call_openai", fake_call)

    payload = {"products": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}
    result = gpt_orchestrator.run_task("desire", prompt_text="Completa", json_payload=payload)

    assert len(contexts) == 2
    assert len(contexts[0]["products"]) == 2
    assert len(contexts[1]["products"]) == 1
    data = result["data"]
    assert data["results"] == {
        "a": {"desire": "Alta"},
        "b": {"desire": "Media"},
        "c": {"desire": "Baja"},
    }
    assert data["prompt_version"] == "v2"
    assert result["ok"] is True
