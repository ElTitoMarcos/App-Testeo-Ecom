import json

from product_research_app import gpt
from product_research_app.prompts import registry


def test_build_messages_task_a():
    context = {
        "products": [
            {"id": 1, "name": "Foco led"},
            {"id": 2, "name": "Silla ergonómica"},
        ]
    }
    messages = gpt.build_messages("A", context_json=context)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == registry.PROMPT_MASTER_V3_SYSTEM
    assert messages[1]["role"] == "user"
    user_payload = messages[1]["content"]
    assert user_payload.startswith(registry.PROMPT_A)
    assert "### CONTEXT_JSON" in user_payload
    json_block = user_payload.split("### CONTEXT_JSON\n", 1)[1]
    parsed = json.loads(json_block)
    assert parsed == context


def test_build_messages_task_b():
    aggregates = {"scores": {"momentum": 70, "margin": 55}}
    messages = gpt.build_messages("B", aggregates=aggregates)
    assert len(messages) == 2
    assert messages[0]["content"] == registry.PROMPT_MASTER_V3_SYSTEM
    user_payload = messages[1]["content"]
    assert user_payload.startswith(registry.PROMPT_B)
    assert "### AGGREGATES" in user_payload
    assert "### CONTEXT_JSON" not in user_payload
    assert "### DATA" not in user_payload
    json_block = user_payload.split("### AGGREGATES\n", 1)[1]
    parsed = json.loads(json_block)
    assert parsed == aggregates


def test_prepare_params_with_schema_json_mode():
    schema = {"name": "demo", "schema": {"type": "object"}, "strict": True}
    payload = gpt.prepare_params(
        model="gpt-5-mini",
        messages=[{"role": "system", "content": "hola"}],
        strict_json=True,
        json_schema=schema,
    )
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"] == schema
    assert "temperature" not in payload
