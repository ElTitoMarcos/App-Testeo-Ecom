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
    assert messages[0]["content"] == registry.PROMPT_MASTER_V4_SYSTEM
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
    assert messages[0]["content"] == registry.PROMPT_MASTER_V4_SYSTEM
    user_payload = messages[1]["content"]
    assert user_payload.startswith(registry.PROMPT_B)
    assert "### AGGREGATES" in user_payload
    assert "### CONTEXT_JSON" not in user_payload
    assert "### DATA" not in user_payload
    json_block = user_payload.split("### AGGREGATES\n", 1)[1]
    parsed = json.loads(json_block)
    assert parsed == aggregates


def test_build_messages_task_desire():
    context = {"notes": "dato"}
    data = {"signals": ["alza"]}
    messages = gpt.build_messages("desire", context_json=context, data=data)
    assert len(messages) == 2
    assert messages[0]["content"] == registry.PROMPT_MASTER_V4_SYSTEM
    user_payload = messages[1]["content"]
    assert user_payload.startswith(registry.PROMPT_DESIRE)
    assert "### AGGREGATES" not in user_payload
    assert "### CONTEXT_JSON\n" in user_payload
    assert "### DATA\n" in user_payload
    _, context_and_data = user_payload.rsplit("### CONTEXT_JSON\n", 1)
    context_block, data_section = context_and_data.split("\n\n### DATA\n", 1)
    data_block = data_section
    assert json.loads(context_block) == context
    assert json.loads(data_block) == data
