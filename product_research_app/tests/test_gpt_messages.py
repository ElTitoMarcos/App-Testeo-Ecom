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


def test_parse_message_content_basic_string():
    raw = {"choices": [{"message": {"content": "{\"foo\": 1}"}}]}
    parsed, text = gpt._parse_message_content(raw)
    assert parsed == {"foo": 1}
    assert text == '{"foo": 1}'


def test_parse_message_content_code_fence_and_list():
    raw = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "```json\n{\"bar\": 2}\n```"},
                    ]
                }
            }
        ]
    }
    parsed, text = gpt._parse_message_content(raw)
    assert parsed == {"bar": 2}
    assert text == '{"bar": 2}'


def test_parse_message_content_tool_call_arguments():
    raw = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "foo", "arguments": "{\"baz\": 3}"},
                        }
                    ],
                }
            }
        ]
    }
    parsed, text = gpt._parse_message_content(raw)
    assert parsed == {"baz": 3}
    assert text == '{"baz": 3}'


def test_parse_message_content_uses_parsed_field():
    raw = {
        "choices": [
            {
                "message": {
                    "parsed": {"qux": 4},
                    "content": None,
                }
            }
        ]
    }
    parsed, text = gpt._parse_message_content(raw)
    assert parsed == {"qux": 4}
    assert json.loads(text) == {"qux": 4}
