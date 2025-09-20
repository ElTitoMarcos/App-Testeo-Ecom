import re

from product_research_app.services.ai_columns import RNG_SENTINEL, _build_payload


def test_build_payload_template_replaces_placeholders():
    batch = [
        {
            "id": 123,
            "title": "Widget",
            "description": "Descripción extensa",
            "category": "Gadgets",
            "brand": "Acme",
            "price": 19.99,
            "rating": 4.7,
            "units_sold": 42,
            "revenue": 1234.56,
            "oldness": 0.3,
        }
    ]
    weights = {"desire": 0.8, "winner_score": 0.6}

    payload = _build_payload(batch, weights)
    messages = payload["messages"]
    assert len(messages) >= 2
    user_content = messages[1]["content"]

    assert '{"results":' in user_content
    assert "0-100" in user_content
    assert not re.findall(r"\{[A-Za-z_][A-Za-z0-9_]*\}", user_content)
    assert RNG_SENTINEL not in user_content
