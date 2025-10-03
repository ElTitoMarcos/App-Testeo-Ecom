import asyncio

from product_research_app import gpt
from product_research_app.services import ai_columns


def test_refine_desire_fallback_low_confidence(monkeypatch):
    candidate = ai_columns.Candidate(
        id=701,
        sig_hash="hash701",
        payload={"name": "Producto"},
        extra={"title": "Producto"},
    )

    calls = []

    async def fake_call_prompt(*args, **kwargs):
        calls.append(kwargs.get("extra_user"))
        if len(calls) == 1:
            exc = gpt.InvalidJSONError("vacío")
            setattr(exc, "sanitized_text", "")
            raise exc
        return {"content": {}}

    monkeypatch.setattr(ai_columns.gpt, "call_prompt_task_async", fake_call_prompt)

    result = asyncio.run(ai_columns._refine_desire_statement(candidate, ""))

    assert result["low_confidence"] is True
    assert result["desire_statement"] == ""
    assert len(calls) == 2
    assert "Responde exclusivamente" in (calls[1] or "")
