from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import pytest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from product_research_app.services import ai_columns


def test_microbatch_splits_until_single_item(monkeypatch):
    candidates: List[ai_columns.Candidate] = [
        ai_columns.Candidate(
            id=101,
            sig_hash="a",
            payload={"name": "Item 101"},
            extra={"title": "Item 101"},
        ),
        ai_columns.Candidate(
            id=102,
            sig_hash="b",
            payload={"name": "Item 102"},
            extra={"title": "Item 102"},
        ),
    ]

    batch = ai_columns._build_batch_request(
        "001",
        candidates,
        trunc_title=120,
        trunc_desc=240,
    )

    responses: List[Dict[str, Any]] = []

    truncated_response = {
        "choices": [
            {
                "message": {"content": ""},
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 800},
    }
    responses.append(truncated_response)

    def _success_payload(pid: int) -> Dict[str, Any]:
        item = {
            "id": pid,
            "desire": 0.8,
            "desire_reason": "Alta demanda",
            "competition": 0.2,
            "competition_level": "Low",
            "revenue": 120.0,
            "units_sold": 10,
            "price": 20.0,
            "oldness": 0.1,
            "rating": 4.5,
        }
        return {
            "choices": [
                {
                    "message": {
                        "content": [
                            {
                                "type": "json",
                                "json": {"items": [item]},
                            }
                        ]
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 200},
        }

    responses.append(_success_payload(101))
    responses.append(_success_payload(102))

    call_sizes: List[int] = []

    async def fake_call_gpt_async(*, messages: List[Dict[str, Any]], **kwargs):
        user_content = messages[-1]["content"]
        idx = user_content.rfind("[")
        payload_count = 0
        if idx >= 0:
            payload = json.loads(user_content[idx:])
            if isinstance(payload, list):
                payload_count = len(payload)
        call_sizes.append(payload_count)
        if not responses:
            raise AssertionError("Se esperaban más respuestas simuladas")
        return responses.pop(0)

    async def fake_refine(*args, **kwargs):
        return None

    monkeypatch.setattr(ai_columns.gpt, "call_gpt_async", fake_call_gpt_async)

    monkeypatch.setattr(ai_columns, "_refine_desire_statement", fake_refine)

    async def run(batch_req: ai_columns.BatchRequest) -> Dict[str, Any]:
        try:
            return await ai_columns._call_batch_with_retries(
                batch_req,
                api_key="test",
                model="gpt-test",
                max_retries=0,
            )
        except ai_columns.BatchAdaptationRequired as exc:
            assert exc.reason == "json_truncated"
            assert len(batch_req.candidates) > 1
            mid = max(1, len(batch_req.candidates) // 2)
            first = ai_columns._build_batch_request(
                f"{batch_req.req_id}a",
                batch_req.candidates[:mid],
                batch_req.trunc_title,
                batch_req.trunc_desc,
                depth=batch_req.depth + 1,
                json_retry_count=batch_req.json_retry_count,
                adapted=True,
            )
            second = ai_columns._build_batch_request(
                f"{batch_req.req_id}b",
                batch_req.candidates[mid:],
                batch_req.trunc_title,
                batch_req.trunc_desc,
                depth=batch_req.depth + 1,
                json_retry_count=batch_req.json_retry_count,
                adapted=True,
            )
            results = []
            for child in (first, second):
                if child.candidates:
                    results.append(await run(child))
            aggregate: Dict[str, Any] = {"ok": {}, "ko": {}}
            for entry in results:
                aggregate["ok"].update(entry.get("ok", {}))
                aggregate["ko"].update(entry.get("ko", {}))
            return aggregate

    result = asyncio.run(run(batch))

    assert sorted(result["ok"].keys()) == ["101", "102"]
    assert result["ko"] == {}
    assert call_sizes == [2, 1, 1]


def test_single_item_truncation_ups_max_tokens(monkeypatch):
    candidate = ai_columns.Candidate(
        id=501,
        sig_hash="sig501",
        payload={"name": "Prod 501"},
        extra={"title": "Prod 501"},
    )

    batch = ai_columns._build_batch_request(
        "solo-501",
        [candidate],
        trunc_title=120,
        trunc_desc=240,
    )

    responses: List[Dict[str, Any]] = [
        {
            "choices": [
                {
                    "message": {"content": ""},
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 600},
        },
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {
                                "type": "json",
                                "json": {
                                    "items": [
                                        {
                                            "id": 501,
                                            "desire_statement": "Alta demanda",
                                            "desire_magnitude": "High",
                                            "competition_level": "Low",
                                        }
                                    ]
                                },
                            }
                        ]
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 200},
        },
    ]

    max_tokens_used: List[int] = []

    async def fake_call_gpt_async(*, max_tokens: int | None = None, **kwargs):
        max_tokens_used.append(max_tokens if max_tokens is not None else 0)
        if not responses:
            raise AssertionError("No quedan respuestas simuladas")
        return responses.pop(0)

    monkeypatch.setattr(ai_columns.gpt, "call_gpt_async", fake_call_gpt_async)

    def fake_parse(raw: Dict[str, Any], expected_ids):
        parsed = {
            int(pid): {
                "desire_statement": "Alta demanda",
                "desire_magnitude": "High",
                "competition_level": "Low",
            }
            for pid in expected_ids
        }
        return parsed, json.dumps({"items": []})

    monkeypatch.setattr(ai_columns, "_parse_strict_json_payload", fake_parse)

    async def fake_finalize(batch_req, strict_map):
        return {str(candidate.id): {}}, {}

    monkeypatch.setattr(ai_columns, "_finalize_batch_payload", fake_finalize)

    result = asyncio.run(
        ai_columns._call_batch_with_retries(
            batch,
            api_key="test",
            model="gpt-test",
            max_retries=0,
        )
    )

    assert str(candidate.id) not in result.get("ko", {})
    assert len(max_tokens_used) == 2
    assert max_tokens_used[1] >= ai_columns.MAX_COMPLETION_TOKENS_JSON
    assert min(max_tokens_used) >= ai_columns.MIN_COMPLETION_TOKENS_JSON

