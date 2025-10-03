from __future__ import annotations

import json

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from product_research_app.services import ai_prompts


def test_parse_score_transforms_new_schema() -> None:
    payload = {
        "content": json.dumps(
            {
                "items": [
                    {
                        "id": 101,
                        "desire": 0.78,
                        "desire_reason": "Alta búsqueda estacional y buenos reviews.",
                        "competition": 0.41,
                        "competition_level": "medium",
                        "revenue": 12450.0,
                        "units_sold": 380,
                        "price": 32.99,
                        "oldness": 0.18,
                        "rating": 4.4,
                    }
                ]
            }
        )
    }

    rows = ai_prompts.parse_score(payload)

    assert rows == [
        {
            "id": 101,
            "desire": "Alta búsqueda estacional y buenos reviews.",
            "desire_magnitude": "High",
            "awareness_level": "Product-Aware",
            "competition_level": "Medium",
        }
    ]
