from __future__ import annotations

import random

from .services.ai_columns import fill_ai_columns_with_recovery


class DummyModel:
    model = "dummy-ia"

    class _Response:
        def __init__(self, text: str) -> None:
            self.text = text

    def complete(self, prompt: str, temperature: float, top_p: float):
        ids: list[int] = []
        for line in prompt.splitlines():
            if "[" in line and "]" in line:
                fragment = line[line.find("[") + 1 : line.find("]")]
                for part in fragment.split(","):
                    part = part.strip()
                    if part.isdigit():
                        ids.append(int(part))
        keep = set(ids)
        if len(ids) >= 8:
            drop = set(random.sample(ids, max(1, len(ids) // 4)))
            keep = set(ids) - drop
        lines = [
            "{" +
            f'"product_id": {pid}, "desire": "high", "desire_magnitude": 0.8, '
            '"awareness_level": "low", "competition_level": "medium"' +
            "}"
            for pid in keep
        ]
        return self._Response("\n".join(lines))


if __name__ == "__main__":
    model = DummyModel()
    ids = list(range(1, 33))
    out = fill_ai_columns_with_recovery(model, ids)
    print("filled:", len(out), "expected:", len(ids))
