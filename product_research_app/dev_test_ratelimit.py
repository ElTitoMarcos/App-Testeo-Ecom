from __future__ import annotations

import itertools
import os
from typing import Any, Dict

from product_research_app.gpt import OpenAIError, call_gpt, call_openai_chat


class Stub:
    """Simula una secuencia de respuestas con errores 429 antes de recuperarse."""

    def __init__(self, fails: int = 3):
        self.counter = itertools.count()
        self.fails = fails
        self.last_index = -1

    def run(self) -> Dict[str, Any]:
        index = next(self.counter)
        self.last_index = index
        if index < self.fails:
            raise OpenAIError(
                "OpenAI API returned status 429: Please try again in 500ms."
            )
        return {
            "choices": [
                {
                    "message": {
                        "content": "{\"desire_statement\": \"texto\", \"desire_magnitude\": 5}"
                    }
                }
            ]
        }


def main() -> None:
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    stub = Stub()
    original = call_openai_chat

    def _fake_call(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return stub.run()

    try:
        # Monkeypatch temporal para simular reintentos.
        import product_research_app.gpt as gpt_module

        gpt_module.call_openai_chat = _fake_call  # type: ignore[assignment]
        result = call_gpt("DESIRE", context_json={"product": {"id": 1}})
        print({"result_ok": result.get("ok"), "attempts": stub.last_index + 1})
    except OpenAIError as exc:
        print({"error": str(exc)})
    finally:
        # Restaurar la función original.
        import product_research_app.gpt as gpt_module

        gpt_module.call_openai_chat = original  # type: ignore[assignment]


if __name__ == "__main__":
    main()

