from __future__ import annotations
import itertools

from product_research_app.gpt import call_gpt

# Para tipado del except en esta prueba
try:
    from product_research_app.gpt import OpenAIError  # type: ignore
except Exception:  # pragma: no cover
    class OpenAIError(Exception): ...

# Para tipado del except en esta prueba
try:
    from product_research_app.gpt import OpenAIError  # type: ignore
except Exception:  # pragma: no cover
    class OpenAIError(Exception): ...

class Stub:
    """Simula 429 X veces y luego OK."""

    def __init__(self, fails: int = 3):
        self.counter = itertools.count()
        self.fails = int(fails)

    def run(self):
        i = next(self.counter)
        if i < self.fails:
            raise OpenAIError("OpenAI API returned status 429: Please try again in 500ms.")
        return {"ok": True, "i": i}


def call_gpt_stubbed(fails: int = 3):
    """Variante de prueba para simular reintentos 429."""
    stub = Stub(fails=fails)
    messages = [
        {"role": "system", "content": "You are a test."},
        {"role": "user", "content": "Hello"},
    ]
    from product_research_app import gpt as _g
    orig = getattr(_g, "call_openai_chat")
    try:
        setattr(_g, "call_openai_chat", lambda **_: stub.run())
        return call_gpt(messages=messages)
    finally:
        setattr(_g, "call_openai_chat", orig)

class _ResponseStub:
    def __init__(self, status: int):
        self.status_code = int(status)
        self.headers = {}


class ServerStub:
    """Simula errores 5xx transitorios antes de recuperarse."""

    def __init__(self, fails: int = 2):
        self.counter = itertools.count()
        self.fails = int(fails)

    def run(self):
        i = next(self.counter)
        if i < self.fails:
            err = OpenAIError(
                "OpenAI API returned status 503: upstream connect error or disconnect/reset before headers."
            )
            setattr(err, "response", _ResponseStub(503))
            raise err
        return {"ok": True, "i": i}


def call_gpt_stubbed_5xx(fails: int = 2):
    """Variante de prueba para simular errores 5xx recuperables."""

    stub = ServerStub(fails=fails)
    messages = [
        {"role": "system", "content": "You are a test."},
        {"role": "user", "content": "Hello"},
    ]
    from product_research_app import gpt as _g

    orig = getattr(_g, "call_openai_chat")
    try:
        setattr(_g, "call_openai_chat", lambda **_: stub.run())
        return call_gpt(messages=messages)
    finally:
        setattr(_g, "call_openai_chat", orig)
        
if __name__ == "__main__":
    print("simulate")
    res = call_gpt_stubbed(fails=3)
    print("result:", res)
    res5 = call_gpt_stubbed_5xx(fails=2)
    print("result_5xx:", res5)
