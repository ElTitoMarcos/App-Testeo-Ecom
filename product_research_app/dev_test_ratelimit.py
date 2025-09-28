from __future__ import annotations
import itertools
from product_research_app.gpt import call_gpt

# Para tipado del except en esta prueba
try:
    from product_research_app.gpt import OpenAIError  # type: ignore
except Exception:  # pragma: no cover
    class OpenAIError(Exception): ...

class Stub:
    """
    Simula 429 X veces y luego OK. Útil para probar el backoff/recovery.
    """
    def __init__(self, fails: int = 3):
        self.counter = itertools.count()
        self.fails = int(fails)
    def run(self):
        i = next(self.counter)
        if i < self.fails:
            # "OpenAI API returned status 429: Please try again in 500ms."
            raise OpenAIError("OpenAI API returned status 429: Please try again in 500ms.")
        return {"ok": True, "i": i}

def call_gpt_stubbed(fails: int = 3):
    """
    Variante de prueba: alinea la firma de call_gpt para inyectar nuestro stub.
    """
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

if __name__ == "__main__":
    print("simulate")
    res = call_gpt_stubbed(fails=3)
    print("result:", res)
