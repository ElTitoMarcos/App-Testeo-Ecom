import json, re

JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*(\{[\s\S]*?\})\s*```",
    re.IGNORECASE
)

def extract_json_object(text: str):
    """Intenta extraer el primer objeto JSON válido de un texto con/ sin fences."""
    if not text:
        return None, "empty"

    m = JSON_BLOCK_RE.search(text)
    if m:
        s = m.group(1)
        try:
            return json.loads(s), None
        except Exception as e:
            last_err = f"fenced_load:{e}"
    else:
        last_err = "no_fence"

    start = text.find("{")
    if start == -1:
        return None, last_err or "no_left_brace"

    depth = 0; end = -1
    for i, ch in enumerate(text[start:], start):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: end = i; break
    if end != -1:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate), None
        except Exception as e:
            return None, f"balanced_load:{e}"

    return None, last_err or "unbalanced"


def coerce_bounds(s: str, mn: int, mx: int) -> str:
    s = (s or "").strip()
    return s[:mx] if len(s) >= mn else s
