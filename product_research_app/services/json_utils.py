import json, re

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
PAIR_RE = re.compile(r'"(?P<id>\d+)"\s*:\s*\{(?P<body>[^{}]*?)\}\s*,?', re.DOTALL)


def fix_newlines_in_strings(text: str) -> str:
    """Reemplaza saltos de línea crudos dentro de strings JSON por \n."""

    out = []
    in_str = False
    esc = False
    for ch in text:
        if esc:
            out.append(ch)
            esc = False
            continue
        if ch == "\\":
            out.append(ch)
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            out.append(ch)
            continue
        if ch in ('\r', '\n') and in_str:
            out.append('\\n')
            continue
        out.append(ch)
    return ''.join(out)


def try_load(s: str):
    try:
        return json.loads(s), None
    except Exception as e:
        return None, str(e)


def extract_json_object_loose(text: str):
    """
    1) Si hay fence ```json ... ```, cargar.
    2) Si no, buscar primer '{' e intentar balancear.
    3) Reparar newlines en strings y reintentar.
    4) Si sigue fallando, salvar pares "id": {...} sueltos.
    Retorna (obj, err). obj puede ser dict por id.
    """

    if not text:
        return None, "empty"

    m = JSON_BLOCK_RE.search(text)
    if m:
        raw = m.group(1)
        obj, err = try_load(raw)
        if obj is not None:
            return obj, None
        raw2 = fix_newlines_in_strings(raw)
        obj, err2 = try_load(raw2)
        if obj is not None:
            return obj, None
        return None, f"fenced:{err2 or err}"

    start = text.find("{")
    if start != -1:
        depth = 0
        end = -1
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            cand = text[start : end + 1]
            obj, err = try_load(cand)
            if obj is not None:
                return obj, None
            cand2 = fix_newlines_in_strings(cand)
            obj, err2 = try_load(cand2)
            if obj is not None:
                return obj, None

    salvage = {}
    fixed = fix_newlines_in_strings(text)
    for match in PAIR_RE.finditer(fixed):
        pid = match.group("id")
        body = "{" + (match.group("body") or "") + "}"
        item, _ = try_load(body)
        if isinstance(item, dict):
            salvage[pid] = item
    if salvage:
        return salvage, None

    return None, "no_json"
