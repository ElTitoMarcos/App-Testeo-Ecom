import re

BAN_PHRASES = [
  "resultados sin invertir tiempo extra ni crear desorden",
  "sin esfuerzo",
  "de forma premium",
]

BAN_PROD = r"(crema|serum|t[oó]nico|jab[oó]n|aspiradora|colch[oó]n|figura|vinilo|cuchillos?|cepillo|parches?|set|kit|pack|combo|x\d+)"
BAN_UNITS = r"(ml|l|litros|w|kw|cm|mm|kg|lbs|oz|pulgadas|quarts?|cuartos?)"

def sanitize_desire_text(s: str) -> str:
    s = re.sub(rf"\b{BAN_PROD}\b", "", s, flags=re.IGNORECASE)
    s = re.sub(rf"\b{BAN_UNITS}\b", "", s, flags=re.IGNORECASE)
    for p in BAN_PHRASES:
        s = re.sub(re.escape(p), "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s{2,}", " ", s).strip(" ,;.-")
    return s

def dedupe_clauses(s: str) -> str:
    # corta repeticiones exactas de cláusulas/frases
    parts = re.split(r"([.;])", s)  # conserva separadores
    seen = set()
    out = []
    buf = ""
    for i in range(0, len(parts), 2):
        clause = parts[i].strip(" ,;.-")
        sep = parts[i+1] if i+1 < len(parts) else ""
        key = re.sub(r"\W+", "", clause.lower())
        if clause and key not in seen:
            seen.add(key)
            out.append(clause + (sep if sep else ""))
    buf = " ".join(out).strip()
    # limpia separadores duplicados
    buf = re.sub(r"\s*([.;])\s*", r"\1 ", buf).strip(" ;.")
    return buf

def needs_regen(payload: dict, min_len=280) -> bool:
    ds = (payload or {}).get("desire_statement") or ""
    return len(ds.strip()) < min_len
