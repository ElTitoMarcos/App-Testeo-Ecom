import re

BAN_PROD = r"(crema|serum|t[oó]nico|jab[oó]n|aspiradora|colch[oó]n|figura|vinilo|cuchillos?|cepillo|parches?|set|kit|pack|combo|x\d+)"
BAN_UNITS = r"(ml|l|litros|w|kw|cm|mm|kg|lbs|oz|pulgadas|quarts?|cuartos?)"

def sanitize_desire_text(s: str) -> str:
    s = re.sub(rf"\b{BAN_PROD}\b", "", s, flags=re.IGNORECASE)
    s = re.sub(rf"\b{BAN_UNITS}\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s{2,}", " ", s).strip(" ,;.-")
    return s

def coerce_len(s: str, min_len: int = 280, max_len: int = 420) -> str:
    s = s.strip()
    if len(s) >= min_len and len(s) <= max_len:
        return s
    # Si viene corto, añade cláusulas concisas sin inflar tokens.
    while len(s) < min_len:
        s += ("; " if not s.endswith((".", "!", "?")) else " ") + \
             "personas que buscan resultados sin invertir tiempo extra ni crear desorden"
        if len(s) > max_len:
            break
    return s[:max_len]
