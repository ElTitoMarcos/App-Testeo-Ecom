import re

UNITS = r"(ml|l|litros|w|kw|cm|mm|kg|lbs|oz|pulgadas|quarts?|cuartos?)"
PACKS = r"(set|pack|kit|combo|duo|trio|x\d+|[0-9]+(?:\s*(?:pcs|pzs|uds)))"
CATTOKS = r"(crema|serum|tónico|jabón|aspiradora|colchón|figura|vinilo|cuchillos?|maquin[ae]|cepillo|parches?)"


def sanitize_desire(s: str, title: str = "") -> str:
    """Remove brand-like tokens, units and category hints from desire statements."""

    if title:
        head = re.escape(" ".join(title.split()[:3]))
        s = re.sub(head, "", s, flags=re.IGNORECASE)
    s = re.sub(rf"\b{UNITS}\b", "", s, flags=re.IGNORECASE)
    s = re.sub(rf"\b{PACKS}\b", "", s, flags=re.IGNORECASE)
    s = re.sub(rf"\b{CATTOKS}\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s{2,}", " ", s).strip(" ,;.-")
    return s
