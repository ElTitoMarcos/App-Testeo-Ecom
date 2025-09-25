import re
import unicodedata

COMMON_CATS = [
  "crema","champ[uú]","ser[úu]m","t[oó]nico","jab[oó]n",
  "cepillo","plancha","secador","aspiradora","colch[oó]n",
  "cuchillos?","maquin[ae]","parches?","figura","vinilo",
  "litter","box","shampoo","conditioner","brush","knife","set","kit","pack"
]
UNITS = ["ml","l","litros","w","kw","cm","mm","kg","lbs","oz","pulgadas"]

def _norm(s:str)->str:
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    return s.lower()

def product_terms_from_title(title:str)->set:
    t = _norm(title or "")
    tokens = re.findall(r"[a-z]{3,}", t)
    base = set(tokens)
    base.update(["set","kit","pack"])
    return base

def looks_like_product_desc(text:str, title:str)->bool:
    txt = _norm(text or "")
    if any(u in txt.split() for u in UNITS): return True
    if re.search(r"\b(" + "|".join(COMMON_CATS) + r")\b", txt): return True
    # solapamiento con el título (marca/modelo/categoría)
    overlap = product_terms_from_title(title).intersection(set(re.findall(r"[a-z]{3,}", txt)))
    return len(overlap) >= 3  # umbral prudente

def cleanse(text:str)->str:
    # quita unidades/packs y espacios sobrantes
    txt = re.sub(r"\b(" + "|".join(UNITS) + r")\b", "", text or "", flags=re.IGNORECASE)
    txt = re.sub(r"\b(set|kit|pack|x\d+)\b", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\s{2,}", " ", txt).strip(" ,;.-")
    return txt


def dedupe_clauses(text: str) -> str:
    if not text:
        return ""
    clauses = re.split(r"(?<=[.!?])\s+", str(text).strip())
    seen: set[str] = set()
    ordered: list[str] = []
    for clause in clauses:
        norm = _norm(clause.strip())
        if not norm or norm in seen:
            continue
        seen.add(norm)
        ordered.append(clause.strip())
    return " ".join(ordered)


def _strip_repeated_spaces(text: str) -> str:
    return re.sub(r"\s{2,}", " ", text).strip()


def sanitize_desire_text(text: str) -> str:
    if not text:
        return ""
    cleaned = cleanse(text)
    return _strip_repeated_spaces(cleaned)

__all__ = [
    "COMMON_CATS",
    "UNITS",
    "_norm",
    "product_terms_from_title",
    "looks_like_product_desc",
    "cleanse",
    "dedupe_clauses",
    "sanitize_desire_text",
]
