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

__all__ = [
    "COMMON_CATS",
    "UNITS",
    "_norm",
    "product_terms_from_title",
    "looks_like_product_desc",
    "cleanse",
]
