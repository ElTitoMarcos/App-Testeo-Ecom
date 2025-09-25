from typing import Dict, List, Tuple, Any


def _norm_id(x):
    # ids llegan como int o str, normaliza a int si posible
    try:
        return int(str(x).strip())
    except Exception:
        return str(x).strip()


def map_desire_results(obj: Any, expected_ids: List[int]) -> Tuple[Dict[int, dict], List[int], List]:
    """
    Acepta distintos layouts de salida y devuelve:
      mapping: { product_id -> payload(dict con claves desire_*) }
      missing_ids: ids de expected_ids sin resultado
      extras: resultados de ids inesperados
    Layouts aceptados:
      A) {"items":[{"id":123, ...}, ...]}
      B) [{"id":123, ...}, ...]
      C) {"123": {...}, "124": {...}}
    """
    mapping: Dict[int, dict] = {}
    got_ids = set()

    if isinstance(obj, dict) and "items" in obj and isinstance(obj["items"], list):
        rows = obj["items"]
    elif isinstance(obj, list):
        rows = obj
    elif isinstance(obj, dict):
        # dict por id
        rows = [{"id": k, **(v or {})} for k, v in obj.items()]
    else:
        rows = []

    for r in rows:
        if not isinstance(r, dict):
            continue
        pid = _norm_id(r.get("id"))
        if isinstance(pid, int):
            mapping[pid] = r
            got_ids.add(pid)

    exp = set(expected_ids)
    missing_ids = sorted(list(exp - got_ids))
    extras = [i for i in got_ids if i not in exp]
    return mapping, missing_ids, extras
