from typing import Any, Dict, List, Tuple


def _norm_id(value: Any):
    try:
        return int(str(value).strip())
    except Exception:
        return None


def map_desire_results(
    obj: Any, expected_ids: List[int]
) -> Tuple[Dict[int, dict], List[int], List[int]]:
    """
    Acepta:
      A) {"items":[{"id":123,...}, ...]}
      B) [{"id":123,...}, ...]
      C) {"123": {...}, "124": {...}}   # dict por id
    Acepta claves: "desire_statement" | "desire" | "s"
    """

    rows: List[dict] = []
    if isinstance(obj, dict) and isinstance(obj.get("items"), list):
        rows = obj["items"]
    elif isinstance(obj, list):
        rows = obj
    elif isinstance(obj, dict):
        rows = [{"id": key, **(value or {})} for key, value in obj.items()]

    mapping: Dict[int, dict] = {}
    seen: set[int] = set()

    for row in rows:
        if not isinstance(row, dict):
            continue
        pid = _norm_id(row.get("id"))
        if pid is None:
            continue
        statement = row.get("desire_statement") or row.get("desire") or row.get("s") or ""
        row["desire_statement"] = statement
        mapping[pid] = row
        seen.add(pid)

    expected = set(expected_ids)
    missing = sorted(list(expected - seen))
    extras = sorted([pid for pid in seen if pid not in expected])
    return mapping, missing, extras
