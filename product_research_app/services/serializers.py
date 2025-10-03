from datetime import date, datetime

DATE_FMT = "%Y-%m-%d"

def _fmt(d):
    if not d:
        return None
    if isinstance(d, (date, datetime)):
        return d.strftime(DATE_FMT)
    try:
        return str(d)[:10]
    except Exception:
        return str(d)

def compute_date_range(first_seen, last_seen):
    a = _fmt(first_seen)
    b = _fmt(last_seen)
    if a and b:
        return f"{a}—{b}"
    return a or b or None

def serialize_product_row(row: dict) -> dict:
    """Normaliza nombres y rellena columnas AI para el frontend."""
    aw = row.get("awareness_level") or row.get("ai_awareness_level") or row.get("awareness")
    comp = row.get("competition_level") or row.get("ai_competition_level") or row.get("competition")
    wscore = row.get("winner_score") or row.get("ai_winner_score") or row.get("score")
    r1 = row.get("first_seen") or row.get("first_date") or row.get("first_seen_at")
    r2 = row.get("last_seen") or row.get("last_date") or row.get("last_seen_at")
    rango = row.get("rango_fechas") or row.get("date_range") or compute_date_range(r1, r2)

    out = dict(row)
    out["awareness_level"] = aw
    out["competition_level"] = comp
    out["winner_score"] = wscore
    out["date_range"] = rango
    out["rango_fechas"] = rango
    return out
