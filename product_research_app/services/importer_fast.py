import csv
import io
from datetime import datetime
from typing import Iterable, Mapping, Sequence

from product_research_app.db import get_db
from product_research_app.database import json_dump

UPSERT_SQL = """
INSERT INTO products (
    id, name, description, category, price, currency, image_url, source,
    import_date, desire, desire_magnitude, awareness_level, competition_level,
    date_range, winner_score, extra
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, json(?))
ON CONFLICT(id) DO UPDATE SET
    name=excluded.name,
    description=excluded.description,
    category=excluded.category,
    price=excluded.price,
    currency=excluded.currency,
    image_url=excluded.image_url,
    source=excluded.source,
    import_date=excluded.import_date,
    desire=excluded.desire,
    desire_magnitude=excluded.desire_magnitude,
    awareness_level=excluded.awareness_level,
    competition_level=excluded.competition_level,
    date_range=excluded.date_range,
    winner_score=COALESCE(excluded.winner_score, products.winner_score),
    extra=excluded.extra;
"""


def _sanitize(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


FIELD_ALIASES: dict[str, Sequence[str]] = {
    "id": ["id"],
    "name": ["name", "nombre", "productname", "product", "title"],
    "description": ["description", "descripcion", "desc"],
    "category": ["category", "categoria", "niche", "segment"],
    "category_path": ["category_path", "categorypath", "path"],
    "price": ["price", "precio", "cost", "unitprice"],
    "currency": ["currency", "moneda"],
    "image_url": [
        "image_url",
        "image",
        "imagen",
        "img",
        "imgurl",
        "picture",
        "imageurl",
        "imagelink",
        "mainimage",
        "mainimageurl",
    ],
    "desire": ["desire", "deseo"],
    "desire_magnitude": ["desire_magnitude", "desiremag", "magnituddeseo"],
    "awareness_level": ["awareness_level", "awareness", "nivelconsciencia"],
    "competition_level": ["competition_level", "competition", "saturacionmercado"],
    "date_range": ["date_range", "daterange", "rangofechas", "fecharango"],
    "launch_date": ["launch_date", "launchdate", "fechalanzamiento"],
    "rating": ["rating", "valoracion", "stars", "productrating"],
    "units_sold": ["units_sold", "unitssold", "units", "itemsold", "items_sold", "sold"],
    "revenue": ["revenue", "sales", "ingresos"],
    "conversion_rate": ["conversion_rate", "conversion", "tasaconversion", "cr", "conversionrate"],
    "winner_score": ["winner_score", "winnerscore"],
    "source": ["source", "fuente"],
}

ALIASES_SANITIZED = {
    field: [_sanitize(alias) for alias in aliases]
    for field, aliases in FIELD_ALIASES.items()
}


def _num(value) -> float:
    if value is None:
        return 0.0
    s = str(value).strip()
    if not s:
        return 0.0
    multiplier = 1.0
    if s.lower().endswith("m"):
        multiplier = 1_000_000.0
        s = s[:-1]
    elif s.lower().endswith("k"):
        multiplier = 1_000.0
        s = s[:-1]
    s = (
        s.replace("€", "")
        .replace("$", "")
        .replace("%", "")
        .replace(".", "")
        .replace(",", ".")
    )
    try:
        return float(s) * multiplier
    except Exception:
        return 0.0


def _parse_optional_number(value, as_int: bool = False):
    if value in (None, ""):
        return None
    num = _num(value)
    if as_int:
        try:
            return int(round(num))
        except Exception:
            return None
    return num


def _pick(row: Mapping[str, object], sanitized: Mapping[str, str], field: str, recognised: set[str]):
    for alias in ALIASES_SANITIZED.get(field, ()):  # type: ignore[arg-type]
        original = sanitized.get(alias)
        if original is None:
            continue
        value = row.get(original)
        if isinstance(value, str):
            value = value.strip()
        if value in (None, ""):
            continue
        recognised.add(original)
        return original, value
    return None, None


def _prepare_rows(records: Iterable[Mapping[str, object]], source: str | None = None):
    prepared = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        row = dict(record)
        sanitized_keys: dict[str, str] = {}
        for key in row.keys():
            if key is None:
                continue
            norm = _sanitize(str(key))
            if not norm:
                continue
            sanitized_keys.setdefault(norm, key)
        recognised: set[str] = set()

        _, raw_id = _pick(row, sanitized_keys, "id", recognised)
        row_id = _parse_optional_number(raw_id, as_int=True)
        if row_id is not None and row_id <= 0:
            row_id = None

        name_key, raw_name = _pick(row, sanitized_keys, "name", recognised)
        if raw_name is None:
            continue
        name = str(raw_name)

        _, raw_description = _pick(row, sanitized_keys, "description", recognised)
        description = str(raw_description).strip() if raw_description not in (None, "") else None

        _, raw_category_path = _pick(row, sanitized_keys, "category_path", recognised)
        category_path = str(raw_category_path).strip() if raw_category_path not in (None, "") else None

        _, raw_category = _pick(row, sanitized_keys, "category", recognised)
        category_value = raw_category if raw_category not in (None, "") else category_path
        category = str(category_value).strip() if category_value not in (None, "") else None

        _, raw_price = _pick(row, sanitized_keys, "price", recognised)
        price = _parse_optional_number(raw_price)

        _, raw_currency = _pick(row, sanitized_keys, "currency", recognised)
        currency = str(raw_currency).strip() if raw_currency not in (None, "") else None

        _, raw_image = _pick(row, sanitized_keys, "image_url", recognised)
        image_url = str(raw_image).strip() if raw_image not in (None, "") else None

        _, raw_desire = _pick(row, sanitized_keys, "desire", recognised)
        desire = str(raw_desire).strip() if raw_desire not in (None, "") else None

        _, raw_desire_mag = _pick(row, sanitized_keys, "desire_magnitude", recognised)
        desire_mag = str(raw_desire_mag).strip() if raw_desire_mag not in (None, "") else None

        _, raw_awareness = _pick(row, sanitized_keys, "awareness_level", recognised)
        awareness = str(raw_awareness).strip() if raw_awareness not in (None, "") else None

        _, raw_competition = _pick(row, sanitized_keys, "competition_level", recognised)
        competition = str(raw_competition).strip() if raw_competition not in (None, "") else None

        _, raw_range = _pick(row, sanitized_keys, "date_range", recognised)
        date_range = str(raw_range).strip() if raw_range not in (None, "") else ""

        _, raw_launch = _pick(row, sanitized_keys, "launch_date", recognised)
        launch_date = str(raw_launch).strip() if raw_launch not in (None, "") else ""
        if launch_date:
            launch_date = launch_date[:10]

        _, raw_rating = _pick(row, sanitized_keys, "rating", recognised)
        rating = _parse_optional_number(raw_rating)

        _, raw_units = _pick(row, sanitized_keys, "units_sold", recognised)
        units_sold = _parse_optional_number(raw_units, as_int=True)

        _, raw_revenue = _pick(row, sanitized_keys, "revenue", recognised)
        revenue = _parse_optional_number(raw_revenue)

        _, raw_conversion = _pick(row, sanitized_keys, "conversion_rate", recognised)
        conversion_rate = _parse_optional_number(raw_conversion)

        _, raw_winner = _pick(row, sanitized_keys, "winner_score", recognised)
        winner_score = _parse_optional_number(raw_winner, as_int=True)

        _, raw_source = _pick(row, sanitized_keys, "source", recognised)
        source_val = str(raw_source).strip() if raw_source not in (None, "") else None
        if not source_val:
            source_val = source or "upload"

        extras: dict[str, object] = {}
        if rating is not None:
            extras["rating"] = rating
        if units_sold is not None:
            extras["units_sold"] = units_sold
        if revenue is not None:
            extras["revenue"] = revenue
        if conversion_rate is not None:
            extras["conversion_rate"] = conversion_rate
        if launch_date:
            extras["launch_date"] = launch_date
        if category_path and (not category or category_path != category):
            extras["category_path"] = category_path

        for key, value in row.items():
            if key in recognised or key is None:
                continue
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    continue
            extras[key] = value

        prepared.append(
            (
                row_id,
                name,
                description,
                category,
                price,
                currency,
                image_url,
                source_val,
                datetime.utcnow().isoformat(),
                desire,
                desire_mag,
                awareness,
                competition,
                date_range,
                winner_score,
                json_dump(extras),
            )
        )
    return prepared


def parse_csv_bytes(payload: bytes, source: str | None = None):
    text = payload.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    return _prepare_rows(reader, source=source)


def prepare_rows(records: Iterable[Mapping[str, object]], source: str | None = None):
    return _prepare_rows(records, source=source)


def _bulk_insert(rows, status_cb):
    db = get_db()
    db.execute("PRAGMA journal_mode=WAL;")
    db.execute("PRAGMA synchronous=NORMAL;")
    db.execute("PRAGMA temp_store=MEMORY;")
    db.execute("PRAGMA cache_size=-20000;")
    db.execute("BEGIN IMMEDIATE;")
    try:
        total = len(rows)
        status_cb(stage="prepare", done=0, total=total)
        batch = 1000
        for idx in range(0, total, batch):
            chunk = rows[idx: idx + batch]
            if not chunk:
                continue
            db.executemany(UPSERT_SQL, chunk)
            status_cb(stage="insert", done=min(idx + len(chunk), total), total=total)
        db.execute("COMMIT;")
        status_cb(stage="commit", done=total, total=total)
        return total
    except Exception:
        db.execute("ROLLBACK;")
        raise
    finally:
        db.execute("PRAGMA synchronous=NORMAL;")


def fast_import(csv_bytes: bytes, status_cb=lambda **_: None, source: str | None = None):
    rows = parse_csv_bytes(csv_bytes, source=source)
    return _bulk_insert(rows, status_cb)


def fast_import_records(records: Iterable[Mapping[str, object]], status_cb=lambda **_: None, source: str | None = None):
    rows = prepare_rows(records, source=source)
    return _bulk_insert(rows, status_cb)
