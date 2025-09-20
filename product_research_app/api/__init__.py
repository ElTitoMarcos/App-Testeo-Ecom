"""Flask application factory and blueprint registration."""

from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from flask import Flask

from .. import database
from ..db import get_db
from ..utils.db import row_to_dict

app = Flask(__name__)


def _parse_extra(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}


def _extra_pick(extras: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in extras and extras[key] not in (None, ""):
            return extras[key]
    lower = {str(k).lower(): v for k, v in extras.items() if isinstance(k, str)}
    for key in keys:
        val = lower.get(str(key).lower())
        if val not in (None, ""):
            return val
    return None


def _ensure_desire(product: dict[str, Any], extras: dict[str, Any]) -> str | None:
    candidates = [
        product.get("desire"),
        _extra_pick(extras, ["desire", "benefit", "claim", "voz_cliente"]),
        product.get("ai_desire"),
        product.get("ai_desire_label"),
        product.get("desire_magnitude"),
    ]
    for value in candidates:
        if value not in (None, ""):
            return str(value)
    return None


def _default_row_provider(ids: Sequence[Any], _columns: Sequence[Any]) -> list[dict[str, Any]]:
    conn = get_db()
    database.initialize_database(conn)
    try:
        cur = conn.cursor()
    except Exception:  # pragma: no cover - defensive
        return []

    ordered_ids: list[int] = []
    for raw in ids or []:
        try:
            ordered_ids.append(int(raw))
        except Exception:
            continue

    if ordered_ids:
        placeholders = ",".join("?" for _ in ordered_ids)
        cur.execute(f"SELECT * FROM products WHERE id IN ({placeholders})", ordered_ids)
        fetched = cur.fetchall()
        rows = {int(row["id"]): row for row in fetched}
        base = [rows[i] for i in ordered_ids if i in rows]
    else:
        base = database.list_products(conn)

    payload: list[dict[str, Any]] = []
    for row in base:
        product = row_to_dict(row) or {}
        extras = _parse_extra(product.get("extra"))
        desire_mag = product.get("desire_magnitude") or _extra_pick(
            extras, ["desire_magnitude", "magnitud_deseo", "magnitude", "magnet"]
        )
        awareness = product.get("awareness_level") or _extra_pick(
            extras, ["awareness_level", "awareness level", "nivel_consciencia"]
        )
        competition = product.get("competition_level") or _extra_pick(
            extras, ["competition_level", "competition", "competition level", "saturacion_mercado"]
        )
        date_range = product.get("date_range") or _extra_pick(
            extras, ["date_range", "date range", "daterange", "rango_fechas", "rango fechas"]
        )
        desire = _ensure_desire(product, extras)
        rating = _extra_pick(extras, ["rating", "product rating", "stars", "valoracion"])
        units = _extra_pick(
            extras,
            [
                "units_sold",
                "units",
                "items_sold",
                "item sold",
                "sold",
                "orders",
                "sales_units",
                "quantity",
            ],
        )
        revenue = _extra_pick(extras, ["revenue", "revenue($)", "sales", "gmv"])
        conversion = _extra_pick(extras, ["conversion_rate", "conversion rate", "conversion"])
        launch_date = _extra_pick(extras, ["launch_date", "launch date", "fecha lanzamiento"])
        image = product.get("image_url") or _extra_pick(
            extras,
            [
                "image",
                "image_url",
                "img",
                "thumbnail",
                "image link",
                "imageurl",
                "main image",
            ],
        )
        product_url = _extra_pick(
            extras,
            ["product_url", "product url", "productlink", "product link"],
        )
        source_url = _extra_pick(extras, ["source_url", "source url"])
        listing_url = _extra_pick(extras, ["listing_url", "listing url"])
        generic_url = _extra_pick(extras, ["url", "link"])
        source_text = product.get("source") or _extra_pick(extras, ["source", "source_name", "fuente"])

        row_payload = {
            "id": product.get("id"),
            "name": product.get("name"),
            "category": product.get("category"),
            "price": product.get("price"),
            "image": image,
            "image_url": image,
            "desire": (str(desire).strip() or None) if desire else None,
            "desire_magnitude": desire_mag,
            "awareness_level": awareness,
            "competition_level": competition,
            "rating": rating,
            "units_sold": units,
            "revenue": revenue,
            "conversion_rate": conversion,
            "launch_date": launch_date,
            "date_range": date_range or "",
            "winner_score": product.get("winner_score"),
            "winner_score_raw": product.get("winner_score_raw"),
            "source": source_text,
            "product_url": product_url or generic_url or source_url or listing_url,
            "source_url": source_url,
            "listing_url": listing_url,
            "url": generic_url,
            "link": generic_url,
            "extras": extras,
        }
        payload.append(row_payload)

    return payload


app.config.setdefault("ROW_PROVIDER", _default_row_provider)

# Import API modules which attach routes to ``app``.
from . import config  # noqa: E402,F401
from .export import bp_export  # noqa: E402
from .winner_score import winner_score_api  # noqa: E402
from ..sse import sse_bp  # noqa: E402

app.register_blueprint(winner_score_api, url_prefix="/api")
app.register_blueprint(bp_export)
app.register_blueprint(sse_bp)


@app.get("/healthz")
def healthz():
    return {"ok": True}


# Log registered routes for easier debugging in start-up logs.
for rule in app.url_map.iter_rules():
    app.logger.info("ROUTE %s %s", ",".join(sorted(rule.methods)), rule.rule)
