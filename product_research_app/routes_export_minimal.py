"""Utilities to expose the Kalodata minimal XLSX export endpoint."""

from __future__ import annotations

import io
import json
import logging
import math
import re
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.formatting.rule import DataBarRule

from . import database
from .utils.db import row_to_dict

logger = logging.getLogger(__name__)

ResponseHandler = Any
EnsureDbFn = Callable[[], Any]

COLUMNS: Sequence[str] = (
    "Product Name",
    "TikTokUrl",
    "KalodataUrl",
    "Img_url",
    "Category",
    "Price($)",
    "Product Rating",
    "Item Sold",
    "Total Revenue($)",
    "Launch Date",
    "Desire",
    "Desire Magnitude",
    "Awareness Level",
    "Competition Level",
)

TEXT_FIELD_KEYS: Dict[str, Sequence[str]] = {
    "Product Name": ("name", "product_name", "title"),
    "TikTokUrl": (
        "tiktok_url",
        "TikTokUrl",
        "tiktokurl",
        "tiktok",
        "tiktok link",
        "TikTok URL",
        "TikTok Url",
    ),
    "KalodataUrl": ("kalodata_url", "KalodataUrl", "kalodataurl", "Kalodata URL"),
    "Img_url": (
        "image_url",
        "image",
        "img_url",
        "img",
        "imageurl",
        "imagelink",
        "Img_url",
        "Img Url",
    ),
    "Category": ("category", "category_path", "Category"),
    "Launch Date": ("launch_date", "Launch Date", "launchdate"),
}

NUMERIC_FIELD_KEYS: Dict[str, Sequence[str]] = {
    "Price($)": ("price", "Price($)", "price$", "price_usd"),
    "Product Rating": ("rating", "Product Rating", "productrating"),
    "Item Sold": ("units_sold", "items_sold", "Item Sold", "sold"),
    "Revenue($)": ("revenue", "Revenue($)", "total_revenue"),
    "Live Revenue($)": ("live_revenue", "Live Revenue($)", "live revenue"),
    "Video Revenue($)": ("video_revenue", "Video Revenue($)", "video revenue"),
    "Shopping Mall Revenue($)": (
        "shopping_mall_revenue",
        "Shopping Mall Revenue($)",
        "shopping mall revenue",
    ),
}

DESIRE_TEXT_KEYS: Sequence[str] = (
    "desire",
    "desires",
    "customer_desire",
    "desire_text",
    "ai_desire_label",
)

DESIRE_SCORE_KEYS: Sequence[str] = (
    "desire_score",
    "_desire_score",
    "ai_desire_score",
    "desire_magnitude",
    "desiremag",
)

AWARENESS_KEYS: Sequence[str] = (
    "awareness_level_label",
    "awareness_level",
    "awareness",
    "awareness_score",
)

COMPETITION_KEYS: Sequence[str] = (
    "competition_level_label",
    "competition_level",
    "competition",
    "competition_score",
)

_NUMERIC_STRIP_RE = re.compile(r"[\s\$,€£%]")

_AWARENESS_LABELS = {
    0: "Unaware",
    1: "Problem-aware",
    2: "Solution-aware",
    3: "Product-aware",
    4: "Most aware",
}


def export_kalodata_minimal(handler: ResponseHandler, ensure_db: EnsureDbFn) -> None:
    start = time.perf_counter()
    length = _read_length(handler.headers)
    raw_body = handler.rfile.read(length) if length else b""
    try:
        payload = json.loads(raw_body.decode("utf-8") or "{}")
    except Exception:
        handler.send_json({"error": "invalid_json"}, 400)
        return

    ids = payload.get("ids")
    if not isinstance(ids, list) or not ids:
        handler.send_json({"error": "ids_required"}, 400)
        return

    try:
        id_list = [int(x) for x in ids]
    except Exception:
        handler.send_json({"error": "invalid_ids"}, 400)
        return

    conn = ensure_db()
    rows: List[Dict[str, Any]] = []
    for pid in id_list:
        product = database.get_product(conn, pid)
        if product is not None:
            rows.append(row_to_dict(product))

    if not rows:
        handler.send_json({"error": "products_not_found"}, 404)
        return

    wb_data = _build_workbook(rows)
    timestamp = datetime.now(timezone.utc)
    filename = f"kalodata_for_analysis_{timestamp:%Y%m%d_%H%M}.xlsx"

    handler.send_response(200)
    handler.send_header(
        "Content-Type",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    handler.send_header("Content-Disposition", f"attachment; filename={filename}")
    handler.send_header("Content-Length", str(len(wb_data)))
    handler.end_headers()
    handler.wfile.write(wb_data)

    duration = time.perf_counter() - start
    logger.info("kalodata minimal export ok products=%d duration=%.3fs", len(rows), duration)


def _build_workbook(rows: Sequence[Mapping[str, Any]]) -> bytes:
    records: List[List[Any]] = []
    for row in rows:
        record, missing = _convert_row(row)
        records.append(record)
        if missing:
            logger.warning(
                "Missing generated fields for product %s: %s",
                row.get("id"),
                ", ".join(missing),
            )

    df = pd.DataFrame(records, columns=COLUMNS)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="products", index=False)
        ws = writer.sheets["products"]
        _apply_sheet_format(ws)

    return buffer.getvalue()


def _apply_sheet_format(ws) -> None:
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    header_fill = PatternFill("solid", fgColor="1F497D")
    header_font = Font(color="FFFFFF", bold=True)
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(vertical="center", wrap_text=True)

    widths = {
        1: 40,
        2: 35,
        3: 35,
        4: 40,
        5: 18,
        6: 12,
        7: 14,
        8: 12,
        9: 16,
        10: 14,
        11: 45,
        12: 18,
        13: 18,
        14: 18,
    }
    for idx, width in widths.items():
        ws.column_dimensions[get_column_letter(idx)].width = width

    text_cols = [1, 4, 5, 11, 13, 14]
    if ws.max_row >= 2:
        for col in text_cols:
            for row in ws.iter_rows(
                min_row=2,
                max_row=ws.max_row,
                min_col=col,
                max_col=col,
            ):
                cell = row[0]
                cell.alignment = Alignment(vertical="top", wrap_text=True)

        num_fmt_int = "#,##0"
        num_fmt_money = "$#,##0.00"
        num_fmt_rating = "0.0"
        num_fmt_date = "yyyy-mm-dd"

        for column in ws.iter_cols(8, 8, 2, ws.max_row):
            for cell in column:
                cell.number_format = num_fmt_int
        for column in ws.iter_cols(6, 6, 2, ws.max_row):
            for cell in column:
                cell.number_format = num_fmt_money
        for column in ws.iter_cols(7, 7, 2, ws.max_row):
            for cell in column:
                cell.number_format = num_fmt_rating
        for column in ws.iter_cols(9, 9, 2, ws.max_row):
            for cell in column:
                cell.number_format = num_fmt_money
        for column in ws.iter_cols(10, 10, 2, ws.max_row):
            for cell in column:
                cell.number_format = num_fmt_date

        dv = DataValidation(
            type="whole",
            operator="between",
            formula1="0",
            formula2="100",
            allow_blank=True,
        )
        ws.add_data_validation(dv)
        dv_range = f"{get_column_letter(12)}2:{get_column_letter(12)}{ws.max_row}"
        dv.add(dv_range)
        bar_rule = DataBarRule(
            start_type="num",
            start_value=0,
            end_type="num",
            end_value=100,
            color="1F497D",
            showValue=None,
        )
        ws.conditional_formatting.add(dv_range, bar_rule)

    table_ref = f"A1:{get_column_letter(ws.max_column)}{ws.max_row}"
    table = Table(displayName="tbl_products", ref=table_ref)
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium9",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    ws.add_table(table)


def _convert_row(row: Mapping[str, Any]) -> tuple[List[Any], List[str]]:
    extras = _ensure_dict(row.get("extra"))
    sources = _prepare_sources(row, extras)

    product_name = _coerce_text(_value_from_sources(sources, TEXT_FIELD_KEYS["Product Name"]))
    tiktok_url = _coerce_text(_value_from_sources(sources, TEXT_FIELD_KEYS["TikTokUrl"]))
    kalodata_url = _coerce_text(_value_from_sources(sources, TEXT_FIELD_KEYS["KalodataUrl"]))
    image_url = _coerce_text(_value_from_sources(sources, TEXT_FIELD_KEYS["Img_url"]))
    category = _coerce_text(_value_from_sources(sources, TEXT_FIELD_KEYS["Category"]))

    price_val = _parse_number(_value_from_sources(sources, NUMERIC_FIELD_KEYS["Price($)"]))
    price = round(price_val, 2) if price_val is not None else None

    rating_val = _parse_number(_value_from_sources(sources, NUMERIC_FIELD_KEYS["Product Rating"]))
    rating = round(rating_val, 2) if rating_val is not None else None

    units_val = _parse_number(_value_from_sources(sources, NUMERIC_FIELD_KEYS["Item Sold"]))
    units = int(round(units_val)) if units_val is not None else None

    total_revenue = _compute_total_revenue(sources)

    launch_date = _normalize_launch_date(
        _value_from_sources(sources, TEXT_FIELD_KEYS["Launch Date"])
    )

    desire_text_raw = _value_from_sources(sources, DESIRE_TEXT_KEYS)
    desire_text = _coerce_text(desire_text_raw)

    desire_score_raw = _value_from_sources(sources, DESIRE_SCORE_KEYS)
    desire_score = _normalize_desire_score(desire_score_raw)

    awareness_raw = _value_from_sources(sources, AWARENESS_KEYS)
    awareness = _normalize_awareness(awareness_raw)

    competition_raw = _value_from_sources(sources, COMPETITION_KEYS)
    competition = _normalize_competition(competition_raw)

    missing: List[str] = []
    if not desire_text:
        missing.append("Desire")
    if desire_score is None:
        missing.append("Desire Magnitude")
    if not awareness:
        missing.append("Awareness Level")
    if not competition:
        missing.append("Competition Level")

    desire_cell: Any
    if desire_score is None:
        desire_cell = math.nan
    else:
        desire_cell = desire_score

    record: List[Any] = [
        product_name,
        tiktok_url,
        kalodata_url,
        image_url,
        category,
        price,
        rating,
        units,
        total_revenue,
        launch_date,
        desire_text,
        desire_cell,
        awareness,
        competition,
    ]

    return record, missing


def _prepare_sources(*dicts: Mapping[str, Any]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for data in dicts:
        mapping = _ensure_dict(data)
        if not mapping:
            continue
        direct: Dict[str, Any] = {}
        lower: Dict[str, Any] = {}
        sanitized: Dict[str, Any] = {}
        for key, value in mapping.items():
            if not isinstance(key, str):
                continue
            direct[key] = value
            lower[key.lower()] = value
            sanitized[_sanitize_key(key)] = value
        sources.extend([direct, lower, sanitized])
    return sources


def _value_from_sources(sources: Sequence[Mapping[str, Any]], keys: Iterable[str]) -> Any:
    for key in keys:
        for variant in _key_variants(key):
            for source in sources:
                if variant in source:
                    value = source[variant]
                    if not _is_missing(value):
                        return value
    return None


def _normalize_desire_score(raw: Any) -> Optional[int]:
    num = _parse_number(raw)
    if num is None:
        return None
    if num <= 1.0:
        num *= 100.0
    num = max(0.0, min(100.0, num))
    return int(round(num))


def _normalize_awareness(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return ""
        parsed = _parse_number(stripped)
        if parsed is not None and not math.isnan(parsed):
            return _awareness_from_numeric(parsed)
        return stripped
    num = _parse_number(raw)
    if num is None or math.isnan(num):
        return ""
    return _awareness_from_numeric(num)


def _awareness_from_numeric(value: float) -> str:
    if value <= 1.0:
        scaled = int(round(value * 4))
    else:
        scaled = int(round(value))
    scaled = max(0, min(4, scaled))
    return _AWARENESS_LABELS.get(scaled, "")


def _normalize_competition(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return ""
        parsed = _parse_number(stripped)
        if parsed is None or math.isnan(parsed):
            return stripped
        return _map_competition(parsed)
    num = _parse_number(raw)
    if num is None or math.isnan(num):
        return ""
    return _map_competition(num)


def _map_competition(value: float) -> str:
    if value <= 1.0:
        score = value
    elif value <= 100.0:
        score = value / 100.0
    else:
        score = value / 100.0
    if score < 0.35:
        return "Low"
    if score < 0.7:
        return "Medium"
    return "High"


def _compute_total_revenue(sources: Sequence[Mapping[str, Any]]) -> float:
    total = 0.0
    for key in (
        "Revenue($)",
        "Live Revenue($)",
        "Video Revenue($)",
        "Shopping Mall Revenue($)",
    ):
        raw = _value_from_sources(sources, NUMERIC_FIELD_KEYS[key])
        total += _parse_money(raw)
    return round(total, 2)


def _normalize_launch_date(raw: Any) -> Any:
    if raw is None:
        return ""
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if isinstance(raw, (int, float)):
        if isinstance(raw, float) and math.isnan(raw):
            return ""
        try:
            base = datetime(1899, 12, 30)
            return (base + timedelta(days=float(raw))).date()
        except Exception:
            return str(raw)
    text = str(raw).strip()
    if not text:
        return ""
    try:
        return datetime.fromisoformat(text).date()
    except Exception:
        pass
    candidate = text.split()[0]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(candidate, fmt).date()
        except Exception:
            continue
    return text


def _parse_money(value: Any) -> float:
    num = _parse_number(value)
    if num is None:
        return 0.0
    return float(num)


def _parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    multiplier = 1.0
    last = text[-1]
    if last in {"k", "K", "m", "M"}:
        multiplier = 1_000.0 if last in {"k", "K"} else 1_000_000.0
        text = text[:-1]
    text = _NUMERIC_STRIP_RE.sub("", text)
    if not text:
        return None
    try:
        return float(text) * multiplier
    except Exception:
        return None


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value)


def _key_variants(key: str) -> List[str]:
    variants = [key]
    lower = key.lower()
    if lower not in variants:
        variants.append(lower)
    sanitized = _sanitize_key(key)
    if sanitized not in variants:
        variants.append(sanitized)
    compact = key.replace(" ", "")
    if compact not in variants:
        variants.append(compact)
    compact_lower = compact.lower()
    if compact_lower not in variants:
        variants.append(compact_lower)
    return variants


def _sanitize_key(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def _ensure_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return True
        return stripped.lower() in {"na", "n/a", "none", "null"}
    return False


def _read_length(headers: MutableMapping[str, str]) -> int:
    try:
        return int(headers.get("Content-Length", "0"))
    except Exception:
        return 0
