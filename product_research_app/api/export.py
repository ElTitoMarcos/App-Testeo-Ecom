from __future__ import annotations

import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import requests
from flask import Blueprint, jsonify, request, send_file
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from PIL import Image as PILImage

from product_research_app import database
from product_research_app.db import get_db
from product_research_app.utils.db import rget, row_to_dict

UA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}
IMG_TARGET = 192
ROW_PAD = 22

export_api = Blueprint("export_api", __name__)


def _px_to_excel_width(px: int) -> float:
    try:
        return max(10.0, (px - 5) / 7.0)
    except Exception:
        return 18.0


def _text_width_px(value: object) -> int:
    if value in (None, ""):
        return 0
    text = str(value)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    longest = max((len(line) for line in lines), default=len(text))
    return min(640, 7 * longest + 24)


def _download_to_png(url: str, out_path: Path) -> bool:
    try:
        resp = requests.get(url, headers=UA_HEADERS, timeout=10)
        resp.raise_for_status()
        with PILImage.open(BytesIO(resp.content)) as img:
            img = img.convert("RGBA")
            if img.width and img.width != IMG_TARGET:
                ratio = IMG_TARGET / float(img.width)
                height = max(1, int(img.height * ratio))
                img = img.resize((IMG_TARGET, height), PILImage.LANCZOS)
            img.save(out_path, format="PNG")
        return True
    except Exception:
        return False


def _coerce_ids(raw_ids: Iterable[object]) -> list[int]:
    ids: list[int] = []
    for raw in raw_ids or []:
        try:
            ids.append(int(raw))
        except Exception:
            continue
    return ids


def _resolve_desire(product: dict, extras: dict) -> str:
    sources = (
        rget(product, "desire"),
        rget(extras, "desire"),
        rget(product, "ai_desire"),
        rget(product, "ai_desire_label"),
        rget(product, "desire_magnitude"),
    )
    for val in sources:
        if val not in (None, ""):
            return str(val)
    return ""


def _winner_score_for(conn, product_id: int | None):
    if product_id is None:
        return None
    try:
        scores = database.get_scores_for_product(conn, int(product_id))
    except Exception:
        scores = []
    if not scores:
        return None
    score_row = row_to_dict(scores[0]) or {}
    val = rget(score_row, "winner_score")
    if val in (None, ""):
        return None
    try:
        return int(round(float(val)))
    except Exception:
        try:
            return int(val)
        except Exception:
            return val


def _get_data(ids, columns):
    conn = get_db()
    requested = _coerce_ids(ids)
    if requested:
        rows = [database.get_product(conn, pid) for pid in requested]
        rows = [row for row in rows if row]
    else:
        rows = database.list_products(conn)

    results: list[dict] = []
    for row in rows:
        prod = row_to_dict(row) or {}
        extras_raw = prod.get("extra") or prod.get("extras") or {}
        if isinstance(extras_raw, str):
            try:
                extras = json.loads(extras_raw)
            except Exception:
                extras = {}
        elif isinstance(extras_raw, dict):
            extras = dict(extras_raw)
        else:
            extras = {}

        alias_pairs = [
            ("rating", "Product Rating"),
            ("units_sold", "Item Sold"),
            ("revenue", "Revenue($)"),
            ("conversion_rate", "Creator Conversion Ratio"),
            ("launch_date", "Launch Date"),
        ]
        for source, target in alias_pairs:
            if source in extras and target not in extras:
                extras[target] = extras[source]

        record = dict(extras)
        record.update({k: v for k, v in prod.items() if v not in (None, "")})
        record["id"] = prod.get("id")
        record["name"] = prod.get("name")
        record["category"] = prod.get("category")
        record["price"] = prod.get("price")
        record["date_range"] = prod.get("date_range") or extras.get("date_range") or ""
        record["desire"] = _resolve_desire(prod, extras)
        record["desire_magnitude"] = prod.get("desire_magnitude") or extras.get("desire_magnitude")
        record["awareness_level"] = prod.get("awareness_level") or extras.get("awareness_level")
        record["competition_level"] = prod.get("competition_level") or extras.get("competition_level")
        record["image_url"] = (
            prod.get("image_url")
            or extras.get("image_url")
            or extras.get("image")
        )
        record.setdefault("image", record.get("image_url"))

        ws_val = prod.get("winner_score")
        if ws_val in (None, ""):
            ws_val = _winner_score_for(conn, record.get("id"))
        else:
            try:
                ws_val = int(round(float(ws_val)))
            except Exception:
                pass
        record["winner_score"] = ws_val

        results.append(record)
    return results


@export_api.route('/export', methods=['POST'])
def export_xlsx():
    data = request.get_json(force=True) or {}
    ids = data.get('ids') or []
    columns = data.get('columns') or []
    if not isinstance(columns, list) or not columns:
        return jsonify({'error': 'Sin columnas para exportar'}), 400

    rows = _get_data(ids, columns)

    wb = Workbook()
    ws = wb.active
    ws.title = 'Productos'

    header_fill = PatternFill('solid', fgColor='1f2347')
    header_font = Font(color='FFFFFF', bold=True)
    cell_align = Alignment(vertical='top', wrap_text=True)
    thin = Side(style='thin', color='333333')
    border = Border(top=thin, left=thin, right=thin, bottom=thin)

    width_targets: dict[str, int] = {}

    for col_index, col in enumerate(columns, start=1):
        title = (col.get('title') or col.get('key') or '').strip()
        cell = ws.cell(row=1, column=col_index, value=title)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border = border
        letter = cell.column_letter
        width_targets[letter] = max(width_targets.get(letter, 0), _text_width_px(title))

    col_index_map = {c.get('key'): idx for idx, c in enumerate(columns, start=1)}
    col_letter_map = {key: ws.cell(row=1, column=idx).column_letter for key, idx in col_index_map.items() if key}
    img_key = next((
        key for key in col_index_map.keys()
        if isinstance(key, str) and key.lower() in {'image', 'image_url', 'img', 'thumbnail'}
    ), None)

    score_keys = ('winner_score', 'winnerScore', 'score', 'winner', 'winner_score_int')

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for row_idx, item in enumerate(rows, start=2):
            for key, col_idx in col_index_map.items():
                val = item.get(key, '') if key else ''
                if key in ('winner_score', 'winnerScore') and val in (None, ''):
                    for alt in score_keys:
                        if item.get(alt) not in (None, ''):
                            val = item.get(alt)
                            break
                cell = ws.cell(row=row_idx, column=col_idx, value=val if val not in (None, '') else '')
                if key in ('winner_score', 'winnerScore'):
                    try:
                        cell.number_format = '0'
                    except Exception:
                        pass
                cell.alignment = cell_align
                cell.border = border
                letter = col_letter_map.get(key)
                if letter and key != img_key:
                    width_targets[letter] = max(width_targets.get(letter, 0), _text_width_px(val))

            if img_key:
                letter = col_letter_map.get(img_key)
                if letter:
                    url = str(item.get(img_key) or item.get('image_url') or '')
                    if url.startswith('http'):
                        png_path = tmp_path / f"img_{row_idx}.png"
                        if _download_to_png(url, png_path):
                            img = XLImage(str(png_path))
                            img.width = IMG_TARGET
                            img.height = IMG_TARGET
                            anchor = f"{letter}{row_idx}"
                            try:
                                ws.add_image(img, anchor)
                            except TypeError:
                                img.anchor = anchor
                                ws.add_image(img)
                            ws.column_dimensions[letter].width = _px_to_excel_width(IMG_TARGET + 20)
                            ws.row_dimensions[row_idx].height = IMG_TARGET + ROW_PAD
                            ws.cell(row=row_idx, column=col_index_map[img_key], value=None)

    for key, col_idx in col_index_map.items():
        letter = col_letter_map.get(key)
        if not letter:
            continue
        if key == img_key:
            continue
        target_px = max(width_targets.get(letter, 0), _text_width_px(columns[col_idx - 1].get('title')))
        ws.column_dimensions[letter].width = _px_to_excel_width(target_px + 12)

    ws.freeze_panes = 'A2'

    bio = BytesIO()
    wb.save(bio)
    bio.seek(0)
    filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return send_file(
        bio,
        as_attachment=True,
        download_name=filename,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
