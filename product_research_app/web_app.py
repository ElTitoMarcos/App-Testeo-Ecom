"""
Simple web interface for the Product Research Copilot.

This module starts an HTTP server that serves a minimal single‑page application
allowing the user to upload CSV/JSON files, configure OpenAI settings, trigger
batch evaluations, and make custom GPT queries.  The UI is written in
plain HTML/JavaScript and features a dark mode.  It is intended to be
platform‑agnostic and requires only the Python standard library.

Limitations:
    - OCR on image uploads is not implemented; when uploading images the user
      will need to input product details manually in the app.
    - For large datasets the evaluation may block the server; consider running
      the batch importer beforehand.

Usage:
    python -m product_research_app.web_app [--host 127.0.0.1] [--port 8000]
Then open http://host:port in a browser.
"""

from __future__ import annotations

import json
import os
import io
import re
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from email.parser import BytesParser
from email.policy import default
import threading
import time
import sqlite3
import math
import hashlib
from typing import Dict, Any, List

from . import database
from . import config
from .services import ai_columns
from .services import winner_v2 as winner_calc
from . import gpt
from . import title_analyzer

WINNER_V2_FIELDS = winner_calc.ALL_METRICS

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "data.sqlite3"
STATIC_DIR = APP_DIR / "static"
logger = logging.getLogger(__name__)

# Heuristic scoring for offline evaluation.
def offline_evaluate(product: dict) -> dict:
    """
    Compute heuristic scores for a product based on available fields when GPT evaluation
    is not available.  Returns a dict with keys: momentum, saturation,
    differentiation, social_proof, margin, logistics, summary, explanations.

    The heuristics use simple rules:
    - Momentum: based on revenue growth rate (extras['Revenue Growth Rate']), scaled around 5.
    - Saturation: based on creator count (extras['Creator Number']): more creators implies higher saturation, thus lower score.
    - Differentiation: fixed base score of 5, as we lack information.
    - Social proof: based on conversion ratio (extras['Creator Conversion Ratio']). Higher ratio yields higher score.
    - Margin: based on price: higher price suggests more margin but saturates.
    - Logistics: based on price: lower priced items are easier to ship, thus higher score.

    All scores are constrained between 1 and 9 to avoid extremes.
    """
    def clamp(val, lo=1.0, hi=9.0):
        return max(lo, min(hi, val))
    extras = {}
    try:
        if isinstance(product.get("extra"), str):
            extras = json.loads(product["extra"]) if product.get("extra") else {}
        elif isinstance(product.get("extra"), dict):
            extras = product["extra"]
    except Exception:
        extras = {}
    # Momentum based on Revenue Growth Rate
    growth_rate = None
    if extras:
        gr_key = None
        for key in extras.keys():
            if 'growth' in key.lower():
                gr_key = key
                break
        if gr_key:
            try:
                gr_str = str(extras[gr_key]).strip().replace('%','').replace(',','.')
                growth_rate = float(gr_str)
            except Exception:
                growth_rate = None
    if growth_rate is None:
        momentum = 5.0
    else:
        # Map growth percent into score; positive growth increases score
        momentum = clamp(5.0 + (growth_rate / 20.0))
    # Saturation based on Creator Number
    creator_num = None
    if extras and 'Creator Number' in extras:
        try:
            creator_num = float(str(extras['Creator Number']).replace(',','').replace('k','000'))
        except Exception:
            creator_num = None
    if creator_num is None:
        saturation = 5.0
    else:
        # More creators indicates higher competition (lower score). Scale: <=500 => 8, >=5000 => 2.
        if creator_num <= 500:
            saturation = 8.0
        elif creator_num >= 5000:
            saturation = 2.0
        else:
            # linear interpolation
            saturation = 8.0 - 6.0 * ((creator_num - 500) / 4500)
        saturation = clamp(saturation)
    # Differentiation: assume medium (5)
    differentiation = 5.0
    # Social proof based on Creator Conversion Ratio
    conv_ratio = None
    if extras and 'Creator Conversion Ratio' in extras:
        try:
            conv_ratio = float(str(extras['Creator Conversion Ratio']).replace('%','').replace(',','.'))
        except Exception:
            conv_ratio = None
    if conv_ratio is None:
        social = 5.0
    else:
        # Higher conversion ratio increases score; typical ratio 20% => 7
        social = clamp(3.0 + conv_ratio / 10.0)
    # Margin based on price or avg unit price
    price = product.get('price')
    if price is None and extras and 'Avg. Unit Price($)' in extras:
        try:
            price = float(str(extras['Avg. Unit Price($)']).replace(',','').replace('$',''))
        except Exception:
            price = None
    if price is None:
        margin = 5.0
    else:
        # Use logarithm to map price to score: cheaper items yield lower margin but easier to sell; mid priced goods moderate; expensive goods moderate.
        # Score between 3 and 8
        import math
        margin = clamp(3.0 + math.log(price + 1, 10) * 5.0)
    # Logistics based on price (proxy for size/weight)
    if price is None:
        logistics = 5.0
    else:
        # Cheaper items likely light and easy to ship (score 7); expensive heavy items score lower.
        if price <= 30:
            logistics = 7.0
        elif price <= 100:
            logistics = 5.0
        else:
            logistics = 3.0
    logistics = clamp(logistics)
    summary = "Evaluación heurística basada en métricas disponibles"
    explanations = {
        "momentum": f"Basado en la tasa de crecimiento de ingresos {growth_rate}% => {momentum:.1f}",
        "saturation": f"Basado en número de creadores {creator_num} => {saturation:.1f}",
        "differentiation": "Valor fijo por falta de datos", 
        "social_proof": f"Basado en ratio de conversión {conv_ratio}% => {social:.1f}",
        "margin": f"Basado en precio {price} => {margin:.1f}",
        "logistics": f"Basado en precio {price} => {logistics:.1f}",
    }
    return {
        'momentum': float(f"{momentum:.2f}"),
        'saturation': float(f"{saturation:.2f}"),
        'differentiation': float(f"{differentiation:.2f}"),
        'social_proof': float(f"{social:.2f}"),
        'margin': float(f"{margin:.2f}"),
        'logistics': float(f"{logistics:.2f}"),
        'summary': summary,
        'explanations': explanations,
    }


def ensure_db():
    conn = database.get_connection(DB_PATH)
    database.initialize_database(conn)
    # Remove any legacy dummy products (IDs 1, 2, 3 or names that suggest test rows)
    try:
        cur = conn.cursor()
        # Remove known dummy/test products by name or id
        cur.execute(
            "DELETE FROM products WHERE id IN (1,2,3) OR lower(name) LIKE '%test%' OR lower(name) LIKE '%prueba%'"
        )
        conn.commit()
    except Exception:
        pass
    return conn


def parse_xlsx(binary: bytes):
    """Parse a minimal XLSX file into a list of dictionaries."""
    import zipfile
    import xml.etree.ElementTree as ET
    from io import BytesIO

    with zipfile.ZipFile(BytesIO(binary)) as z:
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            ss_root = ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in ss_root.findall('.//{*}si'):
                text = ''.join((t.text or '') for t in si.findall('.//{*}t'))
                shared.append(text)
        sheet_name = None
        for name in z.namelist():
            if name.startswith('xl/worksheets/sheet') and name.endswith('.xml'):
                sheet_name = name
                break
        if not sheet_name:
            return []
        root = ET.fromstring(z.read(sheet_name))
        rows = []
        for row in root.findall('.//{*}row'):
            values = []
            last_col_idx = 0
            for c in row.findall('{*}c'):
                cell_ref = c.attrib.get('r', '')
                letters = ''.join(ch for ch in cell_ref if ch.isalpha())
                col_idx = 0
                for ch in letters:
                    col_idx = col_idx * 26 + (ord(ch.upper()) - ord('A') + 1)
                while last_col_idx < col_idx - 1:
                    values.append('')
                    last_col_idx += 1
                val = ''
                cell_type = c.attrib.get('t')
                if cell_type == 's':
                    v = c.find('{*}v')
                    if v is not None:
                        try:
                            idx = int(v.text)
                            val = shared[idx] if idx < len(shared) else ''
                        except Exception:
                            val = ''
                elif cell_type == 'inlineStr':
                    tnode = c.find('{*}is/{*}t')
                    val = tnode.text if tnode is not None else ''
                else:
                    v = c.find('{*}v')
                    val = v.text if v is not None else ''
                values.append(val)
                last_col_idx = col_idx
            rows.append(values)
        while rows and all(not cell for cell in rows[0]):
            rows.pop(0)
        if not rows:
            return []
        headers = rows[0]
        records = []
        for r in rows[1:]:
            rec = {}
            for i, h in enumerate(headers):
                rec[h] = r[i] if i < len(r) else ''
            records.append(rec)
        return records


def _process_import_job(job_id: int, tmp_path: Path, filename: str) -> None:
    """Background task to import XLSX data into the database."""
    conn = ensure_db()
    rows_imported = 0
    inserted_ids: List[int] = []
    try:
        data = tmp_path.read_bytes()
        records = parse_xlsx(data)

        used_cols: set[str] = set()

        def find_key(keys, patterns):
            for k in keys:
                if k in used_cols:
                    continue
                sanitized = ''.join(ch.lower() for ch in k if ch.isalnum())
                for p in patterns:
                    if p in sanitized:
                        used_cols.add(k)
                        return k
            return None

        if records:
            headers = list(records[0].keys())
            # identify columns with tolerant synonyms
            rating_col = find_key(headers, ["rating", "stars", "valoracion", "puntuacion"])
            units_col = find_key(headers, ["unitssold", "units", "ventas", "sold"])
            revenue_col = find_key(headers, ["revenue", "sales", "ingresos"])
            conv_col = find_key(headers, ["conversion", "cr", "tasaconversion"])
            launch_col = find_key(headers, ["launchdate", "fecha", "date", "firstseen"])
            range_col = None
            if "Date Range" in headers:
                range_col = "Date Range"
                used_cols.add(range_col)
            else:
                range_col = find_key(headers, ["daterange", "fecharango", "rangofechas"])
            price_col = find_key(headers, ["price", "precio", "cost", "unitprice"])
            img_col = find_key(headers, ["imageurl", "image", "imagelink", "mainimage", "mainimageurl", "img", "imagen", "picture", "primaryimage"])
            name_col = find_key(headers, ["name", "productname", "title", "product", "producto"])
            desc_col = find_key(headers, ["description", "descripcion", "desc"])
            cat_col = find_key(headers, ["category", "categoria", "niche", "segment"])
            curr_col = find_key(headers, ["currency", "moneda"])

            metric_names = [
                "magnitud_deseo",
                "nivel_consciencia_headroom",
                "evidencia_demanda",
                "tasa_conversion",
                "ventas_por_dia",
                "recencia_lanzamiento",
                "competition_level_invertido",
                "facilidad_anuncio",
                "escalabilidad",
                "durabilidad_recurrencia",
            ]

            def sanitize(name: str) -> str:
                return "".join(ch for ch in name if ch.isalnum())

            metric_cols = {m: find_key(headers, [sanitize(m)]) for m in metric_names}

            def parse_number(val: Any) -> float | None:
                if val in (None, ''):
                    return None
                s = str(val).strip()
                if not s:
                    return None
                percent = '%' in s
                s = s.replace('%', '').replace(' ', '').replace(',', '.')
                s = re.sub(r'[^0-9.+-]', '', s)
                try:
                    num = float(s)
                    if percent:
                        num /= 100.0
                    return num
                except Exception:
                    return None

            def parse_text(val: Any) -> str | None:
                if val is None:
                    return None
                s = str(val).strip()
                return s or None

            numeric_metrics = {
                "evidencia_demanda",
                "tasa_conversion",
                "ventas_por_dia",
                "recencia_lanzamiento",
            }

            cur = conn.cursor()
            cur.execute("BEGIN")
            cur.execute("SELECT COUNT(*) FROM products")
            count = cur.fetchone()[0]
            cur.execute("SELECT COALESCE(MAX(id), -1) FROM products")
            max_id = cur.fetchone()[0]
            is_empty = count == 0
            base_id = 0 if is_empty else (max_id + 1)
            rows_validas = []
            for row in records:
                name = (row.get(name_col) or '').strip() if name_col else None
                if not name:
                    continue
                description = (row.get(desc_col) or '').strip() if desc_col else None
                category = (row.get(cat_col) or '').strip() if cat_col else None
                price = None
                if price_col and row.get(price_col):
                    try:
                        price = float(str(row.get(price_col)).replace(',', '.'))
                    except Exception:
                        price = None
                currency = (row.get(curr_col) or '').strip() if curr_col else None
                image_url = (row.get(img_col) or '').strip() if img_col else None

                date_range = (row.get(range_col) or '').strip() if range_col else ''

                extras = {}
                metrics: dict[str, object] = {}

                rating_val = None
                if rating_col and row.get(rating_col) not in (None, ''):
                    try:
                        s = str(row.get(rating_col)).strip().replace(' ', '').replace(',', '.')
                        s = re.sub(r'[^0-9.]+', '', s)
                        if s.count('.') > 1:
                            parts = s.split('.')
                            s = ''.join(parts[:-1]) + '.' + parts[-1]
                        rating_val = float(s) if s else None
                    except Exception:
                        rating_val = None
                if rating_val is not None:
                    extras['rating'] = rating_val
                    extras['Product Rating'] = rating_val

                units_val = None
                if units_col and row.get(units_col) not in (None, ''):
                    try:
                        s = re.sub(r'[^0-9]+', '', str(row.get(units_col)))
                        units_val = int(s) if s else None
                    except Exception:
                        units_val = None
                if units_val is not None:
                    extras['units_sold'] = units_val
                    extras['Item Sold'] = units_val

                revenue_val = None
                if revenue_col and row.get(revenue_col) not in (None, ''):
                    try:
                        s = str(row.get(revenue_col)).strip().replace(' ', '').replace(',', '.')
                        s = re.sub(r'[^0-9.]+', '', s)
                        if s.count('.') > 1:
                            parts = s.split('.')
                            s = ''.join(parts[:-1]) + '.' + parts[-1]
                        revenue_val = float(s) if s else None
                    except Exception:
                        revenue_val = None
                if revenue_val is None and price is not None and units_val is not None:
                    revenue_val = price * units_val
                if revenue_val is not None:
                    extras['revenue'] = revenue_val
                    extras['Revenue($)'] = revenue_val

                def set_extra(col, key):
                    val = row.get(col) if col else None
                    if val is not None and str(val).strip():
                        extras[key] = str(val).strip()

                set_extra(conv_col, 'conversion_rate')
                set_extra(launch_col, 'launch_date')

                for m in metric_names:
                    col = metric_cols.get(m)
                    raw = row.get(col) if col else None
                    if raw not in (None, ''):
                        if m in numeric_metrics:
                            val = parse_number(raw)
                        else:
                            val = parse_text(raw)
                        if val is not None:
                            metrics[m] = val

                recognized = {
                    name_col,
                    desc_col,
                    cat_col,
                    price_col,
                    curr_col,
                    img_col,
                    rating_col,
                    units_col,
                    revenue_col,
                    conv_col,
                    launch_col,
                    range_col,
                }
                recognized.update(c for c in metric_cols.values() if c)
                for k, v in row.items():
                    if k not in recognized:
                        extras[k] = v

                rows_validas.append(
                    (name, description, category, price, currency, image_url, date_range, extras, metrics)
                )
            for idx, (name, description, category, price, currency, image_url, date_range, extra_cols, metrics) in enumerate(rows_validas):
                row_id = base_id + idx
                database.insert_product(
                    conn,
                    name=name,
                    description=description,
                    category=category,
                    price=price,
                    currency=currency,
                    image_url=image_url,
                    date_range=date_range,
                    source=filename,
                    extra=extra_cols,
                    commit=False,
                    product_id=row_id,
                )
                if metrics:
                    database.update_product(conn, row_id, **metrics)
                rows_imported += 1
                inserted_ids.append(row_id)
            conn.commit()
        
        products_all = [dict(r) for r in database.list_products(conn)]
        ranges = winner_calc.compute_ranges(products_all)
        weights = config.get_scoring_v2_weights()
        total_w = sum(weights.values())
        if total_w <= 0:
            logger.warning(
                "Winner Score import: weight sum <= 0, using uniform weights",
            )
            n = len(weights) or 1
            weights = {k: 1 / n for k in weights}
        else:
            weights = {k: v / total_w for k, v in weights.items()}
        logger.info(
            "Winner Score import: weights=%s sum=%s",
            weights,
            sum(weights.values()),
        )
        updated_scores = 0
        skipped_scores = 0
        for prod in products_all:
            pid = prod["id"]
            existing = database.get_scores_for_product(conn, pid)
            if any((dict(sc).get("winner_score_v2_pct") or 0) > 0 for sc in existing):
                skipped_scores += 1
                continue
            missing: list[str] = []
            used: list[str] = []
            pct_val = winner_calc.score_product(prod, weights, ranges, missing, used)
            used_count = len(used)
            missing_count = len(missing)
            fallback = used_count == 0 or pct_val is None or (isinstance(pct_val, float) and math.isnan(pct_val))
            if fallback:
                pct = 50
            else:
                pct = max(0, min(100, round(pct_val * 100)))
            if missing_count > 0:
                level = logging.WARNING if fallback else logging.INFO
                logger.log(
                    level,
                    "Winner Score: product=%s used=%d missing=%d fallback=%s",
                    pid,
                    used_count,
                    missing_count,
                    str(fallback).lower(),
                )
            database.insert_score(
                conn,
                product_id=pid,
                model="winner_v2",
                total_score=0,
                momentum=0,
                saturation=0,
                differentiation=0,
                social_proof=0,
                margin=0,
                logistics=0,
                summary="",
                explanations={},
                winner_score_v2_pct=pct,
                commit=False,
            )
            updated_scores += 1
        conn.commit()
        logger.info(
            "Winner Score import/backfill: imported=%d updated=%d skipped=%d",
            len(inserted_ids),
            updated_scores,
            skipped_scores,
        )
        if inserted_ids and config.is_auto_fill_ia_on_import_enabled():
            database.start_import_job_ai(conn, job_id, len(inserted_ids))
            cfg_cost = config.get_ai_cost_config()
            res = ai_columns.fill_ai_columns(
                inserted_ids,
                model=cfg_cost.get("model"),
                batch_mode=len(inserted_ids) >= cfg_cost.get("useBatchWhenCountGte", 300),
                cost_cap_usd=cfg_cost.get("costCapUSD"),
            )
            counts = res.get("counts", {})
            database.update_import_job_ai_progress(conn, job_id, counts.get("n_procesados", 0))
            database.set_import_job_ai_counts(conn, job_id, counts, res.get("pending_ids", []))
            if res.get("error"):
                database.set_import_job_ai_error(conn, job_id, "No se pudieron completar las columnas con IA: revisa la API.")
        database.complete_import_job(conn, job_id, rows_imported, updated_scores)
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        database.fail_import_job(conn, job_id, str(exc))
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


def resume_incomplete_imports():
    """Mark stale pending imports as failed and remove orphan temp files."""
    conn = ensure_db()
    database.mark_stale_pending_imports(conn, 5)
    tmp_dir = APP_DIR / 'uploads'
    if tmp_dir.exists():
        cur = conn.cursor()
        cur.execute("SELECT temp_path FROM import_jobs")
        valid = {Path(row[0]) for row in cur.fetchall() if row[0]}
        for f in tmp_dir.glob('import_*'):
            if f not in valid:
                try:
                    f.unlink()
                except Exception:
                    pass


class _SilentWriter:
    """Wrapper around a socket writer that ignores connection errors."""

    def __init__(self, raw):
        self._raw = raw

    def write(self, data):
        try:
            self._raw.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def flush(self):
        try:
            self._raw.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def __getattr__(self, name):
        return getattr(self._raw, name)


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "ProductResearchCopilot/1.0"

    def setup(self):
        super().setup()
        self.wfile = _SilentWriter(self.wfile)

    def _set_json(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def _set_html(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def safe_write(self, func):
        try:
            func()
            return True
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return False

    def send_json(self, obj: Any, status: int = 200):
        self._set_json(status)
        self.wfile.write(json.dumps(obj).encode('utf-8'))

    def _safe_write(self, data: bytes) -> bool:
        return self.safe_write(lambda: self.wfile.write(data))

    def _serve_static(self, rel_path: str):
        file_path = STATIC_DIR / rel_path
        if not file_path.exists():
            self.send_error(404)
            return
        if file_path.suffix == ".js":
            ctype = "application/javascript"
        elif file_path.suffix == ".css":
            ctype = "text/css"
        elif file_path.suffix == ".html":
            ctype = "text/html"
        else:
            ctype = "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.end_headers()
        with open(file_path, "rb") as f:
            self.wfile.write(f.read())

    def _parse_multipart_file(self):
        ctype = self.headers.get('Content-Type', '')
        if not ctype.startswith('multipart/form-data'):
            return None, None
        boundary_key = 'boundary='
        if boundary_key not in ctype:
            return None, None
        boundary = ctype.split(boundary_key, 1)[1].strip().strip('"')
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            length = 0
        if length <= 0:
            return None, None
        body = self.rfile.read(length)
        parser = BytesParser(policy=default)
        header_bytes = f'Content-Type: multipart/form-data; boundary={boundary}\r\n\r\n'.encode('utf-8')
        msg = parser.parsebytes(header_bytes + body)
        for part in msg.iter_parts():
            disp = part.get_content_disposition()
            if disp == 'form-data' and part.get_param('name', header='content-disposition') == 'file':
                filename = part.get_filename()
                data = part.get_payload(decode=True)
                return filename, data
        return None, None

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/" or path == "/index.html":
            self._serve_static("index.html")
            return
        if path.startswith("/static/"):
            rel = path[len("/static/") :]
            self._serve_static(rel)
            return
        if path == "/_import_history":
            params = parse_qs(parsed.query)
            try:
                limit = int(params.get("limit", ["20"])[0])
            except Exception:
                limit = 20
            conn = ensure_db()
            rows = [dict(r) for r in database.get_import_history(conn, limit)]
            self.safe_write(lambda: self.send_json(rows))
            return
        if path == "/_import_status":
            params = parse_qs(parsed.query)
            try:
                task_id = int(params.get("task_id", ["0"])[0])
            except Exception:
                self.safe_write(lambda: self.send_json({"error": "invalid task_id"}, status=400))
                return
            conn = ensure_db()
            row = database.get_import_job(conn, task_id)
            if row:
                data = dict(row)
                try:
                    if data.get("ai_counts"):
                        data["ai_counts"] = json.loads(data["ai_counts"])
                except Exception:
                    data["ai_counts"] = {}
                try:
                    if data.get("ai_pending"):
                        data["pending_ids"] = json.loads(data["ai_pending"])
                    else:
                        data["pending_ids"] = []
                except Exception:
                    data["pending_ids"] = []
                data.pop("ai_pending", None)
                data["message"] = (
                    "Importando productos, por favor espera... El winner score se ha calculado."
                )
                data["imported"] = data.get("rows_imported", 0)
                data["winner_score_updated"] = data.get("winner_score_updated", 0)
                self.safe_write(lambda: self.send_json(data))
            else:
                self.safe_write(lambda: self.send_json({"error": "not found"}, status=404))
            return
        if path == "/products":
            # Return a list of products including extra metadata for UI display
            conn = ensure_db()
            rows = []
            for p in database.list_products(conn):
                scores = database.get_scores_for_product(conn, p["id"])
                score = scores[0] if scores else None
                extra = p["extra"] if "extra" in p.keys() else {}
                try:
                    extra_dict = json.loads(extra) if isinstance(extra, str) else (extra or {})
                except Exception:
                    extra_dict = {}
                if 'rating' in extra_dict and 'Product Rating' not in extra_dict:
                    extra_dict['Product Rating'] = extra_dict['rating']
                if 'units_sold' in extra_dict and 'Item Sold' not in extra_dict:
                    extra_dict['Item Sold'] = extra_dict['units_sold']
                if 'revenue' in extra_dict and 'Revenue($)' not in extra_dict:
                    extra_dict['Revenue($)'] = extra_dict['revenue']
                score_value = None
                if score:
                    key = (
                        "winner_score_v2_pct"
                        if config.is_scoring_v2_enabled()
                        else "total_score"
                    )
                    if key in score.keys():
                        score_value = score[key]
                    else:
                        score_value = None
                    breakdown_data = {}
                    if config.is_scoring_v2_enabled():
                        try:
                            raw_breakdown = (
                                score["winner_score_v2_breakdown"]
                                if "winner_score_v2_breakdown" in score.keys()
                                else None
                            )
                            breakdown_data = json.loads(raw_breakdown or "{}")
                        except Exception:
                            breakdown_data = {}
                dr = p["date_range"]
                if dr is None:
                    dr = extra_dict.get("date_range")
                row = {
                    "id": p["id"],
                    "name": p["name"],
                    "category": p["category"],
                    "price": p["price"],
                    "image_url": p["image_url"],
                    "desire": p["desire"],
                    "desire_magnitude": p["desire_magnitude"],
                    "awareness_level": p["awareness_level"],
                    "competition_level": p["competition_level"],
                    "rating": extra_dict.get("rating"),
                    "units_sold": extra_dict.get("units_sold"),
                    "revenue": extra_dict.get("revenue"),
                    "conversion_rate": extra_dict.get("conversion_rate"),
                    "launch_date": extra_dict.get("launch_date"),
                    "date_range": dr or "",
                    "extras": extra_dict,
                }
                if config.is_scoring_v2_enabled():
                    row["winner_score_v2_pct"] = score_value
                    if score:
                        row["winner_score_v2_breakdown"] = breakdown_data
                else:
                    row["score"] = score_value
                rows.append(row)
            self._set_json()
            self.wfile.write(json.dumps(rows).encode("utf-8"))
            return
        if path == "/config":
            # return stored configuration (without exposing the API key)
            cfg = config.load_config()
            key = cfg.get("api_key") or ""
            data = {
                "model": cfg.get("model", "gpt-4o"),
                "weights": cfg.get("weights", {}),
                "scoring_v2_weights": cfg.get("scoring_v2_weights", {}),
                "has_api_key": bool(key),
            }
            if key:
                data["api_key_last4"] = key[-4:]
                data["api_key_length"] = len(key)
                data["api_key_hash"] = hashlib.sha256(key.encode("utf-8")).hexdigest()
            self._set_json()
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return
        if path == "/settings/winner-score":
            cfg = config.load_config()
            weights = cfg.get("scoring_v2_weights", {})
            self._set_json()
            self.wfile.write(json.dumps(weights).encode("utf-8"))
            return
        if path.startswith("/score/"):
            try:
                pid = int(path.split("/")[-1])
            except ValueError:
                self.send_error(400, "Invalid product ID")
                return
            conn = ensure_db()
            scores = database.get_scores_for_product(conn, pid)
            if not scores:
                self._set_json(404)
                self.wfile.write(json.dumps({"error": "No score"}).encode("utf-8"))
                return
            score = scores[0]
            self._set_json()
            self.wfile.write(json.dumps({key: score[key] for key in score.keys()}).encode("utf-8"))
            return
        if path == "/lists":
            # return all saved groups/lists with product counts
            conn = ensure_db()
            lsts = database.get_lists(conn)
            data = []
            for l in lsts:
                data.append({"id": l["id"], "name": l["name"], "count": l["count"]})
            self._set_json()
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return
        if path.startswith("/list/"):
            # return products belonging to a list
            parts = path.strip("/").split("/")
            if len(parts) == 2 and parts[0] == "list":
                try:
                    lid = int(parts[1])
                except Exception:
                    self.send_error(400, "Invalid list ID")
                    return
                conn = ensure_db()
                prods = database.get_products_in_list(conn, lid)
                rows = []
                for p in prods:
                    scores = database.get_scores_for_product(conn, p["id"])
                    score = scores[0] if scores else None
                    extra = p["extra"] if "extra" in p.keys() else {}
                    try:
                        extra_dict = json.loads(extra) if isinstance(extra, str) else (extra or {})
                    except Exception:
                        extra_dict = {}
                    score_value = None
                    if score:
                        key = (
                            "winner_score_v2_pct"
                            if config.is_scoring_v2_enabled()
                            else "total_score"
                        )
                        if key in score.keys():
                            score_value = score[key]
                        else:
                            score_value = None
                        breakdown_data = {}
                        if config.is_scoring_v2_enabled():
                            try:
                                raw_breakdown = (
                                    score["winner_score_v2_breakdown"]
                                    if "winner_score_v2_breakdown" in score.keys()
                                    else None
                                )
                                breakdown_data = json.loads(raw_breakdown or "{}")
                            except Exception:
                                breakdown_data = {}
                    row = {
                        "id": p["id"],
                        "name": p["name"],
                        "category": p["category"],
                        "price": p["price"],
                        "image_url": p["image_url"],
                        "extras": extra_dict,
                    }
                    if config.is_scoring_v2_enabled():
                        row["winner_score_v2_pct"] = score_value
                        if score:
                            row["winner_score_v2_breakdown"] = breakdown_data
                    else:
                        row["score"] = score_value
                    rows.append(row)
                self._set_json()
                self.wfile.write(json.dumps(rows).encode("utf-8"))
                return
        # trends endpoint: compute analytics for categories, keywords and scatter plots
        if path == "/trends":
            qs = parse_qs(parsed.query)
            start_str = qs.get("start", [None])[0]
            end_str = qs.get("end", [None])[0]
            from datetime import datetime

            def parse_date_str(val: str | None):
                if not val:
                    return None
                for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y"):
                    try:
                        return datetime.strptime(val, fmt)
                    except Exception:
                        continue
                return None

            start_dt = parse_date_str(start_str)
            end_dt = parse_date_str(end_str)

            conn = ensure_db()
            prods = database.list_products(conn)
            from collections import Counter, defaultdict

            cat_rev_growth: Dict[str, float] = defaultdict(float)
            cat_unit_growth: Dict[str, float] = defaultdict(float)
            cat_rev: Dict[str, float] = defaultdict(float)
            cat_units: Dict[str, float] = defaultdict(float)
            cat_product_count: Dict[str, int] = defaultdict(int)
            cat_price_total: Dict[str, float] = defaultdict(float)
            cat_price_count: Dict[str, int] = defaultdict(int)
            cat_rating_total: Dict[str, float] = defaultdict(float)
            cat_rating_count: Dict[str, int] = defaultdict(int)
            word_counter = Counter()
            brand_counter = Counter()
            scatter_rating_revenue = []
            scatter_price_revenue = []
            total_revenue = 0.0
            total_units = 0.0
            price_sum = 0.0
            price_count = 0
            top_product_name = None
            top_product_rev = 0.0

            stopwords = set([
                "the", "and", "for", "with", "a", "an", "de", "la", "el", "para", "y", "con", "un", "una", "los", "las", "en", "por", "to", "of",
            ])

            def parse_float(val):
                try:
                    return float(str(val).replace("%", "").replace("$", "").replace(",", "").strip())
                except Exception:
                    return None
            import re

            for p in prods:
                try:
                    extras = json.loads(p["extra"]) if p["extra"] else {}
                except Exception:
                    extras = {}

                launch_val = extras.get("Launch Date")
                launch_dt = parse_date_str(str(launch_val)) if launch_val else None
                if start_dt and (launch_dt is None or launch_dt < start_dt):
                    continue
                if end_dt and (launch_dt is None or launch_dt > end_dt):
                    continue
                cat = (p["category"] or "").strip().lower()
                if cat:
                    cat_product_count[cat] += 1
                rev_growth = None
                unit_growth = None
                for k, v in extras.items():
                    lk = k.lower()
                    if rev_growth is None and "revenue" in lk and "growth" in lk:
                        rev_growth = parse_float(v)
                    if unit_growth is None and ("item" in lk or "unit" in lk) and "growth" in lk:
                        unit_growth = parse_float(v)
                if rev_growth is not None and cat:
                    cat_rev_growth[cat] += rev_growth
                if unit_growth is not None and cat:
                    cat_unit_growth[cat] += unit_growth
                revenue = None
                for key in ["Revenue($)", "Revenue"]:
                    if key in extras:
                        revenue = parse_float(extras[key])
                        if revenue is not None:
                            break
                item_sold = parse_float(extras.get("Item Sold"))
                if revenue is not None:
                    total_revenue += revenue
                    if cat:
                        cat_rev[cat] += revenue
                    if revenue > top_product_rev:
                        top_product_rev = revenue
                        top_product_name = p["name"]
                if item_sold is not None:
                    total_units += item_sold
                    if cat:
                        cat_units[cat] += item_sold
                name = (p["name"] or "").lower()
                words = re.split(r"[^a-záéíóúüñ0-9]+", name)
                for w in words:
                    if not w or w in stopwords or len(w) < 3:
                        continue
                    word_counter[w] += 1
                tokens = re.split(r"[^A-Za-z0-9]+", p["name"] or "")
                if tokens:
                    brand = tokens[0].lower()
                    if brand and brand not in stopwords and len(brand) >= 3:
                        brand_counter[brand] += 1
                rating = parse_float(extras.get("Product Rating"))
                if rating is not None:
                    if revenue is not None and item_sold is not None:
                        scatter_rating_revenue.append({
                            "x": rating,
                            "y": revenue,
                            "r": item_sold,
                            "label": p["name"],
                            "units": item_sold,
                            "rating": rating,
                            "revenue": revenue,
                        })
                    if cat:
                        cat_rating_total[cat] += rating
                        cat_rating_count[cat] += 1
                avg_price = None
                for key in ["Avg. Unit Price($)", "Avg Unit Price($)", "Avg. Unit Price"]:
                    if key in extras:
                        avg_price = parse_float(extras[key])
                        if avg_price is not None:
                            break
                if avg_price is not None:
                    price_sum += avg_price
                    price_count += 1
                    if cat:
                        cat_price_total[cat] += avg_price
                        cat_price_count[cat] += 1
                    if revenue is not None:
                        scatter_price_revenue.append({
                            "x": avg_price,
                            "y": revenue,
                            "label": p["name"],
                            "units": item_sold,
                            "rating": rating,
                            "revenue": revenue,
                        })

            cat_rev_per_unit = []
            for cat, rev in cat_rev.items():
                units = cat_units.get(cat, 0)
                if units:
                    cat_rev_per_unit.append((cat, rev / units))

            top_rev_growth = sorted(cat_rev_growth.items(), key=lambda x: x[1], reverse=True)[:10]
            top_unit_growth = sorted(cat_unit_growth.items(), key=lambda x: x[1], reverse=True)[:10]
            cat_rev_per_unit.sort(key=lambda x: x[1], reverse=True)
            top_words = word_counter.most_common(10)
            top_brands = [(b.title(), c) for b, c in brand_counter.most_common(10)]
            avg_price = price_sum / price_count if price_count else 0.0
            top_cat = None
            if cat_rev:
                top_cat = max(cat_rev.items(), key=lambda x: x[1])[0]
            category_compare = []
            category_summary = []
            for cat, count in cat_product_count.items():
                total_r = cat_rev.get(cat, 0.0)
                total_u = cat_units.get(cat, 0.0)
                avg_rev = total_r / count if count else 0.0
                avg_units = total_u / count if count else 0.0
                avg_p = cat_price_total[cat] / cat_price_count[cat] if cat_price_count[cat] else 0.0
                avg_r = cat_rating_total[cat] / cat_rating_count[cat] if cat_rating_count[cat] else 0.0
                category_compare.append({
                    "category": cat.title(),
                    "products": count,
                    "avg_revenue": avg_rev,
                    "avg_units": avg_units,
                    "avg_price": avg_p,
                    "total_revenue": total_r,
                    "total_units": total_u,
                    "avg_rating": avg_r,
                })
                
                category_summary.append({
                    "category": cat.title(),
                    "products": count,
                    "total_units": total_u,
                    "total_revenue": total_r,
                    "avg_price": avg_p,
                    "avg_rating": avg_r,
                })

            rows = []
            for p in prods:
                scores = database.get_scores_for_product(conn, p["id"])
                score_val = None
                if scores:
                    key = (
                        "winner_score_v2_pct"
                        if config.is_scoring_v2_enabled()
                        else "total_score"
                    )
                    try:
                        score_val = scores[0][key]
                    except Exception:
                        score_val = None
                if score_val is not None:
                    rows.append((p["id"], p["name"], score_val))
            rows.sort(key=lambda x: x[2], reverse=True)
            key_name = "winner_score_v2_pct" if config.is_scoring_v2_enabled() else "score"
            top_products = [{"id": r[0], "name": r[1], key_name: r[2]} for r in rows[:10]]

            self._set_json()
            self.wfile.write(json.dumps({
                "kpis": {
                    "total_revenue": total_revenue,
                    "total_units": total_units,
                    "avg_price": avg_price,
                    "top_category": top_cat.title() if top_cat else None,
                    "top_product": top_product_name,
                },
                "category_compare": category_compare,
                "category_summary": category_summary,
                "cat_revenue_growth": top_rev_growth,
                "cat_units_growth": top_unit_growth,
                "cat_rev_per_unit": cat_rev_per_unit[:10],
                "keywords": top_words,
                "brands": top_brands,
                "top_products": top_products,
                "scatter_rating_revenue": scatter_rating_revenue,
                "scatter_price_revenue": scatter_price_revenue,
            }).encode("utf-8"))
            return
# export selected or all products
        if path == "/export":
            qs = parse_qs(parsed.query)
            fmt = qs.get('format', ['csv'])[0]
            id_str = qs.get('ids', [None])[0]
            conn = ensure_db()
            items: list[sqlite3.Row] = []
            if id_str:
                try:
                    ids = [int(x) for x in id_str.split(',') if x]
                except Exception:
                    ids = []
                for pid in ids:
                    p = database.get_product(conn, pid)
                    if p:
                        items.append(p)
            else:
                items = database.list_products(conn)
            rows = []
            for p in items:
                scores = database.get_scores_for_product(conn, p['id'])
                score_val = None
                if scores:
                    sc = scores[0]
                    if 'winner_score_v2_pct' in sc.keys():
                        score_val = sc['winner_score_v2_pct']
                rows.append(
                    [
                        p['id'],
                        p['name'],
                        score_val,
                        p['desire'],
                        p['desire_magnitude'],
                        p['awareness_level'],
                        p['competition_level'],
                        p['date_range'],
                    ]
                )
            headers = ["id", "name", "Winner Score v2", "Desire", "Desire Magnitude", "Awareness Level", "Competition Level", "Date Range"]
            if fmt == 'xlsx':
                try:
                    from openpyxl import Workbook
                except Exception:
                    self._set_json(500)
                    self.wfile.write(json.dumps({"error": "openpyxl not installed"}).encode('utf-8'))
                    return
                wb = Workbook()
                ws = wb.active
                ws.append(headers)
                for r in rows:
                    ws.append(r)
                from io import BytesIO
                bio = BytesIO()
                wb.save(bio)
                data = bio.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                self.send_header("Content-Disposition", "attachment; filename=export.xlsx")
                self.end_headers()
                self.wfile.write(data)
                return
            else:
                import csv
                from io import StringIO
                output = StringIO()
                writer = csv.writer(output)
                writer.writerow(headers)
                writer.writerows(rows)
                csv_data = output.getvalue().encode('utf-8')
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", "attachment; filename=export.csv")
                self.end_headers()
                self.wfile.write(csv_data)
                return
        # unknown
        self.send_error(404)
        # unknown
        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/analyze/titles":
            self.handle_analyze_titles()
            return
        if path == "/upload":
            self.handle_upload()
            return
        if path == "/evaluate_all":
            self.handle_evaluate_all()
            return
        if path == "/setconfig":
            self.handle_setconfig()
            return
        if path == "/custom_gpt":
            self.handle_custom_gpt()
            return
        if path == "/api/ba/insights":
            self.handle_ba_insights()
            return
        if path == "/api/ia/batch-columns":
            self.handle_ia_batch_columns()
            return
        if path == "/auto_weights":
            self.handle_auto_weights()
            return
        if path == "/scoring/v2/auto-weights-gpt":
            self.handle_scoring_v2_auto_weights_gpt()
            return
        if path == "/scoring/v2/auto-weights-stat":
            self.handle_scoring_v2_auto_weights_stat()
            return
        if path == "/scoring/v2/gpt-evaluate":
            self.handle_scoring_v2_gpt_evaluate()
            return
        if path == "/scoring/v2/gpt-summary":
            self.handle_scoring_v2_gpt_summary()
            return
        if path == "/scoring/v2/generate":
            self.handle_scoring_v2_generate()
            return
        if path == "/delete":
            self.handle_delete()
            return
        if path == "/remove_from_list":
            self.handle_remove_from_list()
            return
        if path == "/create_list":
            self.handle_create_list()
            return
        if path == "/delete_list":
            self.handle_delete_list()
            return
        if path == "/add_to_list":
            self.handle_add_to_list()
            return
        if path == "/shutdown":
            self.handle_shutdown()
            return
        if path == "/products":
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length).decode('utf-8')
            try:
                data = json.loads(body)
                if not isinstance(data, dict):
                    raise ValueError
            except Exception:
                self._set_json(400)
                self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode('utf-8'))
                return
            if "price" in data and data.get("source") != "import":
                logger.info(
                    "price field is read-only; ignoring on create (source=%s)",
                    data.get("source"),
                )
                data.pop("price", None)
            conn = ensure_db()
            pid = database.insert_product(
                conn,
                name=data.get("name", ""),
                description=data.get("description"),
                category=data.get("category"),
                price=data.get("price"),
                currency=data.get("currency"),
                image_url=data.get("image_url"),
                source=data.get("source"),
                desire=data.get("desire"),
                desire_magnitude=data.get("desire_magnitude"),
                awareness_level=data.get("awareness_level"),
                competition_level=data.get("competition_level"),
                extra=data.get("extras"),
            )
            product = database.get_product(conn, pid)
            self._set_json()
            self.wfile.write(json.dumps(dict(product)).encode('utf-8'))
            return
        self.send_error(404)

    def do_PUT(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path.startswith("/products/"):
            try:
                pid = int(path.split("/")[-1])
            except Exception:
                self._set_json(400)
                self.wfile.write(json.dumps({"error": "Invalid ID"}).encode('utf-8'))
                return
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length).decode('utf-8')
            try:
                data = json.loads(body)
                if not isinstance(data, dict):
                    raise ValueError
            except Exception:
                self._set_json(400)
                self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode('utf-8'))
                return
            if "price" in data and data.get("source") != "import":
                logger.info(
                    "price field is read-only; ignoring update for product %s (source=%s)",
                    pid,
                    data.get("source"),
                )
                data.pop("price", None)
            conn = ensure_db()
            database.update_product(conn, pid, **data)
            product = database.get_product(conn, pid)
            if product:
                self._set_json()
                self.wfile.write(json.dumps(dict(product)).encode('utf-8'))
            else:
                self.send_error(404)
            return
        if path == "/settings/winner-score":
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length).decode('utf-8')
            try:
                data = json.loads(body)
                if not isinstance(data, dict):
                    raise ValueError
            except Exception:
                self._set_json(400)
                self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode('utf-8'))
                return
            cfg = config.load_config()
            weights_v2 = cfg.get('scoring_v2_weights', {})
            for k, v in data.items():
                try:
                    weights_v2[k] = float(v)
                except Exception:
                    continue
            cfg['scoring_v2_weights'] = weights_v2
            config.save_config(cfg)
            self._set_json()
            self.wfile.write(json.dumps({"status": "ok"}).encode('utf-8'))
            return
        self.send_error(404)

    def handle_analyze_titles(self):
        """Endpoint for Title Analyzer.

        Accepts either JSON array of objects or a CSV/XLSX file upload under
        multipart/form-data.  Each item must include a ``title`` field and may
        optionally include ``price`` and ``rating``.  Returns a JSON response
        with the normalized items and placeholder analysis results.
        """
        ctype = self.headers.get('Content-Type', '')
        items = []
        if ctype.startswith('application/json'):
            length = int(self.headers.get('Content-Length', 0))
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw.decode('utf-8'))
                if isinstance(data, list):
                    for obj in data:
                        title = (obj.get('title') or obj.get('name') or '').strip()
                        if not title:
                            continue
                        item = {'title': title}
                        if obj.get('price') is not None:
                            item['price'] = obj.get('price')
                        if obj.get('rating') is not None:
                            item['rating'] = obj.get('rating')
                        items.append(item)
                else:
                    raise ValueError('Expected list')
            except Exception:
                self.send_error(400, 'Invalid JSON')
                return
        elif ctype.startswith('multipart/form-data'):
            filename, data = self._parse_multipart_file()
            if not filename or data is None:
                self.send_error(400, 'No file provided')
                return
            filename = Path(filename).name
            ext = Path(filename).suffix.lower()

            def find_key(keys, patterns):
                for k in keys:
                    sanitized = ''.join(ch.lower() for ch in k if ch.isalnum())
                    for p in patterns:
                        if p in sanitized:
                            return k
                return None

            if ext == '.csv':
                import csv
                text = data.decode('utf-8', errors='ignore')
                reader = csv.DictReader(text.splitlines())
                headers = reader.fieldnames or []
                title_col = find_key(headers, ['title', 'name', 'productname', 'product_name'])
                price_col = find_key(headers, ['price'])
                rating_col = find_key(headers, ['rating', 'stars'])
                for row in reader:
                    title = (row.get(title_col) or '').strip() if title_col else ''
                    if not title:
                        continue
                    item = {'title': title}
                    if price_col and row.get(price_col):
                        try:
                            item['price'] = float(str(row[price_col]).replace(',', '.'))
                        except Exception:
                            pass
                    if rating_col and row.get(rating_col):
                        try:
                            item['rating'] = float(str(row[rating_col]).replace(',', '.'))
                        except Exception:
                            pass
                    items.append(item)
            elif ext in ('.xlsx', '.xlsm', '.xltx', '.xltm'):
                try:
                    import openpyxl
                except Exception:
                    self.send_error(500, 'openpyxl is required for XLSX files')
                    return
                wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True)
                ws = wb.active
                rows = ws.iter_rows(values_only=True)
                try:
                    headers = [str(h).strip() if h else '' for h in next(rows)]
                except StopIteration:
                    headers = []
                title_col = find_key(headers, ['title', 'name', 'productname', 'product_name'])
                price_col = find_key(headers, ['price'])
                rating_col = find_key(headers, ['rating', 'stars'])
                title_idx = headers.index(title_col) if title_col in headers else None
                price_idx = headers.index(price_col) if price_col in headers else None
                rating_idx = headers.index(rating_col) if rating_col in headers else None
                for row in rows:
                    if title_idx is None or title_idx >= len(row):
                        continue
                    title = (str(row[title_idx]).strip() if row[title_idx] else '')
                    if not title:
                        continue
                    item = {'title': title}
                    if price_idx is not None and price_idx < len(row) and row[price_idx] is not None:
                        try:
                            item['price'] = float(str(row[price_idx]).replace(',', '.'))
                        except Exception:
                            pass
                    if rating_idx is not None and rating_idx < len(row) and row[rating_idx] is not None:
                        try:
                            item['rating'] = float(str(row[rating_idx]).replace(',', '.'))
                        except Exception:
                            pass
                    items.append(item)
            else:
                self.send_error(400, 'Unsupported file type')
                return
        else:
            self.send_error(400, 'Unsupported Content-Type')
            return

        result = title_analyzer.analyze_titles(items)
        resp = json.dumps({'items': result}).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(resp)

    def handle_upload(self):
        filename, data = self._parse_multipart_file()
        if not filename or data is None:
            self.send_error(400, "No file provided")
            return
        filename = Path(filename).name
        ext = Path(filename).suffix.lower()
        if ext in (".xlsx", ".xls"):
            tmp_dir = APP_DIR / "uploads"
            tmp_dir.mkdir(exist_ok=True)
            tmp_path = tmp_dir / f"import_{int(time.time()*1000)}{ext}"
            with open(tmp_path, "wb") as f:
                f.write(data)
            conn = ensure_db()
            job_id = database.create_import_job(conn, str(tmp_path))
            threading.Thread(target=_process_import_job, args=(job_id, tmp_path, filename), daemon=True).start()
            self.safe_write(lambda: self.send_json({"task_id": job_id}, status=202))
            return
        conn = ensure_db()
        inserted = 0
        inserted_ids = []
        try:
            # helper to find a column key ignoring spaces and punctuation
            def find_key(keys, patterns):
                for k in keys:
                    sanitized = ''.join(ch.lower() for ch in k if ch.isalnum())
                    for p in patterns:
                        if p in sanitized:
                            return k
                return None
            if ext == ".csv":
                import csv
                text = data.decode('utf-8', errors='ignore')
                reader = csv.DictReader(text.splitlines())
                headers = reader.fieldnames or []
                # find matching columns
                name_col = find_key(headers, ["name", "nombre", "productname", "product", "title"])
                desc_col = find_key(headers, ["description", "descripcion", "desc"])
                cat_col = find_key(headers, ["category", "categoria", "cat"])
                price_col = find_key(headers, ["price", "precio", "cost", "unitprice"])
                curr_col = find_key(headers, ["currency", "moneda"])
                img_col = find_key(headers, ["image", "imagen", "img", "picture", "imgurl"])
                desire_col = find_key(headers, ["desire", "deseo"])
                desire_mag_col = find_key(headers, ["desiremagnitude", "desiremag", "magnituddeseo"])
                awareness_col = find_key(headers, ["awarenesslevel", "awareness", "nivelconsciencia"])
                competition_col = find_key(headers, ["competitionlevel", "competition", "saturacionmercado"])
                # collect names for potential simplification
                rows_data = []
                names_list = []
                for row in reader:
                    name = (row.get(name_col) or '').strip() if name_col else None
                    if not name:
                        continue
                    names_list.append(name)
                    rows_data.append(row)
                # call OpenAI to simplify names once per file
                name_map = {}
                api_key = config.get_api_key() or os.environ.get('OPENAI_API_KEY')
                model = config.get_model()
                if api_key and model and names_list:
                    try:
                        name_map = gpt.simplify_product_names(api_key, model, names_list)
                    except Exception:
                        name_map = {}
                for row in rows_data:
                    original_name = (row.get(name_col) or '').strip() if name_col else None
                    if not original_name:
                        continue
                    simplified = name_map.get(original_name) or original_name
                    description = (row.get(desc_col) or '').strip() if desc_col else None
                    category = (row.get(cat_col) or '').strip() if cat_col else None
                    price = None
                    if price_col and row.get(price_col):
                        try:
                            price = float(str(row.get(price_col)).replace(',', '.'))
                        except ValueError:
                            price = None
                    currency = (row.get(curr_col) or '').strip() if curr_col else None
                    image_url = (row.get(img_col) or '').strip() if img_col else None
                    desire = (row.get(desire_col) or '').strip() if desire_col else None
                    desire = desire or None
                    desire_mag = (row.get(desire_mag_col) or '').strip() if desire_mag_col else None
                    desire_mag = desire_mag or None
                    awareness = (row.get(awareness_col) or '').strip() if awareness_col else None
                    awareness = awareness or None
                    competition = (row.get(competition_col) or '').strip() if competition_col else None
                    competition = competition or None
                    extra_cols = {
                        k: v
                        for k, v in row.items()
                        if k
                        not in {
                            name_col,
                            desc_col,
                            cat_col,
                            price_col,
                            curr_col,
                            img_col,
                            desire_col,
                            desire_mag_col,
                            awareness_col,
                            competition_col,
                        }
                        and k.lower() not in {"product name", "source", "decision"}
                    }
                    # mark if duplicate
                    dupe = database.find_product_by_name(conn, simplified)
                    if simplified != original_name:
                        extra_cols['original_name'] = original_name
                    if dupe:
                        extra_cols['duplicate_of'] = dupe['id']
                    pid = database.insert_product(
                        conn,
                        name=simplified,
                        description=description,
                        category=category,
                        price=price,
                        currency=currency,
                        image_url=image_url,
                        source=filename,
                        desire=desire,
                        desire_magnitude=desire_mag,
                        awareness_level=awareness,
                        competition_level=competition,
                        extra=extra_cols,
                    )
                    inserted += 1
                    inserted_ids.append(pid)
            elif ext == ".json":
                items = json.loads(data.decode('utf-8', errors='ignore'))
                if isinstance(items, list):
                    # collect names for simplification
                    names_list = []
                    processed = []
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        keys = list(item.keys())
                        name_key = find_key(keys, ["name", "nombre", "productname", "product", "title"])
                        if not name_key:
                            continue
                        name = item.get(name_key)
                        if not name:
                            continue
                        names_list.append(str(name).strip())
                        processed.append((item, name_key))
                    # call GPT to simplify names once per file
                    name_map = {}
                    api_key = config.get_api_key() or os.environ.get('OPENAI_API_KEY')
                    model = config.get_model()
                    weights_map = (
                        config.get_scoring_v2_weights()
                        if config.is_scoring_v2_enabled()
                        else {}
                    )
                    if api_key and model and names_list:
                        try:
                            name_map = gpt.simplify_product_names(api_key, model, names_list)
                        except Exception:
                            name_map = {}
                    for item, name_key in processed:
                        original_name = str(item.get(name_key)).strip()
                        simplified = name_map.get(original_name) or original_name
                        keys = list(item.keys())
                        desc_key = find_key(keys, ["description", "descripcion", "desc"])
                        category_key = find_key(keys, ["category", "categoria", "cat"])
                        price_key = find_key(keys, ["price", "precio", "cost", "unitprice"])
                        currency_key = find_key(keys, ["currency", "moneda"])
                        image_key = find_key(keys, ["image", "imagen", "img", "picture", "imgurl"])
                        description = item.get(desc_key) if desc_key else None
                        category = item.get(category_key) if category_key else None
                        price = None
                        if price_key and item.get(price_key):
                            try:
                                price = float(str(item.get(price_key)).replace(',', '.'))
                            except Exception:
                                price = None
                        currency = item.get(currency_key) if currency_key else None
                        image_url = item.get(image_key) if image_key else None
                        desire = item.get("desire") or item.get("desire_text")
                        desire_mag = item.get("desire_magnitude") or item.get("magnitud_deseo")
                        awareness = item.get("awareness_level") or item.get("nivel_consciencia")
                        competition = item.get("competition_level") or item.get("saturacion_mercado")
                        extra = {
                            k: v
                            for k, v in item.items()
                            if k
                            not in {
                                name_key,
                                desc_key,
                                category_key,
                                price_key,
                                currency_key,
                                image_key,
                                "desire",
                                "desire_text",
                                "desire_magnitude",
                                "awareness_level",
                                "competition_level",
                                "magnitud_deseo",
                                "nivel_consciencia",
                                "saturacion_mercado",
                                "source",
                                "decision",
                            }
                        }
                        if simplified != original_name:
                            extra['original_name'] = original_name
                        dupe = database.find_product_by_name(conn, simplified)
                        if dupe:
                            extra['duplicate_of'] = dupe['id']
                        pid = database.insert_product(
                            conn,
                            name=simplified,
                            description=str(description).strip() if description else None,
                            category=str(category).strip() if category else None,
                            price=price,
                            currency=str(currency).strip() if currency else None,
                            image_url=str(image_url).strip() if image_url else None,
                            source=filename,
                            desire=desire,
                            desire_magnitude=desire_mag,
                            awareness_level=awareness,
                            competition_level=competition,
                            extra=extra,
                        )
                        inserted += 1
                        inserted_ids.append(pid)
            elif False and ext in (".xlsx", ".xls"):
                # legacy xlsx processing (handled asynchronously earlier)
                try:
                    import zipfile
                    import xml.etree.ElementTree as ET
                    from io import BytesIO

                    def parse_xlsx(binary: bytes):
                        """Parse a minimal XLSX file into a list of dicts using the first worksheet.

                        Supports shared strings and inline strings.  Skips leading empty rows.
                        """
                        with zipfile.ZipFile(BytesIO(binary)) as z:
                            shared = []
                            if 'xl/sharedStrings.xml' in z.namelist():
                                ss_root = ET.fromstring(z.read('xl/sharedStrings.xml'))
                                for si in ss_root.findall('.//{*}si'):
                                    text = ''.join((t.text or '') for t in si.findall('.//{*}t'))
                                    shared.append(text)
                            sheet_name = None
                            for name in z.namelist():
                                if name.startswith('xl/worksheets/sheet') and name.endswith('.xml'):
                                    sheet_name = name
                                    break
                            if not sheet_name:
                                return []
                            root = ET.fromstring(z.read(sheet_name))
                            rows = []
                            for row in root.findall('.//{*}row'):
                                values = []
                                last_col_idx = 0
                                for c in row.findall('{*}c'):
                                    cell_ref = c.attrib.get('r', '')
                                    letters = ''.join(ch for ch in cell_ref if ch.isalpha())
                                    col_idx = 0
                                    for ch in letters:
                                        col_idx = col_idx * 26 + (ord(ch.upper()) - ord('A') + 1)
                                    while last_col_idx < col_idx - 1:
                                        values.append('')
                                        last_col_idx += 1
                                    val = ''
                                    cell_type = c.attrib.get('t')
                                    if cell_type == 's':
                                        v = c.find('{*}v')
                                        if v is not None:
                                            try:
                                                idx = int(v.text)
                                                val = shared[idx] if idx < len(shared) else ''
                                            except Exception:
                                                val = ''
                                    elif cell_type == 'inlineStr':
                                        tnode = c.find('{*}is/{*}t')
                                        val = tnode.text if tnode is not None else ''
                                    else:
                                        v = c.find('{*}v')
                                        val = v.text if v is not None else ''
                                    values.append(val)
                                    last_col_idx = col_idx
                                rows.append(values)
                            # drop leading empty rows
                            while rows and all(not cell for cell in rows[0]):
                                rows.pop(0)
                            if not rows:
                                return []
                            headers = rows[0]
                            records = []
                            for r in rows[1:]:
                                rec = {}
                                for i, h in enumerate(headers):
                                    rec[h] = r[i] if i < len(r) else ''
                                records.append(rec)
                            return records
                    records = parse_xlsx(data)
                    # collect names for simplification
                    items_list = []
                    names_list = []
                    for item in records:
                        if not isinstance(item, dict):
                            continue
                        keys = list(item.keys())
                        name_key = find_key(keys, ["name", "nombre", "productname", "product", "title"])
                        if not name_key:
                            continue
                        name = item.get(name_key)
                        if not name:
                            continue
                        names_list.append(str(name).strip())
                        items_list.append((item, name_key))
                    # call GPT to simplify names
                    name_map = {}
                    api_key = config.get_api_key() or os.environ.get('OPENAI_API_KEY')
                    model = config.get_model()
                    weights_map = (
                        config.get_scoring_v2_weights()
                        if config.is_scoring_v2_enabled()
                        else {}
                    )
                    if api_key and model and names_list:
                        try:
                            name_map = gpt.simplify_product_names(api_key, model, names_list)
                        except Exception:
                            name_map = {}
                    metric_vals = {k: [] for k in gpt.NUMERIC_FIELD_MAP}
                    for item, _ in items_list:
                        for m in gpt.NUMERIC_FIELD_MAP:
                            v = item.get(m)
                            if v is None or v == "":
                                continue
                            try:
                                f = float(str(v).replace(',', '.'))
                            except Exception:
                                continue
                            metric_vals[m].append(f)
                    metric_ranges = {
                        m: (min(vals), max(vals))
                        for m, vals in metric_vals.items()
                        if vals
                    }
                    for item, name_key in items_list:
                        original_name = str(item.get(name_key)).strip()
                        simplified = name_map.get(original_name) or original_name
                        keys = list(item.keys())
                        desc_key = find_key(keys, ["description", "descripcion", "desc"])
                        category_key = find_key(keys, ["category", "categoria", "cat"])
                        price_key = find_key(keys, ["price", "precio", "cost", "unitprice"])
                        currency_key = find_key(keys, ["currency", "moneda"])
                        image_key = find_key(keys, ["image", "imagen", "img", "picture", "imgurl"])
                        description = item.get(desc_key) if desc_key else None
                        category = item.get(category_key) if category_key else None
                        price = None
                        if price_key and item.get(price_key):
                            try:
                                price = float(str(item.get(price_key)).replace(',', '.'))
                            except Exception:
                                price = None
                        currency = item.get(currency_key) if currency_key else None
                        image_url = item.get(image_key) if image_key else None
                        desire = item.get("desire") or item.get("desire_text")
                        desire_mag = item.get("desire_magnitude") or item.get("magnitud_deseo")
                        awareness = item.get("awareness_level") or item.get("nivel_consciencia")
                        competition = item.get("competition_level") or item.get("saturacion_mercado")
                        extra = {
                            k: v
                            for k, v in item.items()
                            if k
                            not in {
                                name_key,
                                desc_key,
                                category_key,
                                price_key,
                                currency_key,
                                image_key,
                                "desire",
                                "desire_text",
                                "desire_magnitude",
                                "awareness_level",
                                "competition_level",
                                "magnitud_deseo",
                                "nivel_consciencia",
                                "saturacion_mercado",
                                "source",
                                "decision",
                            }
                        }
                        if simplified != original_name:
                            extra['original_name'] = original_name
                        dupe = database.find_product_by_name(conn, simplified)
                        if dupe:
                            extra['duplicate_of'] = dupe['id']
                        pid = database.insert_product(
                            conn,
                            name=str(simplified).strip(),
                            description=str(description).strip() if description else None,
                            category=str(category).strip() if category else None,
                            price=price,
                            currency=str(currency).strip() if currency else None,
                            image_url=str(image_url).strip() if image_url else None,
                            source=filename,
                            desire=desire,
                            desire_magnitude=desire_mag,
                            awareness_level=awareness,
                            competition_level=competition,
                            extra=extra,
                        )
                        if config.is_scoring_v2_enabled() and weights_map:
                            scores, justifs, sources = gpt.compute_numeric_scores(extra, metric_ranges)
                            need = [f for f in WINNER_V2_FIELDS if f not in scores]
                            if need and api_key and model:
                                try:
                                    resp = gpt.evaluate_winner_score(
                                        api_key,
                                        model,
                                        {
                                            "title": simplified,
                                            "description": description,
                                            "category": category,
                                            "metrics": extra,
                                        },
                                    )
                                    rs = resp.get("scores", {})
                                    js = resp.get("justifications", {})
                                    for f in need:
                                        scores[f] = rs.get(f, 3)
                                        justifs[f] = js.get(f, "")
                                        sources[f] = "gpt"
                                except Exception:
                                    pass
                            if scores:
                                weighted = sum(
                                    scores.get(f, 3) * weights_map.get(f, 0.0)
                                    for f in WINNER_V2_FIELDS
                                )
                                raw_score = weighted * 8.0
                                pct = ((raw_score - 8.0) / 32.0) * 100.0
                                pct = max(0, min(100, round(pct)))
                                breakdown = {
                                    "scores": scores,
                                    "justifications": justifs,
                                    "weights": weights_map,
                                    "sources": sources,
                                }
                                database.insert_score(
                                    conn,
                                    product_id=pid,
                                    model=model or "",
                                    total_score=0,
                                    momentum=0,
                                    saturation=0,
                                    differentiation=0,
                                    social_proof=0,
                                    margin=0,
                                    logistics=0,
                                    summary="",
                                    explanations={},
                                    winner_score_v2_raw=raw_score,
                                    winner_score_v2_pct=pct,
                                    winner_score_v2_breakdown=breakdown,
                                )
                        inserted += 1
                        inserted_ids.append(pid)
                except Exception as exc:
                    self._set_json(500)
                    self._safe_write(json.dumps({"error": f"Error al procesar XLSX: {exc}"}).encode('utf-8'))
                    return
            else:
                # treat as image; save temporarily and attempt to extract products using GPT vision
                tmp_dir = APP_DIR / 'uploads'
                tmp_dir.mkdir(exist_ok=True)
                tmp_path = tmp_dir / filename
                with open(tmp_path, 'wb') as f:
                    f.write(data)
                inserted = 0
                # attempt automatic extraction if API key and vision model configured
                api_key = config.get_api_key() or os.environ.get('OPENAI_API_KEY')
                model = config.get_model()
                if api_key and model:
                    try:
                        # call vision extraction
                        products = gpt.extract_products_from_image(api_key, model, str(tmp_path))
                        for item in products:
                            name = item.get('name')
                            if not name:
                                continue
                            desc = item.get('description')
                            cat = item.get('category')
                            price = item.get('price')
                            price_val = None
                            if price:
                                try:
                                    price_val = float(str(price).replace(',', '.'))
                                except Exception:
                                    price_val = None
                            pid = database.insert_product(
                                conn,
                                name=name,
                                description=desc,
                                category=cat,
                                price=price_val,
                                currency=None,
                                image_url=str(tmp_path),
                                source=filename,
                                extra={},
                            )
                            inserted += 1
                            inserted_ids.append(pid)
                    except Exception:
                        # ignore extraction errors
                        inserted = 0
                # respond with info about image and inserted count
                self._set_json()
                self._safe_write(json.dumps({"uploaded_image": f"/uploads/{filename}", "inserted": inserted}).encode('utf-8'))
                return
        except Exception as exc:
            self._set_json(500)
            self._safe_write(json.dumps({"error": str(exc)}).encode('utf-8'))
            return
        # Automatically evaluate newly inserted products with offline heuristic if any
        if inserted_ids:
            conn_eval = ensure_db()
            weights_map = config.get_weights()
            sum_weights = sum(weights_map.values()) or 1.0
            for pid in inserted_ids:
                # skip if already has a score
                if database.get_scores_for_product(conn_eval, pid):
                    continue
                p_rec = database.get_product(conn_eval, pid)
                if not p_rec:
                    continue
                offline = offline_evaluate(dict(p_rec))
                metrics = {
                    'momentum': offline['momentum'],
                    'saturation': offline['saturation'],
                    'differentiation': offline['differentiation'],
                    'social_proof': offline['social_proof'],
                    'margin': offline['margin'],
                    'logistics': offline['logistics'],
                }
                weighted_total = (
                    metrics['momentum'] * weights_map.get('momentum', 1.0)
                    + metrics['saturation'] * weights_map.get('saturation', 1.0)
                    + metrics['differentiation'] * weights_map.get('differentiation', 1.0)
                    + metrics['social_proof'] * weights_map.get('social_proof', 1.0)
                    + metrics['margin'] * weights_map.get('margin', 1.0)
                    + metrics['logistics'] * weights_map.get('logistics', 1.0)
                ) / sum_weights
                # convert to score out of 100
                try:
                    score_int = int(round(((weighted_total - 1.0) / 8.0) * 100))
                    if score_int < 0:
                        score_int = 0
                    if score_int > 100:
                        score_int = 100
                except Exception:
                    score_int = int(round(weighted_total * 10))
                database.insert_score(
                    conn_eval,
                    product_id=pid,
                    model='heuristic',
                    total_score=score_int,
                    momentum=metrics['momentum'],
                    saturation=metrics['saturation'],
                    differentiation=metrics['differentiation'],
                    social_proof=metrics['social_proof'],
                    margin=metrics['margin'],
                    logistics=metrics['logistics'],
                    summary=offline['summary'],
                    explanations=offline['explanations'],
                )
        # Recalculate Winner Score using numeric metrics
        if inserted_ids:
            conn_ws = ensure_db()
            products_all = [dict(r) for r in database.list_products(conn_ws)]
            ranges = winner_calc.compute_ranges(products_all)
            weights = {k: 1.0 for k in winner_calc.ALL_METRICS}
            for prod in products_all:
                missing: list[str] = []
                used: list[str] = []
                pct_val = winner_calc.score_product(prod, weights, ranges, missing, used)
                used_count = len(used)
                missing_count = len(missing)
                fallback = used_count == 0 or pct_val is None or (isinstance(pct_val, float) and math.isnan(pct_val))
                if fallback:
                    pct = 50
                else:
                    pct = max(0, min(100, round(pct_val * 100)))
                if missing_count > 0:
                    level = logging.WARNING if fallback else logging.INFO
                    logger.log(
                        level,
                        "Winner Score: product=%s used=%d missing=%d fallback=%s",
                        prod['id'],
                        used_count,
                        missing_count,
                        str(fallback).lower(),
                    )
                database.insert_score(
                    conn_ws,
                    product_id=prod['id'],
                    model='winner_v2',
                    total_score=0,
                    momentum=0,
                    saturation=0,
                    differentiation=0,
                    social_proof=0,
                    margin=0,
                    logistics=0,
                    summary='',
                    explanations={},
                    winner_score_v2_pct=pct,
                    commit=False,
                )
            conn_ws.commit()
        pending = []
        cost_msg = None
        cost_est = None
        if inserted_ids and config.is_auto_fill_ia_on_import_enabled():
            cfg_cost = config.get_ai_cost_config()
            res_ai = ai_columns.fill_ai_columns(
                inserted_ids,
                model=cfg_cost.get("model"),
                batch_mode=len(inserted_ids) >= cfg_cost.get("useBatchWhenCountGte", 300),
                cost_cap_usd=cfg_cost.get("costCapUSD"),
            )
            pending = res_ai.get("pending_ids", [])
            cost_msg = res_ai.get("ui_cost_message")
            cost_est = res_ai.get("cost_estimated_usd")
            if res_ai.get("error"):
                logger.info("No se pudieron completar las columnas con IA: revisa la API.")
        self._set_json()
        payload: Dict[str, Any] = {}
        if cost_est is not None:
            payload["cost_estimated_usd"] = cost_est
        if cost_msg:
            payload["ui_cost_message"] = cost_msg
        if pending:
            payload["pending_ids"] = pending
        self._safe_write(json.dumps(payload).encode('utf-8'))

    def handle_evaluate_all(self):
        conn = ensure_db()
        api_key = config.get_api_key() or os.environ.get('OPENAI_API_KEY')
        model = config.get_model()
        evaluated = 0
        if config.is_scoring_v2_enabled():
            weights_map = config.get_scoring_v2_weights()
            for p in database.list_products(conn):
                if database.get_scores_for_product(conn, p['id']):
                    continue
                if not (api_key and model):
                    continue
                try:
                    try:
                        extra = json.loads(p.get("extra") or "{}")
                    except Exception:
                        extra = {}
                    resp = gpt.evaluate_winner_score(
                        api_key,
                        model,
                        {
                            "title": p.get("name"),
                            "description": p.get("description"),
                            "category": p.get("category"),
                            "metrics": extra,
                        },
                    )
                    scores = resp.get("scores", {})
                    justifs = resp.get("justifications", {})
                    weighted = sum(
                        scores.get(f, 3) * weights_map.get(f, 0.0)
                        for f in WINNER_V2_FIELDS
                    )
                    raw_score = weighted * 8.0
                    pct = ((raw_score - 8.0) / 32.0) * 100.0
                    pct = max(0, min(100, round(pct)))
                    breakdown = {
                        "scores": scores,
                        "justifications": justifs,
                        "weights": weights_map,
                    }
                    database.insert_score(
                        conn,
                        product_id=p['id'],
                        model=model,
                        total_score=0,
                        momentum=0,
                        saturation=0,
                        differentiation=0,
                        social_proof=0,
                        margin=0,
                        logistics=0,
                        summary="",
                        explanations={},
                        winner_score_v2_raw=raw_score,
                        winner_score_v2_pct=pct,
                        winner_score_v2_breakdown=breakdown,
                    )
                    evaluated += 1
                except Exception:
                    continue
            self._set_json()
            self.wfile.write(json.dumps({"evaluated": evaluated}).encode('utf-8'))
            return
        # Fallback to legacy evaluation if v2 is disabled
        evaluated = 0
        weights_map = config.get_weights()
        sum_weights = sum(weights_map.values()) or 1.0
        for p in database.list_products(conn):
            if database.get_scores_for_product(conn, p['id']):
                continue
            metrics = None
            if api_key:
                try:
                    result = gpt.evaluate_product(api_key, model, dict(p))
                    metrics = {
                        'momentum': float(result.get('momentum_score', 5.0)),
                        'saturation': float(result.get('saturation_score', 5.0)),
                        'differentiation': float(result.get('differentiation_score', 5.0)),
                        'social_proof': float(result.get('social_proof_score', 5.0)),
                        'margin': float(result.get('margin_score', 5.0)),
                        'logistics': float(result.get('logistics_score', 5.0)),
                    }
                    summary = result.get('summary', '')
                    explanations = {
                        'momentum': result.get('momentum_explanation'),
                        'saturation': result.get('saturation_explanation'),
                        'differentiation': result.get('differentiation_explanation'),
                        'social_proof': result.get('social_proof_explanation'),
                        'margin': result.get('margin_explanation'),
                        'logistics': result.get('logistics_explanation'),
                    }
                except Exception:
                    metrics = None
            if metrics is None:
                offline = offline_evaluate(dict(p))
                metrics = {
                    'momentum': offline['momentum'],
                    'saturation': offline['saturation'],
                    'differentiation': offline['differentiation'],
                    'social_proof': offline['social_proof'],
                    'margin': offline['margin'],
                    'logistics': offline['logistics'],
                }
                summary = offline['summary']
                explanations = offline['explanations']
            weighted_total = (
                metrics['momentum'] * weights_map.get('momentum', 1.0)
                + metrics['saturation'] * weights_map.get('saturation', 1.0)
                + metrics['differentiation'] * weights_map.get('differentiation', 1.0)
                + metrics['social_proof'] * weights_map.get('social_proof', 1.0)
                + metrics['margin'] * weights_map.get('margin', 1.0)
                + metrics['logistics'] * weights_map.get('logistics', 1.0)
            ) / sum_weights
            try:
                score_int = int(round(((weighted_total - 1.0) / 8.0) * 100))
                if score_int < 0:
                    score_int = 0
                if score_int > 100:
                    score_int = 100
            except Exception:
                score_int = int(round(weighted_total * 10))
            database.insert_score(
                conn,
                product_id=p['id'],
                model=model or 'heuristic',
                total_score=score_int,
                momentum=metrics['momentum'],
                saturation=metrics['saturation'],
                differentiation=metrics['differentiation'],
                social_proof=metrics['social_proof'],
                margin=metrics['margin'],
                logistics=metrics['logistics'],
                summary=summary,
                explanations=explanations,
            )
            evaluated += 1
        self._set_json()
        self.wfile.write(json.dumps({"evaluated": evaluated}).encode('utf-8'))

    def handle_setconfig(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode('utf-8')
        try:
            data = json.loads(body)
        except Exception:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode('utf-8'))
            return
        cfg = config.load_config()
        if 'api_key' in data:
            key = str(data.get('api_key', '')).strip()
            if not key:
                self._set_json(400)
                self.wfile.write(json.dumps({"error": "empty_api_key"}).encode('utf-8'))
                return
            cfg['api_key'] = key
        if 'model' in data and data['model']:
            cfg['model'] = data['model']
        if 'weights' in data and isinstance(data['weights'], dict):
            # update only known keys to avoid arbitrary injection
            weights = cfg.get('weights', {})
            for k, v in data['weights'].items():
                try:
                    weights[k] = float(v)
                except Exception:
                    continue
            cfg['weights'] = weights
        if 'scoring_v2_weights' in data and isinstance(data['scoring_v2_weights'], dict):
            weights_v2 = cfg.get('scoring_v2_weights', {})
            for k, v in data['scoring_v2_weights'].items():
                try:
                    weights_v2[k] = float(v)
                except Exception:
                    continue
            cfg['scoring_v2_weights'] = weights_v2
        if 'autoFillIAOnImport' in data:
            cfg['autoFillIAOnImport'] = bool(data['autoFillIAOnImport'])
        config.save_config(cfg)
        self._set_json()
        self.wfile.write(json.dumps({"status": "ok"}).encode('utf-8'))

    def handle_custom_gpt(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode('utf-8')
        try:
            data = json.loads(body)
            prompt = data['prompt']
        except Exception:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "Invalid request"}).encode('utf-8'))
            return
        api_key = config.get_api_key() or os.environ.get('OPENAI_API_KEY')
        model = config.get_model()
        if not api_key:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "No API key configured"}).encode('utf-8'))
            return
        try:
            resp = gpt.call_openai_chat(api_key, model, [
                {"role": "system", "content": "Eres un asistente útil."},
                {"role": "user", "content": prompt},
            ])
            content = resp['choices'][0]['message']['content']
            self._set_json()
            self.wfile.write(json.dumps({"response": content}).encode('utf-8'))
        except Exception as exc:
            self._set_json(500)
            self.wfile.write(json.dumps({"error": str(exc)}).encode('utf-8'))

    def handle_ba_insights(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode('utf-8')
        try:
            payload = json.loads(body)
            product = payload.get("product")
            model = payload.get("model") or "gpt-4o-mini-2024-07-18"
        except Exception:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode('utf-8'))
            return
        if not isinstance(product, dict) or not product.get("id"):
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "Missing product"}).encode('utf-8'))
            return
        api_key = config.get_api_key() or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            self._set_json(503)
            self.wfile.write(json.dumps({"error": "OpenAI no disponible"}).encode('utf-8'))
            return
        try:
            grid_updates, usage, duration = gpt.generate_ba_insights(api_key, model, product)
            logger.info("/api/ba/insights tokens=%s duration=%.2fs", usage.get('total_tokens'), duration)
            self._set_json()
            self.wfile.write(json.dumps({"grid_updates": grid_updates}).encode('utf-8'))
        except gpt.InvalidJSONError:
            self._set_json(502)
            self.wfile.write(json.dumps({"error": "Respuesta IA no es JSON"}).encode('utf-8'))
        except Exception:
            self._set_json(503)
            self.wfile.write(json.dumps({"error": "OpenAI no disponible"}).encode('utf-8'))

    def handle_ia_batch_columns(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode('utf-8')
        try:
            payload = json.loads(body)
            items = payload.get("items")
            model = payload.get("model") or "gpt-4o-mini-2024-07-18"
            if not isinstance(items, list):
                raise ValueError
        except Exception:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode('utf-8'))
            return
        api_key = config.get_api_key() or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            self._set_json(503)
            self.wfile.write(json.dumps({"error": "OpenAI no disponible"}).encode('utf-8'))
            return
        try:
            ok, ko, usage, duration = gpt.generate_batch_columns(api_key, model, items)
            logger.info("/api/ia/batch-columns tokens=%s duration=%.2fs", usage.get('total_tokens'), duration)
            self._set_json()
            self.wfile.write(json.dumps({"ok": ok, "ko": ko}).encode('utf-8'))
        except gpt.InvalidJSONError:
            self._set_json(502)
            self.wfile.write(json.dumps({"error": "Respuesta IA no es JSON"}).encode('utf-8'))
        except Exception:
            self._set_json(503)
            self.wfile.write(json.dumps({"error": "OpenAI no disponible"}).encode('utf-8'))

    def handle_auto_weights(self):
        """
        Compute recommended weights based on existing scores.
        The idea is to give higher weight to metrics that are on average low
        (indicating scarcity) and lower weight to metrics that are high on average.
        Returns a mapping of metric -> weight.  If no scores exist, returns
        default weights (1.0 for each).
        """
        conn = ensure_db()
        # gather all scores
        rows = database.list_products(conn)
        metrics = ["momentum", "saturation", "differentiation", "social_proof", "margin", "logistics"]
        sums = {m: 0.0 for m in metrics}
        count = 0
        for p in rows:
            scores = database.get_scores_for_product(conn, p["id"])
            if not scores:
                continue
            s = scores[0]
            count += 1
            for m in metrics:
                try:
                    val = float(s.get(m) or 0.0)
                except Exception:
                    val = 0.0
                sums[m] += val
        if count == 0:
            weights = {m: 1.0 for m in metrics}
        else:
            avg = {m: sums[m] / count for m in metrics}
            # attempt to call GPT to propose weights
            api_key = config.get_api_key() or os.environ.get('OPENAI_API_KEY')
            model = config.get_model()
            recommended = None
            if api_key and model:
                try:
                    prompt = (
                        f"Eres un asistente experto en análisis de productos. Dadas las medias actuales de las métricas "
                        f"(momentum={avg['momentum']:.2f}, saturación={avg['saturation']:.2f}, "
                        f"diferenciación={avg['differentiation']:.2f}, prueba social={avg['social_proof']:.2f}, "
                        f"margen={avg['margin']:.2f}, logística={avg['logistics']:.2f}), "
                        "propón pesos (de 0 a 2) para cada métrica en formato JSON con las claves exactas: "
                        "momentum, saturation, differentiation, social_proof, margin, logistics."
                    )
                    resp = gpt.call_openai_chat(api_key, model, [
                        {"role": "system", "content": "Eres un asistente experto en scoring de productos."},
                        {"role": "user", "content": prompt},
                    ])
                    content = resp['choices'][0]['message']['content']
                    # Attempt to parse JSON from content
                    try:
                        recommended = json.loads(content)
                    except Exception:
                        recommended = None
                except Exception:
                    recommended = None
            if recommended and all(k in recommended for k in metrics):
                # use recommended weights
                weights = {k: float(recommended.get(k, 1.0)) for k in metrics}
            else:
                # fallback: inverse of average (scarce factors get higher weight)
                weights = {m: (1.0 / (avg[m] + 1e-6)) for m in metrics}
            # normalise so sum of weights equals len(metrics)
            total = sum(weights.values()) or 1.0
            factor = float(len(metrics)) / total
            for k in weights:
                weights[k] *= factor
        self._set_json()
        self.wfile.write(json.dumps(weights).encode('utf-8'))

    def _collect_samples_for_weights(self):
        """Gather products with Winner Score v2 breakdown and success metric."""
        conn = ensure_db()
        rows = database.list_products(conn)
        samples = []
        metric_key = None
        for p in rows:
            try:
                extra = json.loads(p["extra"] or "{}")
            except Exception:
                extra = {}
            success = None
            if metric_key and metric_key in extra:
                try:
                    success = float(extra[metric_key])
                except Exception:
                    success = None
            if success is None:
                for key in ("orders", "revenue", "gmv", "sales", "units"):
                    if key in extra:
                        try:
                            success = float(extra[key])
                            metric_key = metric_key or key
                            break
                        except Exception:
                            continue
            if success is None:
                continue
            scores_rows = database.get_scores_for_product(conn, p["id"])
            if not scores_rows:
                continue
            srow = scores_rows[0]
            try:
                breakdown = json.loads(srow["winner_score_v2_breakdown"] or "{}")
                scores = breakdown.get("scores") or {}
            except Exception:
                continue
            if not scores or any(k not in scores for k in WINNER_V2_FIELDS):
                continue
            sample = {k: float(scores[k]) for k in WINNER_V2_FIELDS}
            sample[metric_key] = success
            samples.append(sample)
            if len(samples) >= 50:
                break
        return samples, metric_key

    def handle_scoring_v2_auto_weights_gpt(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length else ""
        try:
            data = json.loads(body) if body else {}
        except Exception:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode('utf-8'))
            return
        features = data.get("features") or WINNER_V2_FIELDS
        samples_in = data.get("data_sample") or []
        target = data.get("target") or ""
        if not samples_in or not target:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "Datos insuficientes"}).encode('utf-8'))
            return
        samples = []
        for s in samples_in:
            if "target" not in s:
                continue
            row = {k: float(s.get(k, 0.0)) for k in features}
            row[target] = float(s.get("target", 0.0))
            samples.append(row)
        api_key = config.get_api_key() or os.environ.get('OPENAI_API_KEY')
        model = config.get_model()
        if not api_key or not model:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "No API key configured"}).encode('utf-8'))
            return
        try:
            result = gpt.recommend_winner_weights(api_key, model, samples, target)
            weights = result.get("weights", {})
            notes = result.get("justification", "")
        except Exception as exc:
            self._set_json(500)
            self.wfile.write(json.dumps({"error": str(exc)}).encode('utf-8'))
            return
        resp = {
            "weights": {k: weights.get(k, 0.0) for k in features},
            "method": "gpt",
            "diagnostics": {"notes": notes},
        }
        self._set_json()
        self.wfile.write(json.dumps(resp).encode('utf-8'))

    def handle_scoring_v2_auto_weights_stat(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length else ""
        try:
            data = json.loads(body) if body else {}
        except Exception:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode('utf-8'))
            return
        features = data.get("features") or WINNER_V2_FIELDS
        samples_in = data.get("data_sample") or []
        target = data.get("target") or ""
        if not samples_in or not target or len(samples_in) < 2:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "Datos insuficientes"}).encode('utf-8'))
            return
        ys = [float(s.get("target", 0.0)) for s in samples_in]
        mean_y = sum(ys) / len(ys)
        denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys)) or 1.0
        weights: Dict[str, float] = {}
        for field in features:
            xs = [float(s.get(field, 0.0)) for s in samples_in]
            mean_x = sum(xs) / len(xs)
            denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs)) or 1.0
            num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
            corr = abs(num / (denom_x * denom_y)) if denom_x and denom_y else 0.0
            weights[field] = corr
        total = sum(weights.values()) or 1.0
        weights = {k: v / total for k, v in weights.items()}
        resp = {"weights": {k: weights.get(k, 0.0) for k in features}, "method": "stat", "diagnostics": {"n": len(samples_in)}}
        self._set_json()
        self.wfile.write(json.dumps(resp).encode('utf-8'))

    def handle_scoring_v2_gpt_evaluate(self):
        """Endpoint that evaluates Winner Score v2 variables via GPT."""

        if not config.is_scoring_v2_enabled():
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "scoring v2 disabled"}).encode("utf-8"))
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length else ""
        try:
            data = json.loads(body) if body else {}
        except Exception:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode("utf-8"))
            return

        title = data.get("title") or data.get("name") or ""
        description = data.get("description") or ""
        category = data.get("category") or ""
        metrics = data.get("metrics") or {}

        api_key = config.get_api_key() or os.environ.get("OPENAI_API_KEY")
        model = config.get_model()
        scores, justifs, sources = gpt.compute_numeric_scores(metrics, {})
        need = [f for f in WINNER_V2_FIELDS if f not in scores]
        if need:
            if not api_key or not model:
                self._set_json(400)
                self.wfile.write(json.dumps({"error": "No API key configured"}).encode("utf-8"))
                return
            try:
                resp = gpt.evaluate_winner_score(
                    api_key,
                    model,
                    {
                        "title": title,
                        "description": description,
                        "category": category,
                        "metrics": metrics,
                    },
                )
            except Exception as exc:
                self._set_json(500)
                self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))
                return
            rs = resp.get("scores", {})
            js = resp.get("justifications", {})
            for f in need:
                scores[f] = rs.get(f, 3)
                justifs[f] = js.get(f, "")
                sources[f] = "gpt"
        out = {**scores, "justificacion": justifs, "source": sources}
        self._set_json()
        self.wfile.write(json.dumps(out).encode("utf-8"))

    def handle_scoring_v2_gpt_summary(self):
        """Generate an executive summary of top products using GPT."""

        if not config.is_scoring_v2_enabled():
            self._set_json(400)
            self.wfile.write(
                json.dumps({"error": "scoring v2 disabled"}).encode("utf-8")
            )
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length else ""
        try:
            data = json.loads(body) if body else {}
        except Exception:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode("utf-8"))
            return

        products = data.get("products") or []
        if not isinstance(products, list) or not products:
            self._set_json(400)
            self.wfile.write(
                json.dumps({"error": "No products provided"}).encode("utf-8")
            )
            return

        api_key = config.get_api_key() or os.environ.get("OPENAI_API_KEY")
        model = config.get_model()
        if not api_key or not model:
            self._set_json(400)
            self.wfile.write(
                json.dumps({"error": "No API key configured"}).encode("utf-8")
            )
            return

        try:
            summary = gpt.summarize_top_products(api_key, model, products)
        except Exception as exc:
            self._set_json(500)
            self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))
            return

        self._set_json()
        self.wfile.write(json.dumps({"summary": summary}).encode("utf-8"))

    def handle_scoring_v2_generate(self):
        """Compute Winner Score for selected products."""

        if not config.is_scoring_v2_enabled():
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "scoring v2 disabled"}).encode("utf-8"))
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length else ""
        try:
            data = json.loads(body) if body else {}
            ids = data.get("ids") or []
            if not isinstance(ids, list):
                raise ValueError
        except Exception:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode("utf-8"))
            return

        logger.info("Winner Score generate: ids_length=%d", len(ids))
        id_set = {int(i) for i in ids if str(i).isdigit()}
        if not id_set:
            logger.info("Winner Score generate: received_ids=0")
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "no_selection"}).encode("utf-8"))
            return

        conn = ensure_db()
        products_all = [dict(r) for r in database.list_products(conn)]
        ranges = winner_calc.compute_ranges(products_all)
        weights = config.get_scoring_v2_weights()
        total_w = sum(weights.values())
        if total_w <= 0:
            logger.warning(
                "Winner Score generate: weight sum <= 0, using uniform weights"
            )
            n = len(weights) or 1
            weights = {k: 1 / n for k in weights}
        else:
            weights = {k: v / total_w for k, v in weights.items()}
        logger.info(
            "Winner Score generate: weights=%s sum=%s",
            weights,
            sum(weights.values()),
        )
        updated: Dict[str, int] = {}
        skipped = 0
        details: list[dict[str, int | str]] = []
        used_metrics: set[str] = set()
        missing_metrics: set[str] = set()
        with_partial = 0
        fallback_only = 0
        for prod in products_all:
            pid = prod["id"]
            if pid not in id_set:
                continue
            existing = database.get_scores_for_product(conn, pid)
            if any((dict(sc).get("winner_score_v2_pct") or 0) > 0 for sc in existing):
                skipped += 1
                details.append({"id": pid, "reason": "already_scored"})
                continue
            missing: list[str] = []
            used: list[str] = []
            pct_val = winner_calc.score_product(prod, weights, ranges, missing, used)
            used_count = len(used)
            missing_count = len(missing)
            fallback = used_count == 0 or pct_val is None or (isinstance(pct_val, float) and math.isnan(pct_val))
            reason: str | None = None
            if fallback:
                pct = 50
                reason = "no_metrics"
                fallback_only += 1
            else:
                pct = max(0, min(100, round(pct_val * 100)))
                if missing_count > 0:
                    with_partial += 1
            used_metrics.update(used)
            missing_metrics.update(missing)
            if missing_count > 0:
                level = logging.WARNING if fallback else logging.INFO
                logger.log(
                    level,
                    "Winner Score: product=%s used=%d missing=%d fallback=%s",
                    pid,
                    used_count,
                    missing_count,
                    str(fallback).lower(),
                )
            database.insert_score(
                conn,
                product_id=pid,
                model="winner_v2",
                total_score=0,
                momentum=0,
                saturation=0,
                differentiation=0,
                social_proof=0,
                margin=0,
                logistics=0,
                summary="",
                explanations={},
                winner_score_v2_pct=pct,
                commit=False,
            )
            updated[str(pid)] = pct
            if reason:
                details.append({"id": pid, "reason": reason})
            else:
                details.append({"id": pid})

        conn.commit()
        logger.info(
            "Winner Score generate: received_ids=%d updated=%d skipped=%d",
            len(id_set),
            len(updated),
            skipped,
        )
        self._set_json()
        self.wfile.write(
            json.dumps(
                {
                    "updated": len(updated),
                    "skipped": skipped,
                    "with_partial": with_partial,
                    "fallback_only": fallback_only,
                    "used_metrics": sorted(used_metrics),
                    "missing_metrics": sorted(missing_metrics),
                    "details": details,
                }
            ).encode("utf-8")
        )

    def handle_create_list(self):
        """Create a new user defined list (group) of products."""
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode('utf-8') if length else ''
        try:
            data = json.loads(body) if body else {}
        except Exception:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode('utf-8'))
            return
        name = (data.get('name') or '').strip()
        if not name:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "Nombre no proporcionado"}).encode('utf-8'))
            return
        conn = ensure_db()
        list_id = database.create_list(conn, name)
        self._set_json()
        self.wfile.write(json.dumps({"id": list_id, "name": name}).encode('utf-8'))

    def handle_delete_list(self):
        """Delete an existing list by ID with options to move or remove products."""
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode('utf-8') if length else ''
        try:
            data = json.loads(body) if body else {}
        except Exception:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode('utf-8'))
            return
        try:
            lid = int(data.get('id'))
            mode = data.get('mode', 'remove')
            tgt = data.get('targetGroupId')
            if tgt is not None:
                tgt = int(tgt)
        except Exception:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "Datos inválidos"}).encode('utf-8'))
            return
        conn = ensure_db()
        try:
            result = database.delete_list(conn, lid, mode=mode, target_list_id=tgt)
            self._set_json()
            self.wfile.write(json.dumps(result).encode('utf-8'))
        except Exception as exc:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": str(exc)}).encode('utf-8'))

    def handle_add_to_list(self):
        """Add one or more products to a list."""
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode('utf-8') if length else ''
        try:
            data = json.loads(body) if body else {}
        except Exception:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode('utf-8'))
            return
        try:
            lid = int(data.get('id'))
            ids = [int(x) for x in data.get('ids', [])]
        except Exception:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "Datos inválidos"}).encode('utf-8'))
            return
        conn = ensure_db()
        for pid in ids:
            database.add_product_to_list(conn, lid, pid)
        self._set_json()
        self.wfile.write(json.dumps({"added": len(ids)}).encode('utf-8'))

    def handle_shutdown(self):
        """Shutdown the HTTP server."""
        self._set_json()
        self.wfile.write(json.dumps({"ok": True}).encode('utf-8'))
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def handle_delete(self):
        """Delete one or more products specified in the request body.

        Expects a JSON payload with an "ids" array of product IDs to delete.
        Returns a JSON object with the number of deleted records.  If any
        ID is invalid, it will be skipped.
        """
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode('utf-8') if length > 0 else '{}'
        try:
            data = json.loads(body or '{}')
        except Exception:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode('utf-8'))
            return
        ids = data.get('ids')
        if not isinstance(ids, list):
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "Missing or invalid ids"}).encode('utf-8'))
            return
        conn = ensure_db()
        deleted = 0
        for pid in ids:
            try:
                pid_int = int(pid)
            except Exception:
                continue
            try:
                database.delete_product(conn, pid_int)
                deleted += 1
            except Exception:
                continue
        self._set_json()
        self.wfile.write(json.dumps({"deleted": deleted}).encode('utf-8'))

    def handle_remove_from_list(self):
        """Remove products from a specific list without deleting them globally.

        Expects JSON payload with ``list_id`` and ``ids`` array. Removes each product from the list.
        Returns number of associations removed.
        """
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode('utf-8') if length > 0 else '{}'
        try:
            data = json.loads(body or '{}')
        except Exception:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode('utf-8'))
            return
        list_id = data.get('list_id')
        ids = data.get('ids')
        try:
            list_id_int = int(list_id)
        except Exception:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "Invalid list_id"}).encode('utf-8'))
            return
        if not isinstance(ids, list):
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "Missing or invalid ids"}).encode('utf-8'))
            return
        conn = ensure_db()
        removed = 0
        for pid in ids:
            try:
                pid_int = int(pid)
            except Exception:
                continue
            try:
                database.remove_product_from_list(conn, list_id_int, pid_int)
                removed += 1
            except Exception:
                continue
        self._set_json()
        self.wfile.write(json.dumps({"removed": removed}).encode('utf-8'))


def run(host: str = '127.0.0.1', port: int = 8000):
    ensure_db()
    resume_incomplete_imports()
    httpd = HTTPServer((host, port), RequestHandler)
    print(f"Servidor iniciado en http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Apagando servidor...")
    httpd.server_close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Web UI for Product Research Copilot")
    parser.add_argument('--host', default='127.0.0.1', help='Host IP to bind')
    parser.add_argument('--port', default=8000, type=int, help='Port number')
    args = parser.parse_args()
    run(args.host, args.port)