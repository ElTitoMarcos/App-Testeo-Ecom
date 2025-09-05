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
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import cgi
import threading
import sqlite3
from typing import Dict, Any

from . import database
from . import config
from . import gpt
from . import title_analyzer

WINNER_V2_FIELDS = [
    "magnitud_deseo",
    "nivel_consciencia",
    "saturacion_mercado",
    "facilidad_anuncio",
    "facilidad_logistica",
    "escalabilidad",
    "engagement_shareability",
    "durabilidad_recurrencia",
]

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "data.sqlite3"
STATIC_DIR = APP_DIR / "static"

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


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "ProductResearchCopilot/1.0"

    def _set_json(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()

    def _set_html(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

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
                score_value = None
                if score:
                    key = (
                        "winner_score_v2_pct"
                        if config.is_scoring_v2_enabled()
                        else "total_score"
                    )
                    score_value = score.get(key)
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
                else:
                    row["score"] = score_value
                rows.append(row)
            self._set_json()
            self.wfile.write(json.dumps(rows).encode("utf-8"))
            return
        if path == "/config":
            # return stored configuration (without exposing the API key)
            cfg = config.load_config()
            data = {
                "model": cfg.get("model", "gpt-4o"),
                "weights": cfg.get("weights", {}),
                "has_api_key": bool(cfg.get("api_key")),
                # do not include the API key for security
            }
            self._set_json()
            self.wfile.write(json.dumps(data).encode("utf-8"))
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
            # return all saved groups/lists
            conn = ensure_db()
            lsts = database.get_lists(conn)
            data = []
            for l in lsts:
                data.append({"id": l["id"], "name": l["name"]})
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
                        score_value = score.get(key)
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
            # Export selected products as CSV; ids provided via query params (?ids=1,2,3)
            qs = parse_qs(parsed.query)
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
            # gather all extra keys
            fieldnames = ['id','name','category','price']
            extra_keys = set()
            for p in items:
                try:
                    extra_dict = json.loads(p['extra']) if p['extra'] else {}
                except Exception:
                    extra_dict = {}
                extra_keys.update(extra_dict.keys())
            fieldnames.extend(sorted(extra_keys))
            import csv
            from io import StringIO
            output = StringIO()
            writer = csv.writer(output)
            writer.writerow(fieldnames)
            for p in items:
                row = []
                row.append(p['id'])
                row.append(p['name'])
                row.append(p['category'])
                row.append(p['price'])
                try:
                    extra_dict = json.loads(p['extra']) if p['extra'] else {}
                except Exception:
                    extra_dict = {}
                for key in sorted(extra_keys):
                    row.append(extra_dict.get(key, ''))
                writer.writerow(row)
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
        if path == "/auto_weights":
            self.handle_auto_weights()
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
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD': 'POST'})
            fileitem = form['file'] if 'file' in form else None
            if fileitem is None or not getattr(fileitem, 'filename', None):
                self.send_error(400, 'No file provided')
                return
            filename = Path(fileitem.filename).name
            data = fileitem.file.read()
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
        # handle multipart form data
        ctype, pdict = cgi.parse_header(self.headers.get('Content-Type'))
        if ctype != 'multipart/form-data':
            self.send_error(400, "Expected multipart/form-data")
            return
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD':'POST'})
        # ``form" may contain multiple fields; we look up 'file' key safely
        fileitem = form['file'] if 'file' in form else None
        # ``fileitem" must not be evaluated in boolean context
        if fileitem is None or not getattr(fileitem, 'filename', None):
            self.send_error(400, "No file provided")
            return
        filename = Path(fileitem.filename).name
        data = fileitem.file.read()
        ext = Path(filename).suffix.lower()
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
                    extra_cols = {k: v for k, v in row.items() if k not in {name_col, desc_col, cat_col, price_col, curr_col, img_col}}
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
                        extra = {k: v for k, v in item.items() if k not in {name_key, desc_key, category_key, price_key, currency_key, image_key}}
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
                            extra=extra,
                        )
                        inserted += 1
                        inserted_ids.append(pid)
            elif ext in (".xlsx", ".xls"):
                # parse basic xlsx into list of dicts (first sheet)
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
                        extra = {k: v for k, v in item.items() if k not in {name_key, desc_key, category_key, price_key, currency_key, image_key}}
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
                            extra=extra,
                        )
                        if (
                            config.is_scoring_v2_enabled()
                            and api_key
                            and model
                        ):
                            try:
                                resp = gpt.evaluate_winner_score(
                                    api_key,
                                    model,
                                    {
                                        "title": simplified,
                                        "description": description,
                                        "category": category,
                                    },
                                )
                                scores = {}
                                for field in WINNER_V2_FIELDS:
                                    try:
                                        val = int(resp.get(field, 3))
                                    except Exception:
                                        val = 3
                                    if val < 1:
                                        val = 1
                                    if val > 5:
                                        val = 5
                                    scores[field] = val
                                weighted = sum(
                                    scores[f] * weights_map.get(f, 0.0)
                                    for f in WINNER_V2_FIELDS
                                )
                                raw_score = weighted * 8.0
                                pct = ((raw_score - 8.0) / 32.0) * 100.0
                                breakdown = {"scores": scores, "weights": weights_map}
                                database.insert_score(
                                    conn,
                                    product_id=pid,
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
                            except Exception:
                                pass
                        inserted += 1
                        inserted_ids.append(pid)
                except Exception as exc:
                    self._set_json(500)
                    self.wfile.write(json.dumps({"error": f"Error al procesar XLSX: {exc}"}).encode('utf-8'))
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
                self.wfile.write(json.dumps({"uploaded_image": f"/uploads/{filename}", "inserted": inserted}).encode('utf-8'))
                return
        except Exception as exc:
            self._set_json(500)
            self.wfile.write(json.dumps({"error": str(exc)}).encode('utf-8'))
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
        self._set_json()
        self.wfile.write(json.dumps({"inserted": inserted}).encode('utf-8'))

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
                    resp = gpt.evaluate_winner_score(api_key, model, dict(p))
                    scores = {}
                    for field in WINNER_V2_FIELDS:
                        try:
                            val = int(resp.get(field, 3))
                        except Exception:
                            val = 3
                        if val < 1:
                            val = 1
                        if val > 5:
                            val = 5
                        scores[field] = val
                    weighted = sum(
                        scores[f] * weights_map.get(f, 0.0) for f in WINNER_V2_FIELDS
                    )
                    raw_score = weighted * 8.0
                    pct = ((raw_score - 8.0) / 32.0) * 100.0
                    breakdown = {"scores": scores, "weights": weights_map}
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
        if 'api_key' in data and data['api_key']:
            cfg['api_key'] = data['api_key']
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
        """Delete an existing list by ID."""
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
        except Exception:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "ID inválido"}).encode('utf-8'))
            return
        conn = ensure_db()
        database.delete_list(conn, lid)
        self._set_json()
        self.wfile.write(json.dumps({"status": "deleted"}).encode('utf-8'))

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