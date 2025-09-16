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

import copy
import json
import os
import io
import re
import logging
import unicodedata
import uuid
import requests
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
from datetime import date, datetime, timedelta
from typing import Dict, Any, Iterable, List, Mapping, Optional, Sequence

from . import database
from .db import get_db
from . import config
from .services import winner_score as winner_calc
from .services import trends_service
from .services.importer_unified import (
    import_records as unified_import_records,
    run_import as unified_run_import,
)
from . import gpt
from . import title_analyzer
from .ai import runner
from .utils.db import row_to_dict, rget

WINNER_SCORE_FIELDS = list(winner_calc.FEATURE_MAP.keys())

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "data.sqlite3"
STATIC_DIR = APP_DIR / "static"
ROOT_DIR = APP_DIR.parent
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "app.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

DEBUG = bool(os.environ.get("DEBUG"))

DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y")


_DB_INIT = False
_DB_INIT_PATH: str | None = None
_DB_INIT_LOCK = threading.Lock()

IMPORT_STATUS: Dict[str, Dict[str, Any]] = {}
_IMPORT_STATUS_LOCK = threading.Lock()


POST_IMPORT_TASKS_ALLOWED = {"desire", "imputacion", "winner_score"}
AI_PROGRESS_TASKS = ("desire", "imputacion", "winner_score")
DEFAULT_POST_IMPORT_TASKS = ("desire", "imputacion")


def _normalize_post_import_task(name: Any) -> str | None:
    if name is None:
        return None
    text = str(name).strip()
    if not text:
        return None
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower().strip()
    if ascii_text in {"desire"}:
        return "desire"
    if ascii_text in {"imputacion", "imputation"}:
        return "imputacion"
    if ascii_text in {"winner_score", "winnerscore", "winner-score"}:
        return "winner_score"
    return None


def _dedupe_preserve_order(items: List[Any]) -> List[Any]:
    seen: set[Any] = set()
    result: List[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _enqueue_post_import_tasks(
    task_id: str,
    product_ids: List[int],
    requested_tasks: List[str] | None,
) -> Dict[str, int]:
    if not product_ids:
        return {}
    normalized: List[str] = []
    for task in requested_tasks or []:
        norm = _normalize_post_import_task(task)
        if norm and norm in POST_IMPORT_TASKS_ALLOWED:
            normalized.append(norm)
    normalized = _dedupe_preserve_order(normalized)
    if not normalized:
        return {}
    id_list: List[int] = []
    for pid in product_ids:
        try:
            id_list.append(int(pid))
        except Exception:
            continue
    if not id_list:
        return {}
    conn = ensure_db()
    summary: Dict[str, int] = {}
    ids = [int(pid) for pid in _dedupe_preserve_order(id_list)]
    for task_type in normalized:
        summary[task_type] = database.enqueue_ai_tasks(
            conn,
            task_type,
            ids,
            import_task_id=task_id,
        )
    return summary


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        try:
            return int(float(value or 0))
        except Exception:
            return 0


def _normalize_ai_progress(progress: Optional[Mapping[str, Mapping[str, Any]]]) -> Dict[str, Dict[str, int]]:
    normalized: Dict[str, Dict[str, int]] = {}
    for task_name in AI_PROGRESS_TASKS:
        entry = progress.get(task_name) if progress else None
        normalized[task_name] = {
            "requested": _coerce_int((entry or {}).get("requested")),
            "processed": _coerce_int((entry or {}).get("processed")),
            "failed": _coerce_int((entry or {}).get("failed")),
        }
    return normalized


def _empty_ai_progress() -> Dict[str, Dict[str, int]]:
    return _normalize_ai_progress({})


def _get_post_import_task_flags() -> Dict[str, bool]:
    cfg = config.load_config()
    flags = {name: True for name in AI_PROGRESS_TASKS}

    raw_flags: Mapping[str, Any] | None = None
    if isinstance(cfg.get("postImportTasks"), Mapping):
        raw_flags = cfg.get("postImportTasks")  # type: ignore[assignment]
    elif isinstance(cfg.get("post_import_tasks"), Mapping):
        raw_flags = cfg.get("post_import_tasks")  # type: ignore[assignment]

    if raw_flags:
        for key, value in raw_flags.items():
            norm = _normalize_post_import_task(key)
            if not norm:
                continue
            flags[norm] = bool(value)

    if not config.is_auto_fill_ia_on_import_enabled():
        for key in flags.keys():
            flags[key] = False

    return flags


def _resolve_enabled_post_import_tasks() -> List[str]:
    flags = _get_post_import_task_flags()
    ordered = [name for name in AI_PROGRESS_TASKS if flags.get(name, False)]
    return ordered


def _parse_date(s: str):
    s = (s or "").strip()
    if not s:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None

def ensure_db():
    global _DB_INIT, _DB_INIT_PATH

    target_path = str(DB_PATH)
    conn = get_db(target_path)
    if not _DB_INIT or _DB_INIT_PATH != target_path:
        with _DB_INIT_LOCK:
            if not _DB_INIT or _DB_INIT_PATH != target_path:
                try:
                    database.initialize_database(conn)
                except Exception:
                    logger.exception("Database initialization failed")
                    raise
                try:
                    cur = conn.cursor()
                    cur.execute(
                        "DELETE FROM products WHERE id IN (1,2,3) OR lower(name) LIKE '%test%' OR lower(name) LIKE '%prueba%'"
                    )
                    conn.commit()
                except Exception:
                    pass
                logger.info("Database ready at %s", DB_PATH)
                _DB_INIT = True
                _DB_INIT_PATH = target_path
    return conn


def _update_import_status(task_id: str, **updates) -> Dict[str, Any]:
    with _IMPORT_STATUS_LOCK:
        state = IMPORT_STATUS.setdefault(task_id, {})
        state.update(updates)
        return dict(state)


def _get_import_status(task_id: str) -> Dict[str, Any] | None:
    with _IMPORT_STATUS_LOCK:
        status = IMPORT_STATUS.get(task_id)
        if status is None:
            return None
        result = dict(status)
        result["ai_progress"] = _normalize_ai_progress(result.get("ai_progress"))
        result.setdefault("post_import_ready", False)
        return result


def _run_post_import_auto(task_id: str, product_ids: Sequence[int]) -> None:
    product_list = [
        int(pid)
        for pid in _dedupe_preserve_order(list(product_ids or []))
        if str(pid).strip()
    ]
    progress = _empty_ai_progress()
    try:
        enabled_tasks = _resolve_enabled_post_import_tasks()
        if not enabled_tasks or not product_list:
            _update_import_status(
                task_id,
                ai_progress=copy.deepcopy(progress),
                state="DONE",
                stage="done",
                post_import_ready=False,
            )
            return

        summary = _enqueue_post_import_tasks(task_id, product_list, enabled_tasks)
        if summary:
            _update_import_status(
                task_id,
                post_import={
                    "tasks": summary,
                    "product_count": len(product_list),
                },
            )
        for name in AI_PROGRESS_TASKS:
            progress[name]["requested"] = _coerce_int(summary.get(name))

        _update_import_status(task_id, ai_progress=copy.deepcopy(progress))

        batch_cfg = config.get_ai_batch_config()
        try:
            batch_size = int(batch_cfg.get("BATCH_SIZE", 200) or 200)
        except Exception:
            batch_size = 200
        batch_size = max(1, min(batch_size, 200))
        try:
            max_parallel = int(batch_cfg.get("MAX_CONCURRENCY", 3) or 3)
        except Exception:
            max_parallel = 3
        max_parallel = max(1, min(max_parallel, 8))

        progress_lock = threading.Lock()

        def _on_progress(import_task_id: str, task_type: str, totals: Mapping[str, int]) -> None:
            if import_task_id != task_id:
                return
            with progress_lock:
                entry = progress.setdefault(
                    task_type,
                    {"requested": 0, "processed": 0, "failed": 0},
                )
                if "requested" in totals:
                    entry["requested"] = max(
                        entry.get("requested", 0),
                        _coerce_int(totals.get("requested")),
                    )
                entry["processed"] = _coerce_int(
                    totals.get("processed", entry.get("processed", 0))
                )
                entry["failed"] = _coerce_int(
                    totals.get("failed", entry.get("failed", 0))
                )
                _update_import_status(task_id, ai_progress=copy.deepcopy(progress))

        runner.register_progress_callback(task_id, _on_progress)
        try:
            result = runner.run_auto(
                set(enabled_tasks),
                batch_size=batch_size,
                max_parallel=max_parallel,
            )
        finally:
            runner.unregister_progress_callback(task_id)

        import_summary = result.get(task_id, {})
        for task_name, totals in (import_summary.get("tasks") or {}).items():
            entry = progress.setdefault(
                task_name,
                {"requested": 0, "processed": 0, "failed": 0},
            )
            entry["requested"] = max(
                entry.get("requested", 0), _coerce_int(totals.get("requested"))
            )
            entry["processed"] = _coerce_int(totals.get("processed", entry.get("processed", 0)))
            entry["failed"] = _coerce_int(totals.get("failed", entry.get("failed", 0)))

        errors = [
            str(msg)
            for msg in (import_summary.get("errors") or [])
            if msg
        ]

        final_state = "ERROR" if errors else "DONE"
        final_stage = "ai_post" if errors else "done"
        updates: Dict[str, Any] = {
            "state": final_state,
            "stage": final_stage,
            "post_import_ready": False,
            "ai_progress": copy.deepcopy(progress),
        }
        if errors:
            updates["error"] = "; ".join(dict.fromkeys(errors))
        else:
            updates["error"] = None
        _update_import_status(task_id, **updates)
    except Exception as exc:
        runner.unregister_progress_callback(task_id)
        logger.exception("Post-import automation failed: task_id=%s", task_id)
        progress_snapshot = copy.deepcopy(progress)
        _update_import_status(
            task_id,
            state="ERROR",
            stage="ai_post",
            error=str(exc),
            post_import_ready=False,
            ai_progress=progress_snapshot,
        )

def _ensure_desire(product: Dict[str, Any], extras: Dict[str, Any]) -> str:
    """Return desire value from known sources.

    Precedence: product.desire -> extras.desire -> product.ai_desire ->
    product.ai_desire_label -> product.desire_magnitude.  Normalizes to
    string and logs when no value is found."""

    sources = [
        ("product.desire", rget(product, "desire")),
        ("extras.desire", rget(extras, "desire")),
        ("product.ai_desire", rget(product, "ai_desire")),
        ("product.ai_desire_label", rget(product, "ai_desire_label")),
        ("product.desire_magnitude", rget(product, "desire_magnitude")),
    ]
    desire_val = ""
    source_used = None
    for name, val in sources:
        if val not in (None, ""):
            desire_val = str(val)
            source_used = name
            break
    if desire_val == "":
        logger.info(
            "desire_missing=true sources_checked=%s product=%s",
            [s for s, _ in sources],
            rget(product, "id"),
        )
    else:
        logger.info(
            "product=%s desire=%s source=%s",
            rget(product, "id"),
            desire_val,
            source_used,
        )
    return desire_val


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
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, OPTIONS")
        self.end_headers()

    def _set_html(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, OPTIONS")
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
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, OPTIONS")
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
        if path == "/api/log-path":
            self._set_json()
            self.wfile.write(json.dumps({"path": str(LOG_PATH)}).encode("utf-8"))
            return
        if path == "/api/auth/has-key":
            has_key = bool(config.get_api_key())
            self._set_json()
            self.wfile.write(json.dumps({"ok": True, "has_key": has_key}).encode("utf-8"))
            return
        if path == "/api/trends/summary":
            params = parse_qs(parsed.query)
            qs_from = params.get("from", [""])[0]
            qs_to = params.get("to", [""])[0]
            filters_s = params.get("filters", [None])[0]

            today = date.today()
            d_from = _parse_date(qs_from)
            d_to = _parse_date(qs_to)

            if d_from is None and d_to is None:
                d_to = today
                d_from = today - timedelta(days=29)
            elif d_from is None:
                d_from = d_to - timedelta(days=29)
            elif d_to is None:
                d_to = d_from + timedelta(days=29)

            if d_from > d_to:
                d_from, d_to = d_to, d_from

            start_dt = datetime.combine(d_from, datetime.min.time())
            end_dt = datetime.combine(d_to + timedelta(days=1), datetime.min.time())

            filters = None
            if filters_s:
                try:
                    filters = json.loads(filters_s)
                except Exception:
                    filters = None
            try:
                resp = trends_service.get_trends_summary(start_dt, end_dt, filters)
            except Exception:
                resp = {
                    "categories": [],
                    "timeseries": [],
                    "granularity": "day",
                    "totals": {
                        "unique_products": 0,
                        "units": 0,
                        "revenue": 0,
                        "avg_price": 0,
                        "avg_rating": 0,
                        "rev_per_unit": 0,
                        "delta_revenue_pct": 0,
                        "delta_units_pct": 0,
                    },
                }
            self._set_json()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return
        if path == "/api/config/winner-weights":
            from .services.config import (
                get_winner_weights_raw,
                get_winner_order_raw,
                get_weights_enabled_raw,
                compute_effective_int,
            )

            raw = get_winner_weights_raw()
            order = get_winner_order_raw()
            enabled = get_weights_enabled_raw()
            raw_eff = {k: (raw.get(k, 0) if enabled.get(k, True) else 0) for k in raw}
            eff_int = compute_effective_int(raw_eff, order)
            logger.info("weights_effective_int=%s order=%s", eff_int, order)
            resp = {
                **raw,
                "weights": raw,
                "order": order,
                "effective": {"int": eff_int},
                "weights_enabled": enabled,
                "weights_order": order,
            }
            self._set_json()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return
        if path == "/_import_history":
            params = parse_qs(parsed.query)
            try:
                limit = int(params.get("limit", ["20"])[0])
            except Exception:
                limit = 20
            conn = ensure_db()
            rows = [row_to_dict(r) for r in database.get_import_history(conn, limit)]
            self.safe_write(lambda: self.send_json(rows))
            return
        if path == "/_import_status":
            params = parse_qs(parsed.query)
            task_id_param = params.get("task_id", [""])[0]
            if not task_id_param:
                self.safe_write(lambda: self.send_json({"state": "unknown"}))
                return
            status = _get_import_status(task_id_param)
            if status:
                if "task_id" not in status:
                    status["task_id"] = task_id_param
                self.safe_write(lambda: self.send_json(status))
                return
            try:
                task_id = int(task_id_param)
            except Exception:
                self.safe_write(lambda: self.send_json({"state": "unknown"}))
                return
            conn = ensure_db()
            row = database.get_import_job(conn, task_id)
            if row:
                data = row_to_dict(row)
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
                data["ai_progress"] = _normalize_ai_progress(data.get("ai_counts"))
                data.setdefault("post_import_ready", False)
                self.safe_write(lambda: self.send_json(data))
            else:
                self.safe_write(lambda: self.send_json({"state": "unknown"}))
            return
        if path in ("/products", "/api/products"):
            # Return a list of products including extra metadata for UI display
            conn = ensure_db()
            rows = []
            for p_row in database.list_products(conn):
                p = row_to_dict(p_row)
                extra = rget(p, "extra", {})
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
                score_value = rget(p, "winner_score")
                dr = rget(p, "date_range")
                if dr is None:
                    dr = extra_dict.get("date_range")
                price_val = rget(p, "price")
                desire_db = rget(p, "desire")
                if desire_db in (None, ""):
                    desire_db = _ensure_desire(p, extra_dict)
                desire_val = (desire_db or "").strip() or None
                row = {
                    "id": rget(p, "id"),
                    "name": rget(p, "name"),
                    "category": rget(p, "category"),
                    "price": price_val,
                    "image_url": rget(p, "image_url"),
                    "desire": desire_val,
                    "desire_magnitude": rget(p, "desire_magnitude"),
                    "awareness_level": rget(p, "awareness_level"),
                    "competition_level": rget(p, "competition_level"),
                    "rating": extra_dict.get("rating"),
                    "units_sold": extra_dict.get("units_sold"),
                    "revenue": extra_dict.get("revenue"),
                    "conversion_rate": extra_dict.get("conversion_rate"),
                    "launch_date": extra_dict.get("launch_date"),
                    "date_range": dr or "",
                    "extras": extra_dict,
                }
                if price_val is not None:
                    try:
                        row["price_display"] = round(float(price_val), 2)
                    except Exception:
                        row["price_display"] = price_val
                row["winner_score"] = score_value
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
                "weights": config.get_weights(),
                "has_api_key": bool(key),
                "oldness_preference": cfg.get("oldness_preference", "newer"),
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
            weights = cfg.get("weights", {})
            self._set_json()
            self.wfile.write(json.dumps(weights).encode("utf-8"))
            return
        if path == "/api/winner-score/explain" and DEBUG:
            self.handle_winner_score_explain(parsed)
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
            score = row_to_dict(scores[0])
            self._set_json()
            self.wfile.write(json.dumps(score).encode("utf-8"))
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
                    p_dict = row_to_dict(p)
                    desire_val = _ensure_desire(p_dict, extra_dict)
                    score_dict = row_to_dict(score)
                    score_value = rget(score_dict, "winner_score")
                    breakdown_data = {}
                    if score_dict:
                        try:
                            raw_breakdown = rget(score_dict, "winner_score_breakdown")
                            breakdown_data = json.loads(raw_breakdown or "{}")
                        except Exception:
                            breakdown_data = {}
                    row = {
                        "id": p["id"],
                        "name": p["name"],
                        "category": p["category"],
                        "price": p["price"],
                        "image_url": p["image_url"],
                        "desire": desire_val,
                        "desire_magnitude": rget(p_dict, "desire_magnitude"),
                        "extras": extra_dict,
                    }
                    row["winner_score"] = score_value
                    if score_dict:
                        row["winner_score_breakdown"] = breakdown_data
                    rows.append(row)
                self._set_json()
                self.wfile.write(json.dumps(rows).encode("utf-8"))
                return
        # trends endpoint: compute analytics for categories, keywords and scatter plots
        if path == "/trends":
            qs = parse_qs(parsed.query)
            start_str = qs.get("start", [None])[0]
            end_str = qs.get("end", [None])[0]
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
                    try:
                        score_val = scores[0]["winner_score"]
                    except Exception:
                        score_val = None
                if score_val is not None:
                    rows.append((p["id"], p["name"], score_val))
            rows.sort(key=lambda x: x[2], reverse=True)
            key_name = "winner_score"
            top_products = [{"id": r[0], "name": r[1], "winner_score": r[2]} for r in rows[:10]]

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
                    if 'winner_score' in sc.keys():
                        score_val = sc['winner_score']
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
            headers = ["id", "name", "Winner Score", "Desire", "Desire Magnitude", "Awareness Level", "Competition Level", "Date Range"]
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
        if path == "/api/auth/set-key":
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length).decode('utf-8')
            try:
                data = json.loads(body)
                key = str(data.get("api_key", "")).strip()
            except Exception:
                self._set_json(400)
                self.wfile.write(json.dumps({"ok": False, "has_key": False, "error": "invalid_json"}).encode('utf-8'))
                return
            if not key:
                self._set_json(400)
                self.wfile.write(json.dumps({"ok": False, "has_key": False, "error": "empty_api_key"}).encode('utf-8'))
                return
            try:
                resp = requests.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=15,
                )
                if resp.status_code != 200:
                    raise ValueError(resp.text)
            except Exception as exc:
                self._set_json(400)
                self.wfile.write(json.dumps({"ok": False, "has_key": False, "error": str(exc)}).encode('utf-8'))
                return
            cfg = config.load_config()
            cfg["api_key"] = key
            config.save_config(cfg)
            self._set_json()
            self.wfile.write(json.dumps({"ok": True, "has_key": True}).encode('utf-8'))
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
        if path == "/api/ai/run_post_import":
            self.handle_ai_run_post_import()
            return
        if path == "/auto_weights":
            self.handle_auto_weights()
            return
        if path == "/api/config/winner-weights/ai":
            self.handle_scoring_v2_auto_weights_gpt()
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
        if path == "/scoring/v2/generate" or path == "/api/winner-score/generate":
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
            product = row_to_dict(database.get_product(conn, pid))
            try:
                extra_dict = json.loads(rget(product, "extra") or "{}")
            except Exception:
                extra_dict = {}
            product["desire"] = _ensure_desire(product, extra_dict)
            self._set_json()
            self.wfile.write(json.dumps(product).encode('utf-8'))
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
            product = row_to_dict(database.get_product(conn, pid))
            if product:
                try:
                    extra_dict = json.loads(rget(product, "extra") or "{}")
                except Exception:
                    extra_dict = {}
                desire_db = rget(product, "desire")
                if desire_db in (None, ""):
                    desire_db = _ensure_desire(product, extra_dict)
                product["desire"] = (desire_db or "").strip() or None
                self._set_json()
                self.wfile.write(json.dumps(product).encode('utf-8'))
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
            weights_cfg = cfg.get('weights', {})
            for k, v in data.items():
                try:
                    weights_cfg[k] = float(v)
                except Exception:
                    continue
            cfg['weights'] = weights_cfg
            config.save_config(cfg)
            self._set_json()
            self.wfile.write(json.dumps({"status": "ok"}).encode('utf-8'))
            return
        self.send_error(404)

    def do_PATCH(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path.startswith("/api/products/"):
            try:
                pid = int(path.split("/")[-1])
            except Exception:
                self._set_json(400)
                self.wfile.write(json.dumps({"error": "Invalid ID"}).encode("utf-8"))
                return
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length).decode('utf-8') if length else ""
            try:
                data = json.loads(body or "{}")
                if not isinstance(data, dict):
                    raise ValueError
            except Exception:
                self._set_json(400)
                self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode('utf-8'))
                return
            logger.info("PATCH product_id=%s fields=%s", pid, list(data.keys()))
            conn = ensure_db()
            prod = database.get_product(conn, pid)
            if not prod:
                logger.warning("PATCH not found product_id=%s", pid)
                self._set_json(404)
                self.wfile.write(json.dumps({"error": "Not found"}).encode('utf-8'))
                return
            allowed = {"desire", "desire_magnitude", "awareness_level", "competition_level"}
            fields = {k: v for k, v in data.items() if k in allowed}
            if fields:
                database.update_product(conn, pid, **fields)
            product = row_to_dict(database.get_product(conn, pid))
            self._set_json()
            self.wfile.write(json.dumps(product).encode('utf-8'))
            return
        if path == "/api/config/winner-weights":
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length).decode('utf-8')
            try:
                data = json.loads(body or "{}")
                if not isinstance(data, dict):
                    raise ValueError
            except Exception:
                self._set_json(400)
                self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode('utf-8'))
                return
            from .services.config import (
                set_winner_weights_raw,
                set_winner_order_raw,
                get_winner_order_raw,
                set_weights_enabled_raw,
                get_weights_enabled_raw,
                compute_effective_int,
                ALLOWED_FIELDS,
            )
            from .services import winner_score

            payload_map = (
                data.get("winner_weights")
                or data.get("weights")
                or {k: v for k, v in data.items() if k in ALLOWED_FIELDS}
            )
            saved = set_winner_weights_raw(payload_map)
            order_in = data.get("order") if isinstance(data, dict) else None
            if isinstance(order_in, list):
                order = [k for k in order_in if k in saved]
            else:
                order = get_winner_order_raw()
            if "awareness" not in order:
                order.append("awareness")
            saved_order = set_winner_order_raw(order)
            en_in = data.get("weights_enabled") if isinstance(data, dict) else None
            if isinstance(en_in, dict):
                set_weights_enabled_raw(en_in)
            enabled = get_weights_enabled_raw()
            winner_score.invalidate_weights_cache()
            eff_map = {k: (saved.get(k, 0) if enabled.get(k, True) else 0) for k in saved}
            resp = {
                **saved,
                "weights": saved,
                "order": saved_order,
                "effective": {"int": compute_effective_int(eff_map, saved_order)},
                "weights_enabled": enabled,
                "weights_order": saved_order,
            }
            self._set_json()
            self.wfile.write(json.dumps(resp).encode('utf-8'))
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
        if ext in {".csv", ".xlsx", ".xls"}:
            ensure_db()
            task_id = str(int(time.time() * 1000))
            _update_import_status(
                task_id,
                state="queued",
                stage="queued",
                done=0,
                total=0,
                error=None,
                imported=0,
                filename=filename,
                post_import_ready=False,
                ai_progress=_empty_ai_progress(),
            )
            file_bytes = data
            import_token = f"{task_id}-{uuid.uuid4().hex}"
            def run_import_task():
                _update_import_status(task_id, state="running", stage="running", started_at=time.time())
                product_ids: List[int] = []
                consumed_token = False
                try:
                    def cb(**kwargs):
                        updates = dict(kwargs)
                        if "done" in updates:
                            try:
                                updates["done"] = int(updates.get("done") or 0)
                            except Exception:
                                updates["done"] = 0
                        if "total" in updates:
                            try:
                                updates["total"] = int(updates.get("total") or 0)
                            except Exception:
                                updates["total"] = 0
                        updates.setdefault("state", "running")
                        _update_import_status(task_id, **updates)

                    # GPT calls for desire/imputación used to run here synchronously.
                    # They are now deferred to the post-import task queue to keep the
                    # upload path responsive.
                    count = unified_run_import(
                        file_bytes,
                        filename,
                        status_cb=cb,
                        import_token=import_token,
                    )
                    conn = ensure_db()
                    product_ids = database.consume_import_token(conn, import_token)
                    consumed_token = True
                    snapshot = _get_import_status(task_id) or {}
                    done_val = int(snapshot.get("done", 0) or 0)
                    if done_val < count:
                        done_val = count
                    total_val = int(snapshot.get("total", count) or 0)
                    if total_val < done_val:
                        total_val = done_val
                    _update_import_status(
                        task_id,
                        state="RUNNING",
                        stage="ai_post",
                        done=done_val,
                        total=total_val,
                        imported=count,
                        finished_at=time.time(),
                        post_import_ready=True,
                        ai_progress=_empty_ai_progress(),
                    )
                    threading.Thread(
                        target=_run_post_import_auto,
                        args=(task_id, list(product_ids)),
                        daemon=True,
                    ).start()
                except Exception as exc:
                    logger.exception("Unified import failed: filename=%s", filename)
                    _update_import_status(
                        task_id,
                        state="ERROR",
                        stage="error",
                        error=str(exc),
                        finished_at=time.time(),
                        post_import_ready=False,
                    )
                finally:
                    if not consumed_token:
                        try:
                            conn = ensure_db()
                            if not product_ids:
                                product_ids.extend(
                                    database.consume_import_token(conn, import_token)
                                )
                        except Exception:
                            pass

            threading.Thread(target=run_import_task, daemon=True).start()
            self.safe_write(lambda: self.send_json({"task_id": task_id}, status=202))
            return

        if ext == ".json":
            try:
                payload = json.loads(data.decode("utf-8", errors="ignore"))
            except Exception as exc:
                self.safe_write(lambda: self.send_json({"error": "invalid_json", "detail": str(exc)}, status=400))
                return
            if not isinstance(payload, list):
                self.safe_write(lambda: self.send_json({"error": "invalid_json"}, status=400))
                return
            records = [item for item in payload if isinstance(item, dict)]
            ensure_db()
            task_id = str(int(time.time() * 1000))
            _update_import_status(
                task_id,
                state="queued",
                stage="queued",
                done=0,
                total=len(records),
                error=None,
                imported=0,
                filename=filename,
                post_import_ready=False,
                ai_progress=_empty_ai_progress(),
            )

            import_token = f"{task_id}-{uuid.uuid4().hex}"
            def run_json():
                _update_import_status(task_id, state="running", stage="running", started_at=time.time())
                product_ids: List[int] = []
                consumed_token = False
                try:
                    def cb(**kwargs):
                        updates = dict(kwargs)
                        if "done" in updates:
                            try:
                                updates["done"] = int(updates.get("done") or 0)
                            except Exception:
                                updates["done"] = 0
                        if "total" in updates:
                            try:
                                updates["total"] = int(updates.get("total") or 0)
                            except Exception:
                                updates["total"] = 0
                        updates.setdefault("state", "running")
                        _update_import_status(task_id, **updates)

                    enriched: List[Dict[str, Any]] = []
                    for item in records:
                        row = dict(item)
                        if filename:
                            row.setdefault("source", filename)
                        row.setdefault("winner_score", "0")
                        row.setdefault("_import_token", import_token)
                        enriched.append(row)
                    _update_import_status(
                        task_id,
                        stage="parse_json",
                        done=len(enriched),
                        total=len(enriched),
                    )
                    # Winner score GPT logic is also deferred; only the bulk insert happens here.
                    count = unified_import_records(enriched, status_cb=cb)
                    conn = ensure_db()
                    product_ids = database.consume_import_token(conn, import_token)
                    consumed_token = True
                    snapshot = _get_import_status(task_id) or {}
                    done_val = int(snapshot.get("done", 0) or 0)
                    if done_val < count:
                        done_val = count
                    total_val = int(snapshot.get("total", len(records)) or 0)
                    if total_val < done_val:
                        total_val = done_val
                    _update_import_status(
                        task_id,
                        state="RUNNING",
                        stage="ai_post",
                        done=done_val,
                        total=total_val,
                        imported=count,
                        finished_at=time.time(),
                        post_import_ready=True,
                        ai_progress=_empty_ai_progress(),
                    )
                    threading.Thread(
                        target=_run_post_import_auto,
                        args=(task_id, list(product_ids)),
                        daemon=True,
                    ).start()
                except Exception as exc:
                    logger.exception("Unified JSON import failed: filename=%s", filename)
                    _update_import_status(
                        task_id,
                        state="ERROR",
                        stage="error",
                        error=str(exc),
                        finished_at=time.time(),
                        post_import_ready=False,
                    )
                finally:
                    if not consumed_token:
                        try:
                            conn = ensure_db()
                            if not product_ids:
                                product_ids.extend(
                                    database.consume_import_token(conn, import_token)
                                )
                        except Exception:
                            pass

            threading.Thread(target=run_json, daemon=True).start()
            self.safe_write(lambda: self.send_json({"task_id": task_id}, status=202))
            return

        self.safe_write(lambda: self.send_json({"error": "unsupported_format"}, status=400))

    def handle_evaluate_all(self):
        conn = ensure_db()
        api_key = config.get_api_key() or os.environ.get('OPENAI_API_KEY')
        model = config.get_model()
        evaluated = 0
        weights_map = config.get_weights()
        for p_row in database.list_products(conn):
            p = row_to_dict(p_row)
            pid = rget(p, 'id')
            if database.get_scores_for_product(conn, pid):
                continue
            if not (api_key and model):
                continue
            try:
                try:
                    extra = json.loads(rget(p, "extra") or "{}")
                except Exception:
                    extra = {}
                resp = gpt.evaluate_winner_score(
                    api_key,
                    model,
                    {
                        "title": rget(p, "name"),
                        "description": rget(p, "description"),
                        "category": rget(p, "category"),
                        "metrics": extra,
                    },
                )
                scores = resp.get("scores", {})
                justifs = resp.get("justifications", {})
                weighted = sum(
                    scores.get(f, 3) * weights_map.get(f, 0.0)
                    for f in WINNER_SCORE_FIELDS
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
                    winner_score_raw=raw_score,
                    winner_score=pct,
                    winner_score_breakdown=breakdown,
                )
                evaluated += 1
            except Exception:
                continue
            self._set_json()
            self.wfile.write(json.dumps({"evaluated": evaluated}).encode('utf-8'))
            return
        # Legacy evaluation removed; always use Winner Score above

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
            cfg['weights'] = winner_calc.sanitize_weights(data['weights'])
        if 'autoFillIAOnImport' in data:
            cfg['autoFillIAOnImport'] = bool(data['autoFillIAOnImport'])
        if 'oldness_preference' in data:
            pref = str(data.get('oldness_preference', '')).strip().lower()
            if pref in ("older", "newer"):
                cfg['oldness_preference'] = pref
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

    def handle_ai_run_post_import(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode('utf-8') if length else ""
        if body:
            try:
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError
            except Exception:
                self._set_json(400)
                self.wfile.write(json.dumps({"error": "invalid_json"}).encode('utf-8'))
                return
        else:
            payload = {}

        raw_tasks = payload.get("tasks") or payload.get("task_types")
        if raw_tasks is None:
            requested_tasks = list(DEFAULT_POST_IMPORT_TASKS)
        elif isinstance(raw_tasks, (list, tuple)):
            requested_tasks = [str(t) for t in raw_tasks]
        else:
            requested_tasks = [str(raw_tasks)]

        limit_raw = payload.get("limit", 50)
        try:
            limit = int(limit_raw)
        except Exception:
            limit = 50
        limit = max(1, min(limit, 200))

        api_key = config.get_api_key() or os.environ.get('OPENAI_API_KEY')
        model = config.get_model()
        if not api_key or not model:
            self._set_json(503)
            self.wfile.write(json.dumps({"error": "openai_unavailable"}).encode('utf-8'))
            return

        normalized_tasks = [
            t
            for t in (
                _normalize_post_import_task(task)
                for task in requested_tasks
            )
            if t in POST_IMPORT_TASKS_ALLOWED
        ]
        normalized_tasks = [t for t in normalized_tasks if t != "winner_score"]
        if not normalized_tasks:
            normalized_tasks = list(DEFAULT_POST_IMPORT_TASKS)

        conn = ensure_db()
        pending = database.fetch_pending_ai_tasks(
            conn,
            task_types=normalized_tasks,
            limit=limit,
        )
        if not pending:
            self._set_json()
            self.wfile.write(
                json.dumps({"ok": True, "processed": 0, "completed": 0, "failed": 0}).encode('utf-8')
            )
            return

        task_ids = [row["id"] for row in pending]
        database.mark_ai_tasks_in_progress(conn, task_ids)

        product_id_order = _dedupe_preserve_order([int(row["product_id"]) for row in pending])
        products = database.get_products_by_ids(conn, product_id_order)
        if not products:
            database.fail_ai_tasks(conn, task_ids, "missing_products")
            self._set_json(404)
            self.wfile.write(json.dumps({"error": "missing_products"}).encode('utf-8'))
            return

        items: List[Dict[str, Any]] = []
        for prod in products:
            product = row_to_dict(prod)
            try:
                extra = json.loads(rget(product, "extra") or "{}")
            except Exception:
                extra = {}
            pid = rget(product, "id")
            items.append(
                {
                    "id": pid,
                    "name": rget(product, "name"),
                    "category": rget(product, "category"),
                    "price": rget(product, "price"),
                    "rating": extra.get("rating"),
                    "units_sold": extra.get("units_sold"),
                    "revenue": extra.get("revenue"),
                    "conversion_rate": extra.get("conversion_rate"),
                    "launch_date": extra.get("launch_date"),
                    "date_range": rget(product, "date_range") or extra.get("date_range"),
                    "image_url": rget(product, "image_url") or extra.get("image_url"),
                }
            )

        if not items:
            database.fail_ai_tasks(conn, task_ids, "missing_payload")
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "missing_payload"}).encode('utf-8'))
            return

        try:
            ok_map, ko_map, usage, duration = gpt.generate_batch_columns(api_key, model, items)
        except gpt.InvalidJSONError as exc:
            database.fail_ai_tasks(conn, task_ids, "invalid_json")
            self._set_json(502)
            self.wfile.write(json.dumps({"error": "Respuesta IA no es JSON"}).encode('utf-8'))
            return
        except Exception as exc:
            database.fail_ai_tasks(conn, task_ids, str(exc))
            self._set_json(503)
            self.wfile.write(json.dumps({"error": "OpenAI no disponible"}).encode('utf-8'))
            return

        successes: List[int] = []
        failures: List[int] = []
        completed_products: List[int] = []
        for row in pending:
            pid = int(row["product_id"])
            tid = int(row["id"])
            entry = ok_map.get(str(pid))
            if entry:
                updates = {
                    "desire": entry.get("desire"),
                    "desire_magnitude": entry.get("desire_magnitude"),
                    "awareness_level": entry.get("awareness_level"),
                    "competition_level": entry.get("competition_level"),
                    "ai_columns_completed_at": datetime.utcnow().isoformat(),
                }
                clean_updates = {k: v for k, v in updates.items() if v not in (None, "")}
                if clean_updates:
                    database.update_product(conn, pid, **clean_updates)
                successes.append(tid)
                completed_products.append(pid)
            else:
                failures.append(tid)

        if successes:
            database.complete_ai_tasks(conn, successes)
        if failures:
            database.fail_ai_tasks(conn, failures, "missing_result")

        cur = conn.cursor()
        if normalized_tasks:
            placeholders = ",".join(["?"] * len(normalized_tasks))
            cur.execute(
                f"SELECT COUNT(*) FROM ai_task_queue WHERE state='pending' AND task_type IN ({placeholders})",
                tuple(normalized_tasks),
            )
        else:
            cur.execute("SELECT COUNT(*) FROM ai_task_queue WHERE state='pending'")
        pending_left = int(cur.fetchone()[0] or 0)

        if not isinstance(usage, dict):
            usage = {}

        response = {
            "ok": True,
            "processed": len(task_ids),
            "completed": len(successes),
            "failed": len(failures),
            "usage": usage,
            "duration": duration,
            "pending_left": pending_left,
        }
        if ko_map:
            response["ko"] = ko_map
        if completed_products:
            response["product_ids"] = _dedupe_preserve_order(completed_products)
        self._set_json()
        self.wfile.write(json.dumps(response).encode('utf-8'))

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
            s = row_to_dict(scores[0])
            count += 1
            for m in metrics:
                try:
                    val = float(rget(s, m, 0.0))
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
        """Gather products with Winner Score breakdown and success metric."""
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
                breakdown = json.loads(srow["winner_score_breakdown"] or "{}")
                scores = breakdown.get("scores") or {}
            except Exception:
                continue
            if not scores or any(k not in scores for k in WINNER_SCORE_FIELDS):
                continue
            sample = {k: float(scores[k]) for k in WINNER_SCORE_FIELDS}
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
        features = [f for f in (data.get("features") or WINNER_SCORE_FIELDS) if f in winner_calc.ALLOWED_FIELDS]
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
        # --- pedir a GPT (o devolverá fallback desde gpt.py) ---
        result = gpt.recommend_winner_weights(api_key, model, samples, target)
        weights_raw = result.get("weights", {}) or {}
        notes = result.get("justification", "")

        # Filtra a campos permitidos + clamp 0..100 enteros
        allowed = list(winner_calc.ALLOWED_FIELDS)
        final_weights = {}
        for k in allowed:
            v = weights_raw.get(k, 0)
            try:
                v = float(v)
            except Exception:
                v = 0.0
            v = max(0.0, min(100.0, v))
            final_weights[k] = int(round(v))

        # ORDER: por peso desc; a igualdad conserva el orden previo
        prev_settings = winner_calc.load_settings()
        prev_order = prev_settings.get("weights_order") or list(allowed)
        order = sorted(final_weights.keys(), key=lambda k: (-final_weights[k], prev_order.index(k) if k in prev_order else 999))

        # Logs (útiles para ti): ahora ai_raw/ints son lo mismo (0..100 independientes)
        logger.info(
            "ai_raw=%s enabled_only=%s ints=%s order=%s sum=%s",
            final_weights,
            final_weights,
            final_weights,
            order,
            sum(final_weights.values()),
        )

        # Respuesta para el frontend (este ya hace el PATCH /api/config/winner-weights)
        resp = {
            "weights": final_weights,      # 0..100 independientes
            "weights_order": order,        # prioridad explícita
            "order": order,
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
        features = [f for f in (data.get("features") or WINNER_SCORE_FIELDS) if f in winner_calc.ALLOWED_FIELDS]
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

        # weights como correlación absoluta -> reescala a 0..100 independientes
        weights01 = {k: float(weights.get(k, 0.0)) for k in features}
        maxv = max(weights01.values() or [0.0])
        weights_raw = {
            k: (v / maxv * 100.0 if maxv > 0 else 50.0)
            for k, v in weights01.items()
        }

        allowed = list(winner_calc.ALLOWED_FIELDS)
        final_weights = {}
        for k in allowed:
            v = weights_raw.get(k, 0.0)
            v = max(0.0, min(100.0, float(v)))
            final_weights[k] = int(round(v))

        prev_settings = winner_calc.load_settings()
        prev_order = prev_settings.get("weights_order") or list(allowed)
        order = sorted(final_weights.keys(), key=lambda k: (-final_weights[k], prev_order.index(k) if k in prev_order else 999))

        resp = {
            "weights": final_weights,        # 0..100 independientes
            "weights_order": order,
            "order": order,
            "method": "stat",
            "diagnostics": {"n": len(samples_in)},
        }
        self._set_json()
        self.wfile.write(json.dumps(resp).encode('utf-8'))

    def handle_scoring_v2_gpt_evaluate(self):
        """Endpoint that evaluates Winner Score variables via GPT."""

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
        need = [f for f in WINNER_SCORE_FIELDS if f not in scores]
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

        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        debug = params.get("debug", ["0"])[0] == "1"

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length else ""
        try:
            data = json.loads(body) if body else {}
            ids = data.get("product_ids") or data.get("ids") or []
            if ids and not isinstance(ids, list):
                raise ValueError
        except Exception:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode("utf-8"))
            return

        logger.info("Winner Score generate: ids_length=%d", len(ids))
        conn = ensure_db()
        result = winner_calc.generate_winner_scores(conn, product_ids=ids or None, debug=debug)
        resp = {
            "ok": True,
            "processed": result.get("processed", 0),
            "updated": result.get("updated", 0),
            "weights_all": result.get("weights_all"),
            "weights_eff": result.get("weights_eff"),
        }
        if debug:
            resp["diag"] = result.get("diag", {})
        self._set_json()
        self.wfile.write(json.dumps(resp).encode("utf-8"))

    def handle_winner_score_explain(self, parsed):
        """Return detailed Winner Score components for debugging."""

        params = parse_qs(parsed.query)
        ids_param = params.get("ids", [""])[0]
        ids = [int(x) for x in ids_param.split(",") if x.strip()]

        conn = ensure_db()
        weights, order, enabled = winner_calc.load_winner_settings()

        if ids:
            placeholders = ",".join("?" for _ in ids)
            cur = conn.execute(
                f"SELECT * FROM products WHERE id IN ({placeholders})",
                tuple(ids),
            )
            rows = cur.fetchall()
        else:
            rows = database.list_products(conn)

        winner_calc.prepare_oldness_bounds(rows)

        data: Dict[str, Any] = {}
        for row in rows:
            res = winner_calc.compute_winner_score_v2(row, weights, order, enabled)
            sf = res.get("score_float") or 0.0
            score_raw = max(0.0, min(1.0, sf)) * 100.0
            data[row["id"]] = {
                "present": res.get("present_fields", []),
                "missing": res.get("missing_fields", []),
                "effective_weights": {k: round(v, 3) for k, v in res.get("effective_weights", {}).items()},
                "score_raw": score_raw,
                "score_int": int(round(score_raw)),
            }

        self._set_json()
        self.wfile.write(json.dumps(data).encode("utf-8"))

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
