"""Compatibility shims for the legacy frontend endpoints.

These routes mirror the original ``web_app`` HTTP server contract so the
existing JavaScript UI keeps functioning while the backend is ported to
Flask.  Each handler logs a warning so the new contract usage can be tracked
and gradually migrated.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import queue
import threading
import time
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import math
import requests
from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    request,
    send_file,
)

from product_research_app import config, database, gpt
from product_research_app.db import get_db
from product_research_app.progress_events import subscribe, unsubscribe
from product_research_app.sse import publish_progress
from product_research_app.services import trends_service
from product_research_app.services import winner_score as winner_calc
from product_research_app.services.importer_fast import (
    DEFAULT_BATCH_SIZE,
    fast_import,
    fast_import_records,
)
from product_research_app.services.winner_score import generate_winner_scores
from product_research_app.utils.db import rget, row_to_dict
from product_research_app import web_app


legacy_bp = Blueprint("legacy", __name__)


def _log_legacy(path: str) -> None:
    current_app.logger.warning("legacy endpoint shim hit", extra={"path": path})


def _ensure_conn():
    conn = get_db()
    database.initialize_database(conn)
    return conn


@legacy_bp.get("/config")
def legacy_config() -> Response:
    _log_legacy("/config")
    cfg = config.load_config()
    key = cfg.get("api_key") or ""
    data: Dict[str, Any] = {
        "model": cfg.get("model", "gpt-4o"),
        "weights": config.get_weights(),
        "has_api_key": bool(key),
        "oldness_preference": cfg.get("oldness_preference", "newer"),
    }
    if key:
        data["api_key_last4"] = key[-4:]
        data["api_key_length"] = len(key)
        data["api_key_hash"] = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return jsonify(data)


@legacy_bp.get("/api/auth/has-key")
def legacy_has_key() -> Response:
    _log_legacy("/api/auth/has-key")
    return jsonify({"ok": True, "has_key": bool(config.get_api_key())})


@legacy_bp.post("/api/auth/set-key")
def legacy_set_key() -> Response:
    _log_legacy("/api/auth/set-key")
    payload = request.get_json(silent=True) or {}
    api_key = str(payload.get("api_key", "")).strip()
    if not api_key:
        return jsonify({"ok": False, "has_key": False, "error": "empty_api_key"}), 400
    try:
        resp = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        if resp.status_code != 200:
            raise ValueError(resp.text)
    except Exception as exc:
        return (
            jsonify({"ok": False, "has_key": False, "error": str(exc)}),
            400,
        )
    cfg = config.load_config()
    cfg["api_key"] = api_key
    config.save_config(cfg)
    return jsonify({"ok": True, "has_key": True})


def _serialize_product(row: Dict[str, Any]) -> Dict[str, Any]:
    extras_raw = row.get("extra")
    try:
        extras = json.loads(extras_raw) if isinstance(extras_raw, str) else (extras_raw or {})
    except Exception:
        extras = {}
    desire_db = rget(row, "desire")
    if desire_db in (None, ""):
        desire_db = web_app._ensure_desire(row, extras)
    return {
        "id": row.get("id"),
        "name": row.get("name"),
        "category": row.get("category"),
        "price": row.get("price"),
        "image_url": row.get("image_url"),
        "desire": (desire_db or "").strip() or None,
        "desire_magnitude": row.get("desire_magnitude"),
        "awareness_level": row.get("awareness_level"),
        "competition_level": row.get("competition_level"),
        "rating": extras.get("rating"),
        "units_sold": extras.get("units_sold"),
        "revenue": extras.get("revenue"),
        "conversion_rate": extras.get("conversion_rate"),
        "launch_date": extras.get("launch_date"),
        "date_range": row.get("date_range") or extras.get("date_range") or "",
        "extras": extras,
        "winner_score": row.get("winner_score"),
    }


@legacy_bp.route("/products", methods=["GET"])
@legacy_bp.route("/api/products", methods=["GET"])
def legacy_list_products() -> Response:
    _log_legacy(request.path)
    conn = _ensure_conn()
    rows: List[Dict[str, Any]] = []
    for prod in database.list_products(conn):
        rows.append(_serialize_product(row_to_dict(prod)))
    return jsonify(rows)


def _load_product(conn, product_id: int) -> Dict[str, Any] | None:
    row = database.get_product(conn, product_id)
    if not row:
        return None
    return _serialize_product(row_to_dict(row))


@legacy_bp.put("/products/<int:product_id>")
def legacy_put_product(product_id: int) -> Response:
    _log_legacy("PUT /products")
    payload = request.get_json(silent=True) or {}
    if "price" in payload and payload.get("source") != "import":
        payload.pop("price", None)
    conn = _ensure_conn()
    database.update_product(conn, product_id, **payload)
    product = _load_product(conn, product_id)
    if not product:
        return jsonify({"error": "Not found"}), 404
    return jsonify(product)


@legacy_bp.patch("/api/products/<int:product_id>")
def legacy_patch_product(product_id: int) -> Response:
    _log_legacy("PATCH /api/products/<id>")
    payload = request.get_json(silent=True) or {}
    allowed = {"desire", "desire_magnitude", "awareness_level", "competition_level"}
    fields = {k: v for k, v in payload.items() if k in allowed}
    conn = _ensure_conn()
    if not database.get_product(conn, product_id):
        return jsonify({"error": "Not found"}), 404
    if fields:
        database.update_product(conn, product_id, **fields)
    product = _load_product(conn, product_id)
    return jsonify(product or {})


@legacy_bp.get("/lists")
def legacy_lists() -> Response:
    _log_legacy("/lists")
    conn = _ensure_conn()
    data = [
        {"id": row["id"], "name": row["name"], "count": row["count"]}
        for row in database.get_lists(conn)
    ]
    return jsonify(data)


@legacy_bp.get("/list/<int:list_id>")
def legacy_list_detail(list_id: int) -> Response:
    _log_legacy(f"/list/{list_id}")
    conn = _ensure_conn()
    prods = database.get_products_in_list(conn, list_id)
    rows = []
    for p in prods:
        prod = row_to_dict(p)
        extras_raw = prod.get("extra")
        try:
            extras = json.loads(extras_raw) if isinstance(extras_raw, str) else (extras_raw or {})
        except Exception:
            extras = {}
        desire_val = web_app._ensure_desire(prod, extras)
        scores = database.get_scores_for_product(conn, prod["id"])
        score_dict = row_to_dict(scores[0]) if scores else None
        payload = {
            "id": prod["id"],
            "name": prod.get("name"),
            "category": prod.get("category"),
            "price": prod.get("price"),
            "image_url": prod.get("image_url"),
            "desire": desire_val,
            "desire_magnitude": prod.get("desire_magnitude"),
            "extras": extras,
        }
        if score_dict:
            try:
                breakdown = json.loads(score_dict.get("winner_score_breakdown") or "{}")
            except Exception:
                breakdown = {}
            payload["winner_score"] = score_dict.get("winner_score")
            payload["winner_score_breakdown"] = breakdown
        rows.append(payload)
    return jsonify(rows)


@legacy_bp.post("/create_list")
def legacy_create_list() -> Response:
    _log_legacy("/create_list")
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Nombre no proporcionado"}), 400
    conn = _ensure_conn()
    list_id = database.create_list(conn, name)
    return jsonify({"id": list_id, "name": name})


@legacy_bp.post("/delete_list")
def legacy_delete_list() -> Response:
    _log_legacy("/delete_list")
    payload = request.get_json(silent=True) or {}
    try:
        lid = int(payload.get("id"))
    except Exception:
        return jsonify({"error": "Datos inválidos"}), 400
    mode = payload.get("mode", "remove")
    tgt = payload.get("targetGroupId")
    try:
        tgt_id = int(tgt) if tgt is not None else None
    except Exception:
        return jsonify({"error": "Datos inválidos"}), 400
    conn = _ensure_conn()
    try:
        result = database.delete_list(conn, lid, mode=mode, target_list_id=tgt_id)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


@legacy_bp.post("/add_to_list")
def legacy_add_to_list() -> Response:
    _log_legacy("/add_to_list")
    payload = request.get_json(silent=True) or {}
    try:
        lid = int(payload.get("id"))
        ids = [int(x) for x in payload.get("ids", [])]
    except Exception:
        return jsonify({"error": "Datos inválidos"}), 400
    conn = _ensure_conn()
    for pid in ids:
        database.add_product_to_list(conn, lid, pid)
    return jsonify({"added": len(ids)})


@legacy_bp.post("/remove_from_list")
def legacy_remove_from_list() -> Response:
    _log_legacy("/remove_from_list")
    payload = request.get_json(silent=True) or {}
    try:
        list_id = int(payload.get("list_id"))
        ids = [int(x) for x in payload.get("ids", [])]
    except Exception:
        return jsonify({"error": "Invalid list_id or ids"}), 400
    conn = _ensure_conn()
    removed = 0
    for pid in ids:
        try:
            database.remove_product_from_list(conn, list_id, pid)
            removed += 1
        except Exception:
            continue
    return jsonify({"removed": removed})


@legacy_bp.post("/delete")
def legacy_delete_products() -> Response:
    _log_legacy("/delete")
    payload = request.get_json(silent=True) or {}
    ids = payload.get("ids")
    if not isinstance(ids, list):
        return jsonify({"error": "Missing or invalid ids"}), 400
    conn = _ensure_conn()
    job_id = f"delete-{int(time.time() * 1000)}"
    total = len(ids)
    if total <= 0:
        publish_progress(
            job_id,
            {
                "operation": "delete",
                "job_id": job_id,
                "phase": "delete",
                "percent": 100,
                "imported": 0,
                "enriched": 0,
                "failed": 0,
                "queued": 0,
                "message": "Sin elementos para eliminar",
            },
        )
        return jsonify({"deleted": 0})

    publish_progress(
        job_id,
        {
            "operation": "delete",
            "job_id": job_id,
            "phase": "delete",
            "percent": 0,
            "imported": 0,
            "enriched": 0,
            "failed": 0,
            "queued": total,
            "message": "Eliminando productos",
        },
    )

    deleted = 0
    failures = 0
    start = time.perf_counter()
    for pid in ids:
        try:
            pid_int = int(pid)
        except Exception:
            failures += 1
            continue
        try:
            database.delete_product(conn, pid_int)
            deleted += 1
        except Exception:
            failures += 1
            continue
        processed = deleted + failures
        remaining = max(total - processed, 0)
        eta_ms = None
        elapsed = time.perf_counter() - start
        if processed > 0 and remaining > 0:
            eta_ms = int(max(0.0, (elapsed / processed) * remaining * 1000.0))
        publish_progress(
            job_id,
            {
                "operation": "delete",
                "job_id": job_id,
                "phase": "delete",
                "percent": int(round((processed / total) * 100)) if total else 100,
                "imported": processed,
                "enriched": deleted,
                "failed": failures,
                "queued": remaining,
                "eta_ms": eta_ms,
                "message": f"Eliminando {processed}/{total}",
            },
        )

    publish_progress(
        job_id,
        {
            "operation": "delete",
            "job_id": job_id,
            "phase": "delete",
            "percent": 100,
            "imported": deleted + failures,
            "enriched": deleted,
            "failed": failures,
            "queued": 0,
            "message": f"Eliminados {deleted} de {total}",
        },
    )
    return jsonify({"deleted": deleted})


@legacy_bp.post("/custom_gpt")
def legacy_custom_gpt() -> Response:
    _log_legacy("/custom_gpt")
    payload = request.get_json(silent=True) or {}
    prompt = payload.get("prompt")
    if not prompt:
        return jsonify({"error": "Invalid request"}), 400
    api_key = config.get_api_key() or os.environ.get("OPENAI_API_KEY")
    model = config.get_model()
    if not api_key:
        return jsonify({"error": "No API key configured"}), 400
    try:
        resp = gpt.call_openai_chat(
            api_key,
            model,
            [
                {"role": "system", "content": "Eres un asistente útil."},
                {"role": "user", "content": prompt},
            ],
        )
        content = resp["choices"][0]["message"]["content"]
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({"response": content})


@legacy_bp.post("/api/ia/batch-columns")
def legacy_batch_columns() -> Response:
    _log_legacy("/api/ia/batch-columns")
    payload = request.get_json(silent=True) or {}
    items = payload.get("items")
    model = payload.get("model") or "gpt-4o-mini-2024-07-18"
    if not isinstance(items, list):
        return jsonify({"error": "Invalid JSON"}), 400
    api_key = config.get_api_key() or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "OpenAI no disponible"}), 503
    try:
        ok, ko, usage, duration = gpt.generate_batch_columns(api_key, model, items)
        current_app.logger.info(
            "/api/ia/batch-columns tokens=%s duration=%.2fs",
            usage.get("total_tokens"),
            duration,
        )
        return jsonify({"ok": ok, "ko": ko})
    except gpt.InvalidJSONError:
        return jsonify({"error": "Respuesta IA no es JSON"}), 502
    except Exception:
        return jsonify({"error": "OpenAI no disponible"}), 503


@legacy_bp.post("/scoring/v2/auto-weights-gpt")
def legacy_auto_weights_gpt() -> Response:
    _log_legacy("/scoring/v2/auto-weights-gpt")
    payload = request.get_json(silent=True) or {}
    features = [f for f in (payload.get("features") or web_app.WINNER_SCORE_FIELDS) if f in winner_calc.ALLOWED_FIELDS]
    samples_in = payload.get("data_sample") or []
    target = payload.get("target") or ""
    if not samples_in or not target:
        job_id = f"weights-{int(time.time() * 1000)}"
        publish_progress(
            job_id,
            {
                "operation": "weights",
                "job_id": job_id,
                "phase": "weights",
                "percent": 100,
                "imported": 0,
                "enriched": 0,
                "failed": 0,
                "queued": 0,
                "status": "error",
                "message": "Datos insuficientes",
            },
        )
        return jsonify({"error": "Datos insuficientes"}), 400

    samples: List[Dict[str, float]] = []
    for sample in samples_in:
        if "target" not in sample:
            continue
        row = {k: float(sample.get(k, 0.0)) for k in features}
        row[target] = float(sample.get("target", 0.0))
        samples.append(row)

    job_id = f"weights-{int(time.time() * 1000)}"
    api_key = config.get_api_key() or os.environ.get("OPENAI_API_KEY")
    model = config.get_model()
    if not api_key or not model:
        publish_progress(
            job_id,
            {
                "operation": "weights",
                "job_id": job_id,
                "phase": "weights",
                "percent": 100,
                "imported": 0,
                "enriched": 0,
                "failed": 0,
                "queued": 0,
                "status": "error",
                "message": "Falta API Key",
            },
        )
        return jsonify({"error": "No API key configured"}), 400

    publish_progress(
        job_id,
        {
            "operation": "weights",
            "job_id": job_id,
            "phase": "weights",
            "percent": 20,
            "imported": len(samples),
            "enriched": 0,
            "failed": 0,
            "queued": 0,
            "message": "Analizando muestras",
        },
    )
    try:
        result = gpt.recommend_winner_weights(api_key, model, samples, target)
    except Exception as exc:
        publish_progress(
            job_id,
            {
                "operation": "weights",
                "job_id": job_id,
                "phase": "weights",
                "percent": 100,
                "imported": len(samples),
                "enriched": 0,
                "failed": 1,
                "queued": 0,
                "status": "error",
                "message": f"Error IA: {exc}",
            },
        )
        return jsonify({"error": str(exc)}), 502

    publish_progress(
        job_id,
        {
            "operation": "weights",
            "job_id": job_id,
            "phase": "weights",
            "percent": 60,
            "imported": len(samples),
            "enriched": 0,
            "failed": 0,
            "queued": 0,
            "message": "IA completada",
        },
    )

    weights_raw = result.get("weights", {}) or {}
    notes = result.get("justification", "")
    allowed = list(winner_calc.ALLOWED_FIELDS)
    final_weights: Dict[str, int] = {}
    for key in allowed:
        value = weights_raw.get(key, 0)
        try:
            value = float(value)
        except Exception:
            value = 0.0
        final_weights[key] = int(round(max(0.0, min(100.0, value))))

    prev_settings = winner_calc.load_settings()
    prev_order = prev_settings.get("weights_order") or list(allowed)
    order = sorted(
        final_weights.keys(),
        key=lambda k: (-final_weights[k], prev_order.index(k) if k in prev_order else 999),
    )

    publish_progress(
        job_id,
        {
            "operation": "weights",
            "job_id": job_id,
            "phase": "weights",
            "percent": 90,
            "imported": len(samples),
            "enriched": 0,
            "failed": 0,
            "queued": 0,
            "message": "Guardando pesos",
        },
    )

    resp = {
        "weights": final_weights,
        "weights_order": order,
        "order": order,
        "method": "gpt",
        "diagnostics": {"notes": notes},
        "job_id": job_id,
    }

    publish_progress(
        job_id,
        {
            "operation": "weights",
            "job_id": job_id,
            "phase": "weights",
            "percent": 100,
            "imported": len(samples),
            "enriched": 0,
            "failed": 0,
            "queued": 0,
            "message": "Pesos IA completados",
        },
    )
    return jsonify(resp)


@legacy_bp.post("/scoring/v2/auto-weights-stat")
def legacy_auto_weights_stat() -> Response:
    _log_legacy("/scoring/v2/auto-weights-stat")
    payload = request.get_json(silent=True) or {}
    features = [f for f in (payload.get("features") or web_app.WINNER_SCORE_FIELDS) if f in winner_calc.ALLOWED_FIELDS]
    samples_in = payload.get("data_sample") or []
    target = payload.get("target") or ""
    job_id = f"weights-{int(time.time() * 1000)}"
    if not samples_in or not target or len(samples_in) < 2:
        publish_progress(
            job_id,
            {
                "operation": "weights",
                "job_id": job_id,
                "phase": "weights",
                "percent": 100,
                "imported": 0,
                "enriched": 0,
                "failed": 0,
                "queued": 0,
                "status": "error",
                "message": "Datos insuficientes",
            },
        )
        return jsonify({"error": "Datos insuficientes"}), 400

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

    publish_progress(
        job_id,
        {
            "operation": "weights",
            "job_id": job_id,
            "phase": "weights",
            "percent": 40,
            "imported": len(samples_in),
            "enriched": 0,
            "failed": 0,
            "queued": 0,
            "message": "Calculando correlaciones",
        },
    )

    weights01 = {k: float(weights.get(k, 0.0)) for k in features}
    maxv = max(weights01.values() or [0.0])
    weights_raw = {k: (v / maxv * 100.0 if maxv > 0 else 50.0) for k, v in weights01.items()}

    allowed = list(winner_calc.ALLOWED_FIELDS)
    final_weights: Dict[str, int] = {}
    for key in allowed:
        value = weights_raw.get(key, 0.0)
        final_weights[key] = int(round(max(0.0, min(100.0, float(value)))))

    prev_settings = winner_calc.load_settings()
    prev_order = prev_settings.get("weights_order") or list(allowed)
    order = sorted(
        final_weights.keys(),
        key=lambda k: (-final_weights[k], prev_order.index(k) if k in prev_order else 999),
    )

    publish_progress(
        job_id,
        {
            "operation": "weights",
            "job_id": job_id,
            "phase": "weights",
            "percent": 85,
            "imported": len(samples_in),
            "enriched": 0,
            "failed": 0,
            "queued": 0,
            "message": "Guardando pesos",
        },
    )

    resp = {
        "weights": final_weights,
        "weights_order": order,
        "order": order,
        "method": "stat",
        "diagnostics": {"n": len(samples_in)},
        "job_id": job_id,
    }

    publish_progress(
        job_id,
        {
            "operation": "weights",
            "job_id": job_id,
            "phase": "weights",
            "percent": 100,
            "imported": len(samples_in),
            "enriched": 0,
            "failed": 0,
            "queued": 0,
            "message": "Pesos estadísticos completados",
        },
    )
    return jsonify(resp)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except Exception:
            continue
    return None


@legacy_bp.get("/api/trends/summary")
def legacy_trends_summary() -> Response:
    _log_legacy("/api/trends/summary")
    params = request.args
    qs_from = params.get("from", "")
    qs_to = params.get("to", "")
    filters_raw = params.get("filters")

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
    if filters_raw:
        try:
            filters = json.loads(filters_raw)
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
    return jsonify(resp)


@legacy_bp.get("/export")
def legacy_export() -> Response:
    _log_legacy("/export")
    fmt = request.args.get("format", "csv")
    ids_param = request.args.get("ids")
    conn = _ensure_conn()
    items: List[Any]
    if ids_param:
        try:
            ids = [int(x) for x in ids_param.split(",") if x]
        except Exception:
            ids = []
        items = [database.get_product(conn, pid) for pid in ids if database.get_product(conn, pid)]
    else:
        items = list(database.list_products(conn))

    rows: List[List[Any]] = []
    for p in items:
        prod = row_to_dict(p)
        scores = database.get_scores_for_product(conn, prod["id"])
        score_val = None
        if scores:
            sc = scores[0]
            if "winner_score" in sc.keys():
                score_val = sc["winner_score"]
        rows.append(
            [
                prod["id"],
                prod.get("name"),
                score_val,
                prod.get("desire"),
                prod.get("desire_magnitude"),
                prod.get("awareness_level"),
                prod.get("competition_level"),
                prod.get("date_range"),
            ]
        )

    headers = [
        "id",
        "name",
        "Winner Score",
        "Desire",
        "Desire Magnitude",
        "Awareness Level",
        "Competition Level",
        "Date Range",
    ]

    if fmt == "xlsx":
        try:
            from openpyxl import Workbook
        except Exception:
            return jsonify({"error": "openpyxl not installed"}), 500
        wb = Workbook()
        ws = wb.active
        ws.append(headers)
        for row in rows:
            ws.append(row)
        bio = io.BytesIO()
        wb.save(bio)
        bio.seek(0)
        return send_file(
            bio,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name="export.xlsx",
        )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows(rows)
    csv_data = output.getvalue()
    return Response(
        csv_data,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=export.csv"},
    )


def _handle_upload_xlsx(storage, filename: str) -> Tuple[str, Dict[str, Any]]:
    tmp_dir = Path(web_app.APP_DIR) / "uploads"
    tmp_dir.mkdir(exist_ok=True)
    ext = Path(filename).suffix
    tmp_path = tmp_dir / f"import_{int(time.time()*1000)}{ext}"
    storage.save(tmp_path)
    conn = _ensure_conn()
    job_id = database.create_import_job(conn, str(tmp_path))
    threading.Thread(
        target=web_app._process_import_job,
        args=(job_id, tmp_path, filename),
        daemon=True,
    ).start()
    return "async", {"task_id": job_id}


def _handle_upload_csv(data: bytes, filename: str) -> Tuple[str, Dict[str, Any]]:
    conn = _ensure_conn()
    job_config = {"filename": filename, "batch_size": DEFAULT_BATCH_SIZE}
    job_id = database.create_import_job(
        conn,
        status="running",
        phase="parse",
        total=0,
        processed=0,
        config=job_config,
    )
    task_id = str(job_id)
    web_app._update_import_status(
        task_id,
        job_id=job_id,
        state="queued",
        stage="queued",
        done=0,
        total=0,
        error=None,
        imported=0,
        filename=filename,
    )
    web_app._set_import_progress(task_id, pct=0, message="En cola", state="queued")

    def run_csv():
        web_app._update_import_status(
            task_id,
            job_id=job_id,
            state="running",
            stage="running",
            started_at=time.time(),
        )
        web_app._set_import_progress(task_id, pct=5, message="Preparando importación")

        try:
            def cb(**kwargs):
                stage = kwargs.get("stage")
                done = int(kwargs.get("done", 0) or 0)
                total = int(kwargs.get("total", 0) or 0)
                extra = {k: v for k, v in kwargs.items() if k not in {"stage", "done", "total"}}
                if stage == "prepare":
                    web_app._set_import_progress(
                        task_id,
                        pct=8,
                        message="Analizando archivo",
                        done=done,
                        total=total,
                        **extra,
                    )
                elif stage == "insert":
                    frac = done / max(total, 1) if total else 0.0
                    pct = 20 + min(60, int(round(60 * frac)))
                    msg = f"Insertando registros ({done}/{total})" if total else "Insertando registros"
                    web_app._set_import_progress(
                        task_id,
                        pct=pct,
                        message=msg,
                        done=done,
                        total=total,
                        **extra,
                    )
                elif stage == "commit":
                    web_app._set_import_progress(
                        task_id,
                        pct=90,
                        message="Guardando cambios",
                        done=done,
                        total=total,
                        **extra,
                    )
                else:
                    web_app._update_import_status(task_id, **kwargs)

            imported_count = fast_import(
                data,
                job_id=job_id,
                status_cb=cb,
                source=filename,
            )
            job_row = database.get_import_job(conn, job_id)
            snapshot = web_app._job_payload_from_row(job_row) or {}
            done_val = int(snapshot.get("processed") or imported_count or 0)
            total_val = int(snapshot.get("total") or done_val)
            imported_val = int(snapshot.get("rows_imported") or imported_count or done_val)
            web_app._set_import_progress(
                task_id,
                pct=100,
                message="Completado",
                done=done_val,
                total=total_val,
                imported=imported_val,
                finished_at=time.time(),
            )
        except Exception as exc:
            current_app.logger.exception("Fast CSV import failed: filename=%s", filename)
            web_app._update_import_status(
                task_id,
                job_id=job_id,
                state="error",
                stage="error",
                error=str(exc),
                finished_at=time.time(),
                pct=100,
                message=f"Error: {exc}",
            )

    threading.Thread(target=run_csv, daemon=True).start()
    return "async", {"task_id": task_id, "job_id": job_id}


def _handle_upload_json(data: bytes, filename: str) -> Tuple[str, Dict[str, Any]]:
    try:
        payload = json.loads(data.decode("utf-8", errors="ignore"))
    except Exception as exc:
        raise ValueError(str(exc))
    if not isinstance(payload, list):
        raise ValueError("invalid_json")
    records = [item for item in payload if isinstance(item, dict)]
    conn = _ensure_conn()
    total_records = len(records)
    job_config = {
        "filename": filename,
        "batch_size": DEFAULT_BATCH_SIZE,
        "expected": total_records,
    }
    job_id = database.create_import_job(
        conn,
        status="running",
        phase="parse",
        total=total_records,
        processed=0,
        config=job_config,
    )
    task_id = str(job_id)
    web_app._update_import_status(
        task_id,
        job_id=job_id,
        state="queued",
        stage="queued",
        done=0,
        total=total_records,
        error=None,
        imported=0,
        filename=filename,
    )
    web_app._set_import_progress(task_id, pct=0, message="En cola", state="queued")

    def run_json():
        web_app._update_import_status(
            task_id,
            job_id=job_id,
            state="running",
            stage="running",
            started_at=time.time(),
        )
        web_app._set_import_progress(
            task_id,
            pct=5,
            message="Preparando importación",
            total=total_records,
        )
        try:
            def cb(**kwargs):
                stage = kwargs.get("stage")
                done = int(kwargs.get("done", 0) or 0)
                total = int(kwargs.get("total", total_records) or total_records)
                extra = {k: v for k, v in kwargs.items() if k not in {"stage", "done", "total"}}
                if stage == "prepare":
                    web_app._set_import_progress(
                        task_id,
                        pct=8,
                        message="Analizando archivo",
                        done=done,
                        total=total,
                        **extra,
                    )
                elif stage == "insert":
                    frac = done / max(total, 1) if total else 0.0
                    pct = 20 + min(60, 60 * frac)
                    msg = f"Insertando registros ({done}/{total})" if total else "Insertando registros"
                    web_app._set_import_progress(
                        task_id,
                        pct=pct,
                        message=msg,
                        done=done,
                        total=total,
                        **extra,
                    )
                elif stage == "commit":
                    web_app._set_import_progress(
                        task_id,
                        pct=82,
                        message="Guardando cambios",
                        done=done,
                        total=total,
                        **extra,
                    )
                else:
                    web_app._update_import_status(task_id, **kwargs)

            imported_count = fast_import_records(
                records,
                job_id=job_id,
                status_cb=cb,
                source=filename,
            )
            job_row = database.get_import_job(conn, job_id)
            snapshot = web_app._job_payload_from_row(job_row) or {}
            done_val = int(snapshot.get("processed") or imported_count or total_records)
            total_val = int(snapshot.get("total") or total_records)
            imported_val = int(snapshot.get("rows_imported") or imported_count or done_val)
            web_app._set_import_progress(
                task_id,
                pct=95,
                message="Finalizando importación",
                done=done_val,
                total=total_val,
                imported=imported_val,
                finished_at=time.time(),
            )
        except Exception as exc:
            current_app.logger.exception("JSON import failed: filename=%s", filename)
            web_app._update_import_status(
                task_id,
                job_id=job_id,
                state="error",
                stage="error",
                error=str(exc),
                finished_at=time.time(),
                pct=100,
                message=f"Error: {exc}",
            )

    threading.Thread(target=run_json, daemon=True).start()
    return "async", {"task_id": task_id, "job_id": job_id}


@legacy_bp.post("/upload")
def legacy_upload() -> Response:
    _log_legacy("/upload")
    storage = request.files.get("file")
    if not storage or storage.filename == "":
        return jsonify({"error": "No file provided"}), 400
    filename = Path(storage.filename).name
    ext = Path(filename).suffix.lower()
    try:
        if ext in {".xlsx", ".xls"}:
            kind, result = _handle_upload_xlsx(storage, filename)
        elif ext == ".csv":
            data = storage.read()
            kind, result = _handle_upload_csv(data, filename)
        elif ext == ".json":
            data = storage.read()
            kind, result = _handle_upload_json(data, filename)
        else:
            return jsonify({"error": "Unsupported file type"}), 400
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    status_code = 200 if kind == "sync" else 202
    return jsonify(result), status_code


@legacy_bp.get("/_import_status")
@legacy_bp.get("/import/status")
def legacy_import_status() -> Response:
    path = request.path
    _log_legacy(path)
    params = request.args
    if path == "/import/status":
        target = params.get("job_id") or params.get("task_id") or ""
    else:
        target = params.get("task_id") or params.get("job_id") or ""
    if not target:
        return jsonify({"state": "unknown"})
    status = web_app._get_import_status(str(target))
    if status is None and str(target).isdigit():
        conn = _ensure_conn()
        row = database.get_import_job(conn, int(target))
        status = web_app._job_payload_from_row(row)
    if status:
        status.setdefault("task_id", str(status.get("task_id") or target))
        if str(target).isdigit():
            status.setdefault("job_id", int(target))
        return jsonify(status)
    return jsonify({"state": "unknown"})


@legacy_bp.post("/shutdown")
def legacy_shutdown() -> Response:
    _log_legacy("/shutdown")
    return jsonify({"ok": True})


@legacy_bp.get("/events/poll")
def legacy_events_poll() -> Response:
    _log_legacy("/events/poll")
    subscriber = subscribe()
    events: List[Dict[str, Any]] = []
    try:
        try:
            event = subscriber.queue.get(timeout=0.5)
            events.append(event)
        except queue.Empty:
            pass
        while True:
            try:
                events.append(subscriber.queue.get_nowait())
            except queue.Empty:
                break
    finally:
        unsubscribe(subscriber)
    return jsonify({"events": events})


@legacy_bp.post("/api/winner-score/generate")
def legacy_winner_score_generate() -> Response:
    _log_legacy("/api/winner-score/generate")
    debug = request.args.get("debug") == "1"
    payload = request.get_json(silent=True) or {}
    ids = payload.get("product_ids") or payload.get("ids") or []
    if ids and not isinstance(ids, list):
        return jsonify({"error": "Invalid JSON"}), 400
    conn = _ensure_conn()
    result = generate_winner_scores(conn, product_ids=ids or None, debug=debug)
    resp = {
        "ok": True,
        "processed": result.get("processed", 0),
        "updated": result.get("updated", 0),
        "weights_all": result.get("weights_all"),
        "weights_eff": result.get("weights_eff"),
    }
    if debug:
        resp["diag"] = result.get("diag", {})
    return jsonify(resp)


@legacy_bp.get("/settings/winner-score")
def legacy_winner_score_settings() -> Response:
    _log_legacy("/settings/winner-score")
    cfg = config.load_config()
    return jsonify(cfg.get("weights", {}))


__all__ = ["legacy_bp"]

