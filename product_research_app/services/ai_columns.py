from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx
from .. import config, database
from ..gpt import call_gpt_json
from ..sse import publish_event
from ..utils.signature import compute_sig_hash

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent.parent
DB_PATH = APP_DIR / "data.sqlite3"

AI_FIELDS = ["desire", "desire_magnitude", "awareness_level", "competition_level"]

BATCH_SIZE = 32
PARALLELISM = 3
MAX_RETRIES = 3
MODEL_NAME = "gpt-4o-mini"
SYSTEM_PROMPT = "Eres un analista. Respondes UN JSON válido."


def _ensure_conn():
    conn = database.get_connection(DB_PATH)
    database.initialize_database(conn)
    return conn


def _apply_ai_updates(conn, updates: Dict[int, Dict[str, Any]]) -> None:
    if not updates:
        return
    now_iso = datetime.utcnow().isoformat()
    cur = conn.cursor()
    began = False
    try:
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
            began = True
        for pid, payload in updates.items():
            cols: List[str] = []
            vals: List[Any] = []
            for field in AI_FIELDS:
                if field in payload and payload[field] is not None:
                    cols.append(f"{field}=?")
                    vals.append(payload[field])
            cols.append("ai_columns_completed_at=?")
            vals.append(now_iso)
            vals.append(int(pid))
            cur.execute(
                f"UPDATE products SET {', '.join(cols)} WHERE id=?",
                vals,
            )
        if began:
            conn.commit()
    except Exception:
        if began and conn.in_transaction:
            conn.rollback()
        raise


def broadcast_products_patch(conn, rows: List[Dict[str, Any]]) -> None:
    del conn  # compatibility placeholder
    if not rows:
        return
    try:
        publish_event("products.patch", {"rows": rows})
    except Exception:  # pragma: no cover - best effort broadcast
        logger.debug("Failed to publish products.patch SSE", exc_info=True)


def _emit_ai_updates(updates: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not updates:
        return []
    rows: List[Dict[str, Any]] = []
    for pid, payload in updates.items():
        if not isinstance(payload, dict):
            continue
        row_data: Dict[str, Any] = {"id": pid}
        has_value = False
        for field in AI_FIELDS:
            if field in payload:
                row_data[field] = payload[field]
                has_value = True
        if has_value:
            rows.append(row_data)
    if not rows:
        return []
    try:
        publish_event(
            "ai-columns",
            {
                "columns": AI_FIELDS,
                "rows": rows,
            },
        )
    except Exception:  # pragma: no cover - best effort broadcast
        logger.debug("Failed to publish ai-columns SSE", exc_info=True)
    try:
        broadcast_products_patch(None, rows)
    except Exception:  # pragma: no cover - best effort broadcast
        logger.debug("Failed to broadcast products.patch SSE", exc_info=True)
    return rows


def _deserialize_extra(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            logger.debug("Failed to parse extra JSON", exc_info=True)
    return {}


def _rows_to_payloads(rows: Sequence[dict]) -> List[dict]:
    payloads: List[dict] = []
    for row in rows:
        extra = row.get("extra") or {}
        payloads.append(
            {
                "id": row["id"],
                "name": row.get("name") or extra.get("title"),
                "category": row.get("category") or extra.get("category"),
                "price": row.get("price"),
                "rating": extra.get("rating"),
                "units_sold": extra.get("units_sold"),
                "revenue": extra.get("revenue"),
                "date_range": row.get("date_range"),
                "description": row.get("description")
                or extra.get("body")
                or extra.get("long_description"),
            }
        )
    return payloads


def _chunked(seq: Sequence[int], size: int):
    chunk_size = max(1, int(size or 1))
    for i in range(0, len(seq), chunk_size):
        yield list(seq[i : i + chunk_size])


def _build_prompt(chunk_rows: List[Dict[str, Any]]) -> str:
    schema_example = {
        "42": {
            "desire": "frase clara (280-420 chars) del deseo humano",
            "desire_magnitude": "Low|Medium|High",
            "awareness_level": "Unaware|Problem-Aware|Solution-Aware|Product-Aware|Most Aware",
            "competition_level": "Low|Medium|High",
        }
    }
    data_payload = {str(row["id"]): row for row in chunk_rows}
    return (
        "Para cada item, genera desire/labels. "
        "Salida = UN objeto JSON: {id: {...}}. "
        f"Ejemplo de propiedades: {json.dumps(schema_example, ensure_ascii=False)}\n"
        f"DATA: {json.dumps(data_payload, ensure_ascii=False)}"
    )


def _merge_cache(
    conn,
    model: str,
    version: int,
    sig_map: Dict[int, str],
) -> Dict[int, Dict[str, Any]]:
    sigs = [sig for sig in sig_map.values() if sig]
    cache = database.get_ai_cache_entries(conn, sigs, model=model, version=version)
    ready: Dict[int, Dict[str, Any]] = {}
    for pid, sig in sig_map.items():
        if not sig:
            continue
        row = cache.get(sig)
        if not row:
            continue
        ready[pid] = {
            "desire": row["desire"],
            "desire_magnitude": row["desire_magnitude"],
            "awareness_level": row["awareness_level"],
            "competition_level": row["competition_level"],
        }
    return ready


def run_ai_fill_job(
    job_id: Optional[int],
    product_ids: Sequence[int],
    *,
    microbatch: int = BATCH_SIZE,
    parallelism: int = PARALLELISM,
    cost_cap_usd: Optional[float] = None,
    status_cb=None,
    apply_updates_flag: bool = True,
) -> Dict[str, Any]:
    del job_id, parallelism, cost_cap_usd  # compatibility placeholders
    conn = _ensure_conn()
    try:
        seen: set[int] = set()
        normalized: List[int] = []
        for pid in product_ids:
            try:
                num = int(pid)
            except Exception:
                continue
            if num in seen:
                continue
            seen.add(num)
            normalized.append(num)

        if not normalized:
            return {
                "ok": {},
                "ko": {},
                "counts": {
                    "queued": 0,
                    "ok": 0,
                    "cached": 0,
                    "ko": 0,
                    "retried": 0,
                    "cost_spent_usd": 0.0,
                },
                "pending_ids": [],
                "error": None,
                "total_requested": 0,
                "inspected": 0,
            }

        placeholders = ",".join(["?"] * len(normalized))
        cur = conn.cursor()
        cur.execute(
            f"SELECT * FROM products WHERE id IN ({placeholders}) ORDER BY id",
            normalized,
        )
        rows = [dict(row) for row in cur.fetchall()]
        row_map = {int(row["id"]): row for row in rows}

        ok: Dict[int, Dict[str, Any]] = {}
        ko: Dict[int, str] = {}

        for pid in normalized:
            if pid not in row_map:
                ko[pid] = "missing"

        prepared_rows: List[dict] = []
        sig_updates: List[tuple[str, int]] = []
        sig_map: Dict[int, str] = {}
        for row in rows:
            pid = int(row["id"])
            extra = _deserialize_extra(row.get("extra"))
            row["extra"] = extra
            name = row.get("name") or extra.get("title")
            if not name:
                ko[pid] = "missing_name"
                continue
            sig = row.get("sig_hash")
            if not sig:
                sig = compute_sig_hash(
                    str(name),
                    extra.get("brand"),
                    extra.get("asin"),
                    extra.get("product_url"),
                )
                if sig:
                    sig_updates.append((sig, pid))
            if sig:
                sig_map[pid] = sig
            prepared_rows.append(row)

        if sig_updates:
            cur.executemany(
                "UPDATE OR IGNORE products SET sig_hash=? WHERE id=?",
                sig_updates,
            )
            if conn.in_transaction:
                conn.commit()

        model = config.get_model() or MODEL_NAME
        cache_version = 1
        cached_updates = _merge_cache(conn, model, cache_version, sig_map)
        pending_apply: Dict[int, Dict[str, Any]] = {}
        if cached_updates:
            if apply_updates_flag:
                _apply_ai_updates(conn, cached_updates)
                _emit_ai_updates(cached_updates)
            else:
                pending_apply.update({int(pid): data for pid, data in cached_updates.items()})

        counts = {
            "queued": len(normalized),
            "ok": 0,
            "cached": len(cached_updates),
            "ko": len(ko),
            "retried": 0,
            "cost_spent_usd": 0.0,
        }

        if status_cb:
            try:
                status_cb({"ok": counts["ok"], "ko": counts["ko"], "cached": counts["cached"]})
            except Exception:
                logger.debug("status callback failed", exc_info=True)

        pending_rows = [row for row in prepared_rows if int(row["id"]) not in cached_updates and int(row["id"]) not in ko]
        api_key = config.get_api_key() or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            for row in pending_rows:
                ko[int(row["id"])] = "missing_api_key"
            counts["ko"] = len(ko)
            pending_ids = [pid for pid in normalized if pid not in cached_updates and pid not in ok]
            return {
                "ok": ok,
                "ko": ko,
                "counts": counts,
                "pending_ids": pending_ids,
                "error": "missing_api_key",
                "total_requested": len(normalized),
                "inspected": len(normalized),
            }

        for start in range(0, len(pending_rows), max(1, microbatch)):
            chunk_rows = pending_rows[start : start + max(1, microbatch)]
            if not chunk_rows:
                continue
            payloads = _rows_to_payloads(chunk_rows)
            prompt = _build_prompt(payloads)
            response = None
            attempt = 1
            while attempt <= MAX_RETRIES:
                try:
                    response_payload = call_gpt_json(
                        api_key=api_key,
                        model=model or MODEL_NAME,
                        system=SYSTEM_PROMPT,
                        user=prompt,
                        max_tokens=2500,
                        temperature=0.2,
                    )
                    if not isinstance(response_payload, dict):
                        raise ValueError("json:expected_object")
                    response = response_payload
                    if attempt > 1:
                        counts["retried"] += attempt - 1
                    break
                except httpx.TimeoutException:
                    if attempt >= MAX_RETRIES:
                        for row in chunk_rows:
                            ko[int(row["id"])] = "timeout"
                        counts["ko"] = len(ko)
                        response = None
                        break
                    sleep_for = min(2 ** (attempt - 1), 8)
                    time.sleep(sleep_for)
                    attempt += 1
                except ValueError:
                    if attempt >= MAX_RETRIES:
                        for row in chunk_rows:
                            ko[int(row["id"])] = "invalid_json"
                        counts["ko"] = len(ko)
                        response = None
                        break
                    sleep_for = min(2 ** (attempt - 1), 8)
                    time.sleep(sleep_for)
                    attempt += 1
                except Exception:
                    logger.exception("Batch IA falló")
                    for row in chunk_rows:
                        ko[int(row["id"])] = "error"
                    counts["ko"] = len(ko)
                    response = None
                    break
            if response is None:
                continue

            updates: Dict[int, Dict[str, Any]] = {}
            for row in chunk_rows:
                pid = int(row["id"])
                payload = response.get(str(pid)) or response.get(pid)
                if not isinstance(payload, dict):
                    ko[pid] = "missing"
                    continue
                updates[pid] = {
                    "desire": payload.get("desire"),
                    "desire_magnitude": payload.get("desire_magnitude"),
                    "awareness_level": payload.get("awareness_level"),
                    "competition_level": payload.get("competition_level"),
                }
                sig = sig_map.get(pid)
                if sig:
                    database.upsert_ai_cache_entry(
                        conn,
                        sig,
                        model=model or MODEL_NAME,
                        version=cache_version,
                        desire=updates[pid]["desire"],
                        desire_magnitude=updates[pid]["desire_magnitude"],
                        awareness_level=updates[pid]["awareness_level"],
                        competition_level=updates[pid]["competition_level"],
                    )

            if apply_updates_flag:
                _apply_ai_updates(conn, updates)
                _emit_ai_updates(updates)
            else:
                for pid, payload in updates.items():
                    pending_apply[int(pid)] = payload
            ok.update({pid: data for pid, data in updates.items() if pid not in ko})
            counts["ok"] = len(ok)
            counts["ko"] = len(ko)

            if status_cb:
                try:
                    status_cb({"ok": counts["ok"], "ko": counts["ko"], "cached": counts["cached"]})
                except Exception:
                    logger.debug("status callback failed", exc_info=True)

        pending_ids = [pid for pid in normalized if pid not in ok and pid not in cached_updates]
        result: Dict[str, Any] = {
            "ok": ok,
            "ko": ko,
            "counts": counts,
            "pending_ids": pending_ids,
            "error": None,
            "total_requested": len(normalized),
            "inspected": len(normalized),
        }
        if not apply_updates_flag and pending_apply:
            result["pending_updates"] = pending_apply
        return result
    finally:
        try:
            conn.close()
        except Exception:
            pass


def validate_and_fill_ai_columns(
    *,
    db_conn=None,
    product_ids: Optional[Sequence[int]] = None,
    batch_size: int = 64,
    parallel: int = PARALLELISM,
) -> Dict[str, Any]:
    conn = db_conn or _ensure_conn()
    close_conn = db_conn is None
    try:
        cur = conn.cursor()
        if product_ids:
            placeholders = ",".join(["?"] * len(product_ids))
            cur.execute(
                f"SELECT id, name, desire FROM products WHERE id IN ({placeholders})",
                [int(pid) for pid in product_ids],
            )
        else:
            cur.execute("SELECT id, name, desire FROM products")
        rows = [dict(row) for row in cur.fetchall()]
    finally:
        if close_conn:
            try:
                conn.close()
            except Exception:
                pass

    pending: List[int] = []
    for row in rows:
        pid = int(row["id"])
        desire = row.get("desire")
        if not isinstance(desire, str) or len(desire.strip()) < 5:
            pending.append(pid)

    if not pending:
        return {
            "counts": {},
            "pending_ids": [],
            "error": None,
            "ran_job": False,
            "total_requested": 0,
            "inspected": len(rows),
        }

    result = run_ai_fill_job(
        job_id=None,
        product_ids=pending,
        microbatch=batch_size,
        parallelism=parallel,
    )
    result["ran_job"] = True
    return result


def recalc_desire_for_all(*, batch_size: int = 64, parallel: int = PARALLELISM) -> int:
    conn = _ensure_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM products ORDER BY id")
        ids = [int(row["id"]) for row in cur.fetchall()]
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not ids:
        return 0

    result = run_ai_fill_job(
        job_id=None,
        product_ids=ids,
        microbatch=batch_size,
        parallelism=parallel,
    )
    return result.get("counts", {}).get("ok", 0) + result.get("counts", {}).get("cached", 0)


def apply_updates(conn, result: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    raw_updates = result.get("pending_updates") or result.get("ok")
    updates: Dict[int, Dict[str, Any]] = {}
    if isinstance(raw_updates, dict):
        for pid, payload in raw_updates.items():
            try:
                pid_int = int(pid)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            updates[pid_int] = payload
    if not updates:
        return []
    _apply_ai_updates(conn, updates)
    return _emit_ai_updates(updates)


def audit_and_backfill_all(conn, batch_size: int = 50) -> int:
    try:
        size = int(batch_size)
    except Exception:
        size = 50
    size = max(1, size)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id FROM products
        WHERE COALESCE(ai_desire,'')='' OR COALESCE(desire_magnitude,'')=''
           OR COALESCE(awareness_level,'')='' OR COALESCE(competition_level,'')=''
        ORDER BY id
        """
    )
    missing_ids = [int(row["id"]) for row in cur.fetchall() if row["id"] is not None]
    if not missing_ids:
        return 0
    total_applied = 0
    for chunk in _chunked(missing_ids, size):
        try:
            result = run_ai_fill_job(
                job_id=None,
                product_ids=chunk,
                microbatch=BATCH_SIZE,
                parallelism=PARALLELISM,
                apply_updates_flag=False,
            )
        except Exception:
            logger.exception("Audit batch failed for ids", extra={"ids": chunk})
            continue
        applied_rows = apply_updates(conn, result)
        total_applied += len(applied_rows)
    return total_applied
