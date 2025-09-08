import json
import logging
import os
import asyncio
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import config, database, gpt

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent.parent
DB_PATH = APP_DIR / "data.sqlite3"


def _ensure_conn():
    conn = database.get_connection(DB_PATH)
    database.initialize_database(conn)
    return conn


async def _call_with_retries(api_key: str, model: str, items: List[Dict[str, Any]], max_retries: int) -> tuple[Dict[str, Any], Dict[str, str], int]:
    retry = 0
    base = 0.5
    while True:
        try:
            ok, ko, usage, duration = await asyncio.to_thread(gpt.generate_batch_columns, api_key, model, items)
            return ok, ko, retry
        except gpt.OpenAIError as exc:
            msg = str(exc)
            status = 0
            for tok in msg.split():
                if tok.isdigit():
                    status = int(tok)
                    break
            if status in (429, 500, 502, 503, 504) and retry < max_retries:
                delay = base * (2**retry) + random.uniform(0.1, 0.4)
                retry += 1
                await asyncio.sleep(delay)
                continue
            return {}, {str(it["id"]): msg for it in items}, retry


def fill_ai_columns(
    product_ids: List[int],
    *,
    model: str | None = None,
    batch_mode: bool | None = None,
    cost_cap_usd: float | None = None,
) -> Dict[str, Any]:
    start_time = time.time()
    conn = _ensure_conn()

    cfg_batch = config.get_ai_batch_config()
    cfg_cost = config.get_ai_cost_config()
    api_key = config.get_api_key() or os.environ.get("OPENAI_API_KEY")
    model = model or cfg_cost.get("model")
    batch_mode = batch_mode if batch_mode is not None else (len(product_ids) >= cfg_cost.get("useBatchWhenCountGte", 300))
    cost_cap_usd = cost_cap_usd if cost_cap_usd is not None else cfg_cost.get("costCapUSD")

    total_requested = len(product_ids)
    skipped_existing = 0
    to_process: List[Dict[str, Any]] = []
    selected_ids: List[int] = []
    records: Dict[str, Any] = {}
    now_ts = datetime.utcnow().isoformat()

    for pid in product_ids:
        rec = database.get_product(conn, pid)
        if not rec:
            continue
        if rec["ai_columns_completed_at"]:
            skipped_existing += 1
            continue
        if rec["desire"] or rec["desire_magnitude"] or rec["awareness_level"] or rec["competition_level"]:
            skipped_existing += 1
            database.update_product(conn, pid, ai_columns_completed_at=now_ts)
            continue
        try:
            extra = json.loads(rec["extra"]) if rec["extra"] else {}
        except Exception:
            extra = {}
        selected_ids.append(rec["id"])
        item = {
            "id": rec["id"],
            "name": rec["name"],
            "category": rec["category"],
            "price": rec["price"],
            "rating": extra.get("rating"),
            "units_sold": extra.get("units_sold"),
            "revenue": extra.get("revenue"),
            "conversion_rate": extra.get("conversion_rate"),
            "launch_date": extra.get("launch_date"),
            "date_range": rec["date_range"],
            "image_url": rec["image_url"],
        }
        to_process.append(item)
        records[str(rec["id"])] = rec

    count = len(to_process)
    est_in = count * cfg_cost.get("estTokensPerItemIn", 0)
    est_out = count * cfg_cost.get("estTokensPerItemOut", 0)
    price_map = cfg_cost.get("prices", {}).get(model, {})
    price_in = price_map.get("input", 0.15)
    price_out = price_map.get("output", 0.6)
    cost_estimated = (est_in / 1_000_000) * price_in + (est_out / 1_000_000) * price_out
    truncated = False
    pending_ids: List[int] = []
    if cost_cap_usd is not None and cost_estimated > cost_cap_usd:
        per_item_cost = ((cfg_cost.get("estTokensPerItemIn", 0) / 1_000_000) * price_in + (cfg_cost.get("estTokensPerItemOut", 0) / 1_000_000) * price_out)
        max_items = int(cost_cap_usd // per_item_cost) if per_item_cost > 0 else 0
        to_process = to_process[:max_items]
        records = {str(it["id"]): records[str(it["id"])] for it in to_process}
        pending_ids.extend(selected_ids[max_items:])
        count = len(to_process)
        est_in = count * cfg_cost.get("estTokensPerItemIn", 0)
        est_out = count * cfg_cost.get("estTokensPerItemOut", 0)
        cost_estimated = (est_in / 1_000_000) * price_in + (est_out / 1_000_000) * price_out
        truncated = True

    if not api_key or not to_process:
        err_msg = "missing_api_key" if not api_key else None
        logger.info(
            "fill_ai_columns: n_importados=%s n_para_ia=0 n_procesados=0 n_omitidos_por_valor_existente=%s n_reintentados=0 n_error_definitivo=%s truncated=%s cost_estimated_usd=%.4f",
            total_requested,
            skipped_existing,
            len(to_process),
            truncated,
            cost_estimated,
        )
        return {
            "ok": {},
            "ko": {str(it["id"]): err_msg or "skipped" for it in to_process} if err_msg else {},
            "error": err_msg,
            "counts": {
                "n_importados": total_requested,
                "n_para_ia": 0,
                "n_procesados": 0,
                "n_omitidos_por_valor_existente": skipped_existing,
                "n_reintentados": 0,
                "n_error_definitivo": len(to_process) if err_msg else 0,
                "truncated": truncated,
                "cost_estimated_usd": cost_estimated,
            },
            "pending_ids": selected_ids,
        }

    batches = [to_process[i : i + cfg_batch.get("BATCH_SIZE", 10)] for i in range(0, len(to_process), cfg_batch.get("BATCH_SIZE", 10))]

    ok_all: Dict[str, Dict[str, Any]] = {}
    ko_all: Dict[str, str] = {}
    success = 0
    errors = 0
    n_retried = 0

    async def run_batches() -> None:
        nonlocal success, errors, n_retried
        sem = asyncio.Semaphore(cfg_batch.get("MAX_CONCURRENCY", 2))

        async def worker(batch: List[Dict[str, Any]]):
            async with sem:
                ok, ko, retries = await _call_with_retries(api_key, model, batch, cfg_batch.get("MAX_RETRIES", 3))
                return batch, ok, ko, retries

        tasks = [asyncio.create_task(worker(b)) for b in batches]
        done = 0
        for coro in asyncio.as_completed(tasks):
            if time.time() - start_time > cfg_batch.get("TIME_LIMIT_SECONDS", 300):
                for t in tasks:
                    t.cancel()
                break
            try:
                batch, ok, ko, retries = await coro
            except Exception:
                continue
            n_retried += retries
            for pid, updates in ok.items():
                rec = records.get(str(pid))
                if not rec:
                    ko_all[str(pid)] = "not_found"
                    errors += 1
                    continue
                apply: Dict[str, Any] = {}
                if not rec["desire"] and updates.get("desire"):
                    apply["desire"] = updates.get("desire")
                if not rec["desire_magnitude"] and updates.get("desire_magnitude"):
                    apply["desire_magnitude"] = updates.get("desire_magnitude")
                if not rec["awareness_level"] and updates.get("awareness_level"):
                    apply["awareness_level"] = updates.get("awareness_level")
                if not rec["competition_level"] and updates.get("competition_level"):
                    apply["competition_level"] = updates.get("competition_level")
                if apply:
                    apply["ai_columns_completed_at"] = datetime.utcnow().isoformat()
                    database.update_product(conn, int(pid), **apply)
                    ok_all[str(pid)] = {k: v for k, v in apply.items() if k != "ai_columns_completed_at"}
                    success += 1
                else:
                    database.update_product(conn, int(pid), ai_columns_completed_at=datetime.utcnow().isoformat())
                    ko_all[str(pid)] = "existing"
                    errors += 1
            for pid, reason in ko.items():
                ko_all[str(pid)] = reason
                errors += 1
            done += len(batch)

    asyncio.run(run_batches())

    conn.commit()
    logger.info(
        "fill_ai_columns: n_importados=%s n_para_ia=%s n_procesados=%s n_omitidos_por_valor_existente=%s n_reintentados=%s n_error_definitivo=%s truncated=%s cost_estimated_usd=%.4f",
        total_requested,
        len(to_process),
        success,
        skipped_existing,
        n_retried,
        errors,
        truncated,
        cost_estimated,
    )
    processed_ids = {int(pid) for pid in ok_all.keys()} | {int(pid) for pid in ko_all.keys()}
    pending_ids.extend([it["id"] for it in to_process if it["id"] not in processed_ids])
    return {
        "ok": ok_all,
        "ko": ko_all,
        "counts": {
            "n_importados": total_requested,
            "n_para_ia": len(to_process),
            "n_procesados": success,
            "n_omitidos_por_valor_existente": skipped_existing,
            "n_reintentados": n_retried,
            "n_error_definitivo": errors,
            "truncated": truncated,
            "cost_estimated_usd": cost_estimated,
        },
        "pending_ids": pending_ids,
    }
