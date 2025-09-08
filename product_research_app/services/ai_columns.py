import json
import logging
import os
import asyncio
import random
import time
import math
import re
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


def _parse_score(val: Any) -> Optional[float]:
    """Parse a score which may be a string label or numeric value.

    Returns a float in [0, 1] or ``None`` if parsing fails."""

    if val is None:
        return None
    if isinstance(val, (int, float)):
        num = float(val)
        if 0 <= num <= 1:
            return num
        if 0 <= num <= 100:
            return num / 100.0
        return None
    if isinstance(val, str):
        txt = val.strip().lower()
        m = re.search(r"\d+(?:\.\d+)?", txt)
        if m:
            try:
                num = float(m.group())
                if num > 1:
                    num /= 100.0
                if 0 <= num <= 1:
                    return num
            except Exception:
                pass
        if txt.startswith("low"):
            return 0.2
        if txt.startswith("med"):
            return 0.5
        if txt.startswith("high"):
            return 0.8
    return None


def _quantile(data: List[float], q: float) -> float:
    """Return the q-th quantile of data using linear interpolation."""

    if not data:
        return 0.0
    s = sorted(data)
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[int(pos)]
    return s[lo] * (hi - pos) + s[hi] * (pos - lo)


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

    ok_raw: Dict[str, Dict[str, Any]] = {}
    ko_all: Dict[str, str] = {}
    n_retried = 0

    async def run_batches() -> None:
        nonlocal n_retried
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
                ok_raw[str(pid)] = updates
            for pid, reason in ko.items():
                ko_all[str(pid)] = reason
            done += len(batch)

    asyncio.run(run_batches())

    cfg_calib = config.get_ai_calibration_config()
    dist_desire = {"Low": 0, "Medium": 0, "High": 0}
    dist_comp = {"Low": 0, "Medium": 0, "High": 0}
    promoted_low = 0
    promoted_high = 0

    desire_scores: List[tuple[str, float]] = []
    comp_scores: List[tuple[str, float]] = []
    updates_final: Dict[str, Dict[str, Any]] = {}

    for pid, updates in ok_raw.items():
        rec = records.get(pid)
        if not rec:
            ko_all[pid] = "not_found"
            continue
        apply: Dict[str, Any] = {}
        if not rec["desire"] and updates.get("desire"):
            apply["desire"] = updates.get("desire")
        if not rec["awareness_level"] and updates.get("awareness_level"):
            apply["awareness_level"] = updates.get("awareness_level")
        if cfg_calib.get("enabled", True):
            if not rec["desire_magnitude"] and updates.get("desire_magnitude"):
                score = _parse_score(updates.get("desire_magnitude"))
                if score is not None:
                    updates["_desire_score"] = score
                    desire_scores.append((pid, score))
                else:
                    logger.warning("invalid desire_magnitude for %s: %s", pid, updates.get("desire_magnitude"))
            if not rec["competition_level"] and updates.get("competition_level"):
                score = _parse_score(updates.get("competition_level"))
                if score is not None:
                    updates["_competition_score"] = score
                    comp_scores.append((pid, score))
                else:
                    logger.warning("invalid competition_level for %s: %s", pid, updates.get("competition_level"))
        else:
            if not rec["desire_magnitude"] and updates.get("desire_magnitude"):
                apply["desire_magnitude"] = updates.get("desire_magnitude")
            if not rec["competition_level"] and updates.get("competition_level"):
                apply["competition_level"] = updates.get("competition_level")
        updates_final[pid] = apply

    q_low = cfg_calib.get("quantiles", {}).get("low", 0.20)
    q_high = cfg_calib.get("quantiles", {}).get("high", 0.80)
    wins_p = cfg_calib.get("winsorize_pct", 0.05)
    min_low_pct = cfg_calib.get("min_low_pct", 0.05)
    min_high_pct = cfg_calib.get("min_high_pct", 0.05)
    desire_q = {"Q20": None, "Q80": None}
    comp_q = {"Q20": None, "Q80": None}

    if cfg_calib.get("enabled", True) and desire_scores:
        scores = [s for _, s in desire_scores]
        if len(scores) >= 50 and wins_p > 0:
            low_lim = _quantile(scores, wins_p)
            high_lim = _quantile(scores, 1 - wins_p)
            scores = [min(max(s, low_lim), high_lim) for s in scores]
        desire_q20 = _quantile(scores, q_low)
        desire_q80 = _quantile(scores, q_high)
        if len(scores) < 15 or abs(desire_q80 - desire_q20) < 0.05:
            desire_q20, desire_q80 = 0.34, 0.66
        desire_q["Q20"] = desire_q20
        desire_q["Q80"] = desire_q80
        for pid, score in desire_scores:
            if score <= desire_q20:
                label = "Low"
            elif score >= desire_q80:
                label = "High"
            else:
                label = "Medium"
            updates_final[pid]["desire_magnitude"] = label
            dist_desire[label] += 1
        if dist_desire["Low"] == 0 and len(desire_scores) >= 3:
            need = max(1, math.ceil(len(desire_scores) * min_low_pct))
            for pid, _ in sorted(desire_scores, key=lambda x: x[1])[:need]:
                if updates_final[pid].get("desire_magnitude") != "Low":
                    updates_final[pid]["desire_magnitude"] = "Low"
                    promoted_low += 1
            dist_desire = {"Low": 0, "Medium": 0, "High": 0}
            for pid, _ in desire_scores:
                lab = updates_final[pid].get("desire_magnitude", "Medium")
                dist_desire[lab] += 1
        if dist_desire["High"] == 0 and len(desire_scores) >= 3:
            need = max(1, math.ceil(len(desire_scores) * min_high_pct))
            for pid, _ in sorted(desire_scores, key=lambda x: x[1], reverse=True)[:need]:
                if updates_final[pid].get("desire_magnitude") != "High":
                    updates_final[pid]["desire_magnitude"] = "High"
                    promoted_high += 1
            dist_desire = {"Low": 0, "Medium": 0, "High": 0}
            for pid, _ in desire_scores:
                lab = updates_final[pid].get("desire_magnitude", "Medium")
                dist_desire[lab] += 1

    if cfg_calib.get("enabled", True) and comp_scores:
        scores = [s for _, s in comp_scores]
        if len(scores) >= 50 and wins_p > 0:
            low_lim = _quantile(scores, wins_p)
            high_lim = _quantile(scores, 1 - wins_p)
            scores = [min(max(s, low_lim), high_lim) for s in scores]
        comp_q20 = _quantile(scores, q_low)
        comp_q80 = _quantile(scores, q_high)
        if len(scores) < 15 or abs(comp_q80 - comp_q20) < 0.05:
            comp_q20, comp_q80 = 0.34, 0.66
        comp_q["Q20"] = comp_q20
        comp_q["Q80"] = comp_q80
        for pid, score in comp_scores:
            if score <= comp_q20:
                label = "Low"
            elif score >= comp_q80:
                label = "High"
            else:
                label = "Medium"
            updates_final[pid]["competition_level"] = label
            dist_comp[label] += 1
        if dist_comp["Low"] == 0 and len(comp_scores) >= 3:
            need = max(1, math.ceil(len(comp_scores) * min_low_pct))
            for pid, _ in sorted(comp_scores, key=lambda x: x[1])[:need]:
                if updates_final[pid].get("competition_level") != "Low":
                    updates_final[pid]["competition_level"] = "Low"
                    promoted_low += 1
            dist_comp = {"Low": 0, "Medium": 0, "High": 0}
            for pid, _ in comp_scores:
                lab = updates_final[pid].get("competition_level", "Medium")
                dist_comp[lab] += 1
        if dist_comp["High"] == 0 and len(comp_scores) >= 3:
            need = max(1, math.ceil(len(comp_scores) * min_high_pct))
            for pid, _ in sorted(comp_scores, key=lambda x: x[1], reverse=True)[:need]:
                if updates_final[pid].get("competition_level") != "High":
                    updates_final[pid]["competition_level"] = "High"
                    promoted_high += 1
            dist_comp = {"Low": 0, "Medium": 0, "High": 0}
            for pid, _ in comp_scores:
                lab = updates_final[pid].get("competition_level", "Medium")
                dist_comp[lab] += 1

    applied_ok: Dict[str, Dict[str, Any]] = {}
    success = 0
    errors = 0
    for pid, apply in updates_final.items():
        if apply:
            apply["ai_columns_completed_at"] = datetime.utcnow().isoformat()
            database.update_product(conn, int(pid), **apply)
            applied_ok[pid] = {k: v for k, v in apply.items() if k != "ai_columns_completed_at"}
            success += 1
        else:
            database.update_product(conn, int(pid), ai_columns_completed_at=datetime.utcnow().isoformat())
            ko_all[pid] = ko_all.get(pid, "existing")
            errors += 1

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
    logger.info(
        "ai_calibration: dist_desire=%s dist_competition=%s applied_quantiles=%s auto_promoted_low=%s auto_promoted_high=%s",
        dist_desire,
        dist_comp,
        {"desire": desire_q, "competition": comp_q},
        promoted_low,
        promoted_high,
    )
    processed_ids = {int(pid) for pid in applied_ok.keys()} | {int(pid) for pid in ko_all.keys()}
    pending_ids.extend([it["id"] for it in to_process if it["id"] not in processed_ids])
    return {
        "ok": applied_ok,
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
