import json
import logging
import os
import asyncio
import random
import time
import math
import re
from datetime import datetime, timezone
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


def _format_cost_message(cost: float) -> str:
    if cost >= 0.1:
        txt = f"{cost:.2f}"
    else:
        txt = f"{cost:.4f}"
    if "." in txt:
        txt = txt.rstrip("0").rstrip(".")
    return f"importando productos, por favor espere... El coste será de {txt}$"


def _classify_scores(
    pairs: List[tuple[str, float]],
    *,
    winsorize_pct: float,
    min_low_pct: float,
    min_medium_pct: float,
    min_high_pct: float,
) -> tuple[Dict[str, str], Dict[str, int], Dict[str, Any]]:
    labels: Dict[str, str] = {}
    dist = {"Low": 0, "Medium": 0, "High": 0}
    info: Dict[str, Any] = {"q33": None, "q67": None, "fallback": False, "moved_medium": 0, "moved_low": 0, "moved_high": 0}

    if not pairs:
        return labels, dist, info

    values = [s for _, s in pairs]
    n = len(values)
    distinct = len(set(values))

    if n >= 50 and winsorize_pct > 0:
        low_lim = _quantile(values, winsorize_pct)
        high_lim = _quantile(values, 1 - winsorize_pct)
        values = [min(max(s, low_lim), high_lim) for s in values]
        pairs = [(pid, min(max(score, low_lim), high_lim)) for pid, score in pairs]

    q33 = _quantile(values, 1 / 3)
    q67 = _quantile(values, 2 / 3)
    info["q33"] = q33
    info["q67"] = q67

    if n >= 6 and distinct >= 3 and abs(q67 - q33) > 1e-6:
        for pid, score in pairs:
            if score <= q33:
                lab = "Low"
            elif score >= q67:
                lab = "High"
            else:
                lab = "Medium"
            labels[pid] = lab
            dist[lab] += 1
    else:
        info["fallback"] = True
        sorted_pairs = sorted(pairs, key=lambda x: x[1])
        cut1 = round(n / 3)
        cut2 = round(2 * n / 3)
        for idx, (pid, _) in enumerate(sorted_pairs):
            if idx < cut1:
                lab = "Low"
            elif idx < cut2:
                lab = "Medium"
            else:
                lab = "High"
            labels[pid] = lab
            dist[lab] += 1

    min_medium = math.ceil(min_medium_pct * n)
    min_low = math.ceil(min_low_pct * n)
    min_high = math.ceil(min_high_pct * n)

    if dist["Medium"] < min_medium:
        need = min_medium - dist["Medium"]
        candidates = [(abs(score - 0.5), pid) for pid, score in pairs if labels[pid] != "Medium"]
        candidates.sort()
        for _, pid in candidates[:need]:
            prev = labels[pid]
            labels[pid] = "Medium"
            dist["Medium"] += 1
            dist[prev] -= 1
            info["moved_medium"] += 1

    available = max(0, dist["Medium"] - min_medium)
    if dist["Low"] < min_low and available > 0:
        need = min(min_low - dist["Low"], available)
        candidates = [
            (abs(score - q33), pid)
            for pid, score in pairs
            if labels[pid] == "Medium"
        ]
        candidates.sort()
        for _, pid in candidates[:need]:
            labels[pid] = "Low"
            dist["Low"] += 1
            dist["Medium"] -= 1
            info["moved_low"] += 1
        available = max(0, dist["Medium"] - min_medium)

    if dist["High"] < min_high and available > 0:
        need = min(min_high - dist["High"], available)
        candidates = [
            (abs(score - q67), pid)
            for pid, score in pairs
            if labels[pid] == "Medium"
        ]
        candidates.sort()
        for _, pid in candidates[:need]:
            labels[pid] = "High"
            dist["High"] += 1
            dist["Medium"] -= 1
            info["moved_high"] += 1

    return labels, dist, info


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
    now_ts = datetime.now(timezone.utc).isoformat()

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

    cost_msg = _format_cost_message(cost_estimated)

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
            "cost_estimated_usd": cost_estimated,
            "ui_cost_message": cost_msg,
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

    wins_p = cfg_calib.get("winsorize_pct", 0.05)
    min_low_pct = cfg_calib.get("min_low_pct", 0.05)
    min_med_pct = cfg_calib.get("min_medium_pct", 0.05)
    min_high_pct = cfg_calib.get("min_high_pct", 0.05)

    desire_info: Dict[str, Any] = {}
    comp_info: Dict[str, Any] = {}

    if cfg_calib.get("enabled", True) and desire_scores:
        labels, dist_desire, desire_info = _classify_scores(
            desire_scores,
            winsorize_pct=wins_p,
            min_low_pct=min_low_pct,
            min_medium_pct=min_med_pct,
            min_high_pct=min_high_pct,
        )
        for pid, lab in labels.items():
            updates_final[pid]["desire_magnitude"] = lab

    if cfg_calib.get("enabled", True) and comp_scores:
        labels, dist_comp, comp_info = _classify_scores(
            comp_scores,
            winsorize_pct=wins_p,
            min_low_pct=min_low_pct,
            min_medium_pct=min_med_pct,
            min_high_pct=min_high_pct,
        )
        for pid, lab in labels.items():
            updates_final[pid]["competition_level"] = lab

    applied_ok: Dict[str, Dict[str, Any]] = {}
    success = 0
    errors = 0
    for pid, apply in updates_final.items():
        if apply:
            apply["ai_columns_completed_at"] = datetime.now(timezone.utc).isoformat()
            database.update_product(conn, int(pid), **apply)
            applied_ok[pid] = {k: v for k, v in apply.items() if k != "ai_columns_completed_at"}
            success += 1
        else:
            database.update_product(conn, int(pid), ai_columns_completed_at=datetime.now(timezone.utc).isoformat())
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
        "ai_calibration_desire: dist=%s q33=%.4f q67=%.4f fallback=%s moved_medium=%s moved_low=%s moved_high=%s",
        dist_desire,
        desire_info.get("q33") or 0.0,
        desire_info.get("q67") or 0.0,
        desire_info.get("fallback"),
        desire_info.get("moved_medium"),
        desire_info.get("moved_low"),
        desire_info.get("moved_high"),
    )
    logger.info(
        "ai_calibration_competition: dist=%s q33=%.4f q67=%.4f fallback=%s moved_medium=%s moved_low=%s moved_high=%s",
        dist_comp,
        comp_info.get("q33") or 0.0,
        comp_info.get("q67") or 0.0,
        comp_info.get("fallback"),
        comp_info.get("moved_medium"),
        comp_info.get("moved_low"),
        comp_info.get("moved_high"),
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
        "cost_estimated_usd": cost_estimated,
        "ui_cost_message": cost_msg,
    }
