from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .. import config, database, gpt

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent.parent
DB_PATH = APP_DIR / "data.sqlite3"

AI_FIELDS = ("desire", "desire_magnitude", "awareness_level", "competition_level")
StatusCallback = Callable[..., None]

DEFAULT_EST_PROMPT_TOKENS = int(config.DEFAULT_CONFIG["aiCost"]["estTokensPerItemIn"])
DEFAULT_EST_COMPLETION_TOKENS = int(config.DEFAULT_CONFIG["aiCost"]["estTokensPerItemOut"])


@dataclass
class ProductJobItem:
    product_id: int
    digest: str
    payload: Dict[str, Any]
    title: str
    description: Optional[str]


def _ensure_conn():
    conn = database.get_connection(DB_PATH)
    database.initialize_database(conn)
    return conn


def _normalize_for_digest(value: Optional[str]) -> str:
    if not value:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = " ".join(text.strip().lower().split())
    return text


def _truncate_text(value: Optional[str], limit: int) -> str:
    if not value:
        return ""
    text = str(value).strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip()


def _compute_digest(title: str, brand: str, category: str, description: str) -> str:
    parts = [
        _normalize_for_digest(title),
        _normalize_for_digest(brand),
        _normalize_for_digest(category),
        _normalize_for_digest(description),
    ]
    joined = "|".join(parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def _ensure_ai_cache_table(conn) -> None:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ai_cache'")
    row = cur.fetchone()
    if row:
        cur.execute("PRAGMA table_info(ai_cache)")
        columns = {info[1] for info in cur.fetchall()}
        expected = {
            "digest",
            "desire_label",
            "desire_magnitude",
            "awareness_level",
            "competition_level",
            "created_at",
        }
        if columns != expected:
            cur.execute("ALTER TABLE ai_cache RENAME TO ai_cache_legacy")
            conn.commit()
            cur.execute("DROP TABLE IF EXISTS ai_cache_legacy")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_cache (
            digest TEXT PRIMARY KEY,
            desire_label TEXT,
            desire_magnitude REAL,
            awareness_level TEXT,
            competition_level TEXT,
            created_at TEXT
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_ai_cache_created_at ON ai_cache(created_at)"
    )
    conn.commit()


def _fetch_cached_rows(conn, digests: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    values = [d for d in digests if d]
    if not values:
        return {}
    placeholders = ",".join("?" for _ in values)
    cur = conn.execute(
        f"SELECT digest, desire_label, desire_magnitude, awareness_level, competition_level FROM ai_cache WHERE digest IN ({placeholders})",
        tuple(values),
    )
    rows = {}
    for row in cur.fetchall():
        rows[row["digest"]] = {
            "desire": row["desire_label"],
            "desire_magnitude": row["desire_magnitude"],
            "awareness_level": row["awareness_level"],
            "competition_level": row["competition_level"],
        }
    return rows


def _upsert_cache_rows(conn, entries: List[Tuple[str, Any, Any, Any, Any, str]]):
    if not entries:
        return
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO ai_cache (digest, desire_label, desire_magnitude, awareness_level, competition_level, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(digest) DO UPDATE SET
            desire_label=excluded.desire_label,
            desire_magnitude=excluded.desire_magnitude,
            awareness_level=excluded.awareness_level,
            competition_level=excluded.competition_level,
            created_at=excluded.created_at
        """,
        entries,
    )


def _calculate_cost(usage: Dict[str, Any], price_in: float, price_out: float) -> float:
    prompt = usage.get("prompt_tokens")
    if prompt is None:
        prompt = usage.get("input_tokens") or usage.get("tokens_in")
    if prompt is None:
        prompt = usage.get("total_tokens")
    completion = usage.get("completion_tokens")
    if completion is None:
        completion = usage.get("output_tokens") or usage.get("tokens_out")
    if completion is None and prompt is not None:
        try:
            total = float(usage.get("total_tokens", 0))
            completion = max(0.0, total - float(prompt))
        except Exception:
            completion = 0.0
    prompt_val = float(prompt or 0.0)
    completion_val = float(completion or 0.0)
    return (prompt_val / 1_000_000.0) * price_in + (completion_val / 1_000_000.0) * price_out


def _parse_score(val: Any) -> Optional[float]:
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
        m = math.nan
        try:
            for part in txt.replace("%", " " ).split():
                if not part:
                    continue
                num = float(part)
                m = num
                break
        except Exception:
            m = math.nan
        if not math.isnan(m):
            if m > 1:
                m /= 100.0
            if 0 <= m <= 1:
                return m
        if txt.startswith("low"):
            return 0.2
        if txt.startswith("med"):
            return 0.5
        if txt.startswith("high"):
            return 0.8
    return None


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
    info: Dict[str, Any] = {
        "q33": None,
        "q67": None,
        "fallback": False,
        "moved_medium": 0,
        "moved_low": 0,
        "moved_high": 0,
    }

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


def _quantile(data: List[float], q: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[int(pos)]
    return s[lo] * (hi - pos) + s[hi] * (pos - lo)


def _notify_status(
    callback: Optional[StatusCallback],
    *,
    counts: Dict[str, Any],
    pending_ids: Sequence[int],
    state: str,
    message: Optional[str] = None,
) -> None:
    if callback is None:
        return
    payload = {
        "phase": "enrich",
        "ai_counts": counts,
        "ai_total": counts.get("total", 0),
        "ai_done": counts.get("ok", 0),
        "ai_pending": list(pending_ids),
        "pct_ai": int(round((counts.get("ok", 0) / max(counts.get("total", 1), 1)) * 100))
        if counts.get("total")
        else 100,
        "state": state,
    }
    if message:
        payload["message"] = message
    try:
        callback(**payload)
    except Exception:
        logger.debug("status callback failed", exc_info=True)


def _prepare_product_item(
    product_id: int,
    row: Any,
    truncate_cfg: Dict[str, int],
) -> Tuple[Optional[ProductJobItem], Optional[str]]:
    extra: Dict[str, Any] = {}
    if row.get("extra"):
        try:
            extra = json.loads(row["extra"])
        except Exception:
            extra = {}
    title = str(row.get("name") or "").strip()
    if not title:
        return None, "missing_title"
    description = str(row.get("description") or "").strip()
    if not description:
        desc_extra = extra.get("description")
        if isinstance(desc_extra, str):
            description = desc_extra.strip()
    brand = str(extra.get("brand") or "").strip()
    category = str(row.get("category") or extra.get("category_path") or "").strip()
    truncated_title = _truncate_text(title, truncate_cfg.get("title", 120))
    truncated_desc = _truncate_text(description, truncate_cfg.get("description", 240))
    if truncated_desc:
        truncated_desc = " ".join(truncated_desc.split())
    if truncated_desc == "":
        truncated_desc = None
    payload = {
        "id": product_id,
        "name": truncated_title,
        "brand": brand,
        "category": category,
        "price": row.get("price"),
        "rating": extra.get("rating"),
        "units_sold": extra.get("units_sold"),
        "revenue": extra.get("revenue"),
    }
    if truncated_desc:
        payload["description"] = truncated_desc
    digest = _compute_digest(truncated_title, brand, category, truncated_desc or "")
    item = ProductJobItem(
        product_id=product_id,
        digest=digest,
        payload=payload,
        title=truncated_title,
        description=truncated_desc,
    )
    return item, None


# TODO: Manual QA - run with 100 productos (primera vez un request <52s, segunda vez todo cache).
# TODO: Verificar error 400 cuando falta API key y truncado con costCapUSD bajo.

def run_ai_fill_job(
    product_ids: Sequence[int],
    job_id: Optional[int] = None,
    *,
    status_cb: Optional[StatusCallback] = None,
) -> Dict[str, Any]:
    start_ts = time.perf_counter()
    conn = _ensure_conn()
    _ensure_ai_cache_table(conn)

    unique_ids: List[int] = []
    seen: set[int] = set()
    for pid in product_ids:
        try:
            num = int(pid)
        except Exception:
            continue
        if num in seen:
            continue
        seen.add(num)
        unique_ids.append(num)

    total_requested = len(unique_ids)
    counts: Dict[str, Any] = {
        "total": total_requested,
        "ok": 0,
        "ko": 0,
        "cached_hits": 0,
        "cached_misses": 0,
        "pending": total_requested,
        "batches": 0,
        "cost_estimated_usd": 0.0,
        "cost_actual_usd": 0.0,
    }
    pending_ids: set[int] = set(unique_ids)
    failures: Dict[int, str] = {}
    ok_payloads: Dict[int, Dict[str, Any]] = {}
    cache_rows: List[Tuple[str, Any, Any, Any, Any, str]] = []
    digest_map: Dict[int, str] = {}
    desire_scores: List[Tuple[str, float]] = []
    comp_scores: List[Tuple[str, float]] = []

    try:
        api_key = config.get_api_key() or os.environ["OPENAI_API_KEY"]
    except KeyError:
        api_key = None
    if not api_key:
        logger.error("run_ai_fill_job missing API key")
        counts["pending"] = len(pending_ids)
        result = {
            "counts": counts,
            "pending_ids": sorted(pending_ids),
            "error": "missing_api_key",
            "ok": {},
            "ko": {pid: "missing_api_key" for pid in unique_ids},
            "cached_hits": 0,
            "total_requested": total_requested,
            "duration": 0.0,
        }
        _notify_status(status_cb, counts=counts, pending_ids=pending_ids, state="error", message="Falta API key")
        conn.close()
        return result

    runtime_cfg = config.get_ai_runtime_config()
    cost_cfg = config.get_ai_cost_config()
    calib_cfg = config.get_ai_calibration_config()

    model = runtime_cfg.get("model") or cost_cfg.get("model") or config.get_model()
    parallelism = max(1, int(runtime_cfg.get("parallelism", 1) or 1))
    microbatch = max(1, int(runtime_cfg.get("microbatch", 32) or 32))
    max_output_per_item = max(1, int(runtime_cfg.get("maxOutputTokensPerItem", 8) or 8))
    temperature = float(runtime_cfg.get("temperature", 0.0) or 0.0)
    top_p = float(runtime_cfg.get("topP", 0.1) or 0.1)
    truncate_cfg = runtime_cfg.get("truncate") or {"title": 120, "description": 240}
    enable_cache = bool(runtime_cfg.get("enableCache", True))

    price_map = cost_cfg.get("prices", {}).get(model, {}) if isinstance(cost_cfg.get("prices"), dict) else {}
    price_in = float(price_map.get("input", 0.0) or 0.0)
    price_out = float(price_map.get("output", 0.0) or 0.0)
    cost_cap = cost_cfg.get("costCapUSD")
    try:
        cost_cap_val = None if cost_cap is None else float(cost_cap)
    except Exception:
        cost_cap_val = None

    rows = database.get_products_by_ids(conn, unique_ids)
    row_map = {int(row["id"]): row for row in rows}

    items: List[ProductJobItem] = []
    for pid in unique_ids:
        row = row_map.get(pid)
        if row is None:
            failures[pid] = "missing_product"
            pending_ids.discard(pid)
            continue
        item, error = _prepare_product_item(pid, row, truncate_cfg)
        if error:
            failures[pid] = error
            pending_ids.discard(pid)
            logger.warning("AI columns skipped product %s: %s", pid, error)
            continue
        digest_map[pid] = item.digest
        items.append(item)

    counts["pending"] = len(pending_ids)

    if job_id:
        database.start_import_job_ai(conn, int(job_id), counts["total"])
        database.set_import_job_ai_counts(conn, counts, sorted(pending_ids))

    _notify_status(status_cb, counts=counts, pending_ids=pending_ids, state="running", message="Preparando IA")

    remaining_items = items
    if enable_cache:
        cached = _fetch_cached_rows(conn, [item.digest for item in items])
        uncached: List[ProductJobItem] = []
        for item in items:
            cached_row = cached.get(item.digest)
            if not cached_row:
                uncached.append(item)
                continue
            payload = {
                "desire": cached_row.get("desire"),
                "desire_magnitude": gpt._norm_tri(cached_row.get("desire_magnitude")),
                "awareness_level": gpt._norm_awareness(cached_row.get("awareness_level")),
                "competition_level": gpt._norm_tri(cached_row.get("competition_level")),
            }
            ok_payloads[item.product_id] = payload
            pending_ids.discard(item.product_id)
            counts["cached_hits"] += 1
            val_desire = _parse_score(payload.get("desire_magnitude"))
            val_comp = _parse_score(payload.get("competition_level"))
            if val_desire is not None:
                desire_scores.append((str(item.product_id), val_desire))
            if val_comp is not None:
                comp_scores.append((str(item.product_id), val_comp))
        remaining_items = uncached
        counts["pending"] = len(pending_ids)

    counts["cached_misses"] = len(remaining_items)

    est_prompt = len(remaining_items) * DEFAULT_EST_PROMPT_TOKENS
    est_completion = len(remaining_items) * DEFAULT_EST_COMPLETION_TOKENS
    estimated_usage = {"prompt_tokens": est_prompt, "completion_tokens": est_completion}
    counts["cost_estimated_usd"] = _calculate_cost(estimated_usage, price_in, price_out)

    truncated_items: List[ProductJobItem] = remaining_items
    result_error: Optional[str] = None
    if cost_cap_val is not None and cost_cap_val > 0 and counts["cost_estimated_usd"] > cost_cap_val and remaining_items:
        per_item_usage = {"prompt_tokens": DEFAULT_EST_PROMPT_TOKENS, "completion_tokens": DEFAULT_EST_COMPLETION_TOKENS}
        per_item_cost = _calculate_cost(per_item_usage, price_in, price_out)
        if per_item_cost > 0:
            allowed = int(cost_cap_val / per_item_cost)
        else:
            allowed = len(remaining_items)
        if allowed < len(remaining_items):
            truncated_items = remaining_items[:allowed]
            skipped = remaining_items[allowed:]
            result_error = "cost_cap_soft"
            for item in skipped:
                logger.warning("AI columns skipped product %s due to cost cap", item.product_id)
                failures[item.product_id] = "cost_cap_soft"
                pending_ids.discard(item.product_id)
            counts["cached_misses"] = len(truncated_items)
            counts["cost_estimated_usd"] = per_item_cost * len(truncated_items)
            counts["pending"] = len(pending_ids)

    batches: List[List[ProductJobItem]] = []
    if len(truncated_items) <= microbatch:
        if truncated_items:
            batches.append(truncated_items)
    else:
        for idx in range(0, len(truncated_items), microbatch):
            chunk = truncated_items[idx : idx + microbatch]
            if chunk:
                batches.append(chunk)

    async def _execute_batches() -> List[Dict[str, Any]]:
        if not batches:
            return []
        sem = asyncio.Semaphore(parallelism)
        results: List[Dict[str, Any]] = []

        async def _run_single(index: int, batch: List[ProductJobItem]) -> Dict[str, Any]:
            async with sem:
                payload_items = [item.payload for item in batch]
                max_tokens = max(64, len(payload_items) * max_output_per_item)
                try:
                    data, _, usage, duration = await asyncio.to_thread(
                        gpt.generate_batch_columns,
                        api_key,
                        model,
                        payload_items,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                    )
                except Exception as exc:  # pragma: no cover - network failure path
                    return {"index": index, "batch": batch, "data": {}, "usage": {}, "duration": 0.0, "error": exc}
                return {"index": index, "batch": batch, "data": data, "usage": usage, "duration": duration, "error": None}

        tasks = [asyncio.create_task(_run_single(idx + 1, batch)) for idx, batch in enumerate(batches)]
        for coro in asyncio.as_completed(tasks):
            results.append(await coro)
        return results

    loop_results: List[Dict[str, Any]] = []
    if batches:
        loop_results = asyncio.run(_execute_batches())

    for result in loop_results:
        batch_items = result.get("batch", [])
        counts["batches"] += 1
        error = result.get("error")
        usage = result.get("usage", {}) or {}
        counts["cost_actual_usd"] += _calculate_cost(usage, price_in, price_out)
        if error:
            err_text = str(error)
            for item in batch_items:
                failures[item.product_id] = err_text
                pending_ids.discard(item.product_id)
                logger.warning("AI columns error product=%s error=%s", item.product_id, err_text)
            continue
        data = result.get("data", {}) or {}
        for item in batch_items:
            entry = data.get(str(item.product_id))
            if not isinstance(entry, dict):
                failures[item.product_id] = "missing_response"
                pending_ids.discard(item.product_id)
                logger.warning("AI columns missing entry product=%s", item.product_id)
                continue
            desire_val = entry.get("desire")
            payload = {
                "desire": desire_val if desire_val not in (None, "") else None,
                "desire_magnitude": gpt._norm_tri(entry.get("desire_magnitude")),
                "awareness_level": gpt._norm_awareness(entry.get("awareness_level")),
                "competition_level": gpt._norm_tri(entry.get("competition_level")),
            }
            ok_payloads[item.product_id] = payload
            pending_ids.discard(item.product_id)
            now_iso = datetime.utcnow().isoformat()
            cache_rows.append(
                (
                    item.digest,
                    payload.get("desire"),
                    payload.get("desire_magnitude"),
                    payload.get("awareness_level"),
                    payload.get("competition_level"),
                    now_iso,
                )
            )
            val_desire = _parse_score(payload.get("desire_magnitude"))
            val_comp = _parse_score(payload.get("competition_level"))
            if val_desire is not None:
                desire_scores.append((str(item.product_id), val_desire))
            if val_comp is not None:
                comp_scores.append((str(item.product_id), val_comp))
        counts["ok"] = len(ok_payloads)
        counts["ko"] = len(failures)
        counts["pending"] = len(pending_ids)
        if job_id:
            database.update_import_job_ai_progress(conn, int(job_id), counts["ok"])
            database.set_import_job_ai_counts(conn, counts, sorted(pending_ids))
        _notify_status(status_cb, counts=counts, pending_ids=pending_ids, state="running")

    counts["ok"] = len(ok_payloads)
    counts["ko"] = len(failures)
    counts["pending"] = len(pending_ids)

    if calib_cfg.get("enabled", True) and ok_payloads:
        wins = float(calib_cfg.get("winsorize_pct", 0.05) or 0.0)
        min_low = float(calib_cfg.get("min_low_pct", 0.05) or 0.0)
        min_med = float(calib_cfg.get("min_medium_pct", 0.05) or 0.0)
        min_high = float(calib_cfg.get("min_high_pct", 0.05) or 0.0)
        if desire_scores:
            labels, _, _ = _classify_scores(
                desire_scores,
                winsorize_pct=wins,
                min_low_pct=min_low,
                min_medium_pct=min_med,
                min_high_pct=min_high,
            )
            for pid_str, label in labels.items():
                pid = int(pid_str)
                payload = ok_payloads.get(pid)
                if payload:
                    payload["desire_magnitude"] = label
        if comp_scores:
            labels, _, _ = _classify_scores(
                comp_scores,
                winsorize_pct=wins,
                min_low_pct=min_low,
                min_medium_pct=min_med,
                min_high_pct=min_high,
            )
            for pid_str, label in labels.items():
                pid = int(pid_str)
                payload = ok_payloads.get(pid)
                if payload:
                    payload["competition_level"] = label

    now_iso = datetime.utcnow().isoformat()
    updates: List[Tuple[Any, ...]] = []
    for pid, payload in ok_payloads.items():
        updates.append(
            (
                payload.get("desire"),
                payload.get("desire_magnitude"),
                payload.get("awareness_level"),
                payload.get("competition_level"),
                now_iso,
                pid,
            )
        )
        if pid in digest_map:
            cache_rows.append(
                (
                    digest_map[pid],
                    payload.get("desire"),
                    payload.get("desire_magnitude"),
                    payload.get("awareness_level"),
                    payload.get("competition_level"),
                    now_iso,
                )
            )

    if updates or cache_rows:
        cur = conn.cursor()
        try:
            cur.execute("BEGIN")
            if updates:
                cur.executemany(
                    "UPDATE products SET desire=?, desire_magnitude=?, awareness_level=?, competition_level=?, ai_columns_completed_at=? WHERE id=?",
                    updates,
                )
            if cache_rows:
                _upsert_cache_rows(conn, cache_rows)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    counts["ok"] = len(ok_payloads)
    counts["ko"] = len(failures)
    counts["pending"] = len(pending_ids)

    if job_id:
        database.set_import_job_ai_counts(conn, counts, sorted(pending_ids))
        if result_error:
            database.set_import_job_ai_error(conn, int(job_id), result_error)
        database.update_import_job_ai_progress(conn, int(job_id), counts["ok"])

    duration = time.perf_counter() - start_ts
    _notify_status(
        status_cb,
        counts=counts,
        pending_ids=pending_ids,
        state="done" if not result_error else "error",
        message="IA columnas completado" if not result_error else "IA columnas con avisos",
    )

    logger.info(
        "run_ai_fill_job total=%d ok=%d ko=%d cached_hits=%d cached_misses=%d batches=%d duration=%.2fs cost_estimated=%.4f",
        counts["total"],
        counts["ok"],
        counts["ko"],
        counts["cached_hits"],
        counts["cached_misses"],
        counts["batches"],
        duration,
        counts["cost_estimated_usd"],
    )

    result = {
        "counts": counts,
        "pending_ids": sorted(pending_ids),
        "error": result_error,
        "ok": {pid: {k: v for k, v in payload.items() if v is not None} for pid, payload in ok_payloads.items()},
        "ko": failures,
        "cached_hits": counts["cached_hits"],
        "total_requested": total_requested,
        "duration": duration,
    }

    conn.close()
    return result


def fill_ai_columns(
    product_ids: Sequence[int],
    *,
    model: Optional[str] = None,
    batch_mode: Optional[bool] = None,
    cost_cap_usd: Optional[float] = None,
) -> Dict[str, Any]:
    result = run_ai_fill_job(product_ids)
    counts = result.get("counts", {})
    total_requested = result.get("total_requested", len(product_ids))
    legacy_counts = {
        "n_importados": total_requested,
        "n_para_ia": counts.get("total", 0),
        "n_procesados": counts.get("ok", 0),
        "n_omitidos_por_valor_existente": counts.get("cached_hits", 0),
        "n_reintentados": 0,
        "n_error_definitivo": counts.get("ko", 0),
        "truncated": result.get("error") == "cost_cap_soft",
        "cost_estimated_usd": counts.get("cost_estimated_usd", 0.0),
    }
    return {
        "ok": {str(pid): payload for pid, payload in result.get("ok", {}).items()},
        "ko": {str(pid): reason for pid, reason in result.get("ko", {}).items()},
        "counts": legacy_counts,
        "pending_ids": result.get("pending_ids", []),
        "cost_estimated_usd": counts.get("cost_estimated_usd", 0.0),
        "ui_cost_message": None,
        "error": result.get("error"),
    }
