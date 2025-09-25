from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import httpx

from .. import config, database
from ..utils.signature import compute_sig_hash
from .desire_utils import looks_like_product_desc
from . import ai_columns

logger = logging.getLogger(__name__)

SOURCES_CHECKED = [
    "product.desire",
    "extras.desire",
    "product.ai_desire",
    "product.ai_desire_label",
    "product.desire_magnitude",
]


def _coerce_conn(db: Any) -> sqlite3.Connection:
    if hasattr(db, "cursor"):
        return db  # type: ignore[return-value]
    conn = getattr(db, "connection", None)
    if conn is not None and hasattr(conn, "cursor"):
        return conn  # type: ignore[return-value]
    raise TypeError("db must be a sqlite3.Connection or provide .connection")


def _notify(logger_obj: Optional[Any], payload: Dict[str, Any]) -> None:
    if not logger_obj:
        return
    try:
        if callable(logger_obj):
            logger_obj(payload)
        elif hasattr(logger_obj, "info"):
            logger_obj.info("desire_backfill_progress", extra={"desire_backfill": payload})
    except Exception:
        pass


def _row_to_dict(row: Any) -> Dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    try:
        return {k: row[k] for k in row.keys()}  # type: ignore[attr-defined]
    except Exception:
        return dict(row)


def _needs_desire(row: Any) -> bool:
    data = _row_to_dict(row)
    desire_raw = data.get("desire")
    title = data.get("name") or data.get("title") or ""
    desire_txt = str(desire_raw or "").strip()
    if not desire_txt:
        return True
    if len(desire_txt) < 280:
        return True
    return looks_like_product_desc(desire_txt, str(title or ""))


def iter_missing_desire_ids(
    db: Any,
    ids: Optional[Sequence[int]] = None,
    chunk: int = 500,
) -> Iterator[List[int]]:
    """Yield product IDs that require a DESIRE refresh."""

    conn = _coerce_conn(db)
    chunk = max(1, int(chunk or 1))
    pending: List[int] = []
    seen: set[int] = set()

    def _flush() -> Iterator[List[int]]:
        nonlocal pending
        if pending:
            out = pending
            pending = []
            yield out

    def _handle_row(row: Any) -> Optional[int]:
        data = _row_to_dict(row)
        try:
            pid = int(data.get("id"))
        except Exception:
            return None
        if pid in seen:
            return None
        if _needs_desire(row):
            seen.add(pid)
            logger.info(
                "desire_missing=true sources_checked=%s product=%s",
                SOURCES_CHECKED,
                pid,
            )
            return pid
        return None

    cur = conn.cursor()
    if ids is not None:
        id_list: List[int] = []
        for raw in ids:
            try:
                id_list.append(int(raw))
            except Exception:
                continue
        if not id_list:
            return
        for start in range(0, len(id_list), chunk):
            subset = id_list[start : start + chunk]
            placeholders = ",".join(["?"] * len(subset))
            cur.execute(
                f"SELECT id, name, desire FROM products WHERE id IN ({placeholders})",
                tuple(int(pid) for pid in subset),
            )
            for row in cur:
                pid = _handle_row(row)
                if pid is None:
                    continue
                pending.append(pid)
                if len(pending) >= chunk:
                    for out in _flush():
                        yield out
    else:
        cur.execute("SELECT id, name, desire FROM products ORDER BY id")
        for row in cur:
            pid = _handle_row(row)
            if pid is None:
                continue
            pending.append(pid)
            if len(pending) >= chunk:
                for out in _flush():
                    yield out

    for out in _flush():
        yield out


async def _execute_batches(
    conn: sqlite3.Connection,
    candidates: List[ai_columns.Candidate],
    *,
    batch_size: int,
    parallel: int,
    max_retries: int,
    trunc_title: int,
    trunc_desc: int,
    timeout_s: float,
    model: str,
    api_key: str,
    rpm_limit: Optional[int],
    tpm_limit: Optional[int],
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    batches: List[ai_columns.BatchRequest] = []
    req_counter = 0
    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start : start + batch_size]
        if not chunk:
            continue
        req_counter += 1
        batches.append(
            ai_columns._build_batch_request(
                f"{req_counter:03d}",
                chunk,
                trunc_title,
                trunc_desc,
            )
        )

    rate_limiter = ai_columns._AsyncRateLimiter(rpm_limit, tpm_limit)
    stop_event = asyncio.Event()
    semaphore = asyncio.Semaphore(max(1, parallel))
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    limits = httpx.Limits(max_connections=max(20, parallel * 4), max_keepalive_connections=parallel * 2)
    timeout_cfg = httpx.Timeout(timeout_s)

    async with httpx.AsyncClient(
        base_url="https://api.openai.com",
        timeout=timeout_cfg,
        limits=limits,
        headers=headers,
    ) as client:
        tasks = [
            asyncio.create_task(
                ai_columns._call_batch_with_retries(
                    client,
                    batch,
                    model=model,
                    max_retries=max_retries,
                    rate_limiter=rate_limiter,
                    semaphore=semaphore,
                    stop_event=stop_event,
                )
            )
            for batch in batches
        ]
        results: List[Dict[str, Any]] = []
        for coro in asyncio.as_completed(tasks):
            results.append(await coro)
        return results


def _build_candidates(
    conn: sqlite3.Connection,
    product_ids: Sequence[int],
) -> tuple[List[ai_columns.Candidate], List[int]]:
    if not product_ids:
        return [], []
    rows = database.get_products_by_ids(conn, product_ids)
    row_map = {int(row["id"]): dict(row) for row in rows}
    candidates: List[ai_columns.Candidate] = []
    skipped: List[int] = []
    sig_updates: List[tuple[str, int]] = []
    for pid in product_ids:
        row = row_map.get(int(pid))
        if row is None:
            skipped.append(int(pid))
            continue
        extra: Dict[str, Any] = {}
        raw_extra = row.get("extra")
        if raw_extra:
            try:
                extra = json.loads(raw_extra)
            except Exception:
                extra = {}
        name = row.get("name")
        if not name:
            skipped.append(int(pid))
            continue
        brand = extra.get("brand")
        asin = extra.get("asin")
        product_url = extra.get("product_url")
        sig_hash = row.get("sig_hash") or compute_sig_hash(name, brand, asin, product_url)
        if sig_hash and not row.get("sig_hash"):
            sig_updates.append((sig_hash, int(pid)))
        payload = ai_columns._build_payload(row, extra)
        candidates.append(
            ai_columns.Candidate(
                id=int(pid),
                sig_hash=sig_hash,
                payload=payload,
                extra=extra,
            )
        )
    if sig_updates:
        cur = conn.cursor()
        for sig_hash, pid in sig_updates:
            cur.execute(
                "UPDATE OR IGNORE products SET sig_hash=? WHERE id=?",
                (sig_hash, pid),
            )
        if conn.in_transaction:
            conn.commit()
    return candidates, skipped


def run_desire_backfill(
    db: Any,
    ids: Optional[Sequence[int]] = None,
    batch_size: int = 32,
    parallel: int = 3,
    max_retries: int = 1,
    logger: Optional[Any] = None,
) -> Dict[str, int]:
    conn = _coerce_conn(db)
    batch_size = max(1, int(batch_size or 1))
    parallel = max(1, int(parallel or 1))
    max_retries = max(0, int(max_retries if max_retries is not None else 1))

    runtime_cfg = config.get_ai_runtime_config()
    trunc_title = int(runtime_cfg.get("trunc_title", 180) or 180)
    trunc_desc = int(runtime_cfg.get("trunc_desc", 800) or 800)
    timeout_s = float(runtime_cfg.get("timeout", 45) or 45)
    rpm_limit = runtime_cfg.get("rpm_limit")
    tpm_limit = runtime_cfg.get("tpm_limit")
    try:
        if rpm_limit is not None:
            rpm_limit = int(rpm_limit)
    except Exception:
        rpm_limit = None
    try:
        if tpm_limit is not None:
            tpm_limit = int(tpm_limit)
    except Exception:
        tpm_limit = None

    cost_cfg = config.get_ai_cost_config()
    model = cost_cfg.get("model") or config.get_model()
    env_model = os.environ.get("AI_MODEL")
    if env_model:
        model = env_model
    api_key = config.get_api_key() or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("missing_api_key")

    stats = {"queued": 0, "done": 0, "retried": 0, "failed": 0}

    def _emit():
        payload = dict(stats)
        _notify(logger, payload)

    try:
        id_iter: Iterable[List[int]]
        if ids is not None:
            id_iter = iter_missing_desire_ids(conn, ids=ids, chunk=500)
        else:
            id_iter = iter_missing_desire_ids(conn, ids=None, chunk=500)

        for group in id_iter:
            if not group:
                continue
            stats["queued"] += len(group)
            _emit()
            candidates, skipped = _build_candidates(conn, group)
            candidate_map = {cand.id: cand for cand in candidates}
            if skipped:
                stats["failed"] += len(skipped)
                _emit()
            if not candidates:
                continue

            start = time.perf_counter()
            results = asyncio.run(
                _execute_batches(
                    conn,
                    candidates,
                    batch_size=batch_size,
                    parallel=parallel,
                    max_retries=max_retries,
                    trunc_title=trunc_title,
                    trunc_desc=trunc_desc,
                    timeout_s=timeout_s,
                    model=model,
                    api_key=api_key,
                    rpm_limit=rpm_limit,
                    tpm_limit=tpm_limit,
                )
            )
            logger.debug(
                "desire_backfill.group processed ids=%s duration=%.2fs",
                group,
                time.perf_counter() - start,
            )
            for result in results:
                retries = int(result.get("retries", 0) or 0)
                stats["retried"] += retries
                ok_map: Dict[str, Dict[str, Any]] = result.get("ok", {}) or {}
                ko_map: Dict[str, Any] = result.get("ko", {}) or {}
                success_updates: Dict[int, Dict[str, Any]] = {}
                for pid_str, payload in ok_map.items():
                    try:
                        pid = int(pid_str)
                    except Exception:
                        continue
                    success_updates[pid] = payload
                if success_updates:
                    ai_columns._apply_ai_updates(conn, success_updates)
                    for pid, payload in success_updates.items():
                        candidate = candidate_map.get(pid)
                        sig_hash = candidate.sig_hash if candidate else ""
                        if sig_hash:
                            database.upsert_ai_cache_entry(
                                conn,
                                sig_hash,
                                model=model,
                                version=int(runtime_cfg.get("version", 1) or 1),
                                desire=payload.get("desire"),
                                desire_magnitude=payload.get("desire_magnitude"),
                                awareness_level=payload.get("awareness_level"),
                                competition_level=payload.get("competition_level"),
                            )
                stats["done"] += len(success_updates)
                stats["failed"] += len(ko_map)
                _emit()
    finally:
        _emit()

    return dict(stats)
