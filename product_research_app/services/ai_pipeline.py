from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Tuple

from product_research_app import gpt
from product_research_app.services import ai_columns, ai_prompts
from product_research_app.utils import cache as ai_cache

log = logging.getLogger(__name__)

_MICRO = int(os.getenv("PRAPP_AI_MICROBATCH", "12"))
_CONC = int(os.getenv("PRAPP_OPENAI_MAX_CONCURRENCY", "3"))
_TWO_STAGE = os.getenv("PRAPP_AI_TWO_STAGE", "1") not in ("0", "false", "False")
_USE_CACHE = os.getenv("PRAPP_AI_CACHE", "1") not in ("0", "false", "False")
_CACHE_VER = os.getenv("PRAPP_AI_CACHE_VERSION", "v1")
_PROPAGATE = os.getenv("PRAPP_AI_VARIANT_PROPAGATION", "1") not in ("0", "false", "False")
_GROUP_MIN = int(os.getenv("PRAPP_AI_GROUP_MIN_PREFIX", "18"))

_COLOR_SIZE = re.compile(
    r"\b(black|white|blue|red|green|yellow|pink|purple|brown|grey|gray|"
    r"rojo|azul|verde|negro|blanco|gris|marr[oó]n|rosa|morado|amarillo|"
    r"talla\s?[smlx]+|size\s?[smlx]+|xl|xxl|xs|s|m|l|xxs|xxxl|"
    r"\d+\s?(cm|mm|in|inch|pcs|pack|ud|uds))\b",
    re.I,
)


def _chunks(seq: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(seq), max(1, n)):
        yield seq[i : i + n]


def _norm_title(txt: str) -> str:
    t = re.sub(r"[\W_]+", " ", txt or "").lower()
    t = COLOR_SIZE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _group_key(prod: Dict[str, Any]) -> str:
    if "parent_id" in prod and prod.get("parent_id"):
        return f"parent:{prod['parent_id']}"
    base = _norm_title(prod.get("title") or prod.get("name") or "")
    return base[: max(_GROUP_MIN, 24)]


def _input_payload(prod: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": prod.get("title") or prod.get("name") or "",
        "category": prod.get("category") or "",
        "desc": prod.get("description") or "",
    }


def _load_products_missing(limit: int | None = None) -> List[Dict[str, Any]]:
    try:
        from product_research_app import db

        conn = db.get_conn()
        cols = [
            "id",
            "title",
            "name",
            "category",
            "description",
            "desire",
            "desire_magnitude",
            "awareness_level",
            "competition_level",
            "parent_id",
        ]
        sql = "SELECT " + ",".join(cols) + " FROM product WHERE " \
            "(desire IS NULL OR desire='')" \
            " OR (desire_magnitude IS NULL)" \
            " OR (awareness_level IS NULL)" \
            " OR (competition_level IS NULL)"
        params: Tuple[Any, ...] | Tuple[()] = ()
        if limit:
            sql += " LIMIT ?"
            params = (limit,)
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        log.warning("ai_pipeline: fallback loader not available", exc_info=True)
        return []


def _persist_rows(rows: List[Dict[str, Any]]) -> None:
    if hasattr(ai_columns, "_persist_rows"):
        ai_columns._persist_rows(rows)  # type: ignore[attr-defined]
        return
    try:
        from product_research_app import db

        conn = db.get_conn()
        for r in rows:
            conn.execute(
                """
                UPDATE product SET
                  desire = COALESCE(:desire, desire),
                  desire_magnitude = COALESCE(:desire_magnitude, desire_magnitude),
                  awareness_level = COALESCE(:awareness_level, awareness_level),
                  competition_level = COALESCE(:competition_level, competition_level)
                WHERE id = :id
                """,
                r,
            )
        conn.commit()
    except Exception:
        log.error("ai_pipeline: persist fallback failed", exc_info=True)
        raise


def _apply_cache_or_group(batch: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ai_cache.init_cache()
    resolved: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    cache_version = os.getenv("PRAPP_AI_CACHE_VERSION", _CACHE_VER)
    group_rep: Dict[str, int] = {}
    group_values: Dict[str, Dict[str, Any]] = {}

    for p in batch:
        payload = _input_payload(p)
        key = ai_cache.make_key(payload, version=cache_version)
        cached = ai_cache.get(key) if _USE_CACHE else None
        gk = _group_key(p) if _PROPAGATE else None
        if cached:
            resolved.append({"id": p["id"], **cached})
            if gk:
                group_values[gk] = cached
                group_rep.setdefault(gk, p["id"])
            continue
        if gk and gk in group_values:
            resolved.append({"id": p["id"], **group_values[gk]})
            continue
        rep_id = p["id"]
        if gk:
            rep_id = group_rep.get(gk, p["id"])
            group_rep.setdefault(gk, rep_id)
        pending.append({**p, "_cache_key": key, "_group_key": gk, "_group_rep": rep_id})

    return resolved, pending


def _save_to_cache(rows: List[Dict[str, Any]], pending: List[Dict[str, Any]]):
    if not _USE_CACHE and not _PROPAGATE:
        return
    by_id = {r["id"]: r for r in rows}
    rep_rows: Dict[int, Dict[str, Any]] = {}
    for prod in pending:
        rep_id = prod.get("_group_rep") or prod["id"]
        if rep_id in by_id and rep_id not in rep_rows:
            rep_rows[rep_id] = by_id[rep_id]

    for prod in pending:
        row = by_id.get(prod["id"])
        if row is None and _PROPAGATE:
            rep_id = prod.get("_group_rep")
            if rep_id and rep_id in rep_rows:
                row = rep_rows[rep_id]
        if row is None:
            continue
        value = {k: row.get(k) for k in ai_prompts.AI_FIELDS}
        ck = prod.get("_cache_key")
        if _USE_CACHE and ck:
            ai_cache.set_(ck, value)


def _triage(pending: List[Dict[str, Any]]) -> List[int]:
    if not _TWO_STAGE:
        return [p["id"] for p in pending]

    need_ids: List[int] = []
    with ThreadPoolExecutor(max_workers=max(1, _CONC)) as ex:
        futs = [ex.submit(_triage_one_batch, group) for group in _chunks(pending, _MICRO)]
        for f in as_completed(futs):
            try:
                need_ids.extend(f.result())
            except Exception:
                log.exception("ai_pipeline: triage batch failed")
    return need_ids


def _triage_one_batch(batch: List[Dict[str, Any]]) -> List[int]:
    msgs = ai_prompts.build_triage_messages(batch)
    raw = gpt.call_gpt(messages=msgs, model=os.getenv("PRAPP_AI_TRIAGE_MODEL"))
    rows = ai_prompts.parse_triage(raw)
    return [r["id"] for r in rows if r.get("needs_scoring")]


def _score(ids: List[int], loader_index: Dict[int, Dict[str, Any]]):
    items = [loader_index[i] for i in ids if i in loader_index]
    if not items:
        return

    all_rows: List[Dict[str, Any]] = []

    def _score_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        msgs = ai_prompts.build_score_messages(batch)
        raw = gpt.call_gpt(messages=msgs, model=os.getenv("PRAPP_AI_SCORE_MODEL"))
        return ai_prompts.parse_score(raw)

    with ThreadPoolExecutor(max_workers=max(1, _CONC)) as ex:
        futs = [ex.submit(_score_batch, group) for group in _chunks(items, _MICRO)]
        for f in as_completed(futs):
            try:
                all_rows.extend(f.result())
            except Exception:
                log.exception("ai_pipeline: scoring batch failed")

    if not all_rows:
        return

    scored_map = {row["id"]: row for row in all_rows}
    rep_rows: Dict[int, Dict[str, Any]] = {}
    for prod in loader_index.values():
        rep_id = prod.get("_group_rep") or prod["id"]
        if rep_id in scored_map and rep_id not in rep_rows:
            rep_rows[rep_id] = scored_map[rep_id]

    propagated_rows: List[Dict[str, Any]] = []
    for prod in loader_index.values():
        if prod["id"] in scored_map:
            continue
        rep_id = prod.get("_group_rep")
        if not rep_id or rep_id not in rep_rows:
            continue
        base_row = rep_rows[rep_id]
        new_row = {"id": prod["id"]}
        for field in ai_prompts.AI_FIELDS:
            new_row[field] = base_row.get(field)
        propagated_rows.append(new_row)

    rows_to_persist = all_rows + propagated_rows
    if rows_to_persist:
        _persist_rows(rows_to_persist)
        _save_to_cache(rows_to_persist, list(loader_index.values()))


def run_ai_pipeline(limit: int | None = None):
    products = _load_products_missing(limit=limit)
    if not products:
        log.info("ai_pipeline: no missing products")
        return

    resolved, pending = _apply_cache_or_group(products)
    if resolved:
        _persist_rows(resolved)

    if not pending:
        log.info("ai_pipeline: nothing pending after cache/group")
        return

    by_id = {p["id"]: p for p in pending}
    triage_ids = set(_triage(pending))
    if not triage_ids:
        log.info("ai_pipeline: no items require scoring after triage")
        return

    score_ids = set()
    for pid in triage_ids:
        entry = by_id.get(pid)
        if not entry:
            continue
        rep_id = entry.get("_group_rep") or pid
        if rep_id in by_id:
            score_ids.add(rep_id)
        else:
            score_ids.add(pid)

    if not score_ids:
        log.info("ai_pipeline: triage selected ids not found in loader index")
        return

    _score(sorted(score_ids), loader_index=by_id)
