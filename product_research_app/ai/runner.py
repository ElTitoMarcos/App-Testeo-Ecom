"""Background AI post-processing runner."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import sqlite3

from product_research_app.ai.ai_status import update_status
from product_research_app.ai.gpt_orchestrator import (
    ChatCompletionError,
    run_desire_batch,
    run_imputacion_batch,
    run_weights_once,
)
from product_research_app.db import get_db
from product_research_app.services.aggregates import build_weighting_aggregates
from product_research_app.services import winner_score as winner_calc
from product_research_app.services.config import set_winner_order_raw, set_winner_weights_raw
from product_research_app.utils.db import row_to_dict


logger = logging.getLogger(__name__)


@dataclass
class AutoSettings:
    enabled: bool = True
    batch_size: int = 100
    max_parallel: int = 1
    max_calls: int = 3
    imputacion_via_ia: bool = False
    gpt_timeout: int = 20
    gpt_max_retry: int = 1

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> "AutoSettings":
        cfg = dict(config or {})
        settings = AutoSettings()
        try:
            settings.enabled = bool(cfg.get("AI_AUTO_ENABLED", settings.enabled))
        except Exception:
            settings.enabled = True
        key_map = {
            "AI_BATCH_SIZE": "batch_size",
            "AI_MAX_PARALLEL": "max_parallel",
            "AI_MAX_CALLS_PER_IMPORT": "max_calls",
        }
        for key, attr in key_map.items():
            try:
                value = int(cfg.get(key, getattr(settings, attr)))
            except Exception:
                value = getattr(settings, attr)
            value = max(0, value)
            setattr(settings, attr, value)
        try:
            settings.imputacion_via_ia = bool(cfg.get("IMPUTACION_VIA_IA", settings.imputacion_via_ia))
        except Exception:
            settings.imputacion_via_ia = False
        try:
            settings.gpt_timeout = max(1, int(cfg.get("GPT_TIMEOUT", settings.gpt_timeout)))
        except Exception:
            settings.gpt_timeout = 20
        try:
            settings.gpt_max_retry = max(0, int(cfg.get("GPT_MAX_RETRY", settings.gpt_max_retry)))
        except Exception:
            settings.gpt_max_retry = 1
        if settings.batch_size <= 0:
            settings.batch_size = 100
        if settings.max_parallel <= 0:
            settings.max_parallel = 1
        if settings.max_calls <= 0:
            settings.max_calls = 3
        return settings


def run_post_import_auto(task_id: str, product_ids: Sequence[int], config: Optional[Mapping[str, Any]]) -> None:
    config_map = dict(config or {})
    settings = AutoSettings.from_config(config_map)
    if not settings.enabled:
        update_status(task_id, notes=["auto_disabled"], state="DONE")
        return

    ids = _normalise_product_ids(product_ids)
    total = len(ids)

    update_status(
        task_id,
        state="RUNNING",
        desire={"requested": total, "processed": 0, "failed": 0},
        imputacion={
            "requested": total if settings.imputacion_via_ia else 0,
            "processed": 0,
            "failed": 0,
        },
        winner_score={"requested": total, "processed": 0, "failed": 0},
    )

    if not ids:
        update_status(task_id, state="DONE")
        return

    conn = get_db()
    rows = _fetch_rows(conn, ids)
    if not rows:
        update_status(task_id, state="DONE", notes=["no_products_found"])
        return

    rows_by_id: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        data = row_to_dict(row) or {}
        pid = data.get("id")
        if pid is None:
            continue
        try:
            pid_int = int(pid)
        except Exception:
            continue
        rows_by_id[pid_int] = data
    missing = [pid for pid in ids if pid not in rows_by_id]
    if missing:
        note = f"missing_products={len(missing)}"
        update_status(task_id, notes=[note])

    calls_used = 0

    desire_result = _process_desire(
        task_id,
        rows_by_id,
        ids,
        settings,
        conn,
        calls_used,
    )
    calls_used += desire_result.calls

    imputacion_processed = ImputacionResult(processed=0, failed=0, calls=0)
    if settings.imputacion_via_ia:
        imputacion_processed = _process_imputacion(
            task_id,
            rows_by_id,
            ids,
            settings,
            conn,
            calls_used,
        )
        calls_used += imputacion_processed.calls
    else:
        update_status(task_id, imputacion={"processed": 0, "failed": 0})

    _process_winner_score(
        task_id,
        rows_by_id,
        ids,
        settings,
        conn,
        calls_used,
        config_map,
    )

    update_status(task_id, state="DONE")


@dataclass
class BatchResult:
    processed: int
    failed: int
    calls: int = 0
    note: Optional[str] = None


ImputacionResult = BatchResult
WinnerResult = BatchResult


def _normalise_product_ids(product_ids: Sequence[int]) -> List[int]:
    seen: set[int] = set()
    normalised: List[int] = []
    for pid in product_ids:
        try:
            num = int(pid)
        except Exception:
            continue
        if num <= 0 or num in seen:
            continue
        seen.add(num)
        normalised.append(num)
    return normalised


def _fetch_rows(conn: sqlite3.Connection, ids: Sequence[int]) -> List[sqlite3.Row]:
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    query = f"SELECT * FROM products WHERE id IN ({placeholders})"
    cur = conn.execute(query, tuple(ids))
    return list(cur.fetchall())


def _has_compact_ai_desire(value: Optional[str]) -> bool:
    if not value:
        return False
    lines = [line.strip() for line in str(value).splitlines() if line.strip()]
    if not (2 <= len(lines) <= 3):
        return False
    return all(len(line) <= 90 for line in lines)


def _chunk(seq: Sequence[Any], size: int) -> Iterable[List[Any]]:
    size = max(1, int(size))
    for idx in range(0, len(seq), size):
        yield list(seq[idx : idx + size])


def _process_desire(
    task_id: str,
    rows_by_id: Mapping[int, Mapping[str, Any]],
    ids: Sequence[int],
    settings: AutoSettings,
    conn: sqlite3.Connection,
    calls_used: int,
) -> BatchResult:
    processed = 0
    failed = 0
    calls = 0
    pending_items: List[Dict[str, Any]] = []

    for pid in ids:
        row = rows_by_id.get(pid)
        if not row:
            failed += 1
            continue
        ai_desire = row.get("ai_desire")
        if _has_compact_ai_desire(ai_desire):
            processed += 1
            continue
        pending_items.append(
            {
                "id": str(pid),
                "desire": row.get("desire"),
                "title": row.get("name"),
                "description": row.get("description"),
            }
        )

    update_status(
        task_id,
        desire={
            "requested": len(ids),
            "processed": processed,
            "failed": failed,
        },
    )

    if not pending_items:
        return BatchResult(processed=processed, failed=failed, calls=0)

    for batch_index, batch in enumerate(_chunk(pending_items, settings.batch_size), start=1):
        if settings.max_calls and (calls_used + calls) >= settings.max_calls:
            note = "desire_call_limit_reached"
            update_status(task_id, notes=[note])
            break

        logger.info(
            "IA desire batch=%s processed=%s call=%s/%s",
            batch_index,
            len(batch),
            calls_used + calls + 1,
            settings.max_calls,
        )

        try:
            result_map = run_desire_batch(
                batch,
                timeout=settings.gpt_timeout,
                max_retry=settings.gpt_max_retry,
            )
            calls += 1
        except ChatCompletionError as exc:
            calls += 1
            failed += len(batch)
            logger.warning("desire_batch_failed task_id=%s error=%s", task_id, exc)
            update_status(task_id, notes=[f"desire_error:{exc}"], desire={"failed": failed, "processed": processed})
            continue
        except Exception as exc:  # pragma: no cover - defensive
            calls += 1
            failed += len(batch)
            logger.exception("desire_batch_exception task_id=%s", task_id)
            update_status(task_id, notes=["desire_exception"], desire={"failed": failed, "processed": processed})
            continue

        updates: List[tuple[str, str, int, int]] = []
        for item in batch:
            pid = int(item["id"])
            mapped = result_map.get(str(pid))
            if not mapped:
                failed += 1
                continue
            lines = [line.strip()[:90] for line in mapped.get("lines", []) if line and line.strip()]
            if not lines:
                failed += 1
                continue
            label = (mapped.get("class") or "").strip().lower()
            if label not in {"alto", "medio", "bajo"}:
                label = "medio"
            magnitude = mapped.get("magnitude")
            if magnitude is None:
                magnitude = {"alto": 80, "medio": 50, "bajo": 25}.get(label, 50)
            else:
                try:
                    magnitude = int(magnitude)
                except Exception:
                    magnitude = 50
            magnitude = max(0, min(100, int(magnitude)))
            updates.append(("\n".join(lines), label, magnitude, pid))
            processed += 1

        if updates:
            conn.executemany(
                "UPDATE products SET ai_desire=?, ai_desire_label=?, desire_magnitude=? WHERE id=?",
                updates,
            )

        update_status(
            task_id,
            desire={
                "requested": len(ids),
                "processed": processed,
                "failed": failed,
            },
        )

    return BatchResult(processed=processed, failed=failed, calls=calls)


def _process_imputacion(
    task_id: str,
    rows_by_id: Mapping[int, Mapping[str, Any]],
    ids: Sequence[int],
    settings: AutoSettings,
    conn: sqlite3.Connection,
    calls_used: int,
) -> ImputacionResult:
    processed = 0
    failed = 0
    calls = 0

    items: List[Dict[str, Any]] = []
    for pid in ids:
        row = rows_by_id.get(pid)
        if not row:
            failed += 1
            continue
        review_count = row.get("review_count")
        image_count = row.get("image_count")
        if review_count is not None and image_count is not None:
            processed += 1
            continue
        items.append(
            {
                "id": str(pid),
                "title": row.get("name"),
                "description": row.get("description"),
                "category": row.get("category"),
                "review_count": review_count,
                "image_count": image_count,
            }
        )

    if not items:
        update_status(task_id, imputacion={"processed": processed, "failed": failed})
        return BatchResult(processed=processed, failed=failed, calls=0)

    for batch_index, batch in enumerate(_chunk(items, settings.batch_size), start=1):
        if settings.max_calls and (calls_used + calls) >= settings.max_calls:
            note = "imputacion_call_limit_reached"
            update_status(task_id, notes=[note])
            break

        logger.info(
            "IA imputacion batch=%s processed=%s call=%s/%s",
            batch_index,
            len(batch),
            calls_used + calls + 1,
            settings.max_calls,
        )
        try:
            result_map = run_imputacion_batch(
                batch,
                timeout=settings.gpt_timeout,
                max_retry=settings.gpt_max_retry,
            )
            calls += 1
        except ChatCompletionError as exc:
            calls += 1
            failed += len(batch)
            logger.warning("imputacion_batch_failed task_id=%s error=%s", task_id, exc)
            update_status(task_id, notes=[f"imputacion_error:{exc}"], imputacion={"failed": failed, "processed": processed})
            continue
        except Exception:
            calls += 1
            failed += len(batch)
            logger.exception("imputacion_batch_exception task_id=%s", task_id)
            update_status(task_id, notes=["imputacion_exception"], imputacion={"failed": failed, "processed": processed})
            continue

        for item in batch:
            pid = int(item["id"])
            mapped = result_map.get(str(pid))
            if not mapped:
                failed += 1
                continue
            fields: MutableMapping[str, Any] = {}
            if "review_count" in mapped:
                fields["review_count"] = int(mapped["review_count"])
            if "image_count" in mapped:
                fields["image_count"] = int(mapped["image_count"])
            if not fields:
                failed += 1
                continue
            placeholders = ", ".join(f"{col}=?" for col in fields.keys())
            conn.execute(
                f"UPDATE products SET {placeholders} WHERE id=?",
                tuple(fields.values()) + (pid,),
            )
            processed += 1

        update_status(
            task_id,
            imputacion={"processed": processed, "failed": failed, "requested": len(ids)},
        )

    return BatchResult(processed=processed, failed=failed, calls=calls)


def _process_winner_score(
    task_id: str,
    rows_by_id: Mapping[int, Mapping[str, Any]],
    ids: Sequence[int],
    settings: AutoSettings,
    conn: sqlite3.Connection,
    calls_used: int,
    config_map: Mapping[str, Any],
) -> WinnerResult:
    processed = 0
    failed = 0
    calls = 0
    note: Optional[str] = None

    cfg_weights = config_map.get("winner_weights")
    has_weights = isinstance(cfg_weights, Mapping) and bool(cfg_weights)
    if not has_weights:
        if settings.max_calls and (calls_used + calls) >= settings.max_calls:
            note = "weights_skipped_call_limit"
            update_status(task_id, notes=[note])
        else:
            aggregates = _build_aggregates(rows_by_id, ids)
            try:
                result = run_weights_once(
                    aggregates,
                    timeout=settings.gpt_timeout,
                    max_retry=settings.gpt_max_retry,
                )
                calls += 1
            except ChatCompletionError as exc:
                calls += 1
                failed = len(ids)
                note = f"weights_error:{exc}"
                logger.warning("weights_call_failed task_id=%s error=%s", task_id, exc)
                update_status(task_id, notes=[note])
                return WinnerResult(processed=0, failed=failed, calls=calls, note=note)
            except Exception:
                calls += 1
                failed = len(ids)
                note = "weights_exception"
                logger.exception("weights_call_exception task_id=%s", task_id)
                update_status(task_id, notes=[note])
                return WinnerResult(processed=0, failed=failed, calls=calls, note=note)

            fetched_weights = result.get("weights") or {}
            fetched_order = result.get("order") or []
            set_winner_weights_raw(fetched_weights)
            if fetched_order:
                set_winner_order_raw(list(fetched_order))
            winner_calc.invalidate_weights_cache()
            logger.info("IA weights weights_loaded=%s", "new")
    else:
        logger.info("IA weights weights_loaded=%s", "from_cache")

    try:
        winner_calc.generate_winner_scores(conn, product_ids=ids)
        processed = len(ids)
    except Exception:
        failed = len(ids)
        logger.exception("winner_score_generation_failed task_id=%s", task_id)
        update_status(task_id, notes=["winner_score_failed"], winner_score={"processed": processed, "failed": failed})
        return WinnerResult(processed=processed, failed=failed, calls=calls, note=note)

    update_status(task_id, winner_score={"processed": processed, "failed": failed, "requested": len(ids)})
    return WinnerResult(processed=processed, failed=failed, calls=calls, note=note)


def _build_aggregates(
    rows_by_id: Mapping[int, Mapping[str, Any]],
    ids: Sequence[int],
) -> Dict[str, Any]:
    products: List[Dict[str, Any]] = []
    for pid in ids:
        row = rows_by_id.get(pid)
        if not row:
            continue
        data = row_to_dict(row) or {}
        extra = data.get("extra")
        if isinstance(extra, str):
            try:
                extra_data = json.loads(extra)
            except json.JSONDecodeError:
                extra_data = {}
        elif isinstance(extra, Mapping):
            extra_data = dict(extra)
        else:
            extra_data = {}
        if isinstance(extra_data, Mapping):
            data.update(extra_data)
        data.setdefault("title", data.get("name"))
        products.append(data)
    return build_weighting_aggregates(products)
