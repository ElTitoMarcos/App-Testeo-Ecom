"""Background AI post-processing runner."""
from __future__ import annotations

import json
import logging
import sqlite3
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from product_research_app.ai.ai_status import set_error, update_status
from product_research_app.database import get_connection as legacy_get_connection  # type: ignore circular
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
    reserve_desire_calls: int = 1
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
            reserve = int(cfg.get("AI_RESERVE_DESIRE_CALLS", settings.reserve_desire_calls))
        except Exception:
            reserve = settings.reserve_desire_calls
        settings.reserve_desire_calls = max(0, reserve)
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

    root_conn = get_db()
    db_path = _resolve_db_path(root_conn)
    conn = legacy_get_connection(Path(db_path))
    try:
        has_desire_summary = _table_has_column(conn, "products", "desire_summary")
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
            has_desire_summary=has_desire_summary,
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
    finally:
        try:
            conn.close()
        except Exception:
            pass

    update_status(task_id, state="DONE")


def run_desire_only(
    task_id: str,
    product_ids: Sequence[int],
    config: Optional[Mapping[str, Any]] = None,
) -> BatchResult:
    """Execute only the desire pipeline for the provided product ids."""

    config_map = dict(config or {})
    settings = AutoSettings.from_config(config_map)
    ids = _normalise_product_ids(product_ids)
    if not ids:
        update_status(task_id, desire={"requested": 0, "processed": 0, "failed": 0})
        return BatchResult(processed=0, failed=0, calls=0)

    root_conn = get_db()
    db_path = _resolve_db_path(root_conn)
    conn = legacy_get_connection(Path(db_path))
    try:
        rows = _fetch_rows(conn, ids)
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
        has_desire_summary = _table_has_column(conn, "products", "desire_summary")
        update_status(task_id, state="RUNNING")
        result = _process_desire(
            task_id,
            rows_by_id,
            ids,
            settings,
            conn,
            calls_used=0,
            has_desire_summary=has_desire_summary,
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass
    update_status(task_id, state="DONE")
    return result


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


def _chunk(seq: Sequence[Any], size: int) -> Iterable[List[Any]]:
    size = max(1, int(size))
    for idx in range(0, len(seq), size):
        yield list(seq[idx : idx + size])


def _fetch_pending_desire_ids(conn: sqlite3.Connection, ids: Sequence[int]) -> List[int]:
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    query = f"""
        SELECT id
        FROM products
        WHERE id IN ({placeholders})
          AND (
            COALESCE(ai_desire, '') = '' OR
            COALESCE(ai_desire_label, '') = '' OR
            desire_magnitude IS NULL
          )
        ORDER BY id
    """
    cur = conn.execute(query, tuple(ids))
    rows = cur.fetchall()
    pending_ids: List[int] = []
    for row in rows:
        try:
            pending_ids.append(int(row[0]))
        except Exception:
            continue
    return pending_ids


def _select_pending_desire_batch(
    conn: sqlite3.Connection,
    ids: Sequence[int],
    limit: int,
) -> List[Dict[str, Any]]:
    if not ids or limit <= 0:
        return []
    placeholders = ",".join("?" for _ in ids)
    query = f"""
        SELECT
            id,
            desire,
            name AS title,
            description,
            ai_desire,
            ai_desire_label,
            desire_magnitude
        FROM products
        WHERE id IN ({placeholders})
          AND (
            COALESCE(ai_desire, '') = '' OR
            COALESCE(ai_desire_label, '') = '' OR
            desire_magnitude IS NULL
          )
        ORDER BY id
        LIMIT ?
    """
    params = tuple(ids) + (int(limit),)
    cur = conn.execute(query, params)
    rows = cur.fetchall()
    batch: List[Dict[str, Any]] = []
    for row in rows:
        if isinstance(row, sqlite3.Row):
            batch.append(dict(row))
        else:
            batch.append(
                {
                    "id": row[0],
                    "desire": row[1] if len(row) > 1 else None,
                    "title": row[2] if len(row) > 2 else None,
                    "description": row[3] if len(row) > 3 else None,
                    "ai_desire": row[4] if len(row) > 4 else None,
                    "ai_desire_label": row[5] if len(row) > 5 else None,
                    "desire_magnitude": row[6] if len(row) > 6 else None,
                }
            )
    return batch


def _sync_desire_summary(
    conn: sqlite3.Connection,
    ids: Sequence[int],
) -> None:
    if not ids:
        return
    unique_ids = []
    seen: set[int] = set()
    for value in ids:
        try:
            num = int(value)
        except Exception:
            continue
        if num in seen:
            continue
        seen.add(num)
        unique_ids.append(num)
    if not unique_ids:
        return
    for chunk_ids in _chunk(unique_ids, 900):
        placeholders = ",".join("?" for _ in chunk_ids)
        query = f"""
            UPDATE products
            SET desire_summary = ai_desire
            WHERE id IN ({placeholders})
              AND desire_summary IS NULL
              AND ai_desire IS NOT NULL
        """
        conn.execute(query, tuple(chunk_ids))
    conn.commit()


def _process_desire(
    task_id: str,
    rows_by_id: Mapping[int, Mapping[str, Any]],
    ids: Sequence[int],
    settings: AutoSettings,
    conn: sqlite3.Connection,
    calls_used: int,
    *,
    has_desire_summary: bool,
) -> BatchResult:
    total_requested = len(ids)
    existing_ids = {int(pid) for pid in ids if pid in rows_by_id}
    missing_ids = [pid for pid in ids if pid not in rows_by_id]
    failed = len(missing_ids)
    calls = 0
    pending_ids = _fetch_pending_desire_ids(conn, ids)
    pending_count = len(pending_ids)
    processed = max(0, len(existing_ids) - pending_count)

    update_status(
        task_id,
        desire={
            "requested": total_requested,
            "processed": processed,
            "failed": failed,
        },
    )

    logger.info(
        "IA desire start: pending=%s batch_size=%s reserve_calls=%s budget=%s",
        pending_count,
        settings.batch_size,
        settings.reserve_desire_calls,
        settings.max_calls or "inf",
    )
    if pending_ids:
        logger.info("IA desire pending sample=%s", pending_ids[:10])

    if pending_count == 0:
        logger.info(
            "IA desire done: processed=%s failed=%s calls_used=%s",
            processed,
            failed,
            calls_used,
        )
        return BatchResult(processed=processed, failed=failed, calls=0)

    remaining_ids = list(pending_ids)
    touched_ids: List[int] = []
    batch_index = 0

    while remaining_ids:
        if settings.max_calls and (calls_used + calls) >= settings.max_calls:
            if (calls_used + calls) < settings.reserve_desire_calls:
                logger.info(
                    "IA desire reserve_call forcing batch remaining=%s",
                    len(remaining_ids),
                )
            else:
                note = "desire_call_limit_reached"
                logger.info(
                    "IA desire stop: call_budget_exhausted calls_used=%s/%s remaining=%s",
                    calls_used + calls,
                    settings.max_calls,
                    len(remaining_ids),
                )
                update_status(task_id, notes=[note])
                break

        batch_rows = _select_pending_desire_batch(conn, remaining_ids, settings.batch_size)
        if not batch_rows:
            break

        batch_index += 1
        batch_ids: List[int] = []
        items: List[Dict[str, Any]] = []
        for row in batch_rows:
            pid = row.get("id")
            if pid is None:
                continue
            try:
                pid_int = int(pid)
            except Exception:
                continue
            batch_ids.append(pid_int)
            items.append(
                {
                    "id": str(pid_int),
                    "desire": row.get("desire"),
                    "title": row.get("title"),
                    "description": row.get("description"),
                }
            )

        if not items:
            remaining_ids = [pid for pid in remaining_ids if pid not in set(batch_ids)]
            continue

        logger.info(
            "IA desire batch=%s send=%s ids=%s",
            batch_index,
            len(items),
            batch_ids[:10],
        )

        try:
            result_map = run_desire_batch(
                items,
                timeout=settings.gpt_timeout,
                max_retry=settings.gpt_max_retry,
            )
            calls += 1
        except ChatCompletionError as exc:
            calls += 1
            handled = set(batch_ids)
            remaining_ids = [pid for pid in remaining_ids if pid not in handled]
            failed += len(handled)
            raw_message = str(exc).strip()
            if "OPENAI_API_KEY" in raw_message or "not configured" in raw_message.lower():
                note = "no_api_key"
                logger.warning(
                    "desire_batch_failed task_id=%s batch=%s error=%s",
                    task_id,
                    batch_index,
                    exc,
                )
                set_error(task_id, note)
                update_status(
                    task_id,
                    notes=[note],
                    desire={
                        "requested": total_requested,
                        "processed": processed,
                        "failed": failed,
                    },
                )
            else:
                note = f"desire_error:{raw_message}" if raw_message else "desire_error"
                logger.warning(
                    "desire_batch_failed task_id=%s batch=%s error=%s",
                    task_id,
                    batch_index,
                    exc,
                )
                update_status(
                    task_id,
                    notes=[note],
                    desire={
                        "requested": total_requested,
                        "processed": processed,
                        "failed": failed,
                    },
                )
            continue
        except Exception as exc:  # pragma: no cover - defensive
            calls += 1
            handled = set(batch_ids)
            remaining_ids = [pid for pid in remaining_ids if pid not in handled]
            failed += len(handled)
            logger.exception("desire_batch_exception task_id=%s batch=%s", task_id, batch_index)
            trace_tail = _format_trace(exc)
            set_error(task_id, "desire_exception", trace_tail)
            update_status(
                task_id,
                notes=["desire_exception"],
                desire={
                    "requested": total_requested,
                    "processed": processed,
                    "failed": failed,
                },
            )
            continue

        handled = set(batch_ids)
        remaining_ids = [pid for pid in remaining_ids if pid not in handled]

        raw_preview = getattr(result_map, "raw_response", None)
        if not result_map:
            failed += len(batch_ids)
            preview = (raw_preview or "")[:300]
            logger.warning(
                "desire_empty_response task_id=%s batch=%s size=%s preview=%s",
                task_id,
                batch_index,
                len(items),
                preview,
            )
            update_status(
                task_id,
                notes=["desire_empty_response"],
                desire={
                    "requested": total_requested,
                    "processed": processed,
                    "failed": failed,
                },
            )
            continue

        logger.info(
            "IA desire batch=%s received=%s ids=%s",
            batch_index,
            len(result_map),
            list(result_map.keys())[:10],
        )

        received_ids: set[int] = set()
        for key in result_map.keys():
            try:
                received_ids.add(int(key))
            except Exception:
                continue

        updates: List[tuple[str, str, int, int]] = []
        for pid in batch_ids:
            mapped = result_map.get(str(pid))
            if not isinstance(mapped, Mapping):
                continue
            text = str(mapped.get("text") or "").strip()
            label = str(mapped.get("label") or "").strip().lower()
            magnitude_val = mapped.get("magnitude")
            try:
                magnitude_int = int(magnitude_val)
            except Exception:
                magnitude_int = None
            if not text or label not in {"alto", "medio", "bajo"} or magnitude_int is None:
                continue
            magnitude_int = max(0, min(100, magnitude_int))
            updates.append((text, label, magnitude_int, pid))

        batch_success = len(updates)
        batch_failed = max(0, len(batch_ids) - batch_success)
        failed += batch_failed

        if batch_failed:
            valid_ids = {row[3] for row in updates}
            missing_in_response = [pid for pid in batch_ids if pid not in received_ids]
            invalid_in_response = [
                pid for pid in received_ids if pid in batch_ids and pid not in valid_ids
            ]
            logger.warning(
                "IA desire batch=%s missing_ids=%s invalid_ids=%s",
                batch_index,
                missing_in_response[:10],
                invalid_in_response[:10],
            )

        if updates:
            conn.executemany(
                "UPDATE products SET ai_desire=?, ai_desire_label=?, desire_magnitude=? WHERE id=?",
                updates,
            )
            conn.commit()
            processed += batch_success
            touched_ids.extend(int(row[3]) for row in updates)
            logger.info(
                "IA desire updated=%s batch=%s calls_used=%s/%s",
                batch_success,
                batch_index,
                calls_used + calls,
                settings.max_calls or "inf",
            )
        else:
            logger.warning(
                "IA desire batch=%s produced no valid updates task_id=%s",
                batch_index,
                task_id,
            )

        update_status(
            task_id,
            desire={
                "requested": total_requested,
                "processed": processed,
                "failed": failed,
            },
        )

    if has_desire_summary and touched_ids:
        try:
            _sync_desire_summary(conn, touched_ids)
        except Exception:
            logger.exception("desire_summary_sync_failed task_id=%s", task_id)

    logger.info(
        "IA desire done: processed=%s failed=%s calls_used=%s",
        processed,
        failed,
        calls_used + calls,
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
            "IA imputacion batch=%s size=%s processed_total=%s failed_total=%s call=%s/%s",
            batch_index,
            len(batch),
            processed,
            failed,
            calls_used + calls + 1,
            settings.max_calls or "inf",
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
            set_error(task_id, f"imputacion_error:{exc}")
            update_status(task_id, notes=[f"imputacion_error:{exc}"], imputacion={"failed": failed, "processed": processed})
            continue
        except Exception as exc:
            calls += 1
            failed += len(batch)
            logger.exception("imputacion_batch_exception task_id=%s", task_id)
            trace_tail = _format_trace(exc)
            set_error(task_id, "imputacion_exception", trace_tail)
            update_status(task_id, notes=["imputacion_exception"], imputacion={"failed": failed, "processed": processed})
            continue

        for item in batch:
            pid = int(item["id"])
            mapped = result_map.get(str(pid))
            if not mapped:
                failed += 1
                continue
            fields: MutableMapping[str, Any] = {}
            if "review_count" in mapped and mapped["review_count"] not in (None, ""):
                fields["review_count"] = int(mapped["review_count"])
            if "image_count" in mapped and mapped["image_count"] not in (None, ""):
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
        try:
            conn.commit()
        except Exception:
            pass

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
                set_error(task_id, note)
                update_status(task_id, notes=[note])
                return WinnerResult(processed=0, failed=failed, calls=calls, note=note)
            except Exception as exc:
                calls += 1
                failed = len(ids)
                note = "weights_exception"
                logger.exception("weights_call_exception task_id=%s", task_id)
                trace_tail = _format_trace(exc)
                set_error(task_id, note, trace_tail)
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
    except Exception as exc:
        failed = len(ids)
        logger.exception("winner_score_generation_failed task_id=%s", task_id)
        trace_tail = _format_trace(exc)
        set_error(task_id, "winner_score_failed", trace_tail)
        update_status(task_id, notes=["winner_score_failed"], winner_score={"processed": processed, "failed": failed})
        return WinnerResult(processed=processed, failed=failed, calls=calls, note=note)

    logger.info(
        "IA winner_score processed=%s failed=%s calls=%s/%s",
        processed,
        failed,
        calls_used + calls,
        settings.max_calls or "inf",
    )
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


def _table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


def _format_trace(exc: BaseException) -> str:
    lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    tail = [line.rstrip("\n") for line in lines][-40:]
    return "\n".join(tail)


def _resolve_db_path(conn: sqlite3.Connection) -> str:
    try:
        rows = conn.execute("PRAGMA database_list").fetchall()
        for _, name, path in rows:
            if name == "main" and path:
                return path
    except Exception:
        pass
    return "product_research_app/data.sqlite3"
