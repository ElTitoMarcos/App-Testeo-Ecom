import os
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .. import config, database, gpt

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent.parent
DB_PATH = APP_DIR / "data.sqlite3"
BATCH_SIZE = 10


def _ensure_conn():
    conn = database.get_connection(DB_PATH)
    database.initialize_database(conn)
    return conn


def fill_ai_columns(
    product_ids: List[int],
    *,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Fill AI-driven columns for the given products.

    Args:
        product_ids: IDs of products to process.
        progress_cb: Optional callback(progress, total) for progress updates.
        timeout: Max seconds to spend overall.

    Returns:
        Dict with keys ``ok`` (updates), ``ko`` (errors) and ``counts``.
    """
    start_time = time.time()
    conn = _ensure_conn()
    total_requested = len(product_ids)
    skipped = 0
    to_process: List[Dict[str, Any]] = []
    records: Dict[str, Any] = {}

    for pid in product_ids:
        rec = database.get_product(conn, pid)
        if not rec:
            continue
        if rec["desire"] or rec["desire_magnitude"] or rec["awareness_level"] or rec["competition_level"]:
            skipped += 1
            continue
        try:
            extra = json.loads(rec["extra"]) if rec["extra"] else {}
        except Exception:
            extra = {}
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

    if not to_process:
        logger.info("fill_ai_columns: n_importados=%s n_con_ia=0 n_omitidos=%s errores=0", total_requested, skipped)
        return {"ok": {}, "ko": {}, "counts": {"n_importados": total_requested, "n_con_ia": 0, "n_omitidos_por_valor_existente": skipped, "errores": 0}}

    api_key = config.get_api_key() or os.environ.get("OPENAI_API_KEY")
    model = config.get_model()
    if not api_key:
        err_msg = "missing_api_key"
        logger.info(
            "fill_ai_columns: n_importados=%s n_con_ia=0 n_omitidos=%s errores=%s", total_requested, skipped, len(to_process)
        )
        return {
            "ok": {},
            "ko": {str(it["id"]): err_msg for it in to_process},
            "error": err_msg,
            "counts": {
                "n_importados": total_requested,
                "n_con_ia": 0,
                "n_omitidos_por_valor_existente": skipped,
                "errores": len(to_process),
            },
        }

    ok_all: Dict[str, Dict[str, Any]] = {}
    ko_all: Dict[str, str] = {}
    processed = 0
    success = 0
    errors = 0
    total = len(to_process)

    for i in range(0, total, BATCH_SIZE):
        if time.time() - start_time > timeout:
            break
        batch = to_process[i : i + BATCH_SIZE]
        items = batch
        try:
            ok, ko, usage, duration = gpt.generate_batch_columns(api_key, model, items)
        except gpt.OpenAIError as exc:
            msg = str(exc)
            if ("status 429" in msg or "status 5" in msg) and time.time() - start_time < timeout:
                time.sleep(1.5)
                try:
                    ok, ko, usage, duration = gpt.generate_batch_columns(api_key, model, items)
                except Exception as exc2:
                    msg = str(exc2)
                    ok, ko = {}, {str(it["id"]): msg for it in batch}
            else:
                ok, ko = {}, {str(it["id"]): msg for it in batch}
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
                database.update_product(conn, int(pid), **apply)
                ok_all[str(pid)] = apply
                success += 1
            else:
                ko_all[str(pid)] = "existing"
                errors += 1
        for pid, reason in ko.items():
            ko_all[str(pid)] = reason
            errors += 1
        processed += len(batch)
        if progress_cb:
            try:
                progress_cb(processed, total)
            except Exception:
                pass

    conn.commit()
    logger.info(
        "fill_ai_columns: n_importados=%s n_con_ia=%s n_omitidos=%s errores=%s",
        total_requested,
        success,
        skipped,
        errors,
    )
    return {
        "ok": ok_all,
        "ko": ko_all,
        "counts": {
            "n_importados": total_requested,
            "n_con_ia": success,
            "n_omitidos_por_valor_existente": skipped,
            "errores": errors,
        },
    }
