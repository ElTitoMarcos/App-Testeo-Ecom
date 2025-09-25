from __future__ import annotations

import json
import logging
from typing import Dict, Iterable, List, Optional, Sequence

from .. import database
from . import ai_columns
from .desire_utils import looks_like_product_desc

logger = logging.getLogger(__name__)


def _rowdict(row):
    """Devuelve un dict normal a partir de sqlite3.Row o de un dict."""
    if row is None:
        return {}
    if isinstance(row, dict):
        return row
    try:
        return {k: row[k] for k in row.keys()}
    except Exception:
        # último recurso: convertir por iteración
        return dict(row)


REQUIRED_FIELDS: Dict[str, Dict[str, str]] = {
    "desire": {"via": "DESIRE", "kind": "text", "min_len": 280},
    "desire_primary": {"via": "DESIRE", "kind": "enum"},
    "desire_magnitude": {"via": "DESIRE", "kind": "object"},
    "awareness_level": {"via": "DESIRE", "kind": "enum"},
    "competition_level": {"via": "DESIRE", "kind": "enum"},
    "ai_desire_label": {"via": "DERIVED", "kind": "text"},
}


def _normalize_ids(ids: Optional[Sequence[int]]) -> Optional[List[int]]:
    if ids is None:
        return None
    seen: set[int] = set()
    result: List[int] = []
    for raw in ids:
        try:
            num = int(raw)
        except Exception:
            continue
        if num in seen:
            continue
        seen.add(num)
        result.append(num)
    result.sort()
    return result


def _coerce_json(value):
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text in {"null", "None"}:
            return None
        if text.startswith("{") or text.startswith("["):
            try:
                return json.loads(text)
            except Exception:
                return value
    return value


def _is_missing(value, kind: str) -> bool:
    if kind == "text":
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        return False
    if kind == "enum":
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        return False
    if kind == "object":
        if value in (None, "", {}):
            return True
        parsed = _coerce_json(value)
        if parsed in (None, "", {}):
            return True
        if isinstance(parsed, dict):
            if not parsed:
                return True
            overall = parsed.get("overall")
            if isinstance(overall, str):
                return not overall.strip()
            return overall in (None, "", {})
        return False
    return value is None


def needs_fill(row) -> bool:
    data = dict(row)
    desire_raw = data.get("desire")
    desire_txt = (desire_raw or "").strip()
    if not desire_txt or len(desire_txt) < REQUIRED_FIELDS["desire"]["min_len"]:
        return True
    title = (data.get("name") or "").strip()
    try:
        if looks_like_product_desc(desire_txt, title):
            return True
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("looks_like_product_desc failed id=%s", data.get("id"))
    for key, meta in REQUIRED_FIELDS.items():
        if key == "desire":
            continue
        if _is_missing(data.get(key), meta["kind"]):
            return True
    return False


def scan_ids(conn, ids: Optional[Sequence[int]] = None) -> List[int]:
    target_ids = _normalize_ids(ids)
    missing: List[int] = []
    for row in database.iter_products(conn, target_ids):
        try:
            pid = int(row["id"])
        except Exception:
            continue
        if needs_fill(row):
            missing.append(pid)
    return missing


def run_fill_for_ids(
    conn,
    ids: Sequence[int],
    *,
    reason: str = "audit",
) -> Dict[str, int]:
    normalized = _normalize_ids(ids) or []
    metrics = {"queued": len(normalized), "ok": 0, "ko": 0, "retried": 0}
    if not normalized:
        return metrics
    result = ai_columns.run_ai_fill_job(
        0,
        normalized,
        status_cb=None,
        reason=reason,
        commit_each=True,
    )
    counts = result.get("counts", {}) or {}
    metrics["queued"] = int(counts.get("queued", len(normalized)))
    metrics["ok"] = int(counts.get("ok", 0) + counts.get("cached", 0))
    metrics["ko"] = int(counts.get("ko", 0))
    metrics["retried"] = int(counts.get("retried", 0))
    missing_after = scan_ids(conn, normalized)
    logger.info(
        "ai_audit: ids=%s missing=%d ok=%d ko=%d retried=%d",
        normalized,
        len(missing_after),
        metrics["ok"],
        metrics["ko"],
        metrics["retried"],
    )
    return metrics


def _first_words(text, n=8):
    import re

    if not text:
        return ""
    words = re.findall(r"\w[\w'-]*", str(text))
    return " ".join(words[:n])


def _fill_ai_desire_labels(conn, ids: Optional[Iterable[int]]):
    """
    Rellena ai_desire_label cuando está vacío.
    Preferencia: desire_primary; si no existe, primeras 8 palabras de desire.
    Emite SSE por cada actualización.
    """

    target_ids = None if ids is None else list(ids)
    if target_ids is not None and not target_ids:
        return {"updated": 0, "skipped": 0}

    if target_ids is None:
        query = (
            "SELECT id, ai_desire_label, desire_primary, desire FROM products "
            "WHERE ai_desire_label IS NULL OR TRIM(ai_desire_label) = ''"
        )
        params: Sequence[int] = ()
    else:
        placeholders = ",".join("?" for _ in target_ids)
        query = (
            f"SELECT id, ai_desire_label, desire_primary, desire FROM products "
            f"WHERE id IN ({placeholders}) "
            "AND (ai_desire_label IS NULL OR TRIM(ai_desire_label) = '')"
        )
        params = target_ids

    rows = conn.execute(query, params).fetchall()

    updated = 0
    skipped = 0

    for r in rows:
        row = _rowdict(r)
        pid = row["id"]
        label = (row.get("ai_desire_label") or "").strip()
        if label:
            skipped += 1
            continue

        candidate = (row.get("desire_primary") or "").strip()
        if not candidate:
            candidate = _first_words(row.get("desire") or "", 8)

        candidate = candidate.strip()
        if not candidate:
            skipped += 1
            continue

        conn.execute(
            "UPDATE products SET ai_desire_label = ? WHERE id = ?",
            (candidate, pid),
        )
        try:
            conn.commit()
        except Exception:
            logger.exception("commit failed while setting ai_desire_label pid=%s", pid)
            skipped += 1
            continue
        updated += 1

        try:
            from product_research_app.web_app import sse_broadcast

            sse_broadcast("product_update", {"id": pid, "fields": {"ai_desire_label": candidate}})
        except Exception:
            pass

        try:
            ai_columns.emit_update(pid, {"ai_desire_label": candidate}, reason="audit")
        except Exception:
            pass

    return {"updated": updated, "skipped": skipped}


def run_audit(conn, ids: Optional[Sequence[int]] = None) -> Dict[str, int]:
    target_ids = _normalize_ids(ids)
    label_metrics = _fill_ai_desire_labels(conn, target_ids)
    missing = scan_ids(conn, target_ids)
    metrics = {"queued": len(missing), "ok": 0, "ko": 0, "retried": 0}
    if missing:
        metrics = run_fill_for_ids(conn, missing, reason="audit")
        extra_labels = _fill_ai_desire_labels(conn, missing)
        label_metrics["updated"] += extra_labels.get("updated", 0)
        label_metrics["skipped"] += extra_labels.get("skipped", 0)
    summary = {
        **metrics,
        "missing": len(missing),
        "label_updates": label_metrics.get("updated", 0),
        "label_skipped": label_metrics.get("skipped", 0),
    }
    logger.info(
        "ai_audit: ids=%s missing=%d ok=%d ko=%d retried=%d labels=%d",
        "*" if target_ids is None else len(target_ids),
        len(missing),
        summary["ok"],
        summary["ko"],
        summary["retried"],
        summary["label_updates"],
    )
    return summary
