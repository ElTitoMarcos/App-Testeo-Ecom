from __future__ import annotations

import csv
import io
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    import pandas as pd
except Exception:  # pragma: no cover - gracefully handle missing pandas
    pd = None  # type: ignore[assignment]

from . import importer_fast

StatusCallback = Callable[..., None]


def _safe_emit(cb: StatusCallback, **kwargs: Any) -> None:
    try:
        cb(**kwargs)
    except Exception:
        # Status callbacks must never break the import flow.
        pass


def import_csv(bytes_data: bytes, *, source: str, status_cb: StatusCallback) -> List[Dict[str, Any]]:
    """Parse CSV bytes into a list of dictionaries."""
    _safe_emit(status_cb, stage="parse_csv", done=0, total=0)
    text = bytes_data.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    records: List[Dict[str, Any]] = []
    for idx, row in enumerate(reader, start=1):
        records.append(dict(row))
        if idx % 500 == 0:
            _safe_emit(status_cb, stage="parse_csv", done=idx, total=0)
    total = len(records)
    _safe_emit(status_cb, stage="parse_csv", done=total, total=total)
    return records


def import_xlsx(bytes_data: bytes, *, source: str, status_cb: StatusCallback) -> List[Dict[str, Any]]:
    """Parse XLSX bytes into a list of dictionaries using pandas."""
    if pd is None:
        _safe_emit(status_cb, stage="parse_xlsx", done=0, total=0)
        raise RuntimeError("pandas is required for XLSX imports")
    _safe_emit(status_cb, stage="parse_xlsx", done=0, total=0)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Workbook contains no default style, apply openpyxl's default",
            category=UserWarning,
            module="openpyxl.styles.stylesheet",
        )
        df = pd.read_excel(io.BytesIO(bytes_data), dtype=str)
    if df.empty:
        _safe_emit(status_cb, stage="parse_xlsx", done=0, total=0)
        return []
    df = df.where(pd.notnull(df), None)
    records_raw = df.to_dict("records")
    total = len(records_raw)
    records: List[Dict[str, Any]] = []
    for idx, row in enumerate(records_raw, start=1):
        records.append(dict(row))
        if idx % 500 == 0 or idx == total:
            _safe_emit(status_cb, stage="parse_xlsx", done=idx, total=total)
    if total == 0:
        _safe_emit(status_cb, stage="parse_xlsx", done=0, total=0)
    return records


def import_records(records: List[Dict[str, Any]], *, status_cb: StatusCallback) -> int:
    """Bulk insert the prepared records using the fast importer."""

    def _wrapped_status_cb(**kwargs: Any) -> None:
        stage = kwargs.get("stage")
        if stage == "prepare":
            kwargs["stage"] = "db_bulk_prepare"
        elif stage == "insert":
            kwargs["stage"] = "db_bulk_insert"
        elif stage == "commit":
            kwargs["stage"] = "db_bulk_commit"
        _safe_emit(status_cb, **kwargs)

    return importer_fast.fast_import_records(records, status_cb=_wrapped_status_cb)


def run_import(
    file_bytes: bytes,
    filename: str,
    status_cb: StatusCallback,
    *,
    import_token: Optional[str] = None,
) -> int:
    """Dispatch CSV/XLSX imports and return the number of inserted rows."""
    ext = Path(filename).suffix.lower()
    source = filename
    if ext == ".csv":
        records = import_csv(file_bytes, source=source, status_cb=status_cb)
    elif ext in {".xlsx", ".xls"}:
        records = import_xlsx(file_bytes, source=source, status_cb=status_cb)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    for record in records:
        record.setdefault("winner_score", "0")
        if source:
            record.setdefault("source", source)
        if import_token:
            # The token is stored temporarily in ``extra`` so we can collect the
            # product IDs that were upserted during this batch without running
            # any synchronous GPT work.  See ``consume_import_token`` in the
            # database layer for cleanup.
            record.setdefault("_import_token", import_token)

    if not records:
        return 0

    return import_records(records, status_cb=status_cb)


__all__ = [
    "import_csv",
    "import_xlsx",
    "import_records",
    "run_import",
]
