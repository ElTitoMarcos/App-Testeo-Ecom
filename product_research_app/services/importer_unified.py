from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Callable, Dict, List, Any

import pandas as pd

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
    _safe_emit(status_cb, stage="parse_xlsx", done=0, total=0)
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


def run_import(file_bytes: bytes, filename: str, status_cb: StatusCallback) -> int:
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

    if not records:
        return 0

    return import_records(records, status_cb=status_cb)


__all__ = [
    "import_csv",
    "import_xlsx",
    "import_records",
    "run_import",
]
