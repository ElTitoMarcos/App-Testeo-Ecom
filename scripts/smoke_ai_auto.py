#!/usr/bin/env python3
"""Smoke test for automatic AI enrichment after bulk import.

The script uploads a generated XLSX file to the running web server, polls the
import and AI status endpoints, and finally verifies that the imported rows have
been populated with ``ai_desire``, ``desire_magnitude`` and ``winner_score``.

Usage:
    python scripts/smoke_ai_auto.py --base-url http://127.0.0.1:8000 \
        --rows 100 --db-path product_research_app/data.sqlite3

Prerequisites:
    * Run the application server (``python -m product_research_app.web_app``).
    * Configure ``OPENAI_API_KEY`` or set it through the UI before executing the
      smoke test so the background runner can call the AI provider.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Iterable, Tuple

import requests
from openpyxl import Workbook

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_ROWS = 100
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "product_research_app" / "data.sqlite3"


def _build_workbook(target: Path, rows: int) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Products"
    headers = [
        "Name",
        "Description",
        "Category",
        "Price",
        "Currency",
        "Image URL",
        "Date Range",
        "Units Sold",
        "Revenue",
    ]
    ws.append(headers)
    for idx in range(1, rows + 1):
        ws.append(
            [
                f"Producto {idx}",
                f"Descripción de prueba {idx}",
                "Accesorios",
                19.99 + idx,
                "USD",
                f"https://example.com/image/{idx}.jpg",
                "2024-01-01 ~ 2024-01-31",
                100 + idx,
                5000 + (idx * 10),
            ]
        )
    wb.save(target)


def _post_upload(base_url: str, file_path: Path) -> str:
    files = {
        "file": (
            file_path.name,
            file_path.read_bytes(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    }
    resp = requests.post(f"{base_url}/upload", files=files, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    task_id = str(payload.get("task_id"))
    if not task_id:
        raise RuntimeError(f"Upload did not return a task_id: {payload}")
    return task_id


def _poll_import_status(base_url: str, task_id: str, timeout: float = 300.0) -> Dict[str, object]:
    deadline = time.time() + timeout
    last_payload: Dict[str, object] | None = None
    while time.time() < deadline:
        resp = requests.get(f"{base_url}/_import_status", params={"task_id": task_id}, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        last_payload = payload
        state = str(payload.get("state") or payload.get("status") or "").lower()
        if state in {"done", "finished"}:
            return payload
        if state in {"error", "failed"}:
            raise RuntimeError(f"Import failed: {payload}")
        time.sleep(2.0)
    raise TimeoutError(f"Timed out waiting for import status; last payload={last_payload}")


def _poll_ai_status(base_url: str, task_id: str, timeout: float = 600.0) -> Dict[str, object]:
    deadline = time.time() + timeout
    last_payload: Dict[str, object] | None = None
    while time.time() < deadline:
        resp = requests.get(f"{base_url}/_ai_status", params={"task_id": task_id}, timeout=15)
        if resp.status_code == 404:
            time.sleep(2.0)
            continue
        resp.raise_for_status()
        payload = resp.json()
        last_payload = payload
        state = str(payload.get("state") or "").upper()
        if state in {"DONE", "ERROR"}:
            return payload
        time.sleep(min(max(float(payload.get("poll_interval_ms", 2500)) / 1000.0, 2.0), 4.0))
    raise TimeoutError(f"Timed out waiting for AI status; last payload={last_payload}")


def _fetch_new_products(db_path: Path, id_threshold: int) -> Iterable[Tuple[int, str, int, int]]:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT id, COALESCE(ai_desire,''), COALESCE(desire_magnitude,-1), COALESCE(winner_score,-1) "
            "FROM products WHERE id > ? ORDER BY id ASC",
            (id_threshold,),
        )
        yield from cur.fetchall()
    finally:
        conn.close()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Smoke test for automatic AI runner")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL of the running web app")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="Number of rows to generate in the XLSX upload")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to the SQLite database used by the application",
    )
    parser.add_argument("--min-ok", type=int, default=10, help="Minimum rows that must have AI data to pass")
    args = parser.parse_args(argv)

    db_path = args.db_path
    if not db_path.exists():
        raise FileNotFoundError(f"Database path not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute("SELECT COALESCE(MAX(id), 0) FROM products")
        before_max = int(cur.fetchone()[0] or 0)
    finally:
        conn.close()

    with NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _build_workbook(tmp_path, max(args.rows, args.min_ok))
        print(f"Uploading {tmp_path} to {args.base_url}...", flush=True)
        task_id = _post_upload(args.base_url, tmp_path)
        print(f"Import task id: {task_id}", flush=True)

        import_payload = _poll_import_status(args.base_url, task_id)
        print(f"Import completed: {import_payload}", flush=True)

        ai_payload = _poll_ai_status(args.base_url, task_id)
        print(f"AI status final: {ai_payload}", flush=True)

        rows = list(_fetch_new_products(db_path, before_max))
        ok_rows = [row for row in rows if row[1].strip() and row[2] >= 0 and row[3] >= 0]
        print(f"Imported rows inspected: {len(rows)}; rows with AI data: {len(ok_rows)}", flush=True)

        if len(ok_rows) < args.min_ok:
            raise AssertionError(
                f"Expected at least {args.min_ok} rows with AI data, found {len(ok_rows)}"
            )

        print("Smoke test completed successfully.")
        return 0
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
