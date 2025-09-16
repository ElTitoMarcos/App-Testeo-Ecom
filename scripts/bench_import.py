#!/usr/bin/env python3
"""Benchmark the product import pipeline using synthetic CSV data."""
from __future__ import annotations

import argparse
import csv
import io
import random
import string
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Tuple

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:5000"
DEFAULT_ROWS = 10_000
CSV_FILENAME = "synthetic_products.csv"
PHASE_KEY = "phases"


def _random_words(rng: random.Random, count: int) -> str:
    alphabet = string.ascii_lowercase
    return " ".join(
        "".join(rng.choice(alphabet) for _ in range(rng.randint(4, 10))).title()
        for _ in range(count)
    )


def generate_csv_bytes(rows: int, seed: int | None = None) -> bytes:
    """Create a UTF-8 CSV payload with deterministic pseudo-random data."""

    rng = random.Random(seed)
    fieldnames = [
        "id",
        "title",
        "price",
        "units_sold",
        "revenue",
        "rating",
        "desire",
        "competition",
        "oldness",
        "awareness",
        "category",
        "store",
        "description",
        "dateAdded",
    ]

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()

    base_date = datetime.utcnow()
    for index in range(rows):
        product_id = 1_000_000 + index
        price = round(rng.uniform(5.0, 250.0), 2)
        units = rng.randint(1, 750)
        revenue = round(price * units, 2)
        rating = round(rng.uniform(1.0, 5.0), 2)
        oldness = rng.randint(1, 72)
        awareness = round(rng.uniform(0.0, 100.0), 2)
        offset_days = rng.randint(0, 365)
        date_added = base_date - timedelta(days=offset_days)

        writer.writerow(
            {
                "id": product_id,
                "title": _random_words(rng, 3),
                "price": price,
                "units_sold": units,
                "revenue": revenue,
                "rating": rating,
                "desire": rng.choice(["high", "medium", "low"]),
                "competition": rng.choice(["low", "medium", "high"]),
                "oldness": oldness,
                "awareness": awareness,
                "category": rng.choice(["Electronics", "Home", "Sports", "Toys"]),
                "store": f"Store {rng.randint(1, 100)}",
                "description": _random_words(rng, 12),
                "dateAdded": date_added.strftime("%Y-%m-%d"),
            }
        )

    return buffer.getvalue().encode("utf-8")


def _post_upload(base_url: str, csv_bytes: bytes) -> Tuple[str, float]:
    url = base_url.rstrip("/") + "/upload"
    files = {
        "file": (CSV_FILENAME, io.BytesIO(csv_bytes), "text/csv"),
    }

    start = time.perf_counter()
    response = requests.post(url, files=files, timeout=60)
    elapsed = time.perf_counter() - start

    if response.status_code != 202:
        raise RuntimeError(
            f"Upload failed ({response.status_code}): {response.text.strip()}"
        )

    data = response.json()
    task_id = data.get("task_id")
    if not task_id:
        raise RuntimeError("Upload response missing task_id")

    return task_id, elapsed


def _poll_status(base_url: str, task_id: str, poll_interval: float) -> Dict[str, object]:
    url = base_url.rstrip("/") + "/_import_status"
    attempts = 0
    while True:
        response = requests.get(url, params={"task_id": task_id}, timeout=30)
        response.raise_for_status()
        status = response.json()
        state = str(status.get("state", "")).upper()
        if state in {"DONE", "ERROR"}:
            return status
        attempts += 1
        # Be gentle with the server on longer runs.
        time.sleep(min(poll_interval + attempts * 0.1, 2.0))


def _rows_processed(status: Dict[str, object], fallback: int) -> int:
    for key in ("processed", "total", "row_count"):
        try:
            value = int(status.get(key) or 0)
        except Exception:
            value = 0
        if value:
            return value
    return fallback


def benchmark(base_url: str, rows: int, seed: int | None) -> None:
    csv_bytes = generate_csv_bytes(rows, seed=seed)
    print(f"Generated CSV with {rows} rows ({len(csv_bytes)} bytes)")

    overall_start = time.perf_counter()
    task_id, upload_time = _post_upload(base_url, csv_bytes)
    print(f"Upload accepted with task_id={task_id} ({upload_time:.3f}s)")

    status = _poll_status(base_url, task_id, poll_interval=0.5)
    total_elapsed = time.perf_counter() - overall_start

    state = status.get("state")
    if state != "DONE":
        raise RuntimeError(f"Import finished in unexpected state: {state}")

    processed_rows = _rows_processed(status, rows)
    rows_per_second = processed_rows / total_elapsed if total_elapsed else 0.0

    print("\nBenchmark results")
    print("-----------------")
    print(f"Total elapsed time: {total_elapsed:.3f}s")
    if status.get("total_ms"):
        try:
            print(f"Backend reported time: {int(status['total_ms']) / 1000:.3f}s")
        except Exception:
            pass
    print(f"Rows processed: {processed_rows}")
    print(f"Throughput: {rows_per_second:,.0f} rows/s")

    phases: Iterable[Dict[str, object]] = status.get(PHASE_KEY) or []
    if phases:
        print("\nPhases:")
        for phase in phases:
            name = phase.get("name") or "unknown"
            try:
                duration = int(phase.get("ms", 0))
            except Exception:
                duration = 0
            print(f"  - {name}: {duration} ms")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic CSV and benchmark the /upload import pipeline. "
            "Ensure the Flask app is running before executing the script."
        )
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL of the running product research backend (default: %(default)s)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_ROWS,
        help="Number of synthetic rows to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic CSV generation (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    benchmark(args.base_url, args.rows, seed=args.seed)


if __name__ == "__main__":
    main()
