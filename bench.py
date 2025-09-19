#!/usr/bin/env python3
"""Run import/enrichment benchmarks and persist the results in the database."""

from __future__ import annotations

import argparse
import json
import csv
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

from product_research_app import database, product_enrichment
from product_research_app.db import get_db
from product_research_app.services.importer_fast import fast_import


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Existing CSV file to import")
    parser.add_argument("--rows", type=int, default=10_000, help="Synthetic rows to generate if CSV not provided")
    parser.add_argument("--batch-size", type=int, default=2000, help="Importer batch size")
    parser.add_argument("--skip-enrich", action="store_true", help="Skip enrichment phase")
    parser.add_argument(
        "--no-student",
        dest="student",
        action="store_false",
        default=True,
        help="Disable student model during benchmark",
    )
    parser.add_argument(
        "--no-similarity",
        dest="similarity",
        action="store_false",
        default=True,
        help="Disable similarity reuse during benchmark",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic data")
    parser.add_argument("--db", type=Path, default=Path("product_research_app/data.sqlite3"), help="Path to SQLite database")
    return parser.parse_args()


def _ensure_csv(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.csv and args.csv.exists():
        return args.csv, False
    random.seed(args.seed)
    categories = ["home", "beauty", "fitness", "kitchen", "electronics", "outdoors", "pets"]
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "title",
                "description",
                "category",
                "brand",
                "price",
                "rating",
                "units_sold",
                "launch_date",
                "asin",
                "url",
            ],
        )
        writer.writeheader()
        for idx in range(args.rows):
            category = random.choice(categories)
            brand = f"Benchmark{idx % 50:02d}"
            base_price = random.uniform(12.0, 90.0)
            rating = min(5.0, max(1.0, random.gauss(4.2, 0.4)))
            units = max(0, int(random.gauss(500, 120)))
            launch = time.time() - random.randint(0, 720) * 86400
            asin = f"BENCH{idx:08d}"
            writer.writerow(
                {
                    "title": f"{category.title()} Product {idx:05d}",
                    "description": f"Synthetic {category} item {idx} for benchmarking.",
                    "category": category,
                    "brand": brand,
                    "price": f"{base_price:.2f}",
                    "rating": f"{rating:.2f}",
                    "units_sold": str(units),
                    "launch_date": time.strftime("%Y-%m-%d", time.gmtime(launch)),
                    "asin": asin,
                    "url": f"https://benchmark.test/{asin.lower()}",
                }
            )
    return Path(handle.name), True


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    conn = get_db(str(args.db))
    database.initialize_database(conn)
    csv_path, generated = _ensure_csv(args)
    with csv_path.open("rb") as handle:
        csv_bytes = handle.read()
    job_id = database.create_import_job(conn, status="pending", phase="parse")
    import_start = time.perf_counter()
    fast_import(
        csv_bytes,
        source="benchmark",
        job_id=job_id,
        batch_size=args.batch_size,
    )
    import_ms = (time.perf_counter() - import_start) * 1000.0
    job_row = database.get_import_job(conn, job_id)
    import_payload = {
        "job_id": job_id,
        "rows": job_row["processed"],
        "rows_imported": job_row["rows_imported"],
        "duration_ms": import_ms,
        "batch_size": args.batch_size,
    }
    database.record_benchmark_run(conn, "import", import_payload, job_id=job_id)
    if generated:
        try:
            csv_path.unlink()
        except Exception:
            pass
    if args.skip_enrich:
        return {"import": import_payload, "job_id": job_id}
    config = {"enrich": {"student": {"enabled": args.student}, "similarity": {"enabled": args.similarity}}}
    database.update_import_job_progress(conn, job_id, config=config)
    enrich_start = time.perf_counter()
    product_enrichment.run_job_sync(job_id)
    enrich_ms = (time.perf_counter() - enrich_start) * 1000.0
    status = database.get_enrichment_status(conn, job_id) or {}
    metrics = status.get("metrics") or {}
    enrich_payload = {
        "duration_ms": enrich_ms,
        "total": status.get("total"),
        "enriched": status.get("enriched"),
        "failed": status.get("failed"),
        "queued": status.get("queued"),
        "metrics": metrics,
        "config": status.get("config"),
    }
    database.record_benchmark_run(conn, "enrich", enrich_payload, job_id=job_id)
    return {"job_id": job_id, "import": import_payload, "enrich": enrich_payload}


def main() -> int:
    args = parse_args()
    summary = run_benchmark(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
