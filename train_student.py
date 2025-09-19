#!/usr/bin/env python3
"""Train the local student model using previously enriched items."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

from product_research_app import database, student_model
from product_research_app.db import get_db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default="product_research_app/data.sqlite3",
        help="Path to the SQLite database (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of labelled items to use for training",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Override confidence threshold when saving the model",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    conn = get_db(args.db)
    database.initialize_database(conn)
    kwargs: Dict[str, Any] = {}
    if args.limit is not None:
        kwargs["limit"] = args.limit
    if args.confidence is not None:
        kwargs["confidence_threshold"] = args.confidence
    metrics = student_model.train_student_models(conn, **kwargs)
    model_dir = Path(student_model.MODEL_DIR).resolve()
    payload = {
        "model_dir": str(model_dir),
        "metrics": metrics,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
