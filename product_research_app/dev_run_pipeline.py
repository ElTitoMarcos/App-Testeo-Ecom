"""Development runner for the AI pipeline."""

from __future__ import annotations

import argparse
import inspect
import logging
import os
import sqlite3
import sys
from typing import Any, Callable

try:
    from product_research_app.services.ai_pipeline import run_ai_pipeline as _run
    from product_research_app.services import ai_pipeline as _ai_pipeline_module

    if not hasattr(_ai_pipeline_module, "COLOR_SIZE") and hasattr(
        _ai_pipeline_module, "_COLOR_SIZE"
    ):
        _ai_pipeline_module.COLOR_SIZE = _ai_pipeline_module._COLOR_SIZE  # type: ignore[attr-defined]
    _RUN_SOURCE = "ai_pipeline"
except Exception:  # pragma: no cover - fallback for legacy layout
    from product_research_app.services.ai_columns import run_ai_fill_job as _run  # type: ignore

    _RUN_SOURCE = "ai_columns"

from product_research_app.db import get_conn
from product_research_app.db_bootstrap import drop_all, ensure_schema, seed_fake_products


def _supports_limit(func: Callable[..., Any]) -> bool:
    """Return True if ``func`` appears to accept a ``limit`` keyword."""

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True

    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.KEYWORD_ONLY, parameter.POSITIONAL_OR_KEYWORD) and parameter.name == "limit":
            return True
        if parameter.kind == parameter.VAR_KEYWORD:
            return True
    return False


def _parse_limit(arg_limit: int | None) -> int | None:
    if arg_limit is not None:
        return arg_limit
    env_value = os.getenv("PRAPP_PIPELINE_LIMIT")
    if not env_value:
        return None
    try:
        return int(env_value)
    except ValueError:
        logging.getLogger(__name__).warning(
            "Invalid PRAPP_PIPELINE_LIMIT=%s; ignoring", env_value
        )
        return None


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return cursor.fetchone() is not None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the product research pipeline")
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of items to process (defaults to PRAPP_PIPELINE_LIMIT if set)",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Ensure the minimal SQLite schema exists before running the pipeline",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the minimal SQLite schema before running the pipeline",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed the database with synthetic products for development",
    )
    args = parser.parse_args(argv)

    conn = get_conn()

    if args.reset:
        drop_all(conn)
        ensure_schema(conn)
    else:
        if args.bootstrap:
            ensure_schema(conn)
        elif not _table_exists(conn, "product"):
            logging.getLogger(__name__).info(
                "db_bootstrap: product table missing → creating minimal schema."
            )
            ensure_schema(conn)

    if args.seed is not None:
        try:
            seed_fake_products(conn, int(args.seed))
        except Exception:
            logging.getLogger(__name__).exception("Failed to seed fake products")
            return 1

    limit = _parse_limit(args.limit)
    supports_limit = _supports_limit(_run)

    try:
        if limit is not None and not supports_limit:
            os.environ["PRAPP_PIPELINE_LIMIT"] = str(limit)
            _run()
        elif limit is not None:
            _run(limit=limit)
        else:
            _run()
    except TypeError:
        logging.getLogger(__name__).exception("Pipeline function signature mismatch")
        return 1
    except Exception:
        logging.getLogger(__name__).exception("Pipeline execution failed")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
