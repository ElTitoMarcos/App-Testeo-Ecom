#!/usr/bin/env python3
"""Smoke test for AI call budgets and batching performance.

This script simulates synthetic imports containing 100, 1,000 and 10,000
products.  For each dataset size it runs the post-import automation runner
with GPT interactions stubbed out so we can validate how many GPT batches are
required and how long the execution takes.  The goal is to ensure that the
configured batch sizes keep the GPT call volume under control even for large
imports while still exercising the winner-weight refresh path.

Usage::

    python scripts/test_ai_budget.py

Environment variables can be used to override the defaults for the AI limits
(e.g. ``AI_MAX_CALLS_PER_IMPORT`` or ``GPT_MAX_RPS``).  Reasonable defaults are
applied when the variables are missing so the script can be run locally without
additional configuration.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

# Ensure the automation is enabled and the call budget is generous enough for
# the stress scenarios unless the caller explicitly overrides it.
os.environ.setdefault("AI_AUTO_ENABLED", "true")
os.environ.setdefault("AI_MAX_CALLS_PER_IMPORT", "1000")
os.environ.setdefault("AI_MIN_BATCH_SIZE", "100")
os.environ.setdefault("AI_MAX_BATCH_SIZE", "250")
os.environ.setdefault("AI_MAX_PARALLEL", "2")
os.environ.setdefault("AI_COALESCE_MS", "200")
os.environ.setdefault("GPT_MAX_PARALLEL", os.environ.get("AI_MAX_PARALLEL", "2"))
os.environ.setdefault("GPT_MAX_RPS", "5")
os.environ.setdefault("GPT_TIMEOUT", "20")

from product_research_app import config, database, gpt
from product_research_app.ai import gpt_guard, runner
from product_research_app.services import config as winner_config
from product_research_app.services import winner_score


@dataclass
class ScenarioLimits:
    size: int
    max_desire_calls: int
    max_imputacion_calls: int
    max_weight_calls: int
    max_total_calls: int | None = None


@dataclass
class ScenarioResult:
    size: int
    duration: float
    desire_calls: int
    imputacion_calls: int
    weight_calls: int
    desire_batches: List[int] = field(default_factory=list)
    imputacion_batches: List[int] = field(default_factory=list)

    @property
    def total_calls(self) -> int:
        return self.desire_calls + self.imputacion_calls + self.weight_calls

    @property
    def avg_time_per_call(self) -> float:
        return self.duration / self.total_calls if self.total_calls else 0.0

    @property
    def avg_desire_batch(self) -> float:
        return _avg(self.desire_batches)

    @property
    def avg_imputacion_batch(self) -> float:
        return _avg(self.imputacion_batches)


SCENARIOS: Sequence[ScenarioLimits] = (
    ScenarioLimits(size=100, max_desire_calls=2, max_imputacion_calls=2, max_weight_calls=1),
    ScenarioLimits(
        size=1000,
        max_desire_calls=20,
        max_imputacion_calls=20,
        max_weight_calls=1,
        max_total_calls=20,
    ),
    ScenarioLimits(
        size=10_000,
        max_desire_calls=120,
        max_imputacion_calls=120,
        max_weight_calls=1,
        max_total_calls=120,
    ),
)


def main() -> None:
    print("AI budget smoke test")
    print("====================\n")

    for limits in SCENARIOS:
        result = run_scenario(limits.size)
        validate_limits(result, limits)
        print_summary(result)


def run_scenario(size: int) -> ScenarioResult:
    """Execute the post-import automation for a synthetic dataset."""

    temp_dir = tempfile.TemporaryDirectory(prefix=f"ai-budget-{size}-")
    base_path = Path(temp_dir.name)
    db_path = base_path / "data.sqlite3"
    config_path = base_path / "config.json"

    original_runner_db = runner.DB_PATH
    original_guard_db = gpt_guard.DB_PATH
    original_cfg_file = config.CONFIG_FILE
    original_winner_db = winner_config.DB_PATH

    runner.DB_PATH = db_path
    gpt_guard.DB_PATH = db_path
    winner_config.DB_PATH = db_path
    config.CONFIG_FILE = config_path

    # Ensure we start from a clean state for each scenario.
    runner._AI_STATUS.clear()  # type: ignore[attr-defined]

    try:
        # Seed a minimal config so the runner can read API/model values.
        initial_config = {"api_key": "sk-test", "model": "gpt-test", "imputacion_via_ia": True}
        config.save_config(initial_config)

        conn = database.get_connection(db_path)
        database.initialize_database(conn)
        try:
            product_ids = _insert_synthetic_products(conn, size)
            import_id = f"smoke-{size}"
            for task in ("desire", "imputacion", "winner_score"):
                database.enqueue_ai_tasks(conn, task, product_ids, import_task_id=import_id)
        finally:
            try:
                conn.close()
            except Exception:
                pass

        desire_batches: List[int] = []
        imputacion_batches: List[int] = []
        desire_calls = 0
        imputacion_calls = 0
        weight_calls = 0

        def fake_desire_orchestrator(api_key: str, model: str, batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
            nonlocal desire_calls
            desire_calls += 1
            desire_batches.append(len(batch))
            items = []
            for entry in batch:
                pid = entry.get("id")
                items.append(
                    {
                        "id": str(pid),
                        "normalized_text": [f"Desire {pid}", "Linea {pid}"],
                        "keywords": ["growth", "conversion"],
                    }
                )
            return {"items": items}

        def fake_imputacion_orchestrator(api_key: str, model: str, batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
            nonlocal imputacion_calls
            imputacion_calls += 1
            imputacion_batches.append(len(batch))
            items = []
            for entry in batch:
                pid = entry.get("id")
                items.append({"id": str(pid), "review_count": 10, "image_count": 4})
            return {"items": items}

        def fake_weights_recommendation(
            api_key: str, model: str, aggregates: Mapping[str, Any]
        ) -> Dict[str, Any]:
            nonlocal weight_calls
            weight_calls += 1
            weights = {field: 10 for field in winner_score.ALLOWED_FIELDS}
            return {"weights": weights, "order": list(winner_score.ALLOWED_FIELDS), "notes": [], "version": "B.v2"}

        def fake_generate_winner_scores(
            conn_arg, *, product_ids: Iterable[int] | None = None, weights: Mapping[str, Any] | None = None, **_: Any
        ) -> Dict[str, int]:
            ids = [int(pid) for pid in (product_ids or [])]
            now = datetime.utcnow().isoformat()
            if ids:
                conn_local = conn_arg if hasattr(conn_arg, "execute") else database.get_connection(db_path)
                try:
                    conn_local.executemany(
                        "UPDATE products SET winner_score=?, winner_score_raw=?, winner_score_updated_at=? WHERE id=?",
                        [(75, 30.0, now, pid) for pid in ids],
                    )
                    conn_local.commit()
                finally:
                    if conn_local is not conn_arg:
                        conn_local.close()
            return {"processed": len(ids), "updated": len(ids)}

        original_desire = gpt.orchestrate_desire_summary
        original_imputacion = gpt.orchestrate_imputation
        original_weights = gpt.recommend_weights_from_aggregates
        original_generate = winner_score.generate_winner_scores
        gpt.orchestrate_desire_summary = fake_desire_orchestrator
        gpt.orchestrate_imputation = fake_imputacion_orchestrator
        gpt.recommend_weights_from_aggregates = fake_weights_recommendation
        winner_score.generate_winner_scores = fake_generate_winner_scores

        start = time.perf_counter()
        try:
            summary = runner.run_post_import_auto(import_id, product_ids)
        finally:
            elapsed = time.perf_counter() - start
            gpt.orchestrate_desire_summary = original_desire
            gpt.orchestrate_imputation = original_imputacion
            gpt.recommend_weights_from_aggregates = original_weights
            winner_score.generate_winner_scores = original_generate

        _assert_processed(summary, size)

        return ScenarioResult(
            size=size,
            duration=elapsed,
            desire_calls=desire_calls,
            imputacion_calls=imputacion_calls,
            weight_calls=weight_calls,
            desire_batches=desire_batches,
            imputacion_batches=imputacion_batches,
        )
    finally:
        runner.DB_PATH = original_runner_db
        gpt_guard.DB_PATH = original_guard_db
        winner_config.DB_PATH = original_winner_db
        config.CONFIG_FILE = original_cfg_file
        temp_dir.cleanup()


def validate_limits(result: ScenarioResult, limits: ScenarioLimits) -> None:
    _ensure(result.desire_calls <= limits.max_desire_calls, result, "desire", limits.max_desire_calls)
    _ensure(result.imputacion_calls <= limits.max_imputacion_calls, result, "imputacion", limits.max_imputacion_calls)
    _ensure(result.weight_calls <= limits.max_weight_calls, result, "weights", limits.max_weight_calls)
    if limits.max_total_calls is not None:
        _ensure(result.total_calls <= limits.max_total_calls, result, "total", limits.max_total_calls)


def print_summary(result: ScenarioResult) -> None:
    print(f"Scenario: {result.size:,} products")
    print(
        f"  Total time: {result.duration:.2f}s | GPT calls: {result.total_calls} | "
        f"avg time/call: {result.avg_time_per_call:.3f}s"
    )
    if result.desire_calls:
        print(
            f"  Desire      -> calls: {result.desire_calls:>3} | avg batch: {result.avg_desire_batch:.1f} | "
            f"max batch: {max(result.desire_batches):d}"
        )
    else:
        print("  Desire      -> no GPT calls (all cached)")
    if result.imputacion_calls:
        print(
            f"  Imputacion  -> calls: {result.imputacion_calls:>3} | avg batch: {result.avg_imputacion_batch:.1f} | "
            f"max batch: {max(result.imputacion_batches):d}"
        )
    else:
        print("  Imputacion  -> no GPT calls (all cached)")
    print(f"  Winner weights -> calls: {result.weight_calls}")
    print("")


def _insert_synthetic_products(conn, count: int) -> List[int]:
    ids: List[int] = []
    for idx in range(1, count + 1):
        pid = database.insert_product(
            conn,
            name=f"Product {idx}",
            description=f"Synthetic description for product {idx}.",
            category="Synthetic",
            price=float((idx % 20) + 1),
            currency="USD",
            source="smoke_test",
            extra={"rating": 4.2, "units_sold": 100 + idx},
            commit=False,
            product_id=idx,
        )
        ids.append(pid)
    conn.commit()
    return ids


def _assert_processed(summary: Mapping[str, Any], expected: int) -> None:
    tasks = summary.get("tasks") if isinstance(summary, Mapping) else None
    desired = _coerce_counts(tasks, "desire")
    imputacion = _coerce_counts(tasks, "imputacion")
    winner = _coerce_counts(tasks, "winner_score")
    if desired.get("processed") < expected:
        raise AssertionError(f"Desire processed {desired.get('processed')} < {expected}")
    if imputacion.get("processed") < expected:
        raise AssertionError(f"Imputacion processed {imputacion.get('processed')} < {expected}")
    if winner.get("processed") < expected:
        raise AssertionError(f"Winner score processed {winner.get('processed')} < {expected}")


def _coerce_counts(tasks: Mapping[str, Any] | None, key: str) -> Dict[str, int]:
    if not isinstance(tasks, Mapping):
        return {"processed": 0, "failed": 0, "skipped": 0}
    entry = tasks.get(key)
    if not isinstance(entry, Mapping):
        return {"processed": 0, "failed": 0, "skipped": 0}
    return {
        "processed": int(entry.get("processed", 0) or 0),
        "failed": int(entry.get("failed", 0) or 0),
        "skipped": int(entry.get("skipped", 0) or 0),
    }


def _ensure(condition: bool, result: ScenarioResult, label: str, limit: int) -> None:
    if condition:
        return
    raise AssertionError(
        f"Scenario {result.size} exceeded {label} call limit: used={_calls_for_label(result, label)} limit={limit}"
    )


def _calls_for_label(result: ScenarioResult, label: str) -> int:
    if label == "desire":
        return result.desire_calls
    if label == "imputacion":
        return result.imputacion_calls
    if label == "weights":
        return result.weight_calls
    if label == "total":
        return result.total_calls
    return -1


def _avg(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


if __name__ == "__main__":
    main()
