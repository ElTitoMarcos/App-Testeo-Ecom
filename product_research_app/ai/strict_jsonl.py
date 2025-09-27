from __future__ import annotations

import json
from typing import Dict, Iterable, List, Set

REQUIRED_FIELDS = ("desire", "desire_magnitude", "awareness_level", "competition_level")


def parse_jsonl_and_validate(text: str, expected_ids: Iterable[int]) -> Dict[int, dict]:
    """Parse JSONL enforcing a strict structure.

    The helper ensures that the amount of JSON objects matches the number of
    expected identifiers, validates that each record includes ``product_id`` and
    the required analytical fields, and verifies that the returned identifiers
    match exactly the requested set.

    Args:
        text: Raw response in JSON Lines format.
        expected_ids: Iterable with the product identifiers requested to the
            model.

    Returns:
        Mapping from ``product_id`` to the decoded JSON object.

    Raises:
        ValueError: When the payload cannot be parsed or any of the validation
            constraints is violated.
    """

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    exp = list(expected_ids)
    if len(lines) != len(exp):
        raise ValueError(f"JSONL length mismatch: got={len(lines)} expected={len(exp)}")

    result: Dict[int, dict] = {}
    for i, ln in enumerate(lines, 1):
        try:
            obj = json.loads(ln)
        except Exception as e:  # pragma: no cover - defensive path
            raise ValueError(f"Invalid JSON on line {i}: {e}") from e
        if "product_id" not in obj:
            raise ValueError(f"Missing product_id on line {i}")
        pid = obj["product_id"]
        if pid in result:
            raise ValueError(f"Duplicated product_id {pid} on line {i}")
        for key in REQUIRED_FIELDS:
            if key not in obj:
                raise ValueError(f"Missing required field '{key}' for product_id={pid}")
        result[pid] = obj

    exp_set: Set[int] = set(exp)
    got_set: Set[int] = set(result.keys())
    if got_set != exp_set:
        missing = list(exp_set - got_set)
        extra = list(got_set - exp_set)
        raise ValueError(f"product_id set mismatch: missing={missing} extra={extra}")

    return result
