from __future__ import annotations

import logging
from typing import Dict, List


log = logging.getLogger(__name__)


def log_partial_ko(
    stage: str,
    requested_ids: List[int],
    returned_ids: List[int],
    batch_size: int,
    model: str,
    note: str = "",
) -> None:
    missing = [pid for pid in requested_ids if pid not in set(returned_ids)]
    log.warning(
        "ai_columns.partial_ko stage=%s batch_size=%s model=%s missing_count=%s missing_ids=%s note=%s",
        stage,
        batch_size,
        model,
        len(missing),
        missing,
        note,
    )


def log_recovered(
    stage: str,
    retries_used: int,
    final_batch_size: int,
    recovered_ids: List[int],
) -> None:
    log.info(
        "ai_columns.ok_recovered stage=%s retries_used=%s final_batch_size=%s recovered_ids=%s",
        stage,
        retries_used,
        final_batch_size,
        recovered_ids,
    )
