from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .hashing import stable_hash
from .io import append_jsonl


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_qc_event(
    log_path: str,
    step_name: str,
    before_count: int,
    after_count: int,
    reason: str,
    params: dict[str, Any],
) -> None:
    append_jsonl(
        log_path,
        {
            "step_name": step_name,
            "before_count": int(before_count),
            "after_count": int(after_count),
            "reason": reason,
            "params_hash": stable_hash(params),
            "params": params,
            "timestamp": utc_now(),
        },
    )

