from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def assign_decision_state(
    posterior_distribution: Sequence[Mapping[str, Any]],
    *,
    decision_gap_threshold: float = 0.42,
    deep_near_tie_threshold: float = 0.05,
    strong_null_threshold: float = 0.95,
) -> dict[str, Any]:
    """Apply the calibration-fixed gap-first VCCV three-state rule.

    Ambiguity is evaluated before the identity of the MAP branch: a gap at or
    below ``decision_gap_threshold`` requests a panel.  Only a separated null
    MAP abstains; a separated non-null MAP is retained.  The strong-null flag
    is descriptive and never changes the three-state decision.
    """
    if not 0.0 <= deep_near_tie_threshold <= decision_gap_threshold <= 1.0:
        raise ValueError("Decision thresholds must satisfy 0 <= deep <= decision <= 1.")
    if not 0.0 <= strong_null_threshold <= 1.0:
        raise ValueError("strong_null_threshold must be between zero and one.")
    if not posterior_distribution:
        raise ValueError("posterior_distribution must contain at least one active branch.")

    items: list[dict[str, Any]] = []
    for item in posterior_distribution:
        probability = float(item.get("posterior", float("nan")))
        if not math.isfinite(probability) or probability < 0.0:
            raise ValueError("Every active branch must have a finite non-negative posterior.")
        items.append(
            {
                "type": str(item.get("type", "")),
                "target_key": str(item.get("target_key", "")),
                "mode": str(item.get("mode", "")),
                "posterior": probability,
            }
        )

    total = sum(float(item["posterior"]) for item in items)
    if total <= 0.0:
        raise ValueError("At least one active branch must have positive posterior mass.")
    for item in items:
        item["posterior"] = float(item["posterior"]) / total
    ranked = sorted(items, key=lambda item: -float(item["posterior"]))

    top = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else {
        "type": "none",
        "target_key": "",
        "mode": "",
        "posterior": 0.0,
    }
    gap = float(top["posterior"]) - float(runner_up["posterior"])

    if gap <= decision_gap_threshold:
        state = "panel"
        panel_policy = "mfs" if gap <= deep_near_tie_threshold else "coverage"
    elif top["type"] == "null":
        state = "abstain"
        panel_policy = None
    else:
        state = "retain"
        panel_policy = None

    null_probability = sum(
        float(item["posterior"]) for item in ranked if item["type"] == "null"
    )
    return {
        "state": state,
        "panel_policy": panel_policy,
        "posterior_gap": gap,
        "decision_gap_threshold": float(decision_gap_threshold),
        "deep_near_tie_threshold": float(deep_near_tie_threshold),
        "top_hypothesis": top,
        "runner_up_hypothesis": runner_up,
        "null_posterior": null_probability,
        "strong_null": bool(null_probability >= strong_null_threshold),
        "strong_null_threshold": float(strong_null_threshold),
    }
