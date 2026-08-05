from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class BindingMetrics:
    auc: float
    pr_auc: float
    nll: float
    ece: float
    brier: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Historical equal-width-bin ECE used for the paper table."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1.0 - 1e-7)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = max(len(y), 1)
    ece = 0.0
    for index in range(n_bins):
        left = edges[index]
        right = edges[index + 1]
        if index == n_bins - 1:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p >= left) & (p < right)
        if not np.any(mask):
            continue
        accuracy = float(np.mean(y[mask]))
        confidence = float(np.mean(p[mask]))
        ece += float(np.sum(mask)) / total * abs(accuracy - confidence)
    return float(ece)


def summarize_binding(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> BindingMetrics:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1.0 - 1e-7)
    auc = float("nan")
    pr_auc = float("nan")
    if len(np.unique(y)) >= 2:
        auc = float(roc_auc_score(y, p))
        pr_auc = float(average_precision_score(y, p))
    return BindingMetrics(
        auc=auc,
        pr_auc=pr_auc,
        nll=float(
            -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean()
        ),
        ece=expected_calibration_error(y, p, n_bins=15),
        brier=float(np.mean((p - y) ** 2)),
    )


def metric_sort_key(metrics: BindingMetrics) -> tuple[float, float, float, float]:
    auc = metrics.auc if np.isfinite(metrics.auc) else -1.0
    pr_auc = metrics.pr_auc if np.isfinite(metrics.pr_auc) else -1.0
    ece = metrics.ece if np.isfinite(metrics.ece) else 1e9
    nll = metrics.nll if np.isfinite(metrics.nll) else 1e9
    return float(auc), float(pr_auc), -float(ece), -float(nll)
