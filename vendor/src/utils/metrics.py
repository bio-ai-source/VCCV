from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def binary_ece(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 15) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = max(len(y_true), 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = (y_prob >= lo) & (y_prob < hi)
        if not idx.any():
            continue
        acc = y_true[idx].mean()
        conf = y_prob[idx].mean()
        ece += (idx.sum() / n) * abs(acc - conf)
    return float(ece)


def nll_binary(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-7) -> float:
    y = np.asarray(y_true).astype(float)
    p = np.clip(np.asarray(y_prob).astype(float), eps, 1.0 - eps)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def mrr_score(ranks: list[int]) -> float:
    if not ranks:
        return float("nan")
    safe = []
    for r in ranks:
        try:
            rv = int(r)
        except Exception:
            rv = 10**9
        if rv <= 0:
            rv = 10**9
        safe.append(rv)
    return float(np.mean([1.0 / r for r in safe]))


def hits_at_k(ranks: list[int], k: int) -> float:
    if not ranks:
        return float("nan")
    safe = []
    for r in ranks:
        try:
            rv = int(r)
        except Exception:
            rv = 10**9
        if rv <= 0:
            rv = 10**9
        safe.append(rv)
    return float(np.mean([1.0 if r <= k else 0.0 for r in safe]))


@dataclass
class BindingMetrics:
    auc: float
    pr_auc: float
    ece: float
    nll: float
    brier: float


def summarize_binding(y_true: np.ndarray, y_prob: np.ndarray) -> BindingMetrics:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    return BindingMetrics(
        auc=safe_auc(y_true, y_prob),
        pr_auc=safe_pr_auc(y_true, y_prob),
        ece=binary_ece(y_true, y_prob),
        nll=nll_binary(y_true, y_prob),
        brier=float(brier_score_loss(y_true, y_prob)),
    )
