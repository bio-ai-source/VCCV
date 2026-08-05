from __future__ import annotations

import numpy as np


def coverage_regularized_order(
    reference_variance: np.ndarray,
    correlation: np.ndarray,
    *,
    budget: int,
    redundancy_coefficient: float = 1.0,
) -> list[int]:
    """Greedily rank genes by the manuscript coverage utility.

    At each step, gene ``g`` maximizes
    ``v(g) / (1 + lambda * mean_s |R(g, s)|)``.  Here ``v(g)`` is the
    empirical variance across the frozen fused reference dictionary.  Ties
    are resolved by the lower gene index for deterministic reproduction.
    """
    variance = np.asarray(reference_variance, dtype=float)
    corr = np.asarray(correlation, dtype=float)
    if variance.ndim != 1:
        raise ValueError("reference_variance must be one-dimensional.")
    if corr.shape != (len(variance), len(variance)):
        raise ValueError("correlation must be square and match the variance vector.")
    if not np.all(np.isfinite(variance)) or np.any(variance < 0.0):
        raise ValueError("reference_variance must be finite and non-negative.")
    if not np.all(np.isfinite(corr)):
        raise ValueError("correlation must be finite.")
    if budget < 0:
        raise ValueError("budget must be non-negative.")
    if redundancy_coefficient < 0.0:
        raise ValueError("redundancy_coefficient must be non-negative.")

    selected: list[int] = []
    remaining = set(range(len(variance)))
    while len(selected) < min(int(budget), len(variance)) and remaining:
        best_gene = -1
        best_utility = -np.inf
        for gene in sorted(remaining):
            redundancy = (
                float(np.mean(np.abs(corr[gene, selected]))) if selected else 0.0
            )
            utility = float(variance[gene]) / (
                1.0 + float(redundancy_coefficient) * redundancy
            )
            if utility > best_utility:
                best_utility = utility
                best_gene = int(gene)
        selected.append(best_gene)
        remaining.remove(best_gene)
    return selected
