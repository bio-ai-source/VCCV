from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.io import load_yaml
from src.utils.metrics import binary_ece, nll_binary, safe_auc, safe_pr_auc


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "revision_round2.yaml"


@dataclass
class ExtendedBindingMetrics:
    auc: float
    pr_auc: float
    ece: float
    ace: float
    nll: float
    brier: float
    prevalence: float
    n_rows: int


def load_round2_cfg() -> dict:
    return load_yaml(CONFIG_PATH)


def clip_prob(values: Iterable[float], eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float), eps, 1.0 - eps)


def safe_logit(values: Iterable[float]) -> np.ndarray:
    p = clip_prob(values)
    return np.log(p / (1.0 - p))


def safe_sigmoid(values: Iterable[float]) -> np.ndarray:
    x = np.clip(np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def adaptive_calibration_error(y_true: Iterable[int], y_prob: Iterable[float], bins: int = 10) -> float:
    y = np.asarray(list(y_true) if not isinstance(y_true, np.ndarray) else y_true, dtype=float)
    p = clip_prob(y_prob)
    if len(y) == 0:
        return float("nan")
    order = np.argsort(p)
    y = y[order]
    p = p[order]
    splits = np.array_split(np.arange(len(y)), bins)
    ace = 0.0
    n = max(len(y), 1)
    for idx in splits:
        if len(idx) == 0:
            continue
        ace += (len(idx) / n) * abs(float(y[idx].mean()) - float(p[idx].mean()))
    return float(ace)


def summarize_extended_binding(y_true: Iterable[int], y_prob: Iterable[float]) -> ExtendedBindingMetrics:
    y = np.asarray(list(y_true) if not isinstance(y_true, np.ndarray) else y_true, dtype=int)
    p = clip_prob(y_prob)
    return ExtendedBindingMetrics(
        auc=float(safe_auc(y, p)),
        pr_auc=float(safe_pr_auc(y, p)),
        ece=float(binary_ece(y, p)),
        ace=float(adaptive_calibration_error(y, p)),
        nll=float(nll_binary(y, p)),
        brier=float(np.mean((p - y.astype(float)) ** 2)) if len(y) else float("nan"),
        prevalence=float(np.mean(y)) if len(y) else float("nan"),
        n_rows=int(len(y)),
    )


def strict_linked_mask(df: pd.DataFrame, cfg: dict | None = None) -> pd.Series:
    config = (cfg or load_round2_cfg())["linked"]["strict_subset"]
    mask = pd.Series(np.ones(len(df), dtype=bool), index=df.index)
    if bool(config.get("expression_supported_only", False)):
        mask &= pd.to_numeric(df["expression_supported_flag"], errors="coerce").fillna(0.0) > 0.0
    best_quality = str(config.get("best_match_quality", "")).strip()
    if best_quality and "best_match_quality" in df.columns:
        mask &= df["best_match_quality"].fillna("").astype(str).eq(best_quality)
    mask &= pd.to_numeric(df["time_diff_lof"], errors="coerce").fillna(np.inf) <= float(config["max_time_diff_hours"])
    mask &= pd.to_numeric(df["log10_dose_diff_lof"], errors="coerce").fillna(np.inf) <= float(config["max_log10_dose_diff"])
    return mask


def block_id_for_proxy(df: pd.DataFrame) -> pd.Series:
    scenario = df["scenario"].astype(str)
    out = pd.Series(df["seed"].astype(str), index=df.index, dtype=object)
    out = out + "::"
    out = out + np.where(
        scenario.eq("drug_heldout"),
        df["drug_key"].astype(str),
        np.where(
            scenario.eq("target_heldout"),
            df["target_key"].astype(str),
            np.where(
                scenario.eq("context_heldout"),
                df["context_key"].astype(str),
                df["drug_key"].astype(str) + "|" + df["target_key"].astype(str),
            ),
        ),
    )
    return out


def coverage_risk_curve(y_true: Iterable[int], y_prob: Iterable[float], coverage_grid: Iterable[float]) -> pd.DataFrame:
    y = np.asarray(list(y_true) if not isinstance(y_true, np.ndarray) else y_true, dtype=int)
    p = clip_prob(y_prob)
    conf = np.abs(p - 0.5) * 2.0
    order = np.argsort(-conf)
    y = y[order]
    p = p[order]
    rows = []
    for frac in coverage_grid:
        keep = max(1, int(round(len(y) * float(frac))))
        ys = y[:keep]
        ps = p[:keep]
        pred = (ps >= 0.5).astype(int)
        risk = 1.0 - float(np.mean(pred == ys))
        met = summarize_extended_binding(ys, ps)
        rows.append(
            {
                "coverage": float(frac),
                "retained_n": int(keep),
                "risk": risk,
                "auc": met.auc,
                "pr_auc": met.pr_auc,
                "ece": met.ece,
                "ace": met.ace,
                "nll": met.nll,
                "brier": met.brier,
            }
        )
    out = pd.DataFrame(rows)
    out["segment_width"] = out["coverage"].diff().fillna(out["coverage"])
    out["aurc_component"] = out["risk"] * out["segment_width"]
    out["aurc"] = float(out["aurc_component"].sum())
    return out


def block_bootstrap_indices(block_ids: Iterable[str], seed: int, n_boot: int) -> list[np.ndarray]:
    block = np.asarray(list(block_ids), dtype=object)
    uniq = np.unique(block)
    lookup = {u: np.flatnonzero(block == u) for u in uniq}
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    for _ in range(int(n_boot)):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([lookup[str(u)] for u in sampled]) if len(sampled) else np.array([], dtype=int)
        draws.append(idx)
    return draws


def bootstrap_metric_summary(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    block_ids: Iterable[str],
    metric: str,
    seed: int,
    n_boot: int,
) -> tuple[float, float, float]:
    y = np.asarray(list(y_true) if not isinstance(y_true, np.ndarray) else y_true, dtype=int)
    p = clip_prob(y_prob)
    vals = []
    for idx in block_bootstrap_indices(block_ids, seed=seed, n_boot=n_boot):
        met = summarize_extended_binding(y[idx], p[idx])
        vals.append(float(getattr(met, metric)))
    arr = np.asarray(vals, dtype=float)
    return float(np.nanmean(arr)), float(np.nanquantile(arr, 0.025)), float(np.nanquantile(arr, 0.975))


def bootstrap_delta_summary(
    y_true: Iterable[int],
    p_a: Iterable[float],
    p_b: Iterable[float],
    block_ids: Iterable[str],
    metric: str,
    seed: int,
    n_boot: int,
) -> tuple[float, float, float]:
    y = np.asarray(list(y_true) if not isinstance(y_true, np.ndarray) else y_true, dtype=int)
    pa = clip_prob(p_a)
    pb = clip_prob(p_b)
    vals = []
    for idx in block_bootstrap_indices(block_ids, seed=seed, n_boot=n_boot):
        ma = summarize_extended_binding(y[idx], pa[idx])
        mb = summarize_extended_binding(y[idx], pb[idx])
        vals.append(float(getattr(ma, metric) - getattr(mb, metric)))
    arr = np.asarray(vals, dtype=float)
    return float(np.nanmean(arr)), float(np.nanquantile(arr, 0.025)), float(np.nanquantile(arr, 0.975))


def add_bin_column(df: pd.DataFrame, source_col: str, bins: list[float], prefix: str) -> pd.Series:
    values = pd.to_numeric(df[source_col], errors="coerce")
    labels: list[str] = []
    left = -np.inf
    for right in bins:
        if np.isfinite(left):
            labels.append(f"{prefix}({left:.2f},{right:.2f}]")
        else:
            labels.append(f"{prefix}<= {right:.2f}")
        left = right
    labels.append(f"{prefix}> {bins[-1]:.2f}")
    return pd.cut(values, bins=[-np.inf, *bins, np.inf], labels=labels, include_lowest=True).astype(str)
