from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import load_yaml
from src.utils.logging import log_qc_event


def _choose_threshold(values: np.ndarray, grid: list[float]) -> float:
    best_t = grid[0]
    best_score = -1e9
    for t in grid:
        keep = values >= t
        if keep.mean() == 0:
            score = -1e9
        else:
            score = float(values[keep].mean() + 0.25 * keep.mean())
        if score > best_score:
            best_score = score
            best_t = t
    return float(best_t)


def _winsorize(df: pd.DataFrame, cols: list[str], w: float) -> pd.DataFrame:
    out = df.copy()
    out[cols] = out[cols].clip(lower=-w, upper=w)
    return out


def run_qc(repo_root: Path) -> None:
    cfg = load_yaml(repo_root / "configs/qc.yaml")
    logs_path = repo_root / "results/logs/qc_log.jsonl"
    logs_path.unlink(missing_ok=True)

    raw_drug = pd.read_parquet(repo_root / "data/processed/signatures_drug_raw.parquet")
    raw_do = pd.read_parquet(repo_root / "data/processed/signatures_do_raw.parquet")

    gene_cols = sorted([c for c in raw_drug.columns if c.startswith("G") and c[1:].isdigit()], key=lambda x: int(x[1:]))

    theta_drug = _choose_threshold(raw_drug["cc_q75"].to_numpy(float), list(cfg["theta_drug_candidates"]))
    theta_do = _choose_threshold(raw_do["cc_q75"].to_numpy(float), list(cfg["theta_do_candidates"]))
    theta_act = _choose_threshold(raw_drug["activity_l2"].to_numpy(float), list(cfg["activity_threshold_candidates"]))
    winsor = float(cfg["winsorize_candidates"][1])

    drug = raw_drug.copy()
    before = len(drug)
    drug["low_conf_replicate"] = (drug["n_rep"] < int(cfg["min_replicates"])).astype(int)
    log_qc_event(
        str(logs_path),
        "flag_low_replicate_drug",
        before,
        len(drug),
        "mark low confidence instances",
        {"min_replicates": int(cfg["min_replicates"])},
    )

    before = len(drug)
    drug = drug[drug["cc_q75"] >= theta_drug].reset_index(drop=True)
    log_qc_event(
        str(logs_path),
        "filter_replicate_consistency_drug",
        before,
        len(drug),
        "cc_q75 threshold",
        {"theta_drug": theta_drug},
    )

    before = len(drug)
    drug = drug[drug["activity_l2"] >= theta_act].reset_index(drop=True)
    log_qc_event(
        str(logs_path),
        "filter_activity_drug",
        before,
        len(drug),
        "activity proxy threshold",
        {"theta_act": theta_act},
    )

    before = len(drug)
    drug = _winsorize(drug, gene_cols, winsor)
    log_qc_event(
        str(logs_path),
        "winsorize_drug",
        before,
        len(drug),
        "clip to stable range",
        {"winsor": winsor},
    )

    before = len(drug)
    drug = drug.dropna().reset_index(drop=True)
    log_qc_event(
        str(logs_path),
        "dropna_drug",
        before,
        len(drug),
        "remove missing values",
        {},
    )

    do_df = raw_do.copy()
    before = len(do_df)
    do_df["low_conf_replicate"] = (do_df["n_rep"] < int(cfg["min_replicates"])).astype(int)
    log_qc_event(
        str(logs_path),
        "flag_low_replicate_do",
        before,
        len(do_df),
        "mark low confidence do instances",
        {"min_replicates": int(cfg["min_replicates"])},
    )

    before = len(do_df)
    do_df = do_df[do_df["cc_q75"] >= theta_do].reset_index(drop=True)
    log_qc_event(
        str(logs_path),
        "filter_replicate_consistency_do",
        before,
        len(do_df),
        "cc_q75 threshold",
        {"theta_do": theta_do},
    )

    before = len(do_df)
    do_df = _winsorize(do_df, gene_cols, winsor)
    log_qc_event(
        str(logs_path),
        "winsorize_do",
        before,
        len(do_df),
        "clip to stable range",
        {"winsor": winsor},
    )

    before = len(do_df)
    do_df = do_df.dropna().reset_index(drop=True)
    log_qc_event(
        str(logs_path),
        "dropna_do",
        before,
        len(do_df),
        "remove missing values",
        {},
    )

    out_drug = repo_root / "data/processed/signatures_drug.parquet"
    out_do = repo_root / "data/processed/signatures_do.parquet"
    drug.to_parquet(out_drug, index=False)
    do_df.to_parquet(out_do, index=False)

    summary = [
        f"- theta_drug: {theta_drug}",
        f"- theta_do: {theta_do}",
        f"- theta_act: {theta_act}",
        f"- winsor: {winsor}",
        f"- drug_after_qc: {len(drug)}",
        f"- do_after_qc: {len(do_df)}",
    ]
    (repo_root / "results/logs/qc_summary.md").write_text("\n".join(summary), encoding="utf-8")

