from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io import load_yaml

ALL_REAL_MODELS = [
    "VCCV_DTI_Ensemble",
    "DeepDTAGen_2025_Reimpl",
    "DTIAM_2025_Reimpl",
    "EviDTI_2025_Reimpl",
    "MolTrans",
    "DeepDTA",
    "GraphDTA",
]


def _setup_style(cfg):
    plt.rcParams["font.family"] = cfg["style"]["font_family"]
    plt.rcParams["figure.dpi"] = int(cfg["style"]["dpi"])
    plt.rcParams["axes.linewidth"] = float(cfg["style"]["linewidth"])


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_all_figures(repo_root: Path) -> None:
    fig_cfg = load_yaml(repo_root / "configs/figures.yaml")
    _setup_style(fig_cfg)
    fig_dir = repo_root / "results/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    captions = []
    prov = {}

    # Fig 1: overview
    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    ax.axis("off")
    txt = "DTI Prior  ->  Observed/Virtual do  ->  Align  ->  Gibbs Posterior  ->  MFS"
    ax.text(0.02, 0.58, txt, fontsize=14, weight="bold")
    ax.text(0.02, 0.30, "Train/Cal/Test isolation + audit logs + decoy/permutation tests", fontsize=10)
    p = fig_dir / "fig1_overview.pdf"
    _save(fig, p)
    captions.append("Fig.1 VCCVDTI overview pipeline.")
    prov[p.name] = ["configs/base.yaml", "README_repro.md"]

    # Fig 2: dataset statistics
    stats = pd.read_parquet(repo_root / "data/processed/dti_labels.parquet")
    fig, ax = plt.subplots(figsize=(8, 4))
    by_ds = stats.groupby("dataset").size().reset_index(name="n")
    ax.bar(by_ds["dataset"], by_ds["n"], color="#174C6B")
    ax.set_ylabel("Interactions")
    ax.set_title("Data Statistics by Dataset")
    p = fig_dir / "fig2_data_splits.pdf"
    _save(fig, p)
    captions.append("Fig.2 Dataset statistics.")
    prov[p.name] = ["data/processed/dti_labels.parquet"]

    # Fig 3: binding results
    bind_path = repo_root / "results/metrics_tables/binding_metrics_with_frontier_scene_tuned.csv"
    if not bind_path.exists():
        bind_path = repo_root / "results/metrics_tables/binding_metrics_with_frontier.csv"
    if not bind_path.exists():
        bind_path = repo_root / "results/metrics_tables/binding_metrics.csv"
    bind = pd.read_csv(bind_path)
    agg = bind.groupby("model", as_index=False)["auc"].mean()
    present_order = [m for m in ALL_REAL_MODELS if m in set(agg["model"].tolist())]
    if present_order:
        agg["model"] = pd.Categorical(agg["model"], categories=present_order, ordered=True)
        agg = agg.sort_values("model", ascending=True)
    else:
        agg = agg.sort_values("auc", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(agg["model"], agg["auc"], color="#B34A3C")
    ax.invert_yaxis()
    ax.set_xlabel("AUC")
    ax.set_title("Binding Performance (Mean AUC, Unified 7-Model Comparison)")
    p = fig_dir / "fig3_binding_results.pdf"
    _save(fig, p)
    captions.append("Fig.3 Binding results across baselines, 2025 frontier methods, and VCCV ensemble.")
    prov[p.name] = [str(bind_path.relative_to(repo_root))]

    # Fig 4: mechanism on silver-like subset
    mech = pd.read_csv(repo_root / "results/metrics_tables/mechanism_metrics.csv")
    sub = mech[mech["scenario"] == "virtual"].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(sub["model"], sub["mrr"], color="#174C6B")
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("MRR")
    ax.set_title("Mechanism Inference (Virtual)")
    p = fig_dir / "fig4_mechanism_real_silver.pdf"
    _save(fig, p)
    captions.append("Fig.4 Mechanism ranking metrics on silver-like set.")
    prov[p.name] = ["results/metrics_tables/mechanism_metrics.csv"]

    # Fig 5: calibration curve
    cal = pd.read_csv(repo_root / "results/metrics_tables/inference_calibration.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(cal["eta"], cal["cal_nll"], marker="o", color="#174C6B")
    ax.set_xlabel("eta")
    ax.set_ylabel("Cal NLL")
    ax.set_title("Gibbs Temperature Calibration")
    p = fig_dir / "fig5_syn_calibration.pdf"
    _save(fig, p)
    captions.append("Fig.5 Calibration over Gibbs temperature eta.")
    prov[p.name] = ["results/metrics_tables/inference_calibration.csv"]

    # Fig 6: MFS entropy curve
    mfs_dir = repo_root / "results/mfs_panels"
    curves = []
    for fp in sorted(mfs_dir.glob("*.json"))[:20]:
        obj = json.loads(fp.read_text(encoding="utf-8"))
        curves.append(obj.get("entropy_curve_proxy", []))
    fig, ax = plt.subplots(figsize=(8, 4))
    for c in curves[:10]:
        if c:
            ax.plot(np.arange(1, len(c) + 1), np.cumsum(c), alpha=0.35, color="#4D5B66")
    nonempty_curves = [c for c in curves if c]
    if nonempty_curves:
        max_len = max(len(c) for c in nonempty_curves)
        mean_curve = []
        for i in range(max_len):
            vals = [np.cumsum(c)[i] for c in nonempty_curves if len(c) > i]
            mean_curve.append(np.mean(vals))
        ax.plot(np.arange(1, len(mean_curve) + 1), mean_curve, color="#174C6B", linewidth=2.5)
    ax.set_xlabel("Panel size")
    ax.set_ylabel("Cumulative Information Gain")
    ax.set_title("MFS Information Efficiency")
    p = fig_dir / "fig6_mfs_efficiency.pdf"
    _save(fig, p)
    captions.append("Fig.6 MFS information efficiency curves.")
    prov[p.name] = [str(x) for x in mfs_dir.glob("*.json")]

    # ED figures (lightweight)
    files_for_ed = [
        ("ed_fig1_qc_sensitivity.pdf", "results/logs/qc_summary.md"),
        ("ed_fig2_frozen_vs_splitconsistent.pdf", "results/logs/download_check.log"),
        ("ed_fig3_observed_virtual_fuse.pdf", "results/metrics_tables/mechanism_metrics.csv"),
        ("ed_fig4_no_align_bias.pdf", "results/metrics_tables/mechanism_metrics.csv"),
        ("ed_fig5_perm_collapse.pdf", "results/metrics_tables/permutation_overall.csv"),
        ("ed_fig6_decoy.pdf", "results/metrics_tables/decoy_overall.csv"),
        ("ed_fig7_time_dose_shift.pdf", "results/metrics_tables/binding_metrics.csv"),
        ("ed_fig8_kcand_sensitivity.pdf", "results/metrics_tables/mechanism_metrics.csv"),
        ("ed_fig9_null_k_sensitivity.pdf", "results/metrics_tables/permutation_overall.csv"),
        ("ed_fig10_mfs_gap.pdf", "results/logs/mfs_greedy_opt_gap.md"),
    ]
    for name, dep in files_for_ed:
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        ax.axis("off")
        ax.text(0.05, 0.62, name.replace(".pdf", "").replace("_", " "), fontsize=12, weight="bold")
        ax.text(0.05, 0.35, f"Source: {dep}", fontsize=9)
        p = fig_dir / name
        _save(fig, p)
        captions.append(f"{name.replace('.pdf', '')}: autogenerated placeholder plot.")
        prov[p.name] = [dep]

    (fig_dir / "captions.md").write_text("\n".join(captions), encoding="utf-8")
    (fig_dir / "figure_provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
