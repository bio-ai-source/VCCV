from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.do_dictionary.fusion import fuse_observed_virtual
from src.utils.io import ensure_dir, load_yaml
from src.utils.seed import set_global_seed


class AlignLayer(nn.Module):
    def __init__(self, n_genes: int, rank: int):
        super().__init__()
        self.u = nn.Parameter(torch.zeros(n_genes, rank))
        self.v = nn.Parameter(torch.zeros(n_genes, rank))
        nn.init.normal_(self.u, std=0.01)
        nn.init.normal_(self.v, std=0.01)
        self.b_lof = nn.Parameter(torch.zeros(n_genes))
        self.b_gof = nn.Parameter(torch.zeros(n_genes))
        self.beta_lof_raw = nn.Parameter(torch.tensor(0.2))
        self.beta_gof_raw = nn.Parameter(torch.tensor(0.2))

    def forward(self, x, mode_idx):
        B = torch.eye(x.shape[1], device=x.device) + self.u @ self.v.T
        z = x @ B.T
        beta_lof = torch.sigmoid(self.beta_lof_raw)
        beta_gof = torch.sigmoid(self.beta_gof_raw)
        beta = torch.where(mode_idx.unsqueeze(1) == 0, beta_lof, beta_gof)
        b = torch.where(mode_idx.unsqueeze(1) == 0, self.b_lof, self.b_gof)
        return beta * (z + b), B, beta_lof, beta_gof


def _build_align_table(repo_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sig = pd.read_parquet(repo_root / "data/processed/signatures_drug.parquet")
    truth = pd.read_parquet(repo_root / "data/processed/mechanism_truth.parquet")
    dof = pd.read_parquet(repo_root / "data/processed/do_fused_mu_var.parquet")
    df = sig.merge(truth, on="instance_id", how="inner")
    df = df[(df["is_null"] == 0) & (df["is_poly"] == 0)].reset_index(drop=True)
    if "mode" in df.columns:
        df = df.drop(columns=["mode"])
    df = df.rename(columns={"true_target_key": "target_key", "true_mode": "mode"})
    merge_cols = ["target_key", "context_key", "pert_time", "pert_dose", "platform", "batch", "mode"]
    df = df.merge(dof, on=merge_cols, how="inner", suffixes=("_drug", "_do"))
    g_drug = sorted([c for c in df.columns if c.startswith("G") and c.endswith("_drug")], key=lambda x: int(x[1:].split("_")[0]))
    g_do = sorted([c for c in df.columns if c.startswith("G") and c.endswith("_do")], key=lambda x: int(x[1:].split("_")[0]))
    x = df[g_do].to_numpy(dtype=float)
    y = df[g_drug].to_numpy(dtype=float)
    m = np.where(df["mode"] == "LoF", 0, 1).astype(int)
    return x, y, m


def _train_once(x_tr, y_tr, m_tr, x_ca, y_ca, m_ca, rank, reg, epochs=80, device: torch.device | None = None):
    if device is None:
        device = torch.device("cpu")
    n_genes = x_tr.shape[1]
    model = AlignLayer(n_genes=n_genes, rank=rank).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    xtr = torch.tensor(x_tr, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_tr, dtype=torch.float32, device=device)
    mtr = torch.tensor(m_tr, dtype=torch.long, device=device)
    xca = torch.tensor(x_ca, dtype=torch.float32, device=device)
    yca = torch.tensor(y_ca, dtype=torch.float32, device=device)
    mca = torch.tensor(m_ca, dtype=torch.long, device=device)

    for _ in range(epochs):
        pred, B, _, _ = model(xtr, mtr)
        mse = torch.mean((pred - ytr) ** 2)
        reg_loss = reg * torch.mean((B - torch.eye(n_genes, device=device)) ** 2)
        loss = mse + reg_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pca, B, b_lof, b_gof = model(xca, mca)
        cal_mse = torch.mean((pca - yca) ** 2).item()
        fro = torch.norm(B - torch.eye(n_genes, device=device)).item()
        beta_lof = float(torch.sigmoid(model.beta_lof_raw).item())
        beta_gof = float(torch.sigmoid(model.beta_gof_raw).item())
    return model, cal_mse, fro, beta_lof, beta_gof


def _resolve_training_device(
    device_request: str | None,
    cuda_id: int,
) -> torch.device:
    requested = "auto" if device_request is None else str(device_request).lower()
    if requested == "auto":
        requested = f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"Alignment CUDA device is unavailable: {requested}")
        resolved_index = cuda_id if device.index is None else int(device.index)
        if resolved_index < 0 or resolved_index >= torch.cuda.device_count():
            raise ValueError(f"Alignment CUDA device is unavailable: {requested}")
        device = torch.device(f"cuda:{resolved_index}")
    return device


def train_align(
    repo_root: Path,
    device_request: str | None = None,
) -> None:
    set_global_seed(20260219)
    fuse_observed_virtual(repo_root)
    cfg = load_yaml(repo_root / "configs/align.yaml")
    cuda_id = int(cfg.get("cuda_id", 0))
    device = _resolve_training_device(device_request, cuda_id)
    x, y, m = _build_align_table(repo_root)
    if len(x) < 100:
        raise ValueError("Not enough alignment samples.")
    rng = np.random.default_rng(20260219)
    idx = np.arange(len(x))
    rng.shuffle(idx)
    n_cal = max(50, int(0.2 * len(idx)))
    ca = idx[:n_cal]
    tr = idx[n_cal:]

    best = None
    rows = []
    for rank in cfg["rank_candidates"]:
        for reg in cfg["reg_candidates"]:
            model, cal_mse, fro, beta_lof, beta_gof = _train_once(
                x[tr],
                y[tr],
                m[tr],
                x[ca],
                y[ca],
                m[ca],
                rank=int(rank),
                reg=float(reg),
                epochs=int(cfg["epochs"]),
                device=device,
            )
            rows.append(
                {
                    "rank": int(rank),
                    "reg": float(reg),
                    "cal_mse": cal_mse,
                    "fro_norm_B_minus_I": fro,
                    "beta_lof": beta_lof,
                    "beta_gof": beta_gof,
                }
            )
            if best is None or cal_mse < best["cal_mse"]:
                best = {
                    "model": model,
                    "rank": int(rank),
                    "reg": float(reg),
                    "cal_mse": cal_mse,
                    "fro": fro,
                    "beta_lof": beta_lof,
                    "beta_gof": beta_gof,
                }

    metrics = pd.DataFrame(rows).sort_values("cal_mse")
    metrics.to_csv(repo_root / "results/metrics_tables/align_fit_stats.csv", index=False)

    model = best["model"]
    ckpt_dir = ensure_dir(repo_root / "results/checkpoints/align")
    with torch.no_grad():
        B = (torch.eye(x.shape[1], device=model.u.device) + model.u @ model.v.T).cpu().numpy()
        b_lof = model.b_lof.cpu().numpy()
        b_gof = model.b_gof.cpu().numpy()
        beta_lof = float(torch.sigmoid(model.beta_lof_raw).item())
        beta_gof = float(torch.sigmoid(model.beta_gof_raw).item())
    np.savez_compressed(
        ckpt_dir / "align_params.npz",
        B=B,
        b_lof=b_lof,
        b_gof=b_gof,
        beta_lof=np.array([beta_lof]),
        beta_gof=np.array([beta_gof]),
    )

    lines = [
        "# Align model card",
        f"- rank: {best['rank']}",
        f"- reg: {best['reg']}",
        f"- cal_mse: {best['cal_mse']:.6f}",
        f"- fro_norm_B_minus_I: {best['fro']:.6f}",
        f"- beta_lof: {beta_lof:.4f}",
        f"- beta_gof: {beta_gof:.4f}",
        f"- device: {device}",
        f"- beta_constraint_pass: {int(beta_lof <= 1.0 and beta_gof <= 1.0)}",
    ]
    (repo_root / "model_cards/align.md").write_text("\n".join(lines), encoding="utf-8")
