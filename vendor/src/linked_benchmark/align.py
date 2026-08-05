from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.utils.io import ensure_dir, load_yaml, save_json


@dataclass
class AlignVariant:
    name: str
    use_low_rank: bool
    use_bias: bool
    use_beta: bool


class LinkedAlignLayer(nn.Module):
    def __init__(self, n_genes: int, rank: int, variant: AlignVariant):
        super().__init__()
        self.variant = variant
        if variant.use_low_rank:
            self.u = nn.Parameter(torch.zeros(n_genes, rank))
            self.v = nn.Parameter(torch.zeros(n_genes, rank))
            nn.init.normal_(self.u, std=0.01)
            nn.init.normal_(self.v, std=0.01)
        else:
            self.register_buffer("u", torch.zeros(n_genes, rank))
            self.register_buffer("v", torch.zeros(n_genes, rank))
        if variant.use_bias:
            self.b_lof = nn.Parameter(torch.zeros(n_genes))
            self.b_gof = nn.Parameter(torch.zeros(n_genes))
        else:
            self.register_buffer("b_lof", torch.zeros(n_genes))
            self.register_buffer("b_gof", torch.zeros(n_genes))
        if variant.use_beta:
            self.beta_lof_raw = nn.Parameter(torch.tensor(0.2))
            self.beta_gof_raw = nn.Parameter(torch.tensor(0.2))
        else:
            self.register_buffer("beta_lof_raw", torch.tensor(100.0))
            self.register_buffer("beta_gof_raw", torch.tensor(100.0))

    def forward(self, x: torch.Tensor, mode_idx: torch.Tensor):
        if self.variant.use_low_rank:
            B = torch.eye(x.shape[1], device=x.device) + self.u @ self.v.T
        else:
            B = torch.eye(x.shape[1], device=x.device)
        z = x @ B.T
        beta_lof = torch.sigmoid(self.beta_lof_raw)
        beta_gof = torch.sigmoid(self.beta_gof_raw)
        beta = torch.where(mode_idx.unsqueeze(1) == 0, beta_lof, beta_gof)
        b = torch.where(mode_idx.unsqueeze(1) == 0, self.b_lof, self.b_gof)
        return beta * (z + b), B


def _entity_holdout_split(
    df: pd.DataFrame,
    entity_col: str,
    seed: int,
    cal_fraction: float,
    test_fraction: float,
) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    entities = np.array(sorted(df[entity_col].dropna().astype(str).unique().tolist()))
    rng.shuffle(entities)
    n = len(entities)
    n_test = max(1, int(round(n * test_fraction)))
    n_cal = max(1, int(round(n * cal_fraction)))
    test_entities = set(entities[:n_test].tolist())
    cal_entities = set(entities[n_test : n_test + n_cal].tolist())
    train_entities = set(entities[n_test + n_cal :].tolist())
    if not train_entities:
        train_entities = cal_entities
        cal_entities = set()
    return {
        "train": df[df[entity_col].astype(str).isin(train_entities)]["align_pair_id"].astype(str).tolist(),
        "cal": df[df[entity_col].astype(str).isin(cal_entities)]["align_pair_id"].astype(str).tolist(),
        "test": df[df[entity_col].astype(str).isin(test_entities)]["align_pair_id"].astype(str).tolist(),
        "train_entities": sorted(train_entities),
        "cal_entities": sorted(cal_entities),
        "test_entities": sorted(test_entities),
    }


def _variant_specs(names: list[str]) -> list[AlignVariant]:
    all_specs = {
        "identity": AlignVariant("identity", use_low_rank=False, use_bias=False, use_beta=False),
        "scale_bias_only": AlignVariant("scale_bias_only", use_low_rank=False, use_bias=True, use_beta=True),
        "low_rank_only": AlignVariant("low_rank_only", use_low_rank=True, use_bias=False, use_beta=False),
        "full": AlignVariant("full", use_low_rank=True, use_bias=True, use_beta=True),
    }
    return [all_specs[n] for n in names]


def _subset_by_ids(df: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    idset = set(str(x) for x in ids)
    return df[df["align_pair_id"].astype(str).isin(idset)].reset_index(drop=True)


def _mode_index(df: pd.DataFrame) -> np.ndarray:
    return np.where(df["mode"].astype(str) == "LoF", 0, 1).astype(int)


def _tensor_triplets(df: pd.DataFrame, gcols_drug: list[str], gcols_do: list[str], device: torch.device):
    x = torch.tensor(df[gcols_do].to_numpy(dtype=float), dtype=torch.float32, device=device)
    y = torch.tensor(df[gcols_drug].to_numpy(dtype=float), dtype=torch.float32, device=device)
    m = torch.tensor(_mode_index(df), dtype=torch.long, device=device)
    return x, y, m


def _gene_columns(df: pd.DataFrame, suffix: str = "") -> list[str]:
    cols = [c for c in df.columns if c.startswith("G") and c.endswith(suffix)]
    return sorted(cols, key=lambda x: int(x[1:].split("_")[0]))


def _materialize_pair_expression(repo_root: Path, cfg: dict, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    gcols_drug = _gene_columns(df, "_drug")
    gcols_do = _gene_columns(df, "_do")
    if gcols_drug and gcols_do:
        return df, gcols_drug, gcols_do

    data_root = repo_root / str(cfg["output_root"]) / str(cfg["data_dir"])
    cmp_path = data_root / "linked_compound_signatures.parquet"
    tgt_path = data_root / "linked_target_signatures.parquet"
    if not (cmp_path.exists() and tgt_path.exists()):
        raise FileNotFoundError("Lightweight alignment pairs require linked signature tables, but they were not found.")

    cmp = pd.read_parquet(cmp_path)
    tgt = pd.read_parquet(tgt_path)
    gcols = _gene_columns(cmp)
    cmp = cmp[["compound_sig_id", *gcols]].rename(columns={g: f"{g}_drug" for g in gcols})
    tgt = tgt[["target_sig_id", *gcols]].rename(columns={g: f"{g}_do" for g in gcols})
    df = df.merge(cmp, on="compound_sig_id", how="inner").merge(tgt, on="target_sig_id", how="inner")
    return df, _gene_columns(df, "_drug"), _gene_columns(df, "_do")


def _cosine_mean(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    good = (an > 1e-12) & (bn > 1e-12)
    if not np.any(good):
        return 0.0
    cos = np.sum(a[good] * b[good], axis=1) / (an[good] * bn[good])
    return float(np.mean(np.clip(cos, -1.0, 1.0)))


def _nearest_target_retrieval(
    test_df: pd.DataFrame,
    all_df: pd.DataFrame,
    model: LinkedAlignLayer,
    gcols_do: list[str],
    device: torch.device,
) -> tuple[float, float]:
    if test_df.empty:
        return 0.0, 0.0
    target_bank = (
        all_df[["target_key", "cell_id", "mode", *gcols_do]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    mrr = []
    hits1 = []
    with torch.no_grad():
        for row in test_df.itertuples(index=False):
            cand = target_bank[
                (target_bank["cell_id"].astype(str) == str(row.cell_id))
                & (target_bank["mode"].astype(str) == str(row.mode))
            ].copy()
            if cand.empty:
                continue
            x = torch.tensor(
                np.asarray([[getattr(row, c) for c in gcols_do]], dtype=float),
                dtype=torch.float32,
                device=device,
            )
            mode_idx = torch.tensor([0 if str(row.mode) == "LoF" else 1], dtype=torch.long, device=device)
            pred, _ = model(x, mode_idx)
            q = pred.cpu().numpy()[0]
            y = cand[gcols_do].to_numpy(dtype=float)
            dist = np.mean((y - q[None, :]) ** 2, axis=1)
            ranked_targets = cand.iloc[np.argsort(dist)]["target_key"].astype(str).tolist()
            true_target = str(row.target_key)
            if true_target not in ranked_targets:
                continue
            rk = ranked_targets.index(true_target) + 1
            mrr.append(1.0 / rk)
            hits1.append(float(rk == 1))
    if not mrr:
        return 0.0, 0.0
    return float(np.mean(mrr)), float(np.mean(hits1))


def _train_variant(
    tr_df: pd.DataFrame,
    ca_df: pd.DataFrame,
    te_df: pd.DataFrame,
    all_df: pd.DataFrame,
    variant: AlignVariant,
    rank: int,
    reg: float,
    epochs: int,
    lr: float,
    gcols_drug: list[str],
    gcols_do: list[str],
    device: torch.device,
) -> dict[str, object]:
    n_genes = len(gcols_do)
    model = LinkedAlignLayer(n_genes=n_genes, rank=rank, variant=variant).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr) if params else None

    xtr, ytr, mtr = _tensor_triplets(tr_df, gcols_drug, gcols_do, device=device)
    xca, yca, mca = _tensor_triplets(ca_df, gcols_drug, gcols_do, device=device)
    xte, yte, mte = _tensor_triplets(te_df, gcols_drug, gcols_do, device=device)

    if opt is not None and len(tr_df) > 0:
        for _ in range(int(epochs)):
            pred, B = model(xtr, mtr)
            mse = torch.mean((pred - ytr) ** 2)
            reg_loss = torch.tensor(0.0, device=device)
            if variant.use_low_rank:
                reg_loss = float(reg) * torch.mean((B - torch.eye(n_genes, device=device)) ** 2)
            loss = mse + reg_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    with torch.no_grad():
        pca, B = model(xca, mca)
        pte, _ = model(xte, mte)
        cal_mse = float(torch.mean((pca - yca) ** 2).item()) if len(ca_df) else math.nan
        test_mse = float(torch.mean((pte - yte) ** 2).item()) if len(te_df) else math.nan
        fro = float(torch.norm(B - torch.eye(n_genes, device=device)).item())
        beta_lof = float(torch.sigmoid(model.beta_lof_raw).item())
        beta_gof = float(torch.sigmoid(model.beta_gof_raw).item())
    test_cos = _cosine_mean(pte.cpu().numpy(), yte.cpu().numpy()) if len(te_df) else 0.0
    mrr, hits1 = _nearest_target_retrieval(
        test_df=te_df,
        all_df=all_df,
        model=model,
        gcols_do=gcols_do,
        device=device,
    )
    return {
        "model": model,
        "cal_mse": cal_mse,
        "test_mse": test_mse,
        "test_cosine": test_cos,
        "fro_norm_B_minus_I": fro,
        "beta_lof": beta_lof,
        "beta_gof": beta_gof,
        "mrr": mrr,
        "hits1": hits1,
    }


def train_linked_align(repo_root: Path) -> None:
    cfg = load_yaml(repo_root / "configs/linked_benchmark.yaml")
    out_root = ensure_dir(repo_root / str(cfg["alignment"]["output_root"]))
    table_dir = ensure_dir(out_root / "tables")
    log_dir = ensure_dir(out_root / "logs")
    ckpt_dir = ensure_dir(out_root / "checkpoints")

    pair_path = repo_root / str(cfg["output_root"]) / str(cfg["data_dir"]) / "linked_alignment_pairs.parquet"
    if not pair_path.exists():
        (log_dir / "README.md").write_text(
            "# Linked alignment audit\n- status: skipped\n- reason: linked_alignment_pairs.parquet not found\n- interpretation: same-cell positive drug/target perturbation pairs were insufficient under current linkage rules",
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {
                    "scenario": "unavailable",
                    "seed": -1,
                    "entity": "unavailable",
                    "n_train": 0,
                    "n_cal": 0,
                    "n_test": 0,
                    "n_drug": 0,
                    "n_target": 0,
                    "n_cell": 0,
                    "n_family": 0,
                    "heldout_overlap": 0,
                }
            ]
        ).to_csv(table_dir / "linked_align_split_audit.csv", index=False)
        return
    df = pd.read_parquet(pair_path)
    if df.empty:
        (log_dir / "README.md").write_text(
            "# Linked alignment audit\n- status: skipped\n- reason: linked_alignment_pairs.parquet is empty",
            encoding="utf-8",
        )
        pd.DataFrame().to_csv(table_dir / "linked_align_ablation_metrics.csv", index=False)
        return
    df, gcols_drug, gcols_do = _materialize_pair_expression(repo_root, cfg, df)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    rows = []
    audit_rows = []
    scenario_specs = dict(cfg["splits"]["scenarios"])
    if df["family_key"].astype(str).nunique() > 1:
        scenario_specs["family_heldout"] = {"entity": "family_key"}
    seeds = [int(x) for x in cfg["splits"]["seeds"]]
    cal_fraction = float(cfg["splits"]["cal_fraction"])
    test_fraction = float(cfg["splits"]["test_fraction"])
    variants = _variant_specs(list(cfg["alignment"]["variants"]))

    for scenario, scfg in scenario_specs.items():
        split_dir = ensure_dir(out_root / "splits" / str(scenario))
        for seed in seeds:
            split = _entity_holdout_split(
                df,
                entity_col=str(scfg["entity"]),
                seed=int(seed),
                cal_fraction=cal_fraction,
                test_fraction=test_fraction,
            )
            save_json(split_dir / f"seed_{seed}.json", split)

            tr_df = _subset_by_ids(df, split["train"])
            ca_df = _subset_by_ids(df, split["cal"])
            te_df = _subset_by_ids(df, split["test"])
            audit_rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "entity": str(scfg["entity"]),
                    "n_train": len(tr_df),
                    "n_cal": len(ca_df),
                    "n_test": len(te_df),
                    "n_drug": int(te_df["drug_key"].nunique()),
                    "n_target": int(te_df["target_key"].nunique()),
                    "n_cell": int(te_df["cell_id"].nunique()),
                    "n_family": int(te_df["family_key"].astype(str).nunique()),
                    "heldout_overlap": len(set(split["train_entities"]) & set(split["test_entities"])),
                }
            )

            for variant in variants:
                best = None
                for rank in cfg["alignment"]["rank_candidates"]:
                    for reg in cfg["alignment"]["reg_candidates"]:
                        fit = _train_variant(
                            tr_df=tr_df,
                            ca_df=ca_df,
                            te_df=te_df,
                            all_df=df,
                            variant=variant,
                            rank=int(rank),
                            reg=float(reg),
                            epochs=int(cfg["alignment"]["epochs"]),
                            lr=float(cfg["alignment"]["lr"]),
                            gcols_drug=gcols_drug,
                            gcols_do=gcols_do,
                            device=device,
                        )
                        row = {
                            "scenario": scenario,
                            "seed": seed,
                            "variant": variant.name,
                            "rank": int(rank),
                            "reg": float(reg),
                            "cal_mse": fit["cal_mse"],
                            "test_mse": fit["test_mse"],
                            "test_cosine": fit["test_cosine"],
                            "fro_norm_B_minus_I": fit["fro_norm_B_minus_I"],
                            "beta_lof": fit["beta_lof"],
                            "beta_gof": fit["beta_gof"],
                            "mrr": fit["mrr"],
                            "hits1": fit["hits1"],
                            "n_pair": len(tr_df) + len(ca_df),
                            "n_drug": int(tr_df["drug_key"].astype(str).nunique()),
                            "n_target": int(tr_df["target_key"].astype(str).nunique()),
                            "n_cell": int(tr_df["cell_id"].astype(str).nunique()),
                            "n_family": int(tr_df["family_key"].astype(str).nunique()),
                        }
                        rows.append(row)
                        if best is None or float(fit["cal_mse"]) < float(best["cal_mse"]):
                            best = {**fit, "rank": int(rank), "reg": float(reg)}
                if best is not None:
                    model = best["model"]
                    with torch.no_grad():
                        B = (torch.eye(len(gcols_do), device=model.u.device) + model.u @ model.v.T).cpu().numpy()
                        np.savez_compressed(
                            ckpt_dir / f"{scenario}_seed_{seed}_{variant.name}.npz",
                            B=B,
                            beta_lof=np.array([best["beta_lof"]]),
                            beta_gof=np.array([best["beta_gof"]]),
                        )

    pd.DataFrame(rows).to_csv(table_dir / "linked_align_ablation_metrics.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(table_dir / "linked_align_split_audit.csv", index=False)
    lines = [
        "# Linked alignment audit",
        f"- scenarios: {len(scenario_specs)}",
        f"- seeds: {len(seeds)}",
        f"- output_root: {out_root.as_posix()}",
        "- note: this pipeline is additive and does not overwrite legacy align checkpoints",
    ]
    (log_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
