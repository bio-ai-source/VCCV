from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SyntheticConfig:
    n_genes: int
    n_instances: int
    seed: int


def gene_cols(n_genes: int) -> list[str]:
    return [f"G{i}" for i in range(n_genes)]


def var_cols(n_genes: int) -> list[str]:
    return [f"V{i}" for i in range(n_genes)]


def _select_gene_space(do_df: pd.DataFrame, n_genes: int) -> tuple[list[str], list[str]]:
    all_g = sorted(
        [c for c in do_df.columns if c.startswith("G") and c[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )
    if not all_g:
        raise ValueError("No gene columns found in real do signatures.")
    n_use = min(len(all_g), max(32, n_genes))
    sel_g = all_g[:n_use]
    sel_v = []
    for g in sel_g:
        v = f"V{g[1:]}"
        if v in do_df.columns:
            sel_v.append(v)
        else:
            sel_v.append(v)
            do_df[v] = 0.15
    return sel_g, sel_v


def _build_observed_dictionary(do_df: pd.DataFrame, gcols: list[str], vcols: list[str]) -> pd.DataFrame:
    key_cols = ["target_key", "context_key", "pert_time", "pert_dose", "platform", "batch", "mode"]
    agg = (
        do_df.groupby(key_cols, as_index=False)
        .agg(
            **{c: (c, "mean") for c in gcols},
            **{c: (c, "mean") for c in vcols},
            n_rep=("n_rep", "sum"),
            cc_q75=("cc_q75", "mean"),
            qc_do=("qc_do", "mean"),
        )
        .reset_index(drop=True)
    )
    agg["n_rep"] = agg["n_rep"].clip(lower=1).astype(int)
    agg["cc_q75"] = agg["cc_q75"].fillna(0.4).clip(0.0, 1.0)
    agg["qc_do"] = agg["qc_do"].fillna(0.4).clip(0.0, 1.0)
    for c in vcols:
        agg[c] = agg[c].fillna(0.15).clip(lower=1e-4)
    return agg


def _align_params(n_genes: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    rank = max(4, min(16, n_genes // 24))
    u = rng.normal(0.0, 0.025, size=(n_genes, rank))
    v = rng.normal(0.0, 0.025, size=(n_genes, rank))
    b_lof = rng.normal(0.0, 0.02, size=n_genes)
    b_gof = rng.normal(0.0, 0.02, size=n_genes)
    beta_lof = 0.78
    beta_gof = 0.66
    B = np.eye(n_genes) + u @ v.T
    return {
        "B": B,
        "b_LoF": b_lof,
        "b_GoF": b_gof,
        "beta_LoF": np.array([beta_lof]),
        "beta_GoF": np.array([beta_gof]),
    }


def _choose_observed_row(
    obs_by_target_mode: dict[tuple[str, str], pd.DataFrame],
    target_key: str,
    mode: str,
    rng: np.random.Generator,
    ref: pd.Series | None = None,
) -> pd.Series | None:
    cand = obs_by_target_mode.get((target_key, mode))
    if cand is None or cand.empty:
        return None
    if ref is None:
        return cand.iloc[int(rng.integers(0, len(cand)))]
    d = (
        (cand["context_key"] != ref["context_key"]).astype(float) * 2.0
        + np.abs(cand["pert_time"].astype(float) - float(ref["pert_time"])) * 0.5
        + np.abs(np.log10(cand["pert_dose"].astype(float) + 1e-6) - np.log10(float(ref["pert_dose"]) + 1e-6))
    )
    return cand.iloc[int(np.argmin(d.to_numpy(dtype=float)))]


def build_synthetic_mechanism_data(
    dti_df: pd.DataFrame,
    do_df: pd.DataFrame,
    cfg: SyntheticConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    """
    Build semi-synthetic mechanism instances driven by real L1000 do signatures.
    """
    rng = np.random.default_rng(cfg.seed)
    do_work = do_df.copy()
    gcols, vcols = _select_gene_space(do_work, cfg.n_genes)
    n_genes = len(gcols)
    obs = _build_observed_dictionary(do_work, gcols=gcols, vcols=vcols)

    available_targets = set(obs["target_key"].unique().tolist())
    pos = dti_df[(dti_df["y"] == 1) & (dti_df["target_key"].isin(available_targets))].copy()
    if len(pos) < 50:
        raise ValueError("Not enough positive DTI pairs with real do coverage for synthetic mechanism generation.")

    # Keep top positive pairs per drug as candidate mechanism targets.
    pos = pos.sort_values("affinity", ascending=False).reset_index(drop=True)
    top_by_drug: dict[str, list[str]] = (
        pos.groupby("drug_key")["target_key"]
        .apply(lambda x: list(dict.fromkeys(x.tolist()))[:8])
        .to_dict()
    )

    obs_by_target_mode: dict[tuple[str, str], pd.DataFrame] = {}
    for (t, m), sub in obs.groupby(["target_key", "mode"]):
        obs_by_target_mode[(t, m)] = sub.reset_index(drop=True)

    align = _align_params(n_genes=n_genes, rng=rng)
    B = align["B"]
    b_lof = align["b_LoF"]
    b_gof = align["b_GoF"]
    beta = {"LoF": float(align["beta_LoF"][0]), "GoF": float(align["beta_GoF"][0])}

    obs_means = obs[gcols].to_numpy(dtype=float)
    if len(obs_means) < 10:
        raise ValueError("Observed do table is too small to form stress prototypes.")
    stress_centers = obs_means[rng.choice(len(obs_means), size=3, replace=False)]
    stress_weights = np.array([0.4, 0.35, 0.25], dtype=float)

    # Sample mechanism instances from positive DTI pairs with replacement if needed.
    selected = pos.sample(
        n=min(cfg.n_instances, len(pos)),
        random_state=cfg.seed,
        replace=len(pos) < cfg.n_instances,
    ).reset_index(drop=True)

    inst_rows: list[dict[str, object]] = []
    truth_rows: list[dict[str, object]] = []
    for i, dt in selected.iterrows():
        drug = dt["drug_key"]
        true_t = dt["target_key"]
        cands = top_by_drug.get(drug, [])
        if not cands:
            continue

        mode = "LoF" if rng.random() < 0.78 else "GoF"
        base = _choose_observed_row(obs_by_target_mode, true_t, mode, rng=rng, ref=None)
        if base is None:
            # Direction fallback if only one direction is available.
            mode = "GoF" if mode == "LoF" else "LoF"
            base = _choose_observed_row(obs_by_target_mode, true_t, mode, rng=rng, ref=None)
        if base is None:
            continue

        is_null = rng.random() < 0.14
        is_poly = (not is_null) and (rng.random() < 0.24)
        u_vec: np.ndarray
        poly_targets = ""
        poly_weights = ""

        if is_null:
            k = int(rng.choice(3, p=stress_weights))
            sigma = np.sqrt(np.maximum(np.median(obs[vcols].to_numpy(dtype=float), axis=0), 1e-4))
            u_vec = stress_centers[k] + rng.normal(0.0, 1.0, size=n_genes) * sigma
        else:
            do_true = _choose_observed_row(obs_by_target_mode, true_t, mode, rng=rng, ref=base)
            if do_true is None:
                continue
            mu_true = do_true[gcols].to_numpy(dtype=float)
            var_true = np.maximum(do_true[vcols].to_numpy(dtype=float), 1e-4)

            if is_poly:
                alt = None
                for t2 in cands:
                    if t2 == true_t:
                        continue
                    r2 = _choose_observed_row(obs_by_target_mode, t2, mode, rng=rng, ref=base)
                    if r2 is not None:
                        alt = (t2, r2)
                        break
                if alt is not None:
                    t2, r2 = alt
                    raw_w = rng.uniform(0.2, 1.0, size=2)
                    w = raw_w / raw_w.sum()
                    mu_mix = w[0] * mu_true + w[1] * r2[gcols].to_numpy(dtype=float)
                    var_mix = (w[0] ** 2) * var_true + (w[1] ** 2) * np.maximum(
                        r2[vcols].to_numpy(dtype=float), 1e-4
                    )
                    poly_targets = f"{true_t};{t2}"
                    poly_weights = f"{w[0]:.4f};{w[1]:.4f}"
                    mu_true = mu_mix
                    var_true = var_mix
                else:
                    is_poly = False

            bias = b_lof if mode == "LoF" else b_gof
            mu_aligned = beta[mode] * (B @ mu_true + bias)
            std_aligned = np.sqrt(np.maximum((beta[mode] ** 2) * var_true + 0.02, 1e-5))
            u_vec = mu_aligned + rng.normal(0.0, 1.0, size=n_genes) * std_aligned

        inst_id = f"inst_{i:07d}"
        cc = float(np.clip(float(base["cc_q75"]) + rng.normal(0.0, 0.05), 0.0, 1.0))
        out_row: dict[str, object] = {
            "instance_id": inst_id,
            "drug_key": drug,
            "context_key": base["context_key"],
            "pert_time": float(base["pert_time"]),
            "pert_dose": float(base["pert_dose"]),
            "platform": base["platform"],
            "batch": base["batch"],
            "mode": mode,
            "n_rep": int(max(2, int(base["n_rep"]))),
            "cc_q75": cc,
            "activity_l2": float(np.linalg.norm(u_vec)),
        }
        for gn, gv in zip(gcols, u_vec):
            out_row[gn] = float(gv)
        inst_rows.append(out_row)

        truth_rows.append(
            {
                "instance_id": inst_id,
                "true_target_key": true_t,
                "true_mode": mode,
                "is_null": int(is_null),
                "is_poly": int(is_poly),
                "poly_targets": poly_targets,
                "poly_weights": poly_weights,
            }
        )

    if not inst_rows:
        raise ValueError("Failed to generate synthetic mechanism instances from real do table.")

    sig_df = pd.DataFrame(inst_rows).reset_index(drop=True)
    truth_df = pd.DataFrame(truth_rows).reset_index(drop=True)
    do_out_cols = [
        "target_key",
        "context_key",
        "pert_time",
        "pert_dose",
        "platform",
        "batch",
        "mode",
        "n_rep",
        "cc_q75",
        "qc_do",
        *gcols,
        *vcols,
    ]
    do_out = do_work[do_out_cols].copy().reset_index(drop=True)
    return sig_df, do_out, align, truth_df
