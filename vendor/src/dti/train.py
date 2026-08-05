from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import pandas as pd
from DeepPurpose import DTI as dp_dti
from DeepPurpose import utils as dp_utils
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import torch

from src.utils.io import ensure_dir, load_json, load_yaml
from src.utils.metrics import summarize_binding
from src.utils.seed import set_global_seed

FRONTIER_PROB_COLS = [
    "prob_DTIAM_2025_Reimpl",
    "prob_EviDTI_2025_Reimpl",
    "prob_DeepDTAGen_2025_Reimpl",
]

FORCED_SINGLE_BY_SCENARIO_DATASET: dict[tuple[str, str], str] = {}

# Legacy forced-single rules are intentionally disabled after introducing
# frontier experts; scenario-specific choice is now calibration-driven.
FORCED_SINGLE_BY_SCENARIO_SEED_DATASET: dict[tuple[str, int, str], str] = {}


@dataclass
class DTIModelSpec:
    name: str
    drug_encoding: str
    target_encoding: str
    epochs: int
    batch_size: int
    lr: float
    extra_config: dict[str, object]


def _model_specs() -> list[DTIModelSpec]:
    # Real neural models trained via DeepPurpose (no sklearn proxy).
    return [
        DTIModelSpec(
            name="DeepDTA",
            drug_encoding="CNN",
            target_encoding="CNN",
            epochs=3,
            batch_size=192,
            lr=8e-4,
            extra_config={
                "cnn_drug_filters": [32, 64, 96],
                "cnn_drug_kernels": [4, 6, 8],
                "cnn_target_filters": [32, 64, 96],
                "cnn_target_kernels": [4, 8, 12],
            },
        ),
        DTIModelSpec(
            name="GraphDTA",
            drug_encoding="MPNN",
            target_encoding="CNN",
            epochs=3,
            batch_size=160,
            lr=7e-4,
            extra_config={
                "mpnn_hidden_size": 64,
                "mpnn_depth": 3,
                "cnn_target_filters": [32, 64, 96],
                "cnn_target_kernels": [4, 8, 12],
            },
        ),
        DTIModelSpec(
            name="MolTrans",
            drug_encoding="Transformer",
            target_encoding="Transformer",
            epochs=2,
            batch_size=128,
            lr=6e-4,
            extra_config={
                "transformer_emb_size_drug": 64,
                "transformer_intermediate_size_drug": 128,
                "transformer_num_attention_heads_drug": 4,
                "transformer_n_layer_drug": 2,
                "transformer_emb_size_target": 32,
                "transformer_intermediate_size_target": 128,
                "transformer_num_attention_heads_target": 4,
                "transformer_n_layer_target": 2,
            },
        ),
    ]


def _subset_by_ids(df: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    id_set = set(ids)
    return df[df["instance_id"].isin(id_set)].copy().reset_index(drop=True)


def _fit_platt(scores: np.ndarray, y: np.ndarray):
    scores = np.asarray(scores, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=int)
    if len(scores) == 0:
        return lambda x: np.full(len(np.asarray(x).reshape(-1)), 0.5, dtype=float)
    if len(np.unique(y)) < 2:
        c = float(y.mean()) if len(y) > 0 else 0.5
        return lambda x: np.full(len(np.asarray(x).reshape(-1)), c, dtype=float)
    lr = LogisticRegression(max_iter=500)
    lr.fit(scores, y)
    return lambda x: lr.predict_proba(np.asarray(x, dtype=float).reshape(-1, 1))[:, 1]


def _safe_predict(model, df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.empty(0, dtype=float)
    p = model.predict(df.reset_index(drop=True))
    p = np.asarray(p, dtype=float)
    return np.clip(p, 1e-6, 1.0 - 1e-6)


def _safe_cosine(u: np.ndarray, v: np.ndarray, un: float, vn: float) -> float:
    if un <= 1e-12 or vn <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(u, v) / (un * vn), -1.0, 1.0))


def _augment_with_vccv_aux_features(dti_df: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    sig = pd.read_parquet(repo_root / "data/processed/signatures_drug.parquet")
    dof = pd.read_parquet(repo_root / "data/processed/do_fused_mu_var.parquet")
    gcols = sorted([c for c in sig.columns if c.startswith("G") and c[1:].isdigit()], key=lambda x: int(x[1:]))

    sig_tbl = sig[["instance_id", "activity_l2", *gcols]].drop_duplicates("instance_id").reset_index(drop=True)
    sig_vec: dict[str, np.ndarray] = {}
    sig_norm: dict[str, float] = {}
    sig_activity: dict[str, float] = {}
    for row in sig_tbl.itertuples(index=False):
        inst = str(row[0])
        act = float(row[1])
        vec = np.asarray(row[2:], dtype=np.float32)
        sig_vec[inst] = vec
        sig_norm[inst] = float(np.linalg.norm(vec))
        sig_activity[inst] = act

    do_tbl = dof[["target_key", "context_key", "pert_time", "pert_dose", "mode", *gcols]].reset_index(drop=True)
    do_vec: dict[tuple[str, str, float, float, str], tuple[np.ndarray, float]] = {}
    for row in do_tbl.itertuples(index=False):
        key = (str(row[0]), str(row[1]), float(row[2]), float(row[3]), str(row[4]))
        vec = np.asarray(row[5:], dtype=np.float32)
        do_vec[key] = (vec, float(np.linalg.norm(vec)))

    n = len(dti_df)
    aux_lof = np.zeros(n, dtype=np.float32)
    aux_gof = np.zeros(n, dtype=np.float32)
    aux_max = np.zeros(n, dtype=np.float32)
    aux_mean = np.zeros(n, dtype=np.float32)
    aux_gap = np.zeros(n, dtype=np.float32)
    aux_present = np.zeros(n, dtype=np.float32)
    aux_sig_act = np.zeros(n, dtype=np.float32)

    for i, row in enumerate(dti_df.itertuples(index=False)):
        inst = str(row.instance_id)
        t = str(row.target_key)
        c = str(row.context_key)
        tm = float(row.pert_time)
        ds = float(row.pert_dose)
        u = sig_vec.get(inst)
        if u is None:
            continue
        un = sig_norm[inst]
        aux_sig_act[i] = float(sig_activity.get(inst, 0.0))

        vals = []
        for mode in ("LoF", "GoF"):
            vv = do_vec.get((t, c, tm, ds, mode))
            if vv is None:
                if mode == "LoF":
                    aux_lof[i] = 0.0
                else:
                    aux_gof[i] = 0.0
                continue
            v, vn = vv
            cs = _safe_cosine(u, v, un=un, vn=vn)
            if mode == "LoF":
                aux_lof[i] = cs
            else:
                aux_gof[i] = cs
            vals.append(cs)
        if vals:
            aux_present[i] = 1.0
            aux_max[i] = float(np.max(vals))
            aux_mean[i] = float(np.mean(vals))
            aux_gap[i] = float(abs(aux_lof[i] - aux_gof[i]))

    out = dti_df.copy()
    out["aux_do_lof"] = aux_lof
    out["aux_do_gof"] = aux_gof
    out["aux_do_max"] = aux_max
    out["aux_do_mean"] = aux_mean
    out["aux_do_gap"] = aux_gap
    out["aux_do_present"] = aux_present
    out["aux_sig_activity"] = aux_sig_act
    return out


def _add_train_prior_aux_features(tr_meta: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    global_rate = float(tr_meta["y"].mean())
    drug_rate = tr_meta.groupby("drug_key")["y"].mean().to_dict()
    target_rate = tr_meta.groupby("target_key")["y"].mean().to_dict()
    context_rate = tr_meta.groupby("context_key")["y"].mean().to_dict()
    pair_rate = tr_meta.groupby(["drug_key", "target_key"])["y"].mean().to_dict()

    out["aux_train_global_rate"] = global_rate
    out["aux_train_drug_rate"] = out["drug_key"].map(drug_rate).fillna(global_rate).astype(float)
    out["aux_train_target_rate"] = out["target_key"].map(target_rate).fillna(global_rate).astype(float)
    out["aux_train_context_rate"] = out["context_key"].map(context_rate).fillna(global_rate).astype(float)
    out["aux_train_pair_rate"] = [
        float(pair_rate.get((d, t), global_rate)) for d, t in zip(out["drug_key"].tolist(), out["target_key"].tolist())
    ]
    out["aux_is_new_drug"] = (~out["drug_key"].isin(drug_rate)).astype(float)
    out["aux_is_new_target"] = (~out["target_key"].isin(target_rate)).astype(float)
    return out


def _build_stack_features(
    frame: pd.DataFrame,
    model_cols: list[str],
    aux_cols: list[str],
) -> tuple[np.ndarray, list[str]]:
    x_model = frame[model_cols].to_numpy(dtype=float)
    names = list(model_cols)
    blocks = [x_model]
    if aux_cols:
        x_aux = frame[aux_cols].to_numpy(dtype=float)
        blocks.append(x_aux)
        names.extend(aux_cols)

        if "aux_do_max" in frame.columns:
            z = frame["aux_do_max"].to_numpy(dtype=float).reshape(-1, 1)
            for j, col in enumerate(model_cols):
                blocks.append(x_model[:, [j]] * z)
                names.append(f"{col}*aux_do_max")
        if "aux_train_target_rate" in frame.columns:
            z = frame["aux_train_target_rate"].to_numpy(dtype=float).reshape(-1, 1)
            for j, col in enumerate(model_cols):
                blocks.append(x_model[:, [j]] * z)
                names.append(f"{col}*aux_train_target_rate")
    x = np.column_stack(blocks)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    return x, names


def _standardize_with_calibration(x_cal: np.ndarray, x_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x_cal.mean(axis=0, keepdims=True)
    std = x_cal.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x_cal - mu) / std, (x_eval - mu) / std


def _oof_logistic_and_test(
    *,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    x_test: np.ndarray,
    c_val: float,
    max_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray] | None:
    y = np.asarray(y_cal, dtype=int)
    if len(np.unique(y)) < 2:
        return None
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    n_splits = min(max_splits, n_pos, n_neg)
    if n_splits < 2:
        return None

    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20260219)
    for tr_idx, va_idx in skf.split(x_cal, y):
        lr = LogisticRegression(max_iter=800, C=float(c_val))
        lr.fit(x_cal[tr_idx], y[tr_idx])
        oof[va_idx] = np.clip(lr.predict_proba(x_cal[va_idx])[:, 1], 1e-6, 1.0 - 1e-6)

    lr_full = LogisticRegression(max_iter=800, C=float(c_val))
    lr_full.fit(x_cal, y)
    p_test = np.clip(lr_full.predict_proba(x_test)[:, 1], 1e-6, 1.0 - 1e-6)
    return oof, p_test


def _prepare_encoded_tables(dti_df: pd.DataFrame, specs: list[DTIModelSpec]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    base_meta_cols = [
        "instance_id",
        "dataset",
        "drug_key",
        "target_key",
        "context_key",
        "smiles",
        "target_seq",
        "y",
    ]
    base = dti_df[base_meta_cols].reset_index(drop=True)
    for spec in specs:
        enc = dp_utils.data_process(
            X_drug=base["smiles"].tolist(),
            X_target=base["target_seq"].tolist(),
            y=base["y"].astype(int).tolist(),
            drug_encoding=spec.drug_encoding,
            target_encoding=spec.target_encoding,
            split_method="no_split",
        )
        enc = enc.reset_index(drop=True)
        for c in base_meta_cols:
            enc[c] = base[c].values
        tables[spec.name] = enc
    return tables


def _train_model(
    tr: pd.DataFrame,
    spec: DTIModelSpec,
    checkpoint_dir: Path,
) -> object | None:
    y = tr["y"].to_numpy(dtype=int)
    if len(tr) < 200 or len(np.unique(y)) < 2:
        return None
    cuda_id = None
    if torch.cuda.is_available():
        n_cuda = int(torch.cuda.device_count())
        env_cuda = os.getenv("VCCV_CUDA_ID", "1").strip()
        try:
            req_cuda = int(env_cuda)
        except Exception:
            req_cuda = 1
        if req_cuda < 0 or req_cuda >= n_cuda:
            req_cuda = min(1, n_cuda - 1)
        cuda_id = int(req_cuda)
    cfg = dp_utils.generate_config(
        drug_encoding=spec.drug_encoding,
        target_encoding=spec.target_encoding,
        train_epoch=int(spec.epochs),
        batch_size=int(spec.batch_size),
        LR=float(spec.lr),
        num_workers=0,
        cuda_id=cuda_id,
        result_folder=str(checkpoint_dir),
        **spec.extra_config,
    )
    model = dp_dti.model_initialize(**cfg)
    model.train(tr.reset_index(drop=True), None, None, verbose=False)
    model.save_model(str(checkpoint_dir))
    return model


def _add_metric_rows(
    rows: list[dict[str, object]],
    *,
    probs: np.ndarray,
    labels_df: pd.DataFrame,
    model_name: str,
    scenario: str,
    seed: int,
) -> None:
    if labels_df.empty:
        return
    tmp = labels_df.copy()
    tmp["p"] = probs
    for dataset, sub in tmp.groupby("dataset"):
        bm = summarize_binding(sub["y"].to_numpy(dtype=int), sub["p"].to_numpy(dtype=float))
        rows.append(
            {
                "scenario": scenario,
                "seed": int(seed),
                "dataset": dataset,
                "model": model_name,
                "auc": bm.auc,
                "pr_auc": bm.pr_auc,
                "ece": bm.ece,
                "nll": bm.nll,
                "brier": bm.brier,
                "n_test": int(len(sub)),
            }
        )


def _metric_sort_key(metrics) -> tuple[float, float, float, float]:
    auc = float(metrics.auc) if np.isfinite(metrics.auc) else -1.0
    pr_auc = float(metrics.pr_auc) if np.isfinite(metrics.pr_auc) else -1.0
    ece = float(metrics.ece) if np.isfinite(metrics.ece) else 1e9
    nll = float(metrics.nll) if np.isfinite(metrics.nll) else 1e9
    return (auc, pr_auc, -ece, -nll)


def _search_convex_auc_weights(
    *,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    single_aucs: list[float],
    seed: int,
    n_samples: int = 800,
) -> np.ndarray:
    x = np.asarray(x_cal, dtype=float)
    y = np.asarray(y_cal, dtype=int)
    m = int(x.shape[1])
    if m <= 1 or len(np.unique(y)) < 2:
        return np.ones(m, dtype=float) / max(m, 1)

    rs = np.random.RandomState(int(seed) + 20260219)
    auc_arr = np.asarray(single_aucs, dtype=float)
    auc_arr = np.where(np.isfinite(auc_arr), auc_arr, 0.5)
    base = np.clip(auc_arr - 0.5, 0.0, None)
    if float(base.sum()) <= 1e-12:
        base = np.ones(m, dtype=float)
    base = base / base.sum()

    cand_w: list[np.ndarray] = []
    cand_w.append(np.ones(m, dtype=float) / m)  # uniform
    cand_w.append(base.copy())  # auc-weighted prior
    for i in range(m):
        e = np.zeros(m, dtype=float)
        e[i] = 1.0
        cand_w.append(e)
    top = np.argsort(auc_arr)[::-1][: min(4, m)].tolist()
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            w = np.zeros(m, dtype=float)
            w[top[i]] = 0.5
            w[top[j]] = 0.5
            cand_w.append(w)

    alpha1 = np.ones(m, dtype=float)
    alpha2 = 0.6 + 4.0 * base * m
    alpha3 = 0.3 + 2.0 * base * m
    n_each = max(1, int(n_samples // 3))
    for _ in range(n_each):
        cand_w.append(rs.dirichlet(alpha1))
    for _ in range(n_each):
        cand_w.append(rs.dirichlet(alpha2))
    for _ in range(n_samples - 2 * n_each):
        cand_w.append(rs.dirichlet(alpha3))

    best_w = cand_w[0]
    best_auc = -1.0
    for w in cand_w:
        p = np.clip(x @ w, 1e-6, 1.0 - 1e-6)
        met = summarize_binding(y, p)
        auc = float(met.auc) if np.isfinite(float(met.auc)) else -1.0
        if auc > best_auc:
            best_auc = auc
            best_w = w
    return np.asarray(best_w, dtype=float)


def _load_frontier_predictions(
    repo_root: Path,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    cal_path = repo_root / "results/predictions_json/frontier_cal_predictions.parquet"
    te_path = repo_root / "results/predictions_json/frontier_split_predictions.parquet"
    if (not cal_path.exists()) or (not te_path.exists()):
        return None, None
    cal_df = pd.read_parquet(cal_path)
    te_df = pd.read_parquet(te_path)
    for df in (cal_df, te_df):
        df["instance_id"] = df["instance_id"].astype(str)
        df["scenario"] = df["scenario"].astype(str)
        df["seed"] = df["seed"].astype(int)
        for col in FRONTIER_PROB_COLS:
            if col in df.columns:
                df[col] = np.clip(df[col].astype(float), 1e-6, 1.0 - 1e-6)
    return cal_df, te_df


def _choose_dataset_ensemble(
    *,
    cal_sub: pd.DataFrame,
    test_sub: pd.DataFrame,
    model_cols: list[str],
    aux_cols: list[str],
    scenario_seed: int,
):
    y_ca = cal_sub["y"].to_numpy(dtype=int)
    x_ca = cal_sub[model_cols].to_numpy(dtype=float)
    x_te = test_sub[model_cols].to_numpy(dtype=float)

    candidates: list[dict[str, object]] = []

    # Candidate 1..K: each single calibrated model.
    single_aucs: list[float] = []
    for i, col in enumerate(model_cols):
        p_ca = np.clip(x_ca[:, i], 1e-6, 1.0 - 1e-6)
        p_te = np.clip(x_te[:, i], 1e-6, 1.0 - 1e-6)
        met = summarize_binding(y_ca, p_ca)
        candidates.append(
            {
                "name": f"single:{col.replace('prob_', '')}",
                "p_ca": p_ca,
                "p_te": p_te,
                "metrics": met,
                "detail": "",
            }
        )
        single_aucs.append(float(met.auc) if np.isfinite(met.auc) else 0.5)

    best_single_idx = int(np.argmax(single_aucs)) if len(single_aucs) > 0 else 0
    best_single_auc = float(single_aucs[best_single_idx]) if len(single_aucs) > 0 else 0.5
    rank_idx = np.argsort(np.asarray(single_aucs, dtype=float))[::-1].tolist()
    top_k = max(2, min(4, len(model_cols)))
    stack_idx = rank_idx[:top_k]
    x_ca_stack = x_ca[:, stack_idx]
    x_te_stack = x_te[:, stack_idx]
    model_cols_stack = [model_cols[i] for i in stack_idx]

    def _safe_val(v: float, bad: float) -> float:
        return float(v) if np.isfinite(float(v)) else bad

    def _robust_single_score(met) -> float:
        auc = _safe_val(met.auc, 0.0)
        pr = _safe_val(met.pr_auc, 0.0)
        ece = _safe_val(met.ece, 1.0)
        nll = _safe_val(met.nll, 1.0)
        return 0.55 * auc + 0.35 * pr - 0.05 * ece - 0.05 * nll

    novelty_drug = 0.0
    novelty_target = 0.0
    if "aux_is_new_drug" in cal_sub.columns:
        novelty_drug = float(pd.to_numeric(cal_sub["aux_is_new_drug"], errors="coerce").fillna(0.0).mean())
    if "aux_is_new_target" in cal_sub.columns:
        novelty_target = float(pd.to_numeric(cal_sub["aux_is_new_target"], errors="coerce").fillna(0.0).mean())
    novelty = max(novelty_drug, novelty_target)

    # OOD-robust single routing (Cal-only): avoid unstable complex fusion in extreme holdout regimes.
    if novelty_drug >= 0.80:
        allow = {"MolTrans", "DTIAM_2025_Reimpl", "DeepDTAGen_2025_Reimpl", "EviDTI_2025_Reimpl"}
        pool = [c for c in candidates if str(c["name"]).startswith("single:") and str(c["name"]).split("single:", 1)[1] in allow]
        if pool:
            return max(pool, key=lambda d: _robust_single_score(d["metrics"]))
    if novelty_target >= 0.80:
        allow = {"EviDTI_2025_Reimpl", "DeepDTAGen_2025_Reimpl", "DTIAM_2025_Reimpl", "MolTrans"}
        pool = [c for c in candidates if str(c["name"]).startswith("single:") and str(c["name"]).split("single:", 1)[1] in allow]
        if pool:
            return max(pool, key=lambda d: _robust_single_score(d["metrics"]))

    # Candidate: uniform average.
    p_ca_mean = np.clip(x_ca.mean(axis=1), 1e-6, 1.0 - 1e-6)
    p_te_mean = np.clip(x_te.mean(axis=1), 1e-6, 1.0 - 1e-6)
    candidates.append(
        {
            "name": "mean",
            "p_ca": p_ca_mean,
            "p_te": p_te_mean,
            "metrics": summarize_binding(y_ca, p_ca_mean),
            "detail": "",
        }
    )

    # Candidate: AUC-weighted average (calibration-only weights).
    w = np.maximum(np.asarray(single_aucs, dtype=float) - 0.5, 0.0)
    if float(w.sum()) <= 1e-12:
        w = np.ones_like(w)
    w = w / w.sum()
    p_ca_w = np.clip(x_ca @ w, 1e-6, 1.0 - 1e-6)
    p_te_w = np.clip(x_te @ w, 1e-6, 1.0 - 1e-6)
    w_desc = ",".join([f"{c.replace('prob_', '')}:{wi:.4f}" for c, wi in zip(model_cols, w.tolist())])
    candidates.append(
        {
            "name": "weighted_auc",
            "p_ca": p_ca_w,
            "p_te": p_te_w,
            "metrics": summarize_binding(y_ca, p_ca_w),
            "detail": w_desc,
        }
    )

    w_opt = _search_convex_auc_weights(
        x_cal=x_ca,
        y_cal=y_ca,
        single_aucs=single_aucs,
        seed=int(scenario_seed),
        n_samples=800,
    )
    p_ca_opt = np.clip(x_ca @ w_opt, 1e-6, 1.0 - 1e-6)
    p_te_opt = np.clip(x_te @ w_opt, 1e-6, 1.0 - 1e-6)
    opt_desc = ",".join([f"{c.replace('prob_', '')}:{wi:.4f}" for c, wi in zip(model_cols, w_opt.tolist())])
    candidates.append(
        {
            "name": "convex_auc_search",
            "p_ca": p_ca_opt,
            "p_te": p_te_opt,
            "metrics": summarize_binding(y_ca, p_ca_opt),
            "detail": opt_desc,
        }
    )

    # Candidate: logistic stacker on model probabilities.
    if len(np.unique(y_ca)) >= 2 and len(cal_sub) >= max(80, 4 * len(model_cols)):
        best_lr = None
        for c_val in [0.2, 0.5, 1.0, 2.0]:
            try:
                res = _oof_logistic_and_test(
                    x_cal=x_ca_stack,
                    y_cal=y_ca,
                    x_test=x_te_stack,
                    c_val=float(c_val),
                )
                if res is None:
                    continue
                p_ca_lr, p_te_lr = res
                cand = {
                    "name": "logistic_topk",
                    "p_ca": p_ca_lr,
                    "p_te": p_te_lr,
                    "metrics": summarize_binding(y_ca, p_ca_lr),
                    "detail": f"C={c_val:.2f};k={len(model_cols_stack)}",
                }
                if best_lr is None or _metric_sort_key(cand["metrics"]) > _metric_sort_key(best_lr["metrics"]):
                    best_lr = cand
            except Exception:
                continue
        if best_lr is not None:
            candidates.append(best_lr)

    # Candidate: VCCV mechanism-enhanced stacker with structural features.
    allow_ext_stacker = novelty < 0.80

    if allow_ext_stacker and len(aux_cols) > 0 and len(np.unique(y_ca)) >= 2 and len(cal_sub) >= max(120, 8 * len(model_cols_stack)):
        x_ca_ext, feature_names = _build_stack_features(cal_sub, model_cols=model_cols_stack, aux_cols=aux_cols)
        x_te_ext, _ = _build_stack_features(test_sub, model_cols=model_cols_stack, aux_cols=aux_cols)
        x_ca_std, x_te_std = _standardize_with_calibration(x_ca_ext, x_te_ext)

        best_ext = None
        for c_val in [0.2, 0.5, 1.0, 2.0, 5.0]:
            try:
                res = _oof_logistic_and_test(x_cal=x_ca_std, y_cal=y_ca, x_test=x_te_std, c_val=float(c_val))
                if res is None:
                    continue
                p_ca_ext, p_te_ext = res
                met_ext = summarize_binding(y_ca, p_ca_ext)
                item = {
                    "name": "vccv_logistic_ext",
                    "p_te": p_te_ext,
                    "p_ca": p_ca_ext,
                    "metrics": met_ext,
                    "detail": f"C={c_val:.2f};n_feat={len(feature_names)}",
                }
                if best_ext is None or _metric_sort_key(item["metrics"]) > _metric_sort_key(best_ext["metrics"]):
                    best_ext = item
            except Exception:
                continue
        if best_ext is not None:
            candidates.append(
                {
                    "name": best_ext["name"],
                    "p_ca": best_ext["p_ca"],
                    "p_te": best_ext["p_te"],
                    "metrics": best_ext["metrics"],
                    "detail": best_ext["detail"],
                }
            )

            # Blend best single with mechanism-enhanced stacker; lambda picked on calibration only.
            j_best = best_single_idx
            p_ca_single = np.clip(x_ca[:, j_best], 1e-6, 1.0 - 1e-6)
            p_te_single = np.clip(x_te[:, j_best], 1e-6, 1.0 - 1e-6)
            p_ca_ext = np.asarray(best_ext["p_ca"], dtype=float)
            p_te_ext = np.asarray(best_ext["p_te"], dtype=float)
            best_blend = None
            for lam in np.linspace(0.1, 0.9, 9):
                p_ca_b = np.clip((1.0 - lam) * p_ca_single + lam * p_ca_ext, 1e-6, 1.0 - 1e-6)
                p_te_b = np.clip((1.0 - lam) * p_te_single + lam * p_te_ext, 1e-6, 1.0 - 1e-6)
                met_b = summarize_binding(y_ca, p_ca_b)
                cand = {
                    "name": "vccv_blend_ext_single",
                    "p_ca": p_ca_b,
                    "p_te": p_te_b,
                    "metrics": met_b,
                    "detail": f"lambda={lam:.2f};single={model_cols[j_best].replace('prob_', '')}",
                }
                if best_blend is None or _metric_sort_key(cand["metrics"]) > _metric_sort_key(best_blend["metrics"]):
                    best_blend = cand
            if best_blend is not None:
                candidates.append(best_blend)

    best_single_name = f"single:{model_cols[best_single_idx].replace('prob_', '')}"
    best_single = next((c for c in candidates if c["name"] == best_single_name), None)
    best_any = max(candidates, key=lambda d: _metric_sort_key(d["metrics"]))
    if best_single is not None and str(best_any["name"]) != str(best_single["name"]):
        auc_gain = float(best_any["metrics"].auc) - float(best_single["metrics"].auc)
        pr_gain = float(best_any["metrics"].pr_auc) - float(best_single["metrics"].pr_auc)
        if novelty >= 0.80:
            min_auc_gain = 0.010
            min_pr_gain = 0.004
        else:
            min_auc_gain = 0.003
            min_pr_gain = 0.002
        # Robustness guard: only accept complex fusion when calibration gain is non-trivial.
        if (auc_gain < min_auc_gain) and (pr_gain < min_pr_gain):
            return best_single
        std_single = float(np.std(np.asarray(best_single.get("p_ca", []), dtype=float)))
        std_best = float(np.std(np.asarray(best_any.get("p_ca", []), dtype=float)))
        if novelty >= 0.80 and std_single > 1e-8 and std_best < 0.35 * std_single:
            return best_single
    return best_any


def _train_for_split(
    *,
    dti_df: pd.DataFrame,
    encoded_tables: dict[str, pd.DataFrame],
    split: dict,
    seed: int,
    scenario: str,
    repo_root: Path,
    frontier_cal_df: pd.DataFrame | None,
    frontier_test_df: pd.DataFrame | None,
) -> tuple[list[dict[str, object]], pd.DataFrame, list[dict[str, object]]]:
    tr_meta = _subset_by_ids(dti_df, split["train"])
    ca_meta = _subset_by_ids(dti_df, split["cal"])
    te_meta = _subset_by_ids(dti_df, split["test"])
    if len(tr_meta) < 300 or len(ca_meta) < 100 or len(te_meta) < 100:
        return [], pd.DataFrame(), []

    ca_meta = _add_train_prior_aux_features(tr_meta=tr_meta, frame=ca_meta)
    te_meta = _add_train_prior_aux_features(tr_meta=tr_meta, frame=te_meta)
    aux_cols = sorted([c for c in ca_meta.columns if c.startswith("aux_")])

    ckpt_root = ensure_dir(repo_root / "results/checkpoints/dti")
    metrics_rows: list[dict[str, object]] = []
    selector_rows: list[dict[str, object]] = []
    cal_scores: list[pd.DataFrame] = []
    test_scores: list[pd.DataFrame] = []

    for spec in _model_specs():
        tab = encoded_tables[spec.name]
        tr = _subset_by_ids(tab, split["train"])
        ca = _subset_by_ids(tab, split["cal"])
        te = _subset_by_ids(tab, split["test"])
        if tr.empty or ca.empty or te.empty:
            continue

        model_ckpt = ensure_dir(ckpt_root / spec.name / f"{scenario}_seed_{seed}")
        model = _train_model(tr=tr, spec=spec, checkpoint_dir=model_ckpt)
        if model is None:
            continue

        p_ca_raw = _safe_predict(model, ca)
        p_te_raw = _safe_predict(model, te)
        calibrator = _fit_platt(p_ca_raw, ca["y"].to_numpy(dtype=int))
        p_ca = np.clip(calibrator(p_ca_raw), 1e-6, 1.0 - 1e-6)
        p_te = np.clip(calibrator(p_te_raw), 1e-6, 1.0 - 1e-6)

        te_lab = te_meta[["instance_id", "dataset", "y"]].merge(
            pd.DataFrame({"instance_id": te["instance_id"], "p": p_te}),
            on="instance_id",
            how="inner",
        )
        _add_metric_rows(
            metrics_rows,
            probs=te_lab["p"].to_numpy(dtype=float),
            labels_df=te_lab[["instance_id", "dataset", "y"]].assign(p=te_lab["p"].values),
            model_name=spec.name,
            scenario=scenario,
            seed=seed,
        )

        cal_scores.append(
            pd.DataFrame(
                {
                    "instance_id": ca["instance_id"].astype(str).tolist(),
                    f"prob_{spec.name}": p_ca.tolist(),
                }
            )
        )
        test_scores.append(
            pd.DataFrame(
                {
                    "instance_id": te["instance_id"].astype(str).tolist(),
                    f"prob_{spec.name}": p_te.tolist(),
                }
            )
        )

    if not test_scores:
        return metrics_rows, pd.DataFrame(), selector_rows

    cal_stack = ca_meta[["instance_id", "dataset", "y", *aux_cols]].copy()
    test_stack = te_meta[["instance_id", "dataset", "y", "drug_key", "target_key", "context_key", *aux_cols]].copy()
    for df_sc in cal_scores:
        cal_stack = cal_stack.merge(df_sc, on="instance_id", how="inner")
    for df_sc in test_scores:
        test_stack = test_stack.merge(df_sc, on="instance_id", how="inner")

    if frontier_cal_df is not None:
        fca = frontier_cal_df[
            (frontier_cal_df["scenario"] == str(scenario)) & (frontier_cal_df["seed"] == int(seed))
        ].copy()
        if not fca.empty:
            keep_cols = ["instance_id"] + [c for c in FRONTIER_PROB_COLS if c in fca.columns]
            fca = fca[keep_cols].drop_duplicates("instance_id")
            cal_stack = cal_stack.merge(fca, on="instance_id", how="left")

    if frontier_test_df is not None:
        fte = frontier_test_df[
            (frontier_test_df["scenario"] == str(scenario)) & (frontier_test_df["seed"] == int(seed))
        ].copy()
        if not fte.empty:
            keep_cols = ["instance_id"] + [c for c in FRONTIER_PROB_COLS if c in fte.columns]
            fte = fte[keep_cols].drop_duplicates("instance_id")
            test_stack = test_stack.merge(fte, on="instance_id", how="left")

    model_cols = [c for c in test_stack.columns if c.startswith("prob_")]
    keep_model_cols: list[str] = []
    for col in model_cols:
        cov = float(cal_stack[col].notna().mean())
        uniq = int(cal_stack[col].nunique(dropna=True))
        if cov < 0.70 or uniq < 2:
            continue
        fill_val = float(cal_stack[col].mean(skipna=True))
        if (not np.isfinite(fill_val)) or fill_val <= 0.0 or fill_val >= 1.0:
            fill_val = 0.5
        cal_stack[col] = np.clip(cal_stack[col].fillna(fill_val).to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
        test_stack[col] = np.clip(test_stack[col].fillna(fill_val).to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
        keep_model_cols.append(col)
    model_cols = keep_model_cols
    if len(model_cols) == 0:
        return metrics_rows, pd.DataFrame(), selector_rows

    # Global fallback for rare dataset slices without valid calibration rows.
    x_ca_all = cal_stack[model_cols].to_numpy(dtype=float)
    y_ca_all = cal_stack["y"].to_numpy(dtype=int)
    x_te_all = test_stack[model_cols].to_numpy(dtype=float)
    if len(np.unique(y_ca_all)) < 2:
        p_ens = np.clip(x_te_all.mean(axis=1), 1e-6, 1.0 - 1e-6)
    else:
        ens_global = LogisticRegression(max_iter=500, C=1.0)
        ens_global.fit(x_ca_all, y_ca_all)
        p_ens = np.clip(ens_global.predict_proba(x_te_all)[:, 1], 1e-6, 1.0 - 1e-6)

    # Scenario+seed+dataset specific selector (uses calibration only).
    for dataset, te_ds in test_stack.groupby("dataset"):
        ca_ds = cal_stack[cal_stack["dataset"] == dataset]
        if ca_ds.empty:
            continue
        forced_model = FORCED_SINGLE_BY_SCENARIO_SEED_DATASET.get((scenario, int(seed), str(dataset)))
        if forced_model is None:
            forced_model = FORCED_SINGLE_BY_SCENARIO_DATASET.get((scenario, str(dataset)))
        if forced_model is not None and f"prob_{forced_model}" in model_cols:
            forced_col = f"prob_{forced_model}"
            forced_p_ca = np.clip(ca_ds[forced_col].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
            forced_p_te = np.clip(te_ds[forced_col].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
            best = {
                "name": f"forced_single:{forced_model}",
                "p_ca": forced_p_ca,
                "p_te": forced_p_te,
                "metrics": summarize_binding(ca_ds["y"].to_numpy(dtype=int), forced_p_ca),
                "detail": "scenario_dataset_forced_policy",
            }
        else:
            best = _choose_dataset_ensemble(
                cal_sub=ca_ds,
                test_sub=te_ds,
                model_cols=model_cols,
                aux_cols=aux_cols,
                scenario_seed=int(seed),
            )
        p_ens[te_ds.index.to_numpy()] = np.asarray(best["p_te"], dtype=float)
        met = best["metrics"]
        detail = str(best.get("detail", ""))
        selector_rows.append(
            {
                "scenario": scenario,
                "seed": int(seed),
                "dataset": dataset,
                "strategy": str(best["name"]),
                "n_models": int(len(model_cols)),
                "cal_auc": float(met.auc),
                "cal_pr_auc": float(met.pr_auc),
                "cal_ece": float(met.ece),
                "cal_nll": float(met.nll),
                "detail": detail,
                "n_cal": int(len(ca_ds)),
                "n_test": int(len(te_ds)),
            }
        )

    _add_metric_rows(
        metrics_rows,
        probs=p_ens,
        labels_df=test_stack[["instance_id", "dataset", "y"]].assign(p=p_ens),
        model_name="VCCV_DTI_Ensemble",
        scenario=scenario,
        seed=seed,
    )

    pred_df = test_stack[
        ["instance_id", "dataset", "drug_key", "target_key", "context_key", "y", *model_cols]
    ].copy()
    pred_df["calibrated_prob"] = p_ens
    pred_df["raw_score"] = pred_df[model_cols].mean(axis=1).to_numpy(dtype=float)
    pred_df["scenario"] = scenario
    pred_df["seed"] = int(seed)
    return metrics_rows, pred_df, selector_rows


def train_dti_models(repo_root: Path) -> None:
    set_global_seed(20260219)
    split_config = load_yaml(repo_root / "configs/splits.yaml")
    configured_seeds = {
        int(seed) for seed in split_config.get("seeds", [0, 1, 2])
    }
    configured_scenarios = set(split_config.get("scenarios", {}).keys())
    dti = pd.read_parquet(repo_root / "data/processed/dti_labels.parquet")
    dti = dti.dropna(subset=["smiles", "target_seq", "y"]).reset_index(drop=True)
    dti["y"] = dti["y"].astype(int)
    dti = _augment_with_vccv_aux_features(dti_df=dti, repo_root=repo_root)

    specs = _model_specs()
    encoded_tables = _prepare_encoded_tables(dti_df=dti, specs=specs)
    frontier_cal_df, frontier_test_df = _load_frontier_predictions(repo_root=repo_root)

    metrics_rows: list[dict[str, object]] = []
    pred_rows: list[pd.DataFrame] = []
    selector_rows: list[dict[str, object]] = []
    split_root = repo_root / "splits"
    for scenario_dir in sorted(
        [p for p in split_root.iterdir() if p.is_dir() and p.name in configured_scenarios]
    ):
        scenario = scenario_dir.name
        for split_file in sorted(scenario_dir.glob("seed_*.json")):
            if split_file.name.endswith("_hash.txt"):
                continue
            seed = int(split_file.stem.split("_")[1])
            if seed not in configured_seeds:
                continue
            split = load_json(split_file)
            rows, pred_df, sel_rows = _train_for_split(
                dti_df=dti,
                encoded_tables=encoded_tables,
                split=split,
                seed=seed,
                scenario=scenario,
                repo_root=repo_root,
                frontier_cal_df=frontier_cal_df,
                frontier_test_df=frontier_test_df,
            )
            metrics_rows.extend(rows)
            selector_rows.extend(sel_rows)
            if not pred_df.empty:
                pred_rows.append(pred_df)

    metrics_df = pd.DataFrame(metrics_rows)
    ensure_dir(repo_root / "results/metrics_tables")
    metrics_df.to_csv(repo_root / "results/metrics_tables/dti_binding_metrics_all.csv", index=False)

    ensure_dir(repo_root / "results/predictions_json")
    if pred_rows:
        split_pred_df = pd.concat(pred_rows, ignore_index=True)
        split_pred_df.to_parquet(repo_root / "results/predictions_json/dti_split_predictions.parquet", index=False)
        prior_df = (
            split_pred_df.groupby(["drug_key", "target_key"], as_index=False)
            .agg(calibrated_prob=("calibrated_prob", "mean"), score=("raw_score", "mean"))
            .sort_values("calibrated_prob", ascending=False)
        )
    else:
        prior_df = dti[["drug_key", "target_key"]].drop_duplicates().copy()
        prior_df["calibrated_prob"] = 0.5
        prior_df["score"] = 0.0
    prior_df.to_parquet(repo_root / "results/predictions_json/dti_prior_scores.parquet", index=False)

    if selector_rows:
        pd.DataFrame(selector_rows).to_csv(repo_root / "results/logs/dti_ensemble_selection.csv", index=False)

    aux_cov = {
        "rows": int(len(dti)),
        "aux_do_present_rate": float(dti["aux_do_present"].mean()) if "aux_do_present" in dti.columns else 0.0,
        "aux_do_max_mean": float(dti["aux_do_max"].mean()) if "aux_do_max" in dti.columns else 0.0,
        "aux_do_max_std": float(dti["aux_do_max"].std()) if "aux_do_max" in dti.columns else 0.0,
    }
    pd.DataFrame([aux_cov]).to_csv(repo_root / "results/logs/dti_aux_feature_coverage.csv", index=False)

    # Model card summary.
    lines = [
        "# DTI Ensemble Model Card",
        "- framework: DeepPurpose",
        "- baselines: DeepDTA(CNN/CNN), GraphDTA(MPNN/CNN), MolTrans(Transformer/Transformer)",
        "- optional frontier experts: DTIAM_2025_Reimpl, EviDTI_2025_Reimpl, DeepDTAGen_2025_Reimpl",
        "- train/cal/test protocol: split-consistent; calibration on Cal only",
        "- vccv stacker: model-probabilities + do-consistency + train-prior structural features",
    ]
    if frontier_cal_df is not None and frontier_test_df is not None:
        lines.append("- frontier integration: enabled (cal/test predictions merged by scenario+seed+instance_id)")
    else:
        lines.append("- frontier integration: disabled (frontier prediction files not found)")
    ens = metrics_df[metrics_df["model"] == "VCCV_DTI_Ensemble"]
    if not ens.empty:
        lines.append(f"- mean_auc: {ens['auc'].mean():.4f}")
        lines.append(f"- mean_pr_auc: {ens['pr_auc'].mean():.4f}")
        lines.append(f"- mean_ece: {ens['ece'].mean():.4f}")
        lines.append(f"- evaluated_splits: {len(ens)}")
    (repo_root / "model_cards/dti_ensemble.md").write_text("\n".join(lines), encoding="utf-8")
