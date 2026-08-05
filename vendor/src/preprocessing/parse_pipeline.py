from __future__ import annotations

import gzip
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from cmapPy.pandasGEXpress.parse_gctx import parse as parse_gctx

from src.dataloaders.deepdta_loader import (
    add_context_and_condition,
    binarize_affinity,
    canonical_target_symbol,
    load_deepdta_dataset,
)
from src.dataloaders.synthetic_mechanism import (
    SyntheticConfig,
    build_synthetic_mechanism_data,
)
from src.utils.hashing import sha256_text, stable_hash
from src.utils.io import ensure_dir, load_yaml


def _read_tsv_gz(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        return pd.read_csv(f, sep="\t", low_memory=False)


def _extract_numeric(x: object, default: float) -> float:
    if x is None:
        return float(default)
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return float(default)
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if m is None:
        return float(default)
    try:
        return float(m.group(0))
    except Exception:
        return float(default)


def _mode_from_pert_type(pert_type: object) -> str | None:
    s = str(pert_type).lower()
    if any(k in s for k in ["trt_sh", "trt_xpr", "trt_kd", "trt_ko", "crispri"]):
        return "LoF"
    if any(k in s for k in ["trt_oe", "crispra"]):
        return "GoF"
    return None


def _ensure_unzipped_gctx(path_gz: Path) -> Path:
    if not path_gz.name.endswith(".gz"):
        return path_gz
    out = path_gz.with_suffix("")
    if out.exists() and out.stat().st_size > 0:
        return out
    with gzip.open(path_gz, "rb") as f_in, out.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out, 1024 * 1024)
    return out


def parse_geo_metadata(repo_root: Path) -> None:
    interim = ensure_dir(repo_root / "data/interim")

    def _collect(pattern: str, out_name: str) -> None:
        rows = []
        for source in ("geo_gse92742", "geo_gse70138"):
            raw_dir = repo_root / "data/raw" / source
            if not raw_dir.exists():
                continue
            for f in raw_dir.glob(pattern):
                df = _read_tsv_gz(f)
                df["source"] = source
                rows.append(df)
        if rows:
            pd.concat(rows, ignore_index=True).to_parquet(interim / out_name, index=False)

    _collect("*sig_info*.txt.gz", "sig_info.parquet")
    _collect("*sig_metrics*.txt.gz", "sig_metrics.parquet")
    _collect("*gene_info*.txt.gz", "gene_info.parquet")
    _collect("*cell_info*.txt.gz", "cell_info.parquet")
    _collect("*pert_info*.txt.gz", "pert_info.parquet")


def build_real_dti_labels(repo_root: Path) -> pd.DataFrame:
    base_cfg = load_yaml(repo_root / "configs/base.yaml")
    label_cfg = load_yaml(repo_root / "configs/dti_labels.yaml")
    mode = base_cfg["mode"]
    max_pairs = int(label_cfg["sampling"]["max_pairs_per_dataset"][mode])
    raw_root = repo_root / "data/raw"

    all_sets = []
    for ds in label_cfg["dataset_preference"]:
        source_name = f"deepdta_{ds}"
        ds_dir = raw_root / source_name
        if not ds_dir.exists():
            continue
        df = load_deepdta_dataset(raw_root, source_name)
        if ds == "davis" and bool(label_cfg.get("legacy_raw_davis_labels", False)):
            # Apply the configured Davis affinity mode.
            df["affinity"] = df["affinity_raw"].astype(float)
            df["affinity_scale"] = "Kd_nM_legacy_threshold_misuse"
        df = add_context_and_condition(df)
        bin_cfg = label_cfg["binarization"][ds]
        df = binarize_affinity(
            df=df,
            positive_if_gte=float(bin_cfg["positive_if_gte"]),
            negative_if_lte=float(bin_cfg["negative_if_lte"]),
        )
        if max_pairs > 0 and len(df) > max_pairs:
            df = df.sample(n=max_pairs, random_state=base_cfg["seed"]).reset_index(drop=True)
        all_sets.append(df)
    if not all_sets:
        raise FileNotFoundError("No real DTI datasets found in data/raw/deepdta_*.")

    out = pd.concat(all_sets, ignore_index=True)
    out["pair_key"] = [sha256_text(f"{d}|{t}") for d, t in zip(out["drug_key"], out["target_key"])]
    out["row_hash"] = [
        stable_hash({"pair": p, "ctx": c, "t": tm, "d": ds})
        for p, c, tm, ds in zip(out["pair_key"], out["context_key"], out["pert_time"], out["pert_dose"])
    ]
    out["instance_id"] = out["row_hash"]
    return out


def _parse_lincs_do_signatures(
    repo_root: Path,
    target_symbols: set[str],
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out_rows: list[pd.DataFrame] = []
    gene_cols_ref: list[str] | None = None
    var_cols_ref: list[str] | None = None
    gene_id_ref: list[object] | None = None

    source_specs = [
        (
            "geo_gse92742",
            "GSE92742_Broad_LINCS_sig_info.txt.gz",
            "GSE92742_Broad_LINCS_sig_metrics.txt.gz",
            "GSE92742_Broad_LINCS_Level2_GEX_delta_n49216x978.gctx.gz",
        ),
    ]

    for source, sig_name, metric_name, gctx_name in source_specs:
        raw_dir = repo_root / "data/raw" / source
        sig_path = raw_dir / sig_name
        gctx_gz = raw_dir / gctx_name
        if not (sig_path.exists() and gctx_gz.exists()):
            continue
        gctx_path = _ensure_unzipped_gctx(gctx_gz)

        sig = _read_tsv_gz(sig_path)
        if "pert_time" in sig.columns:
            t_raw = sig["pert_time"]
        elif "pert_itime" in sig.columns:
            t_raw = sig["pert_itime"]
        else:
            t_raw = pd.Series(["24 h"] * len(sig))
        if "pert_dose" in sig.columns:
            d_raw = sig["pert_dose"]
        elif "pert_idose" in sig.columns:
            d_raw = sig["pert_idose"]
        else:
            d_raw = pd.Series(["1 uM"] * len(sig))
        sig["pert_time_num"] = t_raw.map(lambda x: _extract_numeric(x, 24.0))
        sig["pert_dose_num"] = d_raw.map(lambda x: _extract_numeric(x, 1.0))
        sig["mode"] = sig["pert_type"].map(_mode_from_pert_type)
        sig["target_symbol"] = sig["pert_iname"].map(canonical_target_symbol)
        sig = sig[(sig["mode"].notna()) & (sig["target_symbol"].isin(target_symbols))].copy()
        if sig.empty:
            continue

        metric_path = raw_dir / metric_name
        if metric_path.exists():
            metrics = _read_tsv_gz(metric_path)
            if "Unnamed: 0" in metrics.columns:
                metrics = metrics.drop(columns=["Unnamed: 0"])
            keep_metric_cols = [
                c
                for c in ["sig_id", "distil_cc_q75", "tas", "distil_nsample", "distil_ss", "pct_self_rank_q25"]
                if c in metrics.columns
            ]
            metrics = metrics[keep_metric_cols].drop_duplicates(subset=["sig_id"])
            sig = sig.merge(metrics, on="sig_id", how="left")

        # Expand row-level signature metadata to matrix column ids (distil replicates).
        sig = sig[sig["distil_id"].notna()].copy()
        sig["matrix_cid"] = sig["distil_id"].astype(str).str.split("|")
        sig = sig.explode("matrix_cid").reset_index(drop=True)

        cmeta = parse_gctx(str(gctx_path), col_meta_only=True)
        cids = set(cmeta.index.astype(str).tolist())
        sig = sig[sig["matrix_cid"].astype(str).isin(cids)].copy()
        if sig.empty:
            continue
        sig = sig.drop_duplicates(subset=["matrix_cid"]).reset_index(drop=True)

        cc_raw = sig.get("distil_cc_q75", pd.Series(np.zeros(len(sig))))
        cc_norm = ((cc_raw.astype(float).fillna(0.0) + 1.0) / 2.0).clip(0.0, 1.0)
        tas = sig.get("tas", pd.Series(np.zeros(len(sig)))).astype(float).fillna(0.0).clip(lower=0.0)
        tas_norm = 1.0 - np.exp(-tas / 3.0)
        sig["qc_do"] = (0.7 * cc_norm + 0.3 * tas_norm).clip(0.0, 1.0)
        sig["n_rep_eff"] = sig.get("distil_nsample", pd.Series(np.ones(len(sig)))).fillna(1.0).astype(float).clip(
            lower=1.0
        )
        sig = sig.sort_values(["qc_do", "n_rep_eff"], ascending=[False, False]).reset_index(drop=True)

        # Cap per-source rows for speed while preserving broad target coverage.
        per_src_max = max(2000, max_rows // 2)
        if len(sig) > per_src_max:
            grp = sig.groupby(["target_symbol", "mode"], sort=False)
            keep_per_grp = max(2, per_src_max // max(len(grp), 1))
            sig_keep = grp.head(keep_per_grp).reset_index(drop=True)
            if len(sig_keep) < per_src_max:
                extra = sig.loc[~sig.index.isin(sig_keep.index)].head(per_src_max - len(sig_keep))
                sig_keep = pd.concat([sig_keep, extra], ignore_index=True)
            sig = sig_keep.head(per_src_max).reset_index(drop=True)

        # Parse expression values in chunks.
        chunk_size = 2048
        all_chunks: list[pd.DataFrame] = []
        cid_list = sig["matrix_cid"].astype(str).tolist()
        for st in range(0, len(cid_list), chunk_size):
            chunk_ids = cid_list[st : st + chunk_size]
            g = parse_gctx(str(gctx_path), cid=chunk_ids)
            data = g.data_df
            if gene_cols_ref is None:
                gene_cols_ref = [f"G{i}" for i in range(len(data.index))]
                var_cols_ref = [f"V{i}" for i in range(len(data.index))]
                gene_id_ref = list(data.index)
            else:
                # Force consistent gene order across chunks/sources.
                if gene_id_ref is None:
                    raise RuntimeError("Internal gene-id reference was not initialized.")
                data = data.loc[gene_id_ref]
            expr = data.T.reset_index().rename(columns={"index": "matrix_cid"})
            expr.columns = ["matrix_cid", *gene_cols_ref]
            all_chunks.append(expr)
        expr_df = pd.concat(all_chunks, ignore_index=True)
        merged = sig.merge(expr_df, on="matrix_cid", how="inner")
        if merged.empty:
            continue

        # Build per-row diagonal variance from QC quality.
        qc = merged["qc_do"].to_numpy(dtype=float)
        nrep = merged["n_rep_eff"].to_numpy(dtype=float)
        base_var = (0.08 + 0.30 * (1.0 - qc) + 0.22 / np.sqrt(nrep)).clip(0.02, 0.9)
        var_block = np.repeat(base_var[:, None], len(var_cols_ref), axis=1)
        var_df = pd.DataFrame(var_block, columns=var_cols_ref)

        out = pd.DataFrame(
            {
                "target_key": merged["target_symbol"].map(sha256_text),
                "target_gene": merged["target_symbol"],
                "context_key": merged["cell_id"].astype(str),
                "pert_time": merged["pert_time_num"].astype(float),
                "pert_dose": merged["pert_dose_num"].astype(float).clip(lower=1e-5),
                "platform": source.upper(),
                "batch": merged["matrix_cid"].astype(str).str.split(":").str[0],
                "mode": merged["mode"].astype(str),
                "n_rep": merged["n_rep_eff"].round().astype(int),
                "cc_q75": ((merged.get("distil_cc_q75", 0.0).astype(float).fillna(0.0) + 1.0) / 2.0)
                .clip(0.0, 1.0)
                .to_numpy(),
                "qc_do": merged["qc_do"].astype(float).clip(0.0, 1.0).to_numpy(),
            }
        )
        out = pd.concat([out, merged[gene_cols_ref].reset_index(drop=True), var_df], axis=1)
        out_rows.append(out)

    if not out_rows:
        raise FileNotFoundError("No real L1000 do signatures matched DTI target set and available matrices.")

    do_df = pd.concat(out_rows, ignore_index=True)
    do_df = do_df.drop_duplicates(
        subset=["target_key", "context_key", "pert_time", "pert_dose", "mode", "batch"]
    ).reset_index(drop=True)
    do_df = do_df.sort_values(["qc_do", "cc_q75"], ascending=[False, False]).reset_index(drop=True)

    if len(do_df) > max_rows:
        grp = do_df.groupby(["target_key", "mode"], sort=False)
        keep_per_grp = max(2, max_rows // max(len(grp), 1))
        keep = grp.head(keep_per_grp).reset_index(drop=True)
        if len(keep) < max_rows:
            extra = do_df.loc[~do_df.index.isin(keep.index)].head(max_rows - len(keep))
            keep = pd.concat([keep, extra], ignore_index=True)
        do_df = keep.head(max_rows).reset_index(drop=True)
    do_df = do_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if gene_id_ref is None or gene_cols_ref is None or var_cols_ref is None:
        raise RuntimeError("LINCS gene-axis metadata was not initialized")
    axis_cfg = load_yaml(repo_root / "configs/base.yaml")
    axis_budget = int(
        axis_cfg["modes"][axis_cfg["mode"]]["num_genes"]
    )
    gene_axis = pd.DataFrame(
        {
            "feature_column": gene_cols_ref,
            "variance_column": var_cols_ref,
            "gctx_row_id": [str(value) for value in gene_id_ref],
            "source": "GSE92742_Level2_GCTX",
            "selected_for_model": [
                index < axis_budget for index in range(len(gene_cols_ref))
            ],
        }
    )
    gene_axis.to_parquet(
        ensure_dir(repo_root / "data/interim") / "gene_axis.parquet",
        index=False,
    )
    return do_df


def build_entity_manifest(dti_df: pd.DataFrame) -> pd.DataFrame:
    drugs = dti_df[["drug_key", "smiles"]].drop_duplicates().copy()
    drugs["entity_type"] = "drug"
    drugs = drugs.rename(columns={"smiles": "entity_value", "drug_key": "entity_key"})

    targets = dti_df[["target_key", "target_gene"]].drop_duplicates().copy()
    targets["entity_type"] = "target"
    targets = targets.rename(columns={"target_gene": "entity_value", "target_key": "entity_key"})

    contexts = dti_df[["context_key"]].drop_duplicates().copy()
    contexts["entity_type"] = "context"
    contexts["entity_value"] = contexts["context_key"]
    contexts = contexts.rename(columns={"context_key": "entity_key"})

    cond = dti_df[["condition_key", "pert_time", "pert_dose", "platform", "batch"]].drop_duplicates().copy()
    cond["entity_type"] = "condition"
    cond["entity_value"] = (
        cond["pert_time"].astype(str)
        + "|"
        + cond["pert_dose"].astype(str)
        + "|"
        + cond["platform"]
        + "|"
        + cond["batch"]
    )
    cond = cond.rename(columns={"condition_key": "entity_key"})

    cols = ["entity_key", "entity_type", "entity_value"]
    out = pd.concat([drugs[cols], targets[cols], contexts[cols], cond[cols]], ignore_index=True).drop_duplicates()
    return out


def parse_and_prepare(repo_root: Path) -> None:
    parse_geo_metadata(repo_root)
    dti_df = build_real_dti_labels(repo_root)

    processed = ensure_dir(repo_root / "data/processed")
    interim = ensure_dir(repo_root / "data/interim")
    logs = ensure_dir(repo_root / "results/logs")

    dti_df.to_parquet(processed / "dti_labels.parquet", index=False)
    build_entity_manifest(dti_df).to_parquet(interim / "entity_manifest.parquet", index=False)

    base_cfg = load_yaml(repo_root / "configs/base.yaml")
    mode_cfg = base_cfg["modes"][base_cfg["mode"]]
    syn_cfg = SyntheticConfig(
        n_genes=int(mode_cfg["num_genes"]),
        n_instances=int(mode_cfg["num_virtual_instances"]),
        seed=int(base_cfg["seed"]),
    )
    do_rows_budget = int(max(4000, mode_cfg["num_virtual_instances"] * 3))
    target_symbols = set(dti_df["target_gene"].astype(str).map(canonical_target_symbol).tolist())
    do_real = _parse_lincs_do_signatures(
        repo_root=repo_root,
        target_symbols=target_symbols,
        max_rows=do_rows_budget,
        seed=int(base_cfg["seed"]),
    )
    sig_drug, sig_do, align_params, truth = build_synthetic_mechanism_data(dti_df, do_real, syn_cfg)

    sig_drug.to_parquet(processed / "signatures_drug_raw.parquet", index=False)
    sig_do.to_parquet(processed / "signatures_do_raw.parquet", index=False)
    truth.to_parquet(processed / "mechanism_truth.parquet", index=False)

    np.savez_compressed(
        processed / "align_ground_truth.npz",
        B=align_params["B"],
        b_LoF=align_params["b_LoF"],
        b_GoF=align_params["b_GoF"],
        beta_LoF=align_params["beta_LoF"],
        beta_GoF=align_params["beta_GoF"],
    )

    gcols = [c for c in sig_do.columns if c.startswith("G") and c[1:].isdigit()]
    stats = {
        "dti_rows": int(len(dti_df)),
        "dti_positive": int((dti_df["y"] == 1).sum()),
        "dti_negative": int((dti_df["y"] == 0).sum()),
        "real_do_rows": int(len(do_real)),
        "do_rows_after_projection": int(len(sig_do)),
        "drug_signatures_syn": int(len(sig_drug)),
        "mechanism_truth_rows": int(len(truth)),
        "num_genes": int(len(gcols)),
    }
    (logs / "parse_stats.md").write_text(
        "\n".join([f"- {k}: {v}" for k, v in stats.items()]),
        encoding="utf-8",
    )
