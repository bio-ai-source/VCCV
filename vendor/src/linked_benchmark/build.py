from __future__ import annotations

import gzip
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from cmapPy.pandasGEXpress.parse_gctx import parse as parse_gctx

from src.dataloaders.deepdta_loader import binarize_affinity, canonical_target_symbol, load_deepdta_dataset
from src.preprocessing.parse_pipeline import _extract_numeric, _mode_from_pert_type
from src.utils.hashing import sha256_text, stable_hash
from src.utils.io import ensure_dir, load_yaml

try:
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None


_INVALID_TOKEN_SET = {"", "nan", "na", "none", "null", "-666", "restricted"}


@dataclass(frozen=True)
class LincsSourceSpec:
    name: str
    sig_info: str
    pert_info: str
    gene_info: str
    gctx: str
    priority: int


def _read_tsv_gz(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        return pd.read_csv(f, sep="\t", low_memory=False)


def _ensure_unzipped_gctx(path_gz: Path) -> Path:
    if not path_gz.name.endswith(".gz"):
        return path_gz
    out = path_gz.with_suffix("")
    if out.exists() and out.stat().st_size > 0:
        return out
    with gzip.open(path_gz, "rb") as f_in, out.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out, 1024 * 1024)
    return out


def _clean_token(x: object) -> str:
    s = str(x).strip()
    return "" if s.lower() in _INVALID_TOKEN_SET else s


def _normalize_smiles(x: object) -> str:
    s = _clean_token(x)
    return re.sub(r"\s+", "", s)


def _canonicalize_smiles(x: object) -> str:
    s = _normalize_smiles(x)
    if not s:
        return ""
    if Chem is None:
        return s
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def _smiles_to_inchi_key(x: object) -> str:
    s = _normalize_smiles(x)
    if not s or Chem is None:
        return ""
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return ""
        return str(Chem.MolToInchiKey(mol)).upper().strip()
    except Exception:
        return ""


def _normalize_inchi_key(x: object) -> str:
    s = _clean_token(x)
    return s.upper().strip()


def _normalize_cell_id(x: object) -> str:
    s = str(x).strip().upper()
    return re.sub(r"[^A-Z0-9]+", "", s)


def _coerce_meta_frame(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    for attr in ("data_df", "row_metadata_df", "col_metadata_df"):
        val = getattr(obj, attr, None)
        if isinstance(val, pd.DataFrame):
            return val
    if hasattr(obj, "index"):
        return pd.DataFrame(index=obj.index)
    raise TypeError(f"Unsupported GCTX metadata object: {type(obj)}")


def _get_available_cids(gctx_path: Path) -> list[str]:
    meta = _coerce_meta_frame(parse_gctx(str(gctx_path), col_meta_only=True))
    return meta.index.astype(str).tolist()


def _build_gene_map(gctx_path: Path, gene_info_path: Path) -> pd.DataFrame:
    cids = _get_available_cids(gctx_path)
    if not cids:
        raise ValueError("No GCTX column metadata available to recover row ids.")
    sample = parse_gctx(str(gctx_path), cid=[str(cids[0])])
    if not hasattr(sample, "data_df"):
        raise TypeError("Expected parse_gctx(..., cid=[...]) to expose data_df.")
    row_ids = pd.Series(sample.data_df.index.astype(str), name="row_id")
    gene_map = pd.DataFrame({"row_id": row_ids, "g_index": [f"G{i}" for i in range(len(row_ids))]})

    gene_info = _read_tsv_gz(gene_info_path).copy()
    if "pr_gene_id" in gene_info.columns:
        gene_info["row_id"] = gene_info["pr_gene_id"].astype(str)
    else:
        gene_info["row_id"] = gene_info.iloc[:, 0].astype(str)
    gene_info = gene_info.drop_duplicates(subset=["row_id"]).reset_index(drop=True)

    out = gene_map.merge(gene_info, on="row_id", how="left")
    if "pr_is_lm" in out.columns:
        out["landmark_flag"] = out["pr_is_lm"].fillna(0).astype(int)
    else:
        out["landmark_flag"] = 1
    out["legacy_256_flag"] = (out.index < 256).astype(int)
    return out


def _iter_lincs_sources(cfg: dict) -> list[LincsSourceSpec]:
    lcfg = cfg["lincs"]
    if "sources" in lcfg:
        return [
            LincsSourceSpec(
                name=str(src["name"]),
                sig_info=str(src["sig_info"]),
                pert_info=str(src["pert_info"]),
                gene_info=str(src["gene_info"]),
                gctx=str(src["gctx"]),
                priority=int(idx),
            )
            for idx, src in enumerate(lcfg["sources"])
        ]
    return [
        LincsSourceSpec(
            name=str(lcfg["source"]),
            sig_info=str(lcfg["sig_info"]),
            pert_info=str(lcfg["pert_info"]),
            gene_info=str(lcfg["gene_info"]),
            gctx=str(lcfg["gctx"]),
            priority=0,
        )
    ]


def _load_real_dti_pairs(repo_root: Path, cfg: dict) -> pd.DataFrame:
    frames = []
    raw_root = repo_root / "data/raw"
    for ds in cfg["dti"]["datasets"]:
        df = load_deepdta_dataset(raw_root, ds)
        alias = ds.replace("deepdta_", "")
        bin_cfg = cfg["dti"]["binarization"][alias]
        df = binarize_affinity(
            df=df,
            positive_if_gte=float(bin_cfg["positive_if_gte"]),
            negative_if_lte=float(bin_cfg["negative_if_lte"]),
        )
        drug_meta = df[["drug_key", "smiles"]].drop_duplicates().copy()
        drug_meta["smiles_norm"] = drug_meta["smiles"].map(_normalize_smiles)
        drug_meta["smiles_canon"] = drug_meta["smiles"].map(_canonicalize_smiles)
        drug_meta["inchi_key"] = drug_meta["smiles"].map(_smiles_to_inchi_key)
        df = df.merge(drug_meta, on=["drug_key", "smiles"], how="left")
        df["pair_key"] = [sha256_text(f"{d}|{t}|{alias}") for d, t in zip(df["drug_key"], df["target_key"])]
        df["dataset"] = alias
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["dataset", "drug_key", "target_key"]).reset_index(drop=True)


def _priority_match(
    sig: pd.DataFrame,
    drug_map: pd.DataFrame,
    *,
    left_col: str,
    right_col: str,
    rule_name: str,
) -> pd.DataFrame:
    left = sig[["sig_row_id", left_col]].copy()
    left[left_col] = left[left_col].fillna("").astype(str)
    left = left[left[left_col] != ""].drop_duplicates()
    if left.empty:
        return pd.DataFrame(columns=["sig_row_id", "drug_key", "compound_match_rule"])
    right = drug_map[["drug_key", right_col]].copy()
    right[right_col] = right[right_col].fillna("").astype(str)
    right = right[right[right_col] != ""].drop_duplicates()
    if right.empty:
        return pd.DataFrame(columns=["sig_row_id", "drug_key", "compound_match_rule"])
    hit = left.merge(right, left_on=left_col, right_on=right_col, how="inner")
    if hit.empty:
        return pd.DataFrame(columns=["sig_row_id", "drug_key", "compound_match_rule"])
    return hit[["sig_row_id", "drug_key"]].assign(compound_match_rule=rule_name)


def _match_compounds_to_drugs(sig: pd.DataFrame, drug_map: pd.DataFrame) -> pd.DataFrame:
    pending = sig.copy()
    hits = []
    for rule_name, left_col, right_col in (
        ("smiles_exact", "smiles_norm", "smiles_norm"),
        ("smiles_canonical", "smiles_canon", "smiles_canon"),
        ("inchi_key", "inchi_key_norm", "inchi_key"),
    ):
        matched = _priority_match(pending, drug_map, left_col=left_col, right_col=right_col, rule_name=rule_name)
        if matched.empty:
            continue
        hits.append(matched)
        pending = pending[~pending["sig_row_id"].isin(set(matched["sig_row_id"].astype(str)))].copy()
        if pending.empty:
            break
    if not hits:
        return sig.iloc[0:0].copy()
    match_df = pd.concat(hits, ignore_index=True).drop_duplicates(subset=["sig_row_id", "drug_key"])
    return sig.merge(match_df, on="sig_row_id", how="inner")


def _cap_compound_rows(sig: pd.DataFrame, max_rows_per_drug: int) -> pd.DataFrame:
    sig = sig.sort_values(
        ["drug_key", "source_priority", "cell_key", "pert_time_num", "pert_dose_num", "matrix_cid"]
    ).reset_index(drop=True)
    if max_rows_per_drug <= 0:
        return sig
    sig["cell_rank"] = sig.groupby(["drug_key", "cell_key"]).cumcount()
    sig = sig.sort_values(
        ["drug_key", "cell_rank", "source_priority", "cell_key", "pert_time_num", "pert_dose_num", "matrix_cid"]
    ).reset_index(drop=True)
    sig = sig.groupby("drug_key", as_index=False, group_keys=False).head(max_rows_per_drug).reset_index(drop=True)
    return sig.drop(columns=["cell_rank"], errors="ignore")


def _cap_target_rows(sig: pd.DataFrame, max_rows_per_target_mode: int) -> pd.DataFrame:
    sig = sig.sort_values(
        ["target_key", "mode", "source_priority", "cell_key", "pert_time_num", "pert_dose_num", "matrix_cid"]
    ).reset_index(drop=True)
    if max_rows_per_target_mode <= 0:
        return sig
    sig["cell_rank"] = sig.groupby(["target_key", "mode", "cell_key"]).cumcount()
    sig = sig.sort_values(
        ["target_key", "mode", "cell_rank", "source_priority", "cell_key", "pert_time_num", "pert_dose_num", "matrix_cid"]
    ).reset_index(drop=True)
    sig = (
        sig.groupby(["target_key", "mode"], as_index=False, group_keys=False)
        .head(max_rows_per_target_mode)
        .reset_index(drop=True)
    )
    return sig.drop(columns=["cell_rank"], errors="ignore")


def _parse_expression_chunks(
    gctx_path: Path,
    matrix_ids: list[str],
    gene_cols_ref: list[str] | None,
    gene_rows_ref: list[str] | None,
    chunk_size: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    rows_ref = gene_rows_ref
    cols_ref = gene_cols_ref
    chunks = []
    unique_ids = list(dict.fromkeys(str(x) for x in matrix_ids))
    for st in range(0, len(unique_ids), chunk_size):
        chunk_ids = unique_ids[st : st + chunk_size]
        g = parse_gctx(str(gctx_path), cid=chunk_ids)
        data = g.data_df
        data.index = data.index.astype(str)
        if rows_ref is None:
            rows_ref = data.index.astype(str).tolist()
            cols_ref = [f"G{i}" for i in range(len(rows_ref))]
        else:
            data = data.reindex(rows_ref).fillna(0.0)
        expr = data.T.reset_index().rename(columns={"index": "matrix_cid"})
        expr.columns = ["matrix_cid", *(cols_ref or [])]
        chunks.append(expr)
    expr_df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["matrix_cid", *(cols_ref or [])])
    return expr_df, (cols_ref or []), (rows_ref or [])


def _parse_compound_signatures(
    repo_root: Path,
    cfg: dict,
    drug_map: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], Path, Path, pd.DataFrame, pd.DataFrame]:
    lcfg = cfg["lincs"]
    compound_types = {str(x).lower() for x in lcfg["compound_pert_types"]}
    chunk_size = int(lcfg["parse_chunk_size"])
    max_per_drug = int(lcfg["max_signatures_per_drug"])

    gene_cols_ref: list[str] | None = None
    gene_rows_ref: list[str] | None = None
    gctx_ref_path: Path | None = None
    gene_info_ref_path: Path | None = None
    frames = []
    source_rows = []

    for spec in _iter_lincs_sources(cfg):
        raw_dir = repo_root / "data/raw" / spec.name
        sig_path = raw_dir / spec.sig_info
        pert_path = raw_dir / spec.pert_info
        gene_info_path = raw_dir / spec.gene_info
        gctx_path = _ensure_unzipped_gctx(raw_dir / spec.gctx)
        if not (sig_path.exists() and pert_path.exists() and gene_info_path.exists() and gctx_path.exists()):
            continue

        sig = _read_tsv_gz(sig_path).copy()
        pert = _read_tsv_gz(pert_path).copy()
        for col in ("canonical_smiles", "inchi_key", "pert_iname", "pert_type"):
            if col not in pert.columns:
                pert[col] = ""
        sig = sig.merge(
            pert[["pert_id", "canonical_smiles", "inchi_key", "pert_iname", "pert_type"]],
            on="pert_id",
            how="left",
            suffixes=("", "_pert"),
        )
        if "pert_type_pert" in sig.columns:
            sig["pert_type"] = sig["pert_type_pert"].fillna(sig.get("pert_type"))

        sig["pert_type_norm"] = sig["pert_type"].astype(str).str.lower()
        sig = sig[sig["pert_type_norm"].isin(compound_types)].copy()
        raw_compound_rows = int(len(sig))
        if sig.empty:
            continue

        sig["sig_row_id"] = [f"{spec.name}::{i}" for i in sig.index]
        sig["lincs_source"] = spec.name
        sig["source_priority"] = int(spec.priority)
        sig["smiles_norm"] = sig["canonical_smiles"].map(_normalize_smiles)
        sig["smiles_canon"] = sig["canonical_smiles"].map(_canonicalize_smiles)
        sig["inchi_key_norm"] = sig["inchi_key"].map(_normalize_inchi_key)
        sig = _match_compounds_to_drugs(sig, drug_map)
        mapped_compound_rows = int(len(sig))
        if sig.empty:
            continue

        sig["pert_time_num"] = sig.get("pert_time", sig.get("pert_itime", pd.Series(["24 h"] * len(sig)))).map(
            lambda x: _extract_numeric(x, 24.0)
        )
        sig["pert_dose_num"] = sig.get("pert_dose", sig.get("pert_idose", pd.Series(["1 uM"] * len(sig)))).map(
            lambda x: _extract_numeric(x, 1.0)
        )
        sig["cell_key"] = sig["cell_id"].map(_normalize_cell_id)
        sig = sig[sig["distil_id"].notna()].copy()
        sig["matrix_cid"] = sig["distil_id"].astype(str).str.split("|")
        sig = sig.explode("matrix_cid").reset_index(drop=True)
        sig["matrix_cid"] = sig["matrix_cid"].astype(str)

        available_cids = set(_get_available_cids(gctx_path))
        sig = sig[sig["matrix_cid"].isin(available_cids)].copy()
        sig = sig.drop_duplicates(subset=["lincs_source", "matrix_cid", "drug_key"]).reset_index(drop=True)
        if sig.empty:
            continue

        sig = _cap_compound_rows(sig, max_per_drug)
        expr_df, gene_cols_ref, gene_rows_ref = _parse_expression_chunks(
            gctx_path=gctx_path,
            matrix_ids=sig["matrix_cid"].astype(str).tolist(),
            gene_cols_ref=gene_cols_ref,
            gene_rows_ref=gene_rows_ref,
            chunk_size=chunk_size,
        )
        sig = sig.merge(expr_df, on="matrix_cid", how="inner")
        if sig.empty:
            continue

        if gctx_ref_path is None:
            gctx_ref_path = gctx_path
            gene_info_ref_path = gene_info_path

        sig["compound_sig_id"] = sig["lincs_source"].astype(str) + "::" + sig["matrix_cid"].astype(str)
        out = sig[
            [
                "compound_sig_id",
                "matrix_cid",
                "lincs_source",
                "source_priority",
                "drug_key",
                "compound_match_rule",
                "smiles_norm",
                "smiles_canon",
                "pert_id",
                "pert_iname",
                "canonical_smiles",
                "inchi_key",
                "cell_id",
                "cell_key",
                "pert_time_num",
                "pert_dose_num",
            ]
            + gene_cols_ref
        ].copy()
        out = out.rename(columns={"pert_time_num": "pert_time", "pert_dose_num": "pert_dose"})
        out["signature_kind"] = "compound"
        frames.append(out)
        source_rows.append(
            {
                "lincs_source": spec.name,
                "raw_compound_signature_rows": raw_compound_rows,
                "mapped_compound_signature_rows": mapped_compound_rows,
                "saved_compound_signature_rows": int(len(out)),
                "n_drug": int(out["drug_key"].nunique()),
                "n_cell": int(out["cell_key"].nunique()),
            }
        )

    if not frames or gene_cols_ref is None or gene_rows_ref is None or gctx_ref_path is None or gene_info_ref_path is None:
        raise FileNotFoundError("No LINCS compound perturbations matched DTI drugs across configured sources.")

    cmp_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["compound_sig_id", "drug_key"]).reset_index(drop=True)
    match_audit = (
        cmp_df.groupby(["lincs_source", "compound_match_rule"], dropna=False)
        .agg(
            signature_rows=("compound_sig_id", "count"),
            n_compound_pert=("pert_id", "nunique"),
            n_drug=("drug_key", "nunique"),
            n_cell=("cell_key", "nunique"),
        )
        .reset_index()
    )
    source_audit = pd.DataFrame(source_rows)
    return cmp_df, gene_cols_ref, gene_rows_ref, gctx_ref_path, gene_info_ref_path, match_audit, source_audit


def _parse_target_signatures(
    repo_root: Path,
    cfg: dict,
    target_symbols: set[str],
    gene_cols_ref: list[str],
    gene_rows_ref: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lcfg = cfg["lincs"]
    target_tokens = [str(x).lower() for x in lcfg["target_pert_types"]]
    chunk_size = int(lcfg["parse_chunk_size"])
    max_per_target_mode = int(lcfg["max_signatures_per_target_mode"])
    frames = []
    source_rows = []

    for spec in _iter_lincs_sources(cfg):
        raw_dir = repo_root / "data/raw" / spec.name
        sig_path = raw_dir / spec.sig_info
        pert_path = raw_dir / spec.pert_info
        gctx_path = _ensure_unzipped_gctx(raw_dir / spec.gctx)
        if not (sig_path.exists() and pert_path.exists() and gctx_path.exists()):
            continue

        sig = _read_tsv_gz(sig_path).copy()
        pert = _read_tsv_gz(pert_path).copy()
        for col in ("pert_iname", "pert_type"):
            if col not in pert.columns:
                pert[col] = ""
        sig = sig.merge(pert[["pert_id", "pert_iname", "pert_type"]], on="pert_id", how="left", suffixes=("", "_pert"))
        if "pert_type_pert" in sig.columns:
            sig["pert_type"] = sig["pert_type_pert"].fillna(sig.get("pert_type"))
        if "pert_iname_pert" in sig.columns:
            sig["pert_iname"] = sig["pert_iname_pert"].fillna(sig.get("pert_iname"))

        sig["pert_type_norm"] = sig["pert_type"].astype(str).str.lower()
        sig = sig[sig["pert_type_norm"].map(lambda s: any(tok in s for tok in target_tokens))].copy()
        raw_target_rows = int(len(sig))
        sig["mode"] = sig["pert_type"].map(_mode_from_pert_type)
        sig["target_gene"] = sig["pert_iname"].map(canonical_target_symbol)
        sig = sig[sig["mode"].notna() & sig["target_gene"].isin(target_symbols)].copy()
        matched_target_rows = int(len(sig))
        if sig.empty:
            continue

        sig["lincs_source"] = spec.name
        sig["source_priority"] = int(spec.priority)
        sig["pert_time_num"] = sig.get("pert_time", sig.get("pert_itime", pd.Series(["24 h"] * len(sig)))).map(
            lambda x: _extract_numeric(x, 24.0)
        )
        sig["pert_dose_num"] = sig.get("pert_dose", sig.get("pert_idose", pd.Series(["1 uM"] * len(sig)))).map(
            lambda x: _extract_numeric(x, 1.0)
        )
        sig["cell_key"] = sig["cell_id"].map(_normalize_cell_id)
        sig = sig[sig["distil_id"].notna()].copy()
        sig["matrix_cid"] = sig["distil_id"].astype(str).str.split("|")
        sig = sig.explode("matrix_cid").reset_index(drop=True)
        sig["matrix_cid"] = sig["matrix_cid"].astype(str)

        available_cids = set(_get_available_cids(gctx_path))
        sig = sig[sig["matrix_cid"].isin(available_cids)].copy()
        sig = sig.drop_duplicates(subset=["lincs_source", "matrix_cid"]).reset_index(drop=True)
        if sig.empty:
            continue

        sig["target_key"] = sig["target_gene"].map(sha256_text)
        sig = _cap_target_rows(sig, max_per_target_mode)
        expr_df, _, _ = _parse_expression_chunks(
            gctx_path=gctx_path,
            matrix_ids=sig["matrix_cid"].astype(str).tolist(),
            gene_cols_ref=gene_cols_ref,
            gene_rows_ref=gene_rows_ref,
            chunk_size=chunk_size,
        )
        sig = sig.merge(expr_df, on="matrix_cid", how="inner")
        if sig.empty:
            continue

        sig["target_sig_id"] = sig["lincs_source"].astype(str) + "::" + sig["matrix_cid"].astype(str)
        out = sig[
            [
                "target_sig_id",
                "matrix_cid",
                "lincs_source",
                "source_priority",
                "target_key",
                "target_gene",
                "cell_id",
                "cell_key",
                "pert_time_num",
                "pert_dose_num",
                "mode",
            ]
            + gene_cols_ref
        ].copy()
        out = out.rename(columns={"pert_time_num": "pert_time", "pert_dose_num": "pert_dose"})
        out["signature_kind"] = "target"
        frames.append(out)
        source_rows.append(
            {
                "lincs_source": spec.name,
                "raw_target_signature_rows": raw_target_rows,
                "matched_target_signature_rows": matched_target_rows,
                "saved_target_signature_rows": int(len(out)),
                "n_target": int(out["target_key"].nunique()),
                "n_cell": int(out["cell_key"].nunique()),
                "n_mode": int(out["mode"].astype(str).nunique()),
            }
        )

    if not frames:
        raise FileNotFoundError("No real LINCS target perturbations matched DTI targets.")
    target_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["target_sig_id"]).reset_index(drop=True)
    return target_df, pd.DataFrame(source_rows)


def _load_expression_support(repo_root: Path, cfg: dict) -> pd.DataFrame | None:
    ecfg = cfg["expression"]
    if not bool(ecfg.get("enabled", False)):
        return None
    path = repo_root / str(ecfg["path"])
    if not path.exists():
        return None
    file_type = str(ecfg.get("file_type", path.suffix.lstrip("."))).lower()
    if file_type == "parquet":
        df = pd.read_parquet(path)
    elif file_type in {"csv", "txt"}:
        sep = "," if file_type == "csv" else "\t"
        df = pd.read_csv(path, sep=sep)
    elif file_type == "tsv":
        df = pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"Unsupported expression file type: {file_type}")
    keep = [str(ecfg["cell_col"]), str(ecfg["gene_col"]), str(ecfg["value_col"])]
    df = df[keep].copy()
    df.columns = ["cell_id", "target_gene", "expr_log2_tpm1"]
    df["cell_key"] = df["cell_id"].map(_normalize_cell_id)
    df["target_gene"] = df["target_gene"].map(canonical_target_symbol)
    return df.drop_duplicates(subset=["cell_key", "target_gene"]).reset_index(drop=True)


def _load_family_map(repo_root: Path, cfg: dict) -> pd.DataFrame | None:
    fcfg = cfg.get("family_map", {})
    if not bool(fcfg.get("enabled", False)):
        return None
    path = repo_root / str(fcfg["path"])
    if not path.exists():
        return None
    df = pd.read_csv(path)
    out = df[[str(fcfg["target_col"]), str(fcfg["family_col"])]].copy()
    out.columns = ["target_gene", "family_key"]
    out["target_gene"] = out["target_gene"].map(canonical_target_symbol)
    return out.drop_duplicates(subset=["target_gene"]).reset_index(drop=True)


def _load_lineage_map(repo_root: Path, cfg: dict) -> pd.DataFrame | None:
    lcfg = cfg.get("lineage_map", {})
    if not bool(lcfg.get("enabled", False)):
        return None
    path = repo_root / str(lcfg["path"])
    if not path.exists():
        return None
    file_type = str(lcfg.get("file_type", path.suffix.lstrip("."))).lower()
    if file_type == "csv":
        df = pd.read_csv(path)
    elif file_type == "tsv":
        df = pd.read_csv(path, sep="\t")
    elif file_type == "parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported lineage file type: {file_type}")
    keep = [str(lcfg["cell_col"]), str(lcfg["lineage_col"]), str(lcfg["subtype_col"])]
    df = df[keep].copy()
    df.columns = ["cell_id", "lineage", "lineage_subtype"]
    df["cell_key"] = df["cell_id"].map(_normalize_cell_id)
    return df.drop_duplicates(subset=["cell_key"]).reset_index(drop=True)


def _safe_cosine(u: np.ndarray, v: np.ndarray) -> float:
    un = float(np.linalg.norm(u))
    vn = float(np.linalg.norm(v))
    if un <= 1e-12 or vn <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(u, v) / (un * vn), -1.0, 1.0))


def _build_target_lookup(target_df: pd.DataFrame) -> dict[str, dict[tuple[str, ...], pd.DataFrame]]:
    return {
        "all": {k: v.reset_index(drop=True) for k, v in target_df.groupby(["target_key", "mode"], sort=False)},
        "cell": {
            k: v.reset_index(drop=True)
            for k, v in target_df.groupby(["target_key", "mode", "cell_key"], sort=False)
        },
        "source": {
            k: v.reset_index(drop=True)
            for k, v in target_df.groupby(["target_key", "mode", "lincs_source"], sort=False)
        },
        "cell_source": {
            k: v.reset_index(drop=True)
            for k, v in target_df.groupby(["target_key", "mode", "cell_key", "lincs_source"], sort=False)
        },
    }


def _nearest_target_match(
    target_lookup: dict[str, dict[tuple[str, ...], pd.DataFrame]],
    *,
    target_key: str,
    cell_key: str,
    lincs_source: str,
    pert_time: float,
    pert_dose: float,
    mode: str,
    same_cell_only: bool,
    prefer_same_source: bool,
    allow_cross_source: bool,
    time_weight: float,
    log10_dose_weight: float,
) -> pd.Series | None:
    sub: pd.DataFrame | None = None
    if same_cell_only and prefer_same_source:
        sub = target_lookup["cell_source"].get((target_key, mode, cell_key, lincs_source))
        if sub is None and allow_cross_source:
            sub = target_lookup["cell"].get((target_key, mode, cell_key))
    elif same_cell_only:
        sub = target_lookup["cell"].get((target_key, mode, cell_key))
    elif prefer_same_source:
        sub = target_lookup["source"].get((target_key, mode, lincs_source))
        if sub is None and allow_cross_source:
            sub = target_lookup["all"].get((target_key, mode))
    else:
        sub = target_lookup["all"].get((target_key, mode))
    if sub is None or sub.empty:
        return None
    d = (
        np.abs(sub["pert_time"].astype(float) - float(pert_time)) * float(time_weight)
        + np.abs(np.log10(sub["pert_dose"].astype(float) + 1e-6) - np.log10(float(pert_dose) + 1e-6))
        * float(log10_dose_weight)
    )
    return sub.iloc[int(np.argmin(d.to_numpy(dtype=float)))]


def _expression_status_map(expr_df: pd.DataFrame | None, row: pd.Series, cfg: dict) -> tuple[str, float | None]:
    if expr_df is None:
        return "unknown", None
    sub = expr_df[
        (expr_df["cell_key"].astype(str) == str(row["cell_key"]))
        & (expr_df["target_gene"].astype(str) == str(row["target_gene"]))
    ]
    if sub.empty:
        return "unknown", None
    expr_val = float(sub.iloc[0]["expr_log2_tpm1"])
    thr = float(cfg["expression"]["thresholds"]["expressed_log2_tpm1"])
    return ("expression-supported" if expr_val >= thr else "expression-unsupported"), expr_val


def _cap_alignment_pairs(align_df: pd.DataFrame, max_pairs: int) -> pd.DataFrame:
    if max_pairs <= 0 or len(align_df) <= max_pairs:
        return align_df.reset_index(drop=True)
    align_df = align_df.sort_values(
        [
            "target_key",
            "mode",
            "cell_key",
            "compound_source",
            "target_source",
            "compound_time",
            "compound_dose",
            "align_pair_id",
        ]
    ).reset_index(drop=True)
    align_df["cell_rank"] = align_df.groupby(["target_key", "mode", "cell_key"]).cumcount()
    align_df = align_df.sort_values(
        [
            "cell_rank",
            "target_key",
            "mode",
            "cell_key",
            "compound_source",
            "target_source",
            "compound_time",
            "compound_dose",
            "align_pair_id",
        ]
    ).reset_index(drop=True)
    grp = align_df.groupby(["target_key", "mode"], sort=False)
    keep_per_grp = max(1, max_pairs // max(len(grp), 1))
    kept = grp.head(keep_per_grp).reset_index(drop=True)
    if len(kept) < max_pairs:
        extra = align_df.loc[~align_df["align_pair_id"].isin(set(kept["align_pair_id"].astype(str)))].head(
            max_pairs - len(kept)
        )
        kept = pd.concat([kept, extra], ignore_index=True)
    return kept.head(max_pairs).drop(columns=["cell_rank"], errors="ignore").reset_index(drop=True)


def _build_source_coverage_summary(dti: pd.DataFrame, cmp_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in sorted(cmp_df["lincs_source"].astype(str).unique().tolist()):
        cmp_sub = cmp_df[cmp_df["lincs_source"].astype(str) == source].copy()
        tgt_sub = target_df[target_df["lincs_source"].astype(str) == source].copy()
        shared_cells = set(cmp_sub["cell_key"].astype(str)) & set(tgt_sub["cell_key"].astype(str))
        cmp_sub = cmp_sub[cmp_sub["cell_key"].astype(str).isin(shared_cells)].copy()
        tgt_sub = tgt_sub[tgt_sub["cell_key"].astype(str).isin(shared_cells)].copy()
        drug_cells = cmp_sub.groupby("drug_key")["cell_key"].apply(lambda s: sorted(set(s.astype(str)))).to_dict()
        tgt_pairs = set(zip(tgt_sub["target_key"].astype(str), tgt_sub["cell_key"].astype(str)))
        mapped = dti[dti["drug_key"].isin(set(drug_cells.keys()))].copy()
        mapped["has_same_cell_do"] = [
            any((str(t), c) in tgt_pairs for c in drug_cells.get(d, [])) for d, t in zip(mapped["drug_key"], mapped["target_key"])
        ]
        rows.append(
            {
                "scope": source,
                "mapped_pairs": int(len(mapped)),
                "mapped_positive_pairs": int((mapped["y"] == 1).sum()),
                "mapped_negative_pairs": int((mapped["y"] == 0).sum()),
                "same_cell_do_pairs": int(mapped["has_same_cell_do"].sum()),
                "same_cell_do_positive_pairs": int(((mapped["y"] == 1) & mapped["has_same_cell_do"]).sum()),
                "same_cell_do_negative_pairs": int(((mapped["y"] == 0) & mapped["has_same_cell_do"]).sum()),
                "shared_cells": int(len(shared_cells)),
            }
        )

    shared_cells = set(cmp_df["cell_key"].astype(str)) & set(target_df["cell_key"].astype(str))
    cmp_union = cmp_df[cmp_df["cell_key"].astype(str).isin(shared_cells)].copy()
    tgt_union = target_df[target_df["cell_key"].astype(str).isin(shared_cells)].copy()
    drug_cells = cmp_union.groupby("drug_key")["cell_key"].apply(lambda s: sorted(set(s.astype(str)))).to_dict()
    tgt_pairs = set(zip(tgt_union["target_key"].astype(str), tgt_union["cell_key"].astype(str)))
    mapped = dti[dti["drug_key"].isin(set(drug_cells.keys()))].copy()
    mapped["has_same_cell_do"] = [
        any((str(t), c) in tgt_pairs for c in drug_cells.get(d, [])) for d, t in zip(mapped["drug_key"], mapped["target_key"])
    ]
    rows.append(
        {
            "scope": "union",
            "mapped_pairs": int(len(mapped)),
            "mapped_positive_pairs": int((mapped["y"] == 1).sum()),
            "mapped_negative_pairs": int((mapped["y"] == 0).sum()),
            "same_cell_do_pairs": int(mapped["has_same_cell_do"].sum()),
            "same_cell_do_positive_pairs": int(((mapped["y"] == 1) & mapped["has_same_cell_do"]).sum()),
            "same_cell_do_negative_pairs": int(((mapped["y"] == 0) & mapped["has_same_cell_do"]).sum()),
            "shared_cells": int(len(shared_cells)),
        }
    )
    return pd.DataFrame(rows)


def _merge_optional_metadata(frame: pd.DataFrame, lineage_df: pd.DataFrame | None) -> pd.DataFrame:
    out = frame.copy()
    if lineage_df is not None:
        out = out.merge(lineage_df[["cell_key", "lineage", "lineage_subtype"]], on="cell_key", how="left")
    if "lineage" not in out.columns:
        out["lineage"] = "unknown"
    else:
        out["lineage"] = out["lineage"].fillna("unknown").astype(str)
    if "lineage_subtype" not in out.columns:
        out["lineage_subtype"] = "unknown"
    else:
        out["lineage_subtype"] = out["lineage_subtype"].fillna("unknown").astype(str)
    return out


def _coverage_table(
    linked_df: pd.DataFrame,
    group_cols: list[str],
    out_name: str,
    tables_dir: Path,
) -> pd.DataFrame:
    df = (
        linked_df.groupby(group_cols, dropna=False)
        .agg(
            n_context=("linked_instance_id", "count"),
            n_pair=("pair_key", "nunique"),
            n_positive=("y", "sum"),
            n_negative=("y", lambda s: int((1 - s).sum())),
            n_target=("target_key", "nunique"),
            n_family=("family_key", "nunique"),
            relaxed_rows=("aux_do_present", "sum"),
            exact_rows=("aux_do_exact_present", "sum"),
            within_source_rows=("matched_within_source_flag", "sum"),
            cross_source_rows=("matched_cross_source_flag", "sum"),
            expression_supported_rows=("expression_supported_flag", "sum"),
        )
        .reset_index()
        .sort_values(["n_context"], ascending=[False])
    )
    df.to_csv(tables_dir / out_name, index=False)
    return df


def build_linked_benchmark(repo_root: Path) -> None:
    cfg = load_yaml(repo_root / "configs/linked_benchmark.yaml")
    out_root = ensure_dir(repo_root / str(cfg["output_root"]))
    tables_dir = ensure_dir(out_root / str(cfg["tables_dir"]))
    data_dir = ensure_dir(out_root / str(cfg["data_dir"]))
    logs_dir = ensure_dir(out_root / str(cfg["logs_dir"]))

    dti = _load_real_dti_pairs(repo_root, cfg)
    drug_map = dti[["drug_key", "smiles_norm", "smiles_canon", "inchi_key"]].drop_duplicates().reset_index(drop=True)
    cmp_df, gcols, gene_rows, gctx_path, gene_info_path, compound_match_audit, compound_source_audit = _parse_compound_signatures(
        repo_root, cfg, drug_map
    )
    target_df, target_source_audit = _parse_target_signatures(
        repo_root=repo_root,
        cfg=cfg,
        target_symbols=set(dti["target_gene"].astype(str).tolist()),
        gene_cols_ref=gcols,
        gene_rows_ref=gene_rows,
    )
    expr_df = _load_expression_support(repo_root, cfg)
    family_df = _load_family_map(repo_root, cfg)
    lineage_df = _load_lineage_map(repo_root, cfg)
    gene_map = _build_gene_map(gctx_path=gctx_path, gene_info_path=gene_info_path)

    cmp_df = _merge_optional_metadata(cmp_df, lineage_df)
    target_df = _merge_optional_metadata(target_df, lineage_df)

    cmp_df.to_parquet(data_dir / "linked_compound_signatures.parquet", index=False)
    target_df.to_parquet(data_dir / "linked_target_signatures.parquet", index=False)
    gene_map.to_csv(tables_dir / "lincs_gene_provenance.csv", index=False)
    compound_match_audit.to_csv(tables_dir / "compound_match_rule_summary.csv", index=False)
    compound_source_audit.to_csv(tables_dir / "compound_source_parse_summary.csv", index=False)
    target_source_audit.to_csv(tables_dir / "target_source_parse_summary.csv", index=False)
    source_coverage = _build_source_coverage_summary(dti, cmp_df, target_df)
    source_coverage.to_csv(tables_dir / "source_pair_coverage_summary.csv", index=False)

    cmp_by_drug = {k: sub.reset_index(drop=True) for k, sub in cmp_df.groupby("drug_key")}
    target_lookup = _build_target_lookup(target_df)
    # Build the expression lookup once.  The prior implementation filtered the
    # entire expression table for every linked candidate row, which was
    # equivalent but quadratic in practice for the full benchmark.
    expression_lookup: dict[tuple[str, str], tuple[str, float | None]] = {}
    if expr_df is not None:
        expression_threshold = float(cfg["expression"]["thresholds"]["expressed_log2_tpm1"])
        expression_unique = expr_df.drop_duplicates(["cell_key", "target_gene"], keep="first")
        for expr_row in expression_unique.itertuples(index=False):
            expr_value = float(expr_row.expr_log2_tpm1)
            expression_lookup[(str(expr_row.cell_key), str(expr_row.target_gene))] = (
                "expression-supported" if expr_value >= expression_threshold else "expression-unsupported",
                expr_value,
            )
    same_cell_only = bool(cfg["matching"]["same_cell_only"])
    prefer_same_source = bool(cfg["matching"].get("prefer_same_source", True))
    allow_cross_source = bool(cfg["matching"].get("allow_cross_source", True))
    time_weight = float(cfg["matching"]["time_weight"])
    log10_dose_weight = float(cfg["matching"]["log10_dose_weight"])
    modes = [str(x) for x in cfg["matching"]["modes"]]
    align_max_pairs = int(cfg["alignment"].get("build_max_pairs", 20000))

    linked_rows: list[dict[str, object]] = []
    align_rows: list[dict[str, object]] = []
    mapped_dti = dti[dti["drug_key"].isin(set(cmp_df["drug_key"].astype(str)))].copy().reset_index(drop=True)
    for row in mapped_dti.itertuples(index=False):
        cmp_sub = cmp_by_drug.get(str(row.drug_key))
        if cmp_sub is None or cmp_sub.empty:
            continue
        for _, cmp_row in cmp_sub.iterrows():
            rec: dict[str, object] = {
                "linked_instance_id": stable_hash({"pair_key": str(row.pair_key), "compound_sig_id": str(cmp_row["compound_sig_id"])}),
                "dataset": str(row.dataset),
                "pair_key": str(row.pair_key),
                "drug_key": str(row.drug_key),
                "target_key": str(row.target_key),
                "target_gene": str(row.target_gene),
                "y": int(row.y),
                "compound_sig_id": str(cmp_row["compound_sig_id"]),
                "compound_source": str(cmp_row["lincs_source"]),
                "compound_match_rule": str(cmp_row["compound_match_rule"]),
                "cell_id": str(cmp_row["cell_id"]),
                "cell_key": str(cmp_row["cell_key"]),
                "pert_time": float(cmp_row["pert_time"]),
                "pert_dose": float(cmp_row["pert_dose"]),
                "dose_stratum": "dose<=1uM"
                if float(cmp_row["pert_dose"]) <= 1.0
                else ("dose>=10uM" if float(cmp_row["pert_dose"]) >= 10.0 else "dose_mid"),
            }

            expr_status, expr_val = expression_lookup.get(
                (str(rec["cell_key"]), str(rec["target_gene"])),
                ("unknown", None),
            )
            rec["expression_status"] = expr_status
            rec["expr_log2_tpm1"] = expr_val if expr_val is not None else np.nan

            cmp_vec = cmp_row[gcols].to_numpy(dtype=float)
            vals = []
            matched_sources = []
            matched_scopes = []
            for mode in modes:
                tgt = _nearest_target_match(
                    target_lookup,
                    target_key=str(row.target_key),
                    cell_key=str(cmp_row["cell_key"]),
                    lincs_source=str(cmp_row["lincs_source"]),
                    pert_time=float(cmp_row["pert_time"]),
                    pert_dose=float(cmp_row["pert_dose"]),
                    mode=mode,
                    same_cell_only=same_cell_only,
                    prefer_same_source=prefer_same_source,
                    allow_cross_source=allow_cross_source,
                    time_weight=time_weight,
                    log10_dose_weight=log10_dose_weight,
                )
                if tgt is None:
                    rec[f"target_sig_id_{mode.lower()}"] = ""
                    rec[f"target_source_{mode.lower()}"] = ""
                    rec[f"source_scope_{mode.lower()}"] = ""
                    rec[f"aux_do_{mode.lower()}"] = 0.0
                    rec[f"matched_time_{mode.lower()}"] = np.nan
                    rec[f"matched_dose_{mode.lower()}"] = np.nan
                    rec[f"time_diff_{mode.lower()}"] = np.nan
                    rec[f"log10_dose_diff_{mode.lower()}"] = np.nan
                    rec[f"exact_time_match_{mode.lower()}"] = 0
                    rec[f"exact_dose_match_{mode.lower()}"] = 0
                    rec[f"exact_time_dose_match_{mode.lower()}"] = 0
                    rec[f"match_quality_{mode.lower()}"] = "none"
                    continue
                tgt_vec = tgt[gcols].to_numpy(dtype=float)
                cos = _safe_cosine(cmp_vec, tgt_vec)
                source_scope = "within_source" if str(tgt["lincs_source"]) == str(cmp_row["lincs_source"]) else "cross_source"
                time_diff = float(abs(float(tgt["pert_time"]) - float(cmp_row["pert_time"])))
                dose_log_diff = float(
                    abs(
                        np.log10(float(tgt["pert_dose"]) + 1e-6)
                        - np.log10(float(cmp_row["pert_dose"]) + 1e-6)
                    )
                )
                exact_time = int(time_diff <= 1e-9)
                exact_dose = int(dose_log_diff <= 1e-9)
                exact_ctx = int(exact_time and exact_dose)
                rec[f"target_sig_id_{mode.lower()}"] = str(tgt["target_sig_id"])
                rec[f"target_source_{mode.lower()}"] = str(tgt["lincs_source"])
                rec[f"source_scope_{mode.lower()}"] = source_scope
                rec[f"aux_do_{mode.lower()}"] = cos
                rec[f"matched_time_{mode.lower()}"] = float(tgt["pert_time"])
                rec[f"matched_dose_{mode.lower()}"] = float(tgt["pert_dose"])
                rec[f"time_diff_{mode.lower()}"] = time_diff
                rec[f"log10_dose_diff_{mode.lower()}"] = dose_log_diff
                rec[f"exact_time_match_{mode.lower()}"] = exact_time
                rec[f"exact_dose_match_{mode.lower()}"] = exact_dose
                rec[f"exact_time_dose_match_{mode.lower()}"] = exact_ctx
                rec[f"match_quality_{mode.lower()}"] = "exact" if exact_ctx else "relaxed"
                vals.append(cos)
                matched_sources.append(str(tgt["lincs_source"]))
                matched_scopes.append(source_scope)
                if int(row.y) == 1:
                    align_rows.append(
                        {
                            "align_pair_id": stable_hash(
                                {
                                    "pair_key": str(row.pair_key),
                                    "compound_sig_id": str(cmp_row["compound_sig_id"]),
                                    "target_sig_id": str(tgt["target_sig_id"]),
                                    "mode": mode,
                                }
                            ),
                            "dataset": str(row.dataset),
                            "drug_key": str(row.drug_key),
                            "target_key": str(row.target_key),
                            "target_gene": str(row.target_gene),
                            "cell_id": str(cmp_row["cell_id"]),
                            "cell_key": str(cmp_row["cell_key"]),
                            "compound_sig_id": str(cmp_row["compound_sig_id"]),
                            "target_sig_id": str(tgt["target_sig_id"]),
                            "compound_source": str(cmp_row["lincs_source"]),
                            "target_source": str(tgt["lincs_source"]),
                            "source_scope": source_scope,
                            "mode": mode,
                            "compound_time": float(cmp_row["pert_time"]),
                            "compound_dose": float(cmp_row["pert_dose"]),
                            "target_time": float(tgt["pert_time"]),
                            "target_dose": float(tgt["pert_dose"]),
                            "time_diff": time_diff,
                            "log10_dose_diff": dose_log_diff,
                            "exact_time_match": exact_time,
                            "exact_dose_match": exact_dose,
                            "exact_time_dose_match": exact_ctx,
                            "match_quality": "exact" if exact_ctx else "relaxed",
                            "expression_status": expr_status,
                            "expr_log2_tpm1": expr_val if expr_val is not None else np.nan,
                        }
                    )

            rec["aux_do_present"] = float(bool(vals))
            rec["aux_do_max"] = float(max(vals)) if vals else 0.0
            rec["aux_do_mean"] = float(np.mean(vals)) if vals else 0.0
            exact_flags = [int(rec.get(f"exact_time_dose_match_{mode.lower()}", 0)) for mode in modes]
            rec["aux_do_exact_present"] = float(any(exact_flags))
            rec["best_match_quality"] = "exact" if any(exact_flags) else ("relaxed" if vals else "none")
            rec["matched_target_sources"] = "|".join(sorted(set(matched_sources)))
            rec["matched_source_scopes"] = "|".join(sorted(set(matched_scopes)))
            rec["matched_within_source_flag"] = int("within_source" in set(matched_scopes))
            rec["matched_cross_source_flag"] = int("cross_source" in set(matched_scopes))
            rec["expression_supported_flag"] = int(rec["expression_status"] == "expression-supported")
            linked_rows.append(rec)

    linked_df = pd.DataFrame(linked_rows)
    if linked_df.empty:
        raise ValueError("Linked benchmark construction produced no evaluable rows.")
    if family_df is not None:
        linked_df = linked_df.merge(family_df, on="target_gene", how="left")
    else:
        linked_df["family_key"] = linked_df["target_gene"]
    linked_df = _merge_optional_metadata(linked_df, lineage_df)

    align_df = pd.DataFrame(align_rows).drop_duplicates(subset=["align_pair_id"]).reset_index(drop=True)
    raw_align_count = int(len(align_df))
    if not align_df.empty:
        if family_df is not None:
            align_df = align_df.merge(family_df, on="target_gene", how="left")
        else:
            align_df["family_key"] = align_df["target_gene"]
        align_df = _merge_optional_metadata(align_df, lineage_df)
        align_df = _cap_alignment_pairs(align_df, align_max_pairs)

    linked_df.to_parquet(data_dir / "linked_context_benchmark.parquet", index=False)
    if not align_df.empty:
        align_df.to_parquet(data_dir / "linked_alignment_pairs.parquet", index=False)

    mapped_pairs = mapped_dti
    pair_scope = (
        linked_df.groupby("pair_key", dropna=False)
        .agg(
            relaxed_present=("aux_do_present", "max"),
            exact_present=("aux_do_exact_present", "max"),
            expression_supported=("expression_supported_flag", "max"),
        )
        .reset_index()
    )
    funnel = pd.DataFrame(
        [
            {"stage": "dti_pairs_total", "count": int(len(dti)), "unit": "pair"},
            {"stage": "dti_positive_pairs_total", "count": int((dti["y"] == 1).sum()), "unit": "pair"},
            {"stage": "dti_negative_pairs_total", "count": int((dti["y"] == 0).sum()), "unit": "pair"},
            {
                "stage": "raw_lincs_compound_signature_rows",
                "count": int(compound_source_audit["raw_compound_signature_rows"].sum()) if not compound_source_audit.empty else 0,
                "unit": "signature",
            },
            {
                "stage": "raw_lincs_target_signature_rows",
                "count": int(target_source_audit["raw_target_signature_rows"].sum()) if not target_source_audit.empty else 0,
                "unit": "signature",
            },
            {"stage": "dti_pairs_with_lincs_drug", "count": int(len(mapped_pairs)), "unit": "pair"},
            {"stage": "mapped_positive_pairs", "count": int((mapped_pairs["y"] == 1).sum()), "unit": "pair"},
            {"stage": "mapped_negative_pairs", "count": int((mapped_pairs["y"] == 0).sum()), "unit": "pair"},
            {"stage": "target_genes_with_do_support", "count": int(target_df["target_gene"].nunique()), "unit": "target"},
            {"stage": "target_families_with_mapping", "count": int(linked_df["family_key"].astype(str).nunique()), "unit": "family"},
            {"stage": "compound_context_instances", "count": int(len(cmp_df)), "unit": "compound_context"},
            {
                "stage": "pair_level_relaxed_matches",
                "count": int(pair_scope["relaxed_present"].sum()),
                "unit": "pair",
            },
            {
                "stage": "pair_level_exact_matches",
                "count": int(pair_scope["exact_present"].sum()),
                "unit": "pair",
            },
            {"stage": "pair_context_with_same_cell_do", "count": int((linked_df["aux_do_present"] > 0).sum()), "unit": "pair_context"},
            {"stage": "pair_context_exact_time_dose", "count": int((linked_df["aux_do_exact_present"] > 0).sum()), "unit": "pair_context"},
            {"stage": "pair_context_expression_supported", "count": int((linked_df["expression_status"] == "expression-supported").sum()), "unit": "pair_context"},
            {"stage": "final_evaluable_context_instances", "count": int((linked_df["aux_do_present"] > 0).sum()), "unit": "pair_context"},
        ]
    )
    funnel.to_csv(tables_dir / "coverage_funnel.csv", index=False)

    strata = (
        linked_df.groupby(["dataset", "expression_status", "dose_stratum"], dropna=False)
        .agg(
            n=("linked_instance_id", "count"),
            positives=("y", "sum"),
            aux_do_present_rate=("aux_do_present", "mean"),
            aux_do_max_mean=("aux_do_max", "mean"),
        )
        .reset_index()
    )
    strata.to_csv(tables_dir / "linked_context_strata_summary.csv", index=False)

    source_scope_summary = (
        linked_df.groupby(["compound_source", "matched_source_scopes", "compound_match_rule"], dropna=False)
        .agg(
            n=("linked_instance_id", "count"),
            positives=("y", "sum"),
            negatives=("y", lambda s: int((1 - s).sum())),
            aux_do_present_rate=("aux_do_present", "mean"),
            n_cell=("cell_key", "nunique"),
        )
        .reset_index()
    )
    source_scope_summary.to_csv(tables_dir / "linked_source_scope_summary.csv", index=False)
    match_quality = (
        linked_df.groupby(["dataset", "best_match_quality", "expression_status"], dropna=False)
        .agg(
            n_context=("linked_instance_id", "count"),
            n_pair=("pair_key", "nunique"),
            positives=("y", "sum"),
            negatives=("y", lambda s: int((1 - s).sum())),
            within_source_rows=("matched_within_source_flag", "sum"),
            cross_source_rows=("matched_cross_source_flag", "sum"),
        )
        .reset_index()
    )
    match_quality.to_csv(tables_dir / "linked_match_quality_summary.csv", index=False)
    _coverage_table(
        linked_df,
        ["cell_id", "lineage", "lineage_subtype"],
        "linked_cell_coverage.csv",
        tables_dir,
    )
    _coverage_table(
        linked_df,
        ["family_key"],
        "linked_family_coverage.csv",
        tables_dir,
    )
    mechanism_cov = (
        linked_df.groupby(["dataset"], dropna=False)
        .agg(
            lof_relaxed_rows=("aux_do_lof", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) != 0.0).sum())),
            gof_relaxed_rows=("aux_do_gof", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) != 0.0).sum())),
            lof_exact_rows=("exact_time_dose_match_lof", "sum"),
            gof_exact_rows=("exact_time_dose_match_gof", "sum"),
        )
        .reset_index()
    )
    mechanism_cov.to_csv(tables_dir / "linked_mechanism_coverage.csv", index=False)

    source_list = sorted(set(cmp_df["lincs_source"].astype(str)) | set(target_df["lincs_source"].astype(str)))
    summary_lines = [
        "# Linked benchmark build summary",
        f"- lincs_sources: {', '.join(source_list)}",
        f"- dti_pairs_total: {len(dti)}",
        f"- mapped_dti_pairs: {len(mapped_pairs)}",
        f"- mapped_positive_pairs: {int((mapped_pairs['y'] == 1).sum())}",
        f"- mapped_negative_pairs: {int((mapped_pairs['y'] == 0).sum())}",
        f"- linked_context_rows: {len(linked_df)}",
        f"- linked_context_positive_rows: {int((linked_df['y'] == 1).sum())}",
        f"- linked_context_negative_rows: {int((linked_df['y'] == 0).sum())}",
        f"- aux_do_present_rate: {linked_df['aux_do_present'].mean():.6f}",
        f"- exact_time_dose_rate: {linked_df['aux_do_exact_present'].mean():.6f}",
        f"- alignment_pairs_raw: {raw_align_count}",
        f"- alignment_pairs_saved: {len(align_df)}",
        f"- expression_table_present: {int(expr_df is not None)}",
        f"- family_map_present: {int(family_df is not None)}",
        f"- lineage_map_present: {int(lineage_df is not None)}",
        f"- n_linked_cell: {int(linked_df['cell_id'].astype(str).nunique())}",
        f"- n_linked_lineage: {int(linked_df['lineage'].astype(str).nunique())}",
        f"- n_linked_family: {int(linked_df['family_key'].astype(str).nunique())}",
        f"- n_expression_supported_rows: {int(linked_df['expression_supported_flag'].sum())}",
        f"- gene_dim: {len(gcols)}",
    ]
    (logs_dir / "build_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
