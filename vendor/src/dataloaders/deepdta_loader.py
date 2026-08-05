from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.hashing import sha256_text


def _read_json_line(path: Path) -> dict[str, str]:
    raw = path.read_text(encoding="utf-8").strip()
    return json.loads(raw)


def _read_pickle_matrix(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def canonical_target_symbol(raw_target_id: str) -> str:
    """
    Normalize DeepDTA target identifiers to a stable gene-like symbol
    so they can be aligned with perturbation targets in LINCS.
    """
    s = str(raw_target_id).upper().strip()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^A-Z0-9_-]", "", s)
    if s.endswith("P") and len(s) > 2 and s[-2].isdigit():
        s = s[:-1]
    return s or "UNKNOWN_TARGET"


def normalize_affinity_scale(dataset_alias: str, value: float) -> tuple[float, str]:
    """Return the benchmark value on the scale used by the label thresholds.

    The DeepDTA Davis matrix stores Kd in nM, whereas the configured Davis
    cutoffs are pKd cutoffs.  KIBA is already distributed on the KIBA-score
    scale and therefore requires no transformation.
    """
    if dataset_alias == "davis":
        if value <= 0:
            raise ValueError(f"Davis Kd must be positive, found {value}")
        return float(9.0 - np.log10(value)), "pKd"
    return float(value), "KIBA_score"


def load_deepdta_dataset(root: Path, dataset: str) -> pd.DataFrame:
    ds_dir = root / dataset
    dataset_alias = dataset.replace("deepdta_", "")
    ligands = _read_json_line(ds_dir / "ligands_can.txt")
    proteins = _read_json_line(ds_dir / "proteins.txt")
    y = _read_pickle_matrix(ds_dir / "Y")

    drug_ids = list(ligands.keys())
    target_ids = list(proteins.keys())
    rows: list[dict[str, object]] = []
    for i, drug_id in enumerate(drug_ids):
        for j, target_id in enumerate(target_ids):
            value = y[i, j]
            if np.isnan(value):
                continue
            smiles = ligands[drug_id]
            seq = proteins[target_id]
            target_symbol = canonical_target_symbol(target_id)
            affinity, affinity_scale = normalize_affinity_scale(dataset_alias, float(value))
            rows.append(
                {
                    "dataset": dataset_alias,
                    "drug_id": drug_id,
                    "target_id": target_id,
                    "smiles": smiles,
                    "target_seq": seq,
                    "affinity_raw": float(value),
                    "affinity": affinity,
                    "affinity_scale": affinity_scale,
                    "drug_key": sha256_text(smiles),
                    "target_key": sha256_text(target_symbol),
                    "target_gene": target_symbol,
                    "target_raw_id": target_id,
                }
            )
    return pd.DataFrame(rows)


def add_context_and_condition(df: pd.DataFrame) -> pd.DataFrame:
    cells = ["A549", "MCF7", "PC3", "HT29", "A375", "VCAP"]
    times = [6, 24, 48]
    doses = [0.1, 1.0, 10.0]

    out = df.copy()
    idx = np.arange(len(out))
    out["context_key"] = [cells[i % len(cells)] for i in idx]
    out["pert_time"] = [times[(i // 2) % len(times)] for i in idx]
    out["pert_dose"] = [doses[(i // 3) % len(doses)] for i in idx]
    out["platform"] = np.where((idx % 2) == 0, "L1000", "L1000B")
    out["batch"] = [f"B{(i % 5) + 1}" for i in idx]
    out["condition_key"] = (
        out["pert_time"].astype(str)
        + "|"
        + out["pert_dose"].astype(str)
        + "|"
        + out["platform"]
        + "|"
        + out["batch"]
    )
    return out


def binarize_affinity(
    df: pd.DataFrame,
    positive_if_gte: float,
    negative_if_lte: float,
) -> pd.DataFrame:
    out = df.copy()
    y = np.full(len(out), fill_value=-1, dtype=int)
    y[out["affinity"].to_numpy() >= positive_if_gte] = 1
    y[out["affinity"].to_numpy() <= negative_if_lte] = 0
    out["y"] = y
    out = out[out["y"] >= 0].reset_index(drop=True)
    return out
