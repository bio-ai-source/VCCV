from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.hashing import stable_hash
from src.utils.io import ensure_dir, load_yaml, save_json


def _entity_holdout_split(
    df: pd.DataFrame,
    entity_col: str,
    seed: int,
    cal_fraction: float,
    test_fraction: float,
    id_col: str,
) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    entities = np.array(sorted(df[entity_col].dropna().unique().tolist()))
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
    train_ids = df[df[entity_col].isin(train_entities)][id_col].tolist()
    cal_ids = df[df[entity_col].isin(cal_entities)][id_col].tolist()
    test_ids = df[df[entity_col].isin(test_entities)][id_col].tolist()
    return {"train": train_ids, "cal": cal_ids, "test": test_ids}


def _time_dose_shift_split(
    df: pd.DataFrame,
    time_train: list[float],
    time_test: list[float],
    dose_train: list[float],
    dose_test: list[float],
    seed: int,
    cal_fraction: float,
    id_col: str,
) -> dict[str, list[str]]:
    test_mask = df["pert_time"].isin(time_test) | df["pert_dose"].isin(dose_test)
    train_pool_mask = df["pert_time"].isin(time_train) & df["pert_dose"].isin(dose_train)
    train_pool = df[train_pool_mask].copy()
    test_df = df[test_mask].copy()
    rng = np.random.default_rng(seed)
    idx = np.arange(len(train_pool))
    rng.shuffle(idx)
    n_cal = max(1, int(round(len(train_pool) * cal_fraction)))
    cal_idx = set(idx[:n_cal].tolist())
    cal_ids = train_pool.iloc[list(cal_idx)][id_col].tolist()
    train_ids = train_pool.iloc[[i for i in range(len(train_pool)) if i not in cal_idx]][id_col].tolist()
    test_ids = test_df[id_col].tolist()
    return {"train": train_ids, "cal": cal_ids, "test": test_ids}


def _audit_intersections(split: dict[str, list[str]]) -> dict[str, int]:
    tr = set(split["train"])
    ca = set(split["cal"])
    te = set(split["test"])
    return {
        "train_cal_intersection": len(tr & ca),
        "train_test_intersection": len(tr & te),
        "cal_test_intersection": len(ca & te),
    }


def generate_all_splits(repo_root: Path) -> None:
    cfg = load_yaml(repo_root / "configs/splits.yaml")
    dti = pd.read_parquet(repo_root / "data/processed/dti_labels.parquet")
    mech = pd.read_parquet(repo_root / "data/processed/signatures_drug.parquet")
    truth = pd.read_parquet(repo_root / "data/processed/mechanism_truth.parquet")
    mech = mech.merge(truth[["instance_id", "true_target_key"]], on="instance_id", how="left")
    mech = mech.rename(columns={"true_target_key": "target_key"})
    mech["dataset"] = "mechanism"
    dti["dataset"] = "binding"
    if "instance_id" not in dti.columns:
        dti["instance_id"] = dti["pair_key"]
    if "instance_id" not in mech.columns:
        raise ValueError("Mechanism signatures missing instance_id.")

    frames = [dti[["instance_id", "drug_key", "target_key", "context_key", "pert_time", "pert_dose", "dataset"]], mech[["instance_id", "drug_key", "target_key", "context_key", "pert_time", "pert_dose", "dataset"]]]
    all_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["instance_id"])

    cal_fraction = float(cfg["cal_fraction"])
    test_fraction = float(cfg["test_fraction"])
    seeds = [int(x) for x in cfg["seeds"]]

    audit_lines: list[str] = []
    for scenario, scfg in cfg["scenarios"].items():
        scenario_dir = ensure_dir(repo_root / "splits" / scenario)
        for seed in seeds:
            if scenario == "time_dose_shift":
                split = _time_dose_shift_split(
                    all_df,
                    time_train=list(scfg["time_train"]),
                    time_test=list(scfg["time_test"]),
                    dose_train=list(scfg["dose_train"]),
                    dose_test=list(scfg["dose_test"]),
                    seed=seed,
                    cal_fraction=cal_fraction,
                    id_col="instance_id",
                )
            else:
                split = _entity_holdout_split(
                    all_df,
                    entity_col=scfg["entity"],
                    seed=seed,
                    cal_fraction=cal_fraction,
                    test_fraction=test_fraction,
                    id_col="instance_id",
                )
            split_path = scenario_dir / f"seed_{seed}.json"
            save_json(split_path, split, indent=2)
            h = stable_hash(split)
            (scenario_dir / f"seed_{seed}_hash.txt").write_text(h, encoding="utf-8")
            ints = _audit_intersections(split)
            audit_lines.extend(
                [
                    f"## {scenario} seed={seed}",
                    f"- train: {len(split['train'])}",
                    f"- cal: {len(split['cal'])}",
                    f"- test: {len(split['test'])}",
                    f"- train_cal_intersection: {ints['train_cal_intersection']}",
                    f"- train_test_intersection: {ints['train_test_intersection']}",
                    f"- cal_test_intersection: {ints['cal_test_intersection']}",
                ]
            )
    (repo_root / "results/logs/split_audit.md").write_text("\n".join(audit_lines), encoding="utf-8")

