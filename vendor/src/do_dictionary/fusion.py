from __future__ import annotations

import hashlib
from itertools import product
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import load_yaml


KEY_COLS = [
    "target_key",
    "context_key",
    "pert_time",
    "pert_dose",
    "platform",
    "batch",
    "mode",
]
PARAMETER_VERSION = 1


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def _normalise_key_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ("target_key", "context_key", "platform", "batch", "mode"):
        out[column] = out[column].astype(str)
    out["pert_time"] = out["pert_time"].astype(float)
    out["pert_dose"] = out["pert_dose"].astype(float)
    return out


def _row_key(row: pd.Series | Any) -> tuple[str, str, float, float, str, str, str]:
    return (
        str(row["target_key"]),
        str(row["context_key"]),
        float(row["pert_time"]),
        float(row["pert_dose"]),
        str(row["platform"]),
        str(row["batch"]),
        str(row["mode"]),
    )


def _record_key(record: dict[str, Any]) -> tuple[str, str, float, float, str, str, str]:
    return (
        str(record["target_key"]),
        str(record["context_key"]),
        float(record["pert_time"]),
        float(record["pert_dose"]),
        str(record["platform"]),
        str(record["batch"]),
        str(record["mode"]),
    )


def _map_distance(row, obs_row, cfg):
    dt = abs(float(row["pert_time"]) - float(obs_row["pert_time"]))
    dd = abs(
        np.log10(float(row["pert_dose"]) + 1e-6)
        - np.log10(float(obs_row["pert_dose"]) + 1e-6)
    )
    dp = 0.0 if row["platform"] == obs_row["platform"] else 1.0
    db = 0.0 if row["batch"] == obs_row["batch"] else 1.0
    return (
        float(cfg["mapping"]["lambda_t"]) * dt
        + float(cfg["mapping"]["lambda_d"]) * dd
        + float(cfg["mapping"]["lambda_p"]) * dp
        + float(cfg["mapping"]["lambda_b"]) * db
    )


def _nearest_row(
    query: pd.Series,
    candidates: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.Series, float] | None:
    if candidates.empty:
        return None
    stable = candidates.sort_values(KEY_COLS, kind="mergesort").reset_index(drop=True)
    distances = stable.apply(lambda candidate: _map_distance(query, candidate, cfg), axis=1)
    best_index = int(np.argmin(distances.to_numpy(dtype=float)))
    return stable.iloc[best_index], float(distances.iloc[best_index])


def _fusion_arrays(
    *,
    mu_obs: np.ndarray,
    var_obs: np.ndarray,
    mu_virtual: np.ndarray,
    var_virtual: np.ndarray,
    reliability: float,
) -> tuple[np.ndarray, np.ndarray]:
    r = float(np.clip(reliability, 0.0, 1.0))
    var_obs = np.maximum(np.asarray(var_obs, dtype=float), 1e-5)
    var_virtual = np.maximum(np.asarray(var_virtual, dtype=float), 1e-5)
    precision = r / var_obs + (1.0 - r) / var_virtual
    var_fused = 1.0 / np.maximum(precision, 1e-8)
    mu_fused = var_fused * (
        r * np.asarray(mu_obs, dtype=float) / var_obs
        + (1.0 - r) * np.asarray(mu_virtual, dtype=float) / var_virtual
    )
    return mu_fused, var_fused


def _deterministic_holdout_keys(obs: pd.DataFrame) -> set[tuple[str, str, float, float, str, str, str]]:
    """Reconstruct the VirtualDO 20% holdout when an old schema is absent."""

    indices = np.arange(len(obs))
    rng = np.random.default_rng(20260219)
    rng.shuffle(indices)
    n_calibration = max(1, int(0.2 * len(obs)))
    if n_calibration >= len(obs):
        n_calibration = max(1, len(obs) - 1)
    return {
        _row_key(obs.iloc[int(index)])
        for index in indices[:n_calibration]
    }


def _load_holdout_keys(
    repo_root: Path,
    obs: pd.DataFrame,
) -> tuple[
    set[tuple[str, str, float, float, str, str, str]],
    str,
]:
    schema_path = repo_root / "results/checkpoints/virtualdo/virtualdo_schema.json"
    if schema_path.exists():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        records = schema.get("calibration", {}).get("holdout_keys", [])
        if records:
            return {_record_key(record) for record in records}, str(
                schema_path.relative_to(repo_root)
            )
    return _deterministic_holdout_keys(obs), "deterministic_virtualdo_split_fallback"


def _build_calibration_cases(
    *,
    obs: pd.DataFrame,
    virtual: pd.DataFrame,
    gcols: list[str],
    vcols: list[str],
    cfg: dict[str, Any],
    holdout_keys: set[tuple[str, str, float, float, str, str, str]],
) -> list[dict[str, Any]]:
    """Create leakage-safe cases: held-out rows are never candidate anchors."""

    holdout_mask = obs.apply(lambda row: _row_key(row) in holdout_keys, axis=1)
    heldout = obs[holdout_mask].copy().reset_index(drop=True)
    anchor_pool = obs[~holdout_mask].copy().reset_index(drop=True)
    cases: list[dict[str, Any]] = []

    for _, truth_row in heldout.iterrows():
        anchor_candidates = anchor_pool[
            (anchor_pool["target_key"] == truth_row["target_key"])
            & (anchor_pool["context_key"] == truth_row["context_key"])
            & (anchor_pool["mode"] == truth_row["mode"])
        ]
        virtual_candidates = virtual[
            (virtual["target_key"] == truth_row["target_key"])
            & (virtual["context_key"] == truth_row["context_key"])
            & (virtual["mode"] == truth_row["mode"])
        ]
        observed_match = _nearest_row(truth_row, anchor_candidates, cfg)
        virtual_match = _nearest_row(truth_row, virtual_candidates, cfg)
        if observed_match is None or virtual_match is None:
            continue
        observed_row, observed_distance = observed_match
        virtual_row, virtual_distance = virtual_match
        cases.append(
            {
                "truth_mu": truth_row[gcols].to_numpy(dtype=float),
                "truth_var": np.maximum(
                    truth_row[vcols].to_numpy(dtype=float),
                    1e-5,
                ),
                "observed_mu": observed_row[gcols].to_numpy(dtype=float),
                "observed_var": np.maximum(
                    observed_row[vcols].to_numpy(dtype=float),
                    1e-5,
                ),
                "virtual_mu": virtual_row[gcols].to_numpy(dtype=float),
                "virtual_var": np.maximum(
                    virtual_row[vcols].to_numpy(dtype=float),
                    1e-5,
                ),
                "qc_do": float(observed_row.get("qc_do", 0.8)),
                "d_map": float(observed_distance),
                "virtual_d_map": float(virtual_distance),
                "holdout_key": _row_key(truth_row),
                "anchor_key": _row_key(observed_row),
            }
        )
    return cases


def select_fusion_parameters(
    *,
    obs: pd.DataFrame,
    virtual: pd.DataFrame,
    gcols: list[str],
    vcols: list[str],
    cfg: dict[str, Any],
    holdout_keys: set[tuple[str, str, float, float, str, str, str]],
) -> tuple[dict[str, float], pd.DataFrame, int]:
    """Exhaustively select r parameters by Gaussian NLL, then MSE."""

    cases = _build_calibration_cases(
        obs=obs,
        virtual=virtual,
        gcols=gcols,
        vcols=vcols,
        cfg=cfg,
        holdout_keys=holdout_keys,
    )
    if not cases:
        raise ValueError(
            "Fusion calibration has no leakage-safe observed holdout cases."
        )

    grids = cfg["r_grid"]
    combinations = list(
        product(
            [float(value) for value in grids["a0"]],
            [float(value) for value in grids["a1"]],
            [float(value) for value in grids["a2"]],
        )
    )
    if not combinations:
        raise ValueError("fusion.r_grid must contain at least one combination.")

    rows: list[dict[str, Any]] = []
    for a0, a1, a2 in combinations:
        nll_values: list[float] = []
        mse_values: list[float] = []
        reliability_values: list[float] = []
        for case in cases:
            reliability = _sigmoid(
                a0 + a1 * float(case["qc_do"]) - a2 * float(case["d_map"])
            )
            mu_fused, var_fused = _fusion_arrays(
                mu_obs=case["observed_mu"],
                var_obs=case["observed_var"],
                mu_virtual=case["virtual_mu"],
                var_virtual=case["virtual_var"],
                reliability=reliability,
            )
            error = np.asarray(case["truth_mu"]) - mu_fused
            # Observation uncertainty is part of the predictive comparison;
            # the held-out mean/variance never enters the fused prediction.
            predictive_var = np.maximum(
                var_fused + np.asarray(case["truth_var"]),
                1e-6,
            )
            nll_values.append(
                float(
                    0.5
                    * np.mean(
                        (error**2) / predictive_var
                        + np.log(predictive_var)
                    )
                )
            )
            mse_values.append(float(np.mean(error**2)))
            reliability_values.append(reliability)
        rows.append(
            {
                "a0": a0,
                "a1": a1,
                "a2": a2,
                "mean_gaussian_nll": float(np.mean(nll_values)),
                "mean_mse": float(np.mean(mse_values)),
                "mean_reliability": float(np.mean(reliability_values)),
                "n_holdout": int(len(cases)),
            }
        )

    calibration = pd.DataFrame(rows)
    calibration = calibration.sort_values(
        ["mean_gaussian_nll", "mean_mse", "a0", "a1", "a2"],
        kind="mergesort",
    ).reset_index(drop=True)
    calibration["selected"] = 0
    calibration.loc[0, "selected"] = 1
    selected = {
        "a0": float(calibration.loc[0, "a0"]),
        "a1": float(calibration.loc[0, "a1"]),
        "a2": float(calibration.loc[0, "a2"]),
    }
    return selected, calibration, len(cases)


def save_fusion_parameters(
    path: Path,
    *,
    parameters: dict[str, float],
    mapping: dict[str, Any],
    calibration: pd.DataFrame,
    holdout_source: str,
    declared_holdout_rows: int,
    evaluated_holdout_rows: int,
) -> None:
    selected_row = calibration.loc[calibration["selected"].eq(1)].iloc[0]
    payload = {
        "format": "vccv.fusion.parameters",
        "version": PARAMETER_VERSION,
        "parameters": {
            key: float(parameters[key])
            for key in ("a0", "a1", "a2")
        },
        "mapping": {
            key: float(value)
            if isinstance(value, (float, int, np.floating, np.integer))
            else value
            for key, value in mapping.items()
        },
        "selection": {
            "objective": "mean_gaussian_nll_then_mean_mse",
            "tie_break": ["a0", "a1", "a2"],
            "mean_gaussian_nll": float(selected_row["mean_gaussian_nll"]),
            "mean_mse": float(selected_row["mean_mse"]),
            "grid_combinations": int(len(calibration)),
        },
        "holdout": {
            "source": holdout_source,
            "declared_rows": int(declared_holdout_rows),
            "evaluated_rows": int(evaluated_holdout_rows),
            "anchor_policy": "virtualdo_train_rows_only",
        },
    }
    _atomic_json(path, payload)


def load_fusion_parameters(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "vccv.fusion.parameters":
        raise ValueError("Unsupported fusion parameter format.")
    if int(payload.get("version", -1)) != PARAMETER_VERSION:
        raise ValueError("Unsupported fusion parameter version.")
    for key in ("a0", "a1", "a2"):
        value = float(payload["parameters"][key])
        if not np.isfinite(value):
            raise ValueError(f"Non-finite fusion parameter: {key}")
        payload["parameters"][key] = value
    return payload


def _observed_only_anchors(
    obs: pd.DataFrame,
    vir: pd.DataFrame,
    gcols: list[str],
    vcols: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return observed anchors that would otherwise be lost by a virtual-only loop."""

    virtual_keys = {
        tuple(row[column] for column in KEY_COLS)
        for _, row in vir[KEY_COLS].drop_duplicates().iterrows()
    }
    rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    for _, row in obs.iterrows():
        key = tuple(row[column] for column in KEY_COLS)
        if key in virtual_keys:
            continue
        out = {column: row[column] for column in KEY_COLS}
        for column in gcols:
            out[column] = float(row[column])
        for column in vcols:
            out[column] = float(max(row[column], 1e-5))
        rows.append(out)
        mapping_rows.append(
            {
                "target_key": row["target_key"],
                "context_key": row["context_key"],
                "pert_time": float(row["pert_time"]),
                "pert_dose": float(row["pert_dose"]),
                "mode": row["mode"],
                "obs_source": (
                    f"{row['target_key']}|{row['context_key']}|{row['pert_time']}|"
                    f"{row['pert_dose']}|{row['mode']}"
                ),
                "d_map": 0.0,
                "qc_do": float(row.get("qc_do", 0.8)),
                "r": 1.0,
                "is_fallback": 0,
            }
        )
    return rows, mapping_rows


def _fuse_tables(
    *,
    obs: pd.DataFrame,
    virtual: pd.DataFrame,
    gcols: list[str],
    vcols: list[str],
    cfg: dict[str, Any],
    parameters: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fused_rows: list[dict[str, object]] = []
    map_rows: list[dict[str, object]] = []
    stable_virtual = virtual.sort_values(KEY_COLS, kind="mergesort").reset_index(drop=True)

    for _, row in stable_virtual.iterrows():
        candidates = obs[
            (obs["target_key"] == row["target_key"])
            & (obs["context_key"] == row["context_key"])
            & (obs["mode"] == row["mode"])
        ]
        observed_match = _nearest_row(row, candidates, cfg)
        if observed_match is None:
            qc_do = 0.0
            d_map = 3.0
            reliability = 0.0
            mu_obs = np.zeros(len(gcols), dtype=float)
            var_obs = np.ones(len(gcols), dtype=float) * 100.0
            obs_source = ""
            is_fallback = 1
        else:
            best, d_map = observed_match
            qc_do = float(best.get("qc_do", 0.8))
            reliability = _sigmoid(
                parameters["a0"]
                + parameters["a1"] * qc_do
                - parameters["a2"] * d_map
            )
            mu_obs = best[gcols].to_numpy(dtype=float)
            var_obs = np.maximum(best[vcols].to_numpy(dtype=float), 1e-5)
            obs_source = (
                f"{best['target_key']}|{best['context_key']}|"
                f"{best['pert_time']}|{best['pert_dose']}|{best['mode']}"
            )
            is_fallback = 0 if d_map == 0 else 1

        mu_virtual = row[gcols].to_numpy(dtype=float)
        var_virtual = np.maximum(row[vcols].to_numpy(dtype=float), 1e-5)
        mu_fused, var_fused = _fusion_arrays(
            mu_obs=mu_obs,
            var_obs=var_obs,
            mu_virtual=mu_virtual,
            var_virtual=var_virtual,
            reliability=reliability,
        )

        out = {column: row[column] for column in KEY_COLS}
        for column, value in zip(gcols, mu_fused):
            out[column] = float(value)
        for column, value in zip(vcols, var_fused):
            out[column] = float(value)
        fused_rows.append(out)
        map_rows.append(
            {
                "target_key": row["target_key"],
                "context_key": row["context_key"],
                "pert_time": float(row["pert_time"]),
                "pert_dose": float(row["pert_dose"]),
                "mode": row["mode"],
                "obs_source": obs_source,
                "d_map": d_map,
                "qc_do": qc_do,
                "r": reliability,
                "is_fallback": is_fallback,
                "a0": float(parameters["a0"]),
                "a1": float(parameters["a1"]),
                "a2": float(parameters["a2"]),
            }
        )

    observed_rows, observed_map_rows = _observed_only_anchors(
        obs,
        stable_virtual,
        gcols,
        vcols,
    )
    fused_rows.extend(observed_rows)
    map_rows.extend(observed_map_rows)
    fused_df = (
        pd.DataFrame(fused_rows)
        .drop_duplicates(subset=KEY_COLS, keep="first")
        .sort_values(KEY_COLS, kind="mergesort")
        .reset_index(drop=True)
    )
    map_df = pd.DataFrame(map_rows).reset_index(drop=True)
    return fused_df, map_df


def fuse_observed_virtual(repo_root: Path) -> None:
    """Fit reliability on a holdout, serialize/reload it, then fuse."""

    config_path = repo_root / "configs/fusion.yaml"
    obs_mu_path = repo_root / "data/processed/observeddo_mu.parquet"
    obs_var_path = repo_root / "data/processed/observeddo_var_diag.parquet"
    virtual_path = repo_root / "data/processed/virtualdo_predictions.parquet"
    f_cfg = load_yaml(config_path)
    obs_mu = pd.read_parquet(obs_mu_path)
    obs_var = pd.read_parquet(obs_var_path)
    virtual = _normalise_key_columns(pd.read_parquet(virtual_path))
    obs = _normalise_key_columns(
        obs_mu.merge(
            obs_var,
            on=KEY_COLS,
            how="inner",
            suffixes=("", "_var"),
        )
    )

    gcols = sorted(
        [column for column in virtual.columns if column.startswith("G") and column[1:].isdigit()],
        key=lambda value: int(value[1:]),
    )
    vcols = sorted(
        [column for column in virtual.columns if column.startswith("V") and column[1:].isdigit()],
        key=lambda value: int(value[1:]),
    )
    if not gcols or len(gcols) != len(vcols):
        raise ValueError("VirtualDO mean/variance gene schemas are empty or misaligned.")
    missing = [
        column
        for column in [*gcols, *vcols]
        if column not in obs.columns
    ]
    if missing:
        raise ValueError(f"ObservedDo is missing VirtualDO gene columns: {missing[:5]}")

    holdout_keys, holdout_source = _load_holdout_keys(repo_root, obs)
    selected, calibration, evaluated_rows = select_fusion_parameters(
        obs=obs,
        virtual=virtual,
        gcols=gcols,
        vcols=vcols,
        cfg=f_cfg,
        holdout_keys=holdout_keys,
    )

    calibration_path = repo_root / "results/metrics_tables/fusion_calibration.csv"
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(calibration_path, index=False)

    params_path = repo_root / "results/checkpoints/fusion/fusion_params.json"
    save_fusion_parameters(
        params_path,
        parameters=selected,
        mapping=dict(f_cfg["mapping"]),
        calibration=calibration,
        holdout_source=holdout_source,
        declared_holdout_rows=len(holdout_keys),
        evaluated_holdout_rows=evaluated_rows,
    )

    # Deliberately discard the in-memory selection and use only the serialized
    # JSON artifact for the final fusion pass.
    loaded = load_fusion_parameters(params_path)
    loaded_cfg = {"mapping": loaded["mapping"]}
    fused_df, map_df = _fuse_tables(
        obs=obs,
        virtual=virtual,
        gcols=gcols,
        vcols=vcols,
        cfg=loaded_cfg,
        parameters=loaded["parameters"],
    )

    fused_path = repo_root / "data/processed/do_fused_mu_var.parquet"
    mapping_path = repo_root / "results/logs/do_mapping_log.parquet"
    fused_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    fused_df.to_parquet(fused_path, index=False)
    map_df.to_parquet(mapping_path, index=False)

    virtual_schema_path = (
        repo_root / "results/checkpoints/virtualdo/virtualdo_schema.json"
    )
    lineage_inputs = {
        str(obs_mu_path.relative_to(repo_root)): _sha256_file(obs_mu_path),
        str(obs_var_path.relative_to(repo_root)): _sha256_file(obs_var_path),
        str(virtual_path.relative_to(repo_root)): _sha256_file(virtual_path),
        str(config_path.relative_to(repo_root)): _sha256_file(config_path),
    }
    if virtual_schema_path.exists():
        lineage_inputs[str(virtual_schema_path.relative_to(repo_root))] = (
            _sha256_file(virtual_schema_path)
        )
    lineage = {
        "stage": "observed_virtual_fusion",
        "selection": loaded["selection"],
        "parameters": loaded["parameters"],
        "holdout": loaded["holdout"],
        "inputs": lineage_inputs,
        "checkpoint": {
            str(params_path.relative_to(repo_root)): _sha256_file(params_path),
        },
        "outputs": {
            str(calibration_path.relative_to(repo_root)): _sha256_file(
                calibration_path
            ),
            str(fused_path.relative_to(repo_root)): _sha256_file(fused_path),
            str(mapping_path.relative_to(repo_root)): _sha256_file(mapping_path),
        },
        "rows": {
            "observed": int(len(obs)),
            "virtual": int(len(virtual)),
            "fused": int(len(fused_df)),
            "mapping": int(len(map_df)),
        },
    }
    _atomic_json(
        repo_root / "results/logs/fusion_lineage.json",
        lineage,
    )
