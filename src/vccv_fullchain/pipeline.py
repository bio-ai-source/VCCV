from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vccv_table1.pipeline import (
    PROB_BASE,
    reproduce_table1,
    sha256_file,
)

from src.align.train import train_align
from src.do_dictionary.observed import build_observed_dictionary
from src.do_dictionary.virtualdo import train_virtualdo
from src.inference.engine import run_inference
from src.preprocessing.parse_pipeline import parse_and_prepare
from src.preprocessing.splits import generate_all_splits
from src.qc.pipeline import run_qc


RAW_MANIFEST = "RAW_INPUT_MANIFEST.json"
PACKAGE_MANIFEST = "PACKAGE_MANIFEST.json"
TABLE_INPUT_FILES = (
    "data/processed/dti_labels.parquet",
    "data/processed/signatures_drug.parquet",
    "data/processed/do_fused_mu_var.parquet",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_record(path: Path) -> dict[str, Any]:
    return {"size": int(path.stat().st_size), "sha256": sha256_file(path)}


def _hash_tree(root: Path, *, excluded_names: set[str] | None = None) -> list[dict[str, Any]]:
    excluded_names = excluded_names or set()
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name in excluded_names:
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _verify_file_manifest(
    package_root: Path,
    manifest_path: Path,
    *,
    values_are_records: bool,
) -> list[dict[str, Any]]:
    manifest = _read_json(manifest_path)
    errors: list[str] = []
    checked: list[dict[str, Any]] = []
    for rel, expected_value in sorted(manifest["files"].items()):
        path = package_root / rel
        if not path.is_file():
            errors.append(f"missing: {rel}")
            continue
        expected = (
            expected_value
            if values_are_records
            else {"sha256": str(expected_value), "size": None}
        )
        actual = _manifest_record(path)
        if actual["sha256"] != expected["sha256"]:
            errors.append(
                f"sha256 mismatch: {rel}; expected={expected['sha256']} "
                f"actual={actual['sha256']}"
            )
        if expected.get("size") is not None and actual["size"] != int(expected["size"]):
            errors.append(
                f"size mismatch: {rel}; expected={expected['size']} "
                f"actual={actual['size']}"
            )
        checked.append({"path": rel, **actual})
    if errors:
        raise RuntimeError(
            f"Manifest verification failed for {manifest_path.name}:\n"
            + "\n".join(errors)
        )
    return checked


def verify_packaged_inputs(package_root: Path) -> dict[str, Any]:
    """Verify immutable source/raw bytes without loading any historical results."""
    package_root = package_root.resolve()
    raw_manifest_path = package_root / RAW_MANIFEST
    package_manifest_path = package_root / PACKAGE_MANIFEST
    if not raw_manifest_path.is_file():
        raise FileNotFoundError(raw_manifest_path)
    if not package_manifest_path.is_file():
        raise FileNotFoundError(package_manifest_path)

    raw_checked = _verify_file_manifest(
        package_root, raw_manifest_path, values_are_records=True
    )
    package_checked = _verify_file_manifest(
        package_root, package_manifest_path, values_are_records=False
    )
    raw_declared = set(_read_json(raw_manifest_path)["files"])
    raw_actual = {
        path.relative_to(package_root).as_posix()
        for path in (package_root / "data/raw").rglob("*")
        if path.is_file()
    }
    if raw_declared != raw_actual:
        raise RuntimeError(
            "RAW_INPUT_MANIFEST.json is not a closed-world inventory; "
            f"missing={sorted(raw_declared - raw_actual)}, "
            f"extra={sorted(raw_actual - raw_declared)}"
        )
    package_declared = set(_read_json(package_manifest_path)["files"])
    package_actual = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file()
        and path.name != PACKAGE_MANIFEST
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
    }
    if package_declared != package_actual:
        raise RuntimeError(
            "PACKAGE_MANIFEST.json is not a closed-world inventory; "
            f"missing={sorted(package_declared - package_actual)}, "
            f"extra={sorted(package_actual - package_declared)}"
        )
    forbidden = [
        path.relative_to(package_root).as_posix()
        for path in (package_root / "data").rglob("*")
        if path.is_file()
        and (
            "predictions" in path.parts
            or "results" in path.parts
            or path.suffix in {".pt", ".pth", ".ckpt"}
        )
    ]
    if forbidden:
        raise RuntimeError(
            "The immutable package contains forbidden historical predictions/checkpoints: "
            + ", ".join(forbidden)
        )
    return {
        "raw_manifest_sha256": sha256_file(raw_manifest_path),
        "package_manifest_sha256": sha256_file(package_manifest_path),
        "raw_files": raw_checked,
        "package_files_checked": len(package_checked),
        "raw_bytes": int(sum(row["size"] for row in raw_checked)),
    }


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, destination, dirs_exist_ok=False)


def _prepare_workspace(package_root: Path, run_root: Path) -> Path:
    workspace = run_root / "workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    _copy_tree(package_root / "data" / "raw", workspace / "data" / "raw")
    _copy_tree(package_root / "configs" / "upstream", workspace / "configs")
    for rel in (
        "data/interim",
        "data/processed",
        "results/logs",
        "results/metrics_tables",
        "results/predictions_json",
        "results/checkpoints",
        "model_cards",
    ):
        (workspace / rel).mkdir(parents=True, exist_ok=True)
    return workspace


def _validate_upstream_data(workspace: Path) -> dict[str, Any]:
    dti = pd.read_parquet(workspace / "data/processed/dti_labels.parquet")
    sig_drug = pd.read_parquet(
        workspace / "data/processed/signatures_drug.parquet"
    )
    sig_do = pd.read_parquet(workspace / "data/processed/signatures_do.parquet")
    truth = pd.read_parquet(workspace / "data/processed/mechanism_truth.parquet")
    if len(dti) != 24_000 or int(dti["y"].sum()) != 23_362:
        raise RuntimeError(
            "Raw reconstruction did not produce the Table 1 label "
            f"population: rows={len(dti)}, positives={int(dti['y'].sum())}"
        )
    expected_cols = {
        "instance_id",
        "drug_key",
        "target_key",
        "context_key",
        "pert_time",
        "pert_dose",
        "y",
    }
    missing = sorted(expected_cols - set(dti.columns))
    if missing:
        raise RuntimeError(f"DTI data missing columns: {missing}")
    if sig_drug.empty or sig_do.empty or truth.empty:
        raise RuntimeError("Raw preprocessing produced an empty mechanism table")
    return {
        "dti_rows": int(len(dti)),
        "dti_positive": int(dti["y"].sum()),
        "dti_duplicate_instance_ids": int(
            dti["instance_id"].astype(str).duplicated().sum()
        ),
        "drug_signatures_after_qc": int(len(sig_drug)),
        "do_signatures_after_qc": int(len(sig_do)),
        "mechanism_truth_rows": int(len(truth)),
        "mechanism_drugs": int(sig_drug["drug_key"].nunique()),
    }


def _build_observed_only_fusion(workspace: Path, output_path: Path) -> None:
    """Build a temporary observed-only table for the independent Table 1 lane.

    Table 1 binding IDs and mechanism signature IDs have zero overlap, so these
    values cannot affect the historical verifier.  The complete fitted fusion is
    produced later in the VCCV lane and never replaced by this bootstrap table.
    """
    mu = pd.read_parquet(workspace / "data/processed/observeddo_mu.parquet")
    var = pd.read_parquet(
        workspace / "data/processed/observeddo_var_diag.parquet"
    )
    keys = [
        "target_key",
        "context_key",
        "pert_time",
        "pert_dose",
        "platform",
        "batch",
        "mode",
    ]
    fused = mu.merge(var, on=keys, how="inner")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fused.to_parquet(output_path, index=False)


def _stage_table1_input(
    package_root: Path, workspace: Path, run_root: Path
) -> Path:
    table_input = run_root / "table1_input"
    (table_input / "data/processed").mkdir(parents=True, exist_ok=False)
    for name in ("dti_labels.parquet", "signatures_drug.parquet"):
        shutil.copy2(
            workspace / "data/processed" / name,
            table_input / "data/processed" / name,
        )
    _build_observed_only_fusion(
        workspace, table_input / "data/processed/do_fused_mu_var.parquet"
    )
    _copy_tree(package_root / "splits", table_input / "splits")
    (table_input / "configs").mkdir(parents=True, exist_ok=True)
    for name in (
        "display_reconstruction_rng_streams.json",
        "historical_verifier_hyperparameters.json",
    ):
        shutil.copy2(
            package_root / "configs" / name, table_input / "configs" / name
        )
    _copy_tree(package_root / "reference", table_input / "reference")

    files: dict[str, dict[str, Any]] = {}
    for rel in TABLE_INPUT_FILES:
        files[rel] = _manifest_record(table_input / rel)
    for base in ("splits", "configs", "reference"):
        for path in sorted((table_input / base).rglob("*")):
            if path.is_file():
                rel = path.relative_to(table_input).as_posix()
                files[rel] = _manifest_record(path)
    _write_json(
        table_input / "INPUT_MANIFEST.json",
        {
            "schema_version": 1,
            "purpose": "freshly reconstructed Table 1 input boundary",
            "files": files,
        },
    )
    return table_input


def _build_evidti_prior(
    workspace: Path, table_output: Path
) -> dict[str, Any]:
    predictions_path = table_output / "evidti_test_predictions_fresh.parquet"
    dti_path = workspace / "data/processed/dti_labels.parquet"
    predictions = pd.read_parquet(predictions_path)
    dti = pd.read_parquet(dti_path)
    canonical = (
        dti[["instance_id", "drug_key", "target_key"]]
        .drop_duplicates("instance_id", keep="last")
        .copy()
    )
    joined = predictions.merge(canonical, on="instance_id", how="inner")
    if joined.empty:
        raise RuntimeError("Fresh EviDTI predictions could not be linked to DTI rows")
    prior = (
        joined.groupby(["drug_key", "target_key"], as_index=False)[PROB_BASE]
        .mean()
        .rename(columns={PROB_BASE: "calibrated_prob"})
        .sort_values(["drug_key", "calibrated_prob"], ascending=[True, False])
        .reset_index(drop=True)
    )
    prior_path = (
        workspace / "results/predictions_json/dti_prior_scores.parquet"
    )
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    prior.to_parquet(prior_path, index=False)

    signatures = pd.read_parquet(
        workspace / "data/processed/signatures_drug.parquet"
    )
    mechanism_drugs = set(signatures["drug_key"].astype(str))
    covered_drugs = set(prior["drug_key"].astype(str))
    missing = sorted(mechanism_drugs - covered_drugs)
    if missing:
        raise RuntimeError(
            f"Fresh EviDTI prior misses {len(missing)} mechanism drugs"
        )
    lineage = {
        "artifact": prior_path.relative_to(workspace).as_posix(),
        "artifact_sha256": sha256_file(prior_path),
        "source_predictions": predictions_path.as_posix(),
        "source_predictions_sha256": sha256_file(predictions_path),
        "source_dti_sha256": sha256_file(dti_path),
        "rows": int(len(prior)),
        "drugs": int(prior["drug_key"].nunique()),
        "targets": int(prior["target_key"].nunique()),
        "mechanism_drug_coverage": 1.0,
        "aggregation": (
            "mean checkpoint-reload EviDTI probability over every fresh "
            "scenario/seed test occurrence for each drug-target pair"
        ),
    }
    _write_json(
        workspace / "results/predictions_json/dti_prior_lineage.json",
        lineage,
    )
    return lineage


def _verify_alignment_reload(workspace: Path) -> dict[str, Any]:
    path = workspace / "results/checkpoints/align/align_params.npz"
    first = np.load(path, allow_pickle=False)
    arrays = {name: np.asarray(first[name]).copy() for name in first.files}
    first.close()
    second = np.load(path, allow_pickle=False)
    for name, expected in arrays.items():
        actual = np.asarray(second[name])
        if not np.array_equal(expected, actual):
            raise RuntimeError(f"Alignment reload mismatch for {name}")
    second.close()
    result = {
        "artifact": path.relative_to(workspace).as_posix(),
        "artifact_sha256": sha256_file(path),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
        "reload_equal": True,
    }
    _write_json(
        workspace / "results/checkpoints/align/reload_validation.json", result
    )
    return result


def _require_fullchain_artifacts(workspace: Path) -> dict[str, Any]:
    required = [
        "data/processed/virtualdo_predictions.parquet",
        "data/processed/do_fused_mu_var.parquet",
        "results/checkpoints/align/align_params.npz",
        "results/checkpoints/posterior/posterior_bundle.json",
        "results/checkpoints/posterior/posterior_bundle.npz",
        "results/predictions_json/mechanism_predictions.parquet",
        "results/predictions_json/mechanism_summary.parquet",
    ]
    # The upgraded components write these directories.  File names inside are
    # deliberately discovered so the bundle format can evolve without silently
    # dropping the serialization boundary.
    required_dirs = [
        "results/checkpoints/virtualdo",
        "results/checkpoints/fusion",
        "results/checkpoints/posterior",
    ]
    missing = [
        rel for rel in required if not (workspace / rel).is_file()
    ]
    for rel in required_dirs:
        path = workspace / rel
        if not path.is_dir() or not any(p.is_file() for p in path.rglob("*")):
            missing.append(rel)
    if missing:
        raise RuntimeError("Full-chain artifacts missing: " + ", ".join(missing))
    summary = pd.read_parquet(
        workspace / "results/predictions_json/mechanism_summary.parquet"
    )
    if summary.empty:
        raise RuntimeError("Posterior inference produced no mechanism predictions")
    pure_predictions = pd.read_parquet(
        workspace / "results/predictions_json/mechanism_predictions.parquet"
    )
    if len(pure_predictions) != len(summary):
        raise RuntimeError(
            "Truth-free posterior predictions and evaluated summary have "
            "different row counts"
        )

    instance_dir = (
        workspace / "results/predictions_json/mechanism_instances"
    )
    non_null_counts: list[int] = []
    candidate_counts: list[int] = []
    top_null = 0
    for path in sorted(instance_dir.glob("*.json")):
        record = _read_json(path)
        items = record.get("posterior_distribution", [])
        non_null_counts.append(
            sum(item.get("type") != "null" for item in items)
        )
        candidate_counts.append(
            int(record.get("input_summary", {}).get("candidate_count", 0))
        )
        top_null += int(
            record.get("top_hypothesis", {}).get("type") == "null"
        )
    if len(non_null_counts) != len(summary):
        raise RuntimeError(
            "Per-instance posterior JSON count does not match prediction count"
        )
    if not non_null_counts or min(non_null_counts) < 1:
        raise RuntimeError(
            "At least one posterior instance has no non-NULL hypothesis"
        )
    if min(candidate_counts) < 1:
        raise RuntimeError("At least one posterior instance has no candidates")
    top_null_rate = float(top_null / len(non_null_counts))
    if top_null_rate >= 0.95:
        raise RuntimeError(
            f"Posterior collapsed to NULL: top-NULL rate={top_null_rate:.3f}"
        )

    fusion_calibration = pd.read_csv(
        workspace / "results/metrics_tables/fusion_calibration.csv"
    )
    selected = fusion_calibration[fusion_calibration["selected"].eq(1)]
    if len(selected) != 1 or int(selected.iloc[0]["n_holdout"]) < 1:
        raise RuntimeError("Fusion has no valid selected holdout calibration")

    posterior_metadata = _read_json(
        workspace
        / "results/checkpoints/posterior/posterior_bundle.json"
    )
    return {
        "mechanism_predictions": int(len(summary)),
        "posterior_columns": summary.columns.tolist(),
        "posterior_support": {
            "minimum_non_null_hypotheses": int(min(non_null_counts)),
            "median_non_null_hypotheses": float(
                np.median(non_null_counts)
            ),
            "maximum_non_null_hypotheses": int(max(non_null_counts)),
            "top_null_count": int(top_null),
            "top_null_rate": top_null_rate,
            "minimum_candidate_count": int(min(candidate_counts)),
        },
        "fusion_calibration": {
            "n_holdout": int(selected.iloc[0]["n_holdout"]),
            "a0": float(selected.iloc[0]["a0"]),
            "a1": float(selected.iloc[0]["a1"]),
            "a2": float(selected.iloc[0]["a2"]),
        },
        "posterior_parent_hashes": posterior_metadata["input_hashes"],
        "required_artifacts": {
            rel: _manifest_record(workspace / rel) for rel in required
        },
    }


def _write_processed_manifest(workspace: Path) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for base in ("data/interim", "data/processed"):
        for path in sorted((workspace / base).rglob("*")):
            if path.is_file():
                records[path.relative_to(workspace).as_posix()] = _manifest_record(
                    path
                )
    manifest = {
        "schema_version": 1,
        "generated_from_raw_in_this_run": True,
        "files": records,
    }
    _write_json(workspace / "PROCESSED_DATA_MANIFEST.json", manifest)
    return manifest


def reproduce_fullchain(
    package_root: Path,
    output_dir: Path,
    *,
    device_request: str = "auto",
    strict_table1: bool = True,
    legacy_rng_compatibility: bool = True,
    train_seed: int | None = None,
) -> dict[str, Any]:
    """Run Table 1 and the complete VCCV chain from packaged raw inputs."""
    package_root = package_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir == package_root or package_root in output_dir.parents:
        raise RuntimeError("Output must be outside the immutable package")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    started = time.time()

    input_verification = verify_packaged_inputs(package_root)
    _write_json(output_dir / "input_verification.json", input_verification)
    workspace = _prepare_workspace(package_root, output_dir)

    print("[1/8] raw parse and synthetic mechanism construction", flush=True)
    parse_and_prepare(workspace)
    print("[2/8] QC, fresh mechanism splits and ObservedDO", flush=True)
    run_qc(workspace)
    generate_all_splits(workspace)
    build_observed_dictionary(workspace)
    upstream_stats = _validate_upstream_data(workspace)

    print("[3/8] fresh EviDTI training/reload and Table 1 lane", flush=True)
    table_input = _stage_table1_input(package_root, workspace, output_dir)
    table_output = output_dir / "table1"
    table_manifest = reproduce_table1(
        package_root=table_input,
        output_dir=table_output,
        device_request=device_request,
        strict=strict_table1,
        legacy_rng_compatibility=legacy_rng_compatibility,
        train_seed=train_seed,
    )

    print("[4/8] build fresh EviDTI DTI prior", flush=True)
    prior_lineage = _build_evidti_prior(workspace, table_output)
    print("[5/8] train, serialize, reload and infer VirtualDO", flush=True)
    train_virtualdo(workspace, device_request=device_request)
    print("[6/8] fit/reload fusion and train/reload alignment", flush=True)
    train_align(workspace, device_request=device_request)
    align_reload = _verify_alignment_reload(workspace)
    print("[7/8] fit, serialize, reload and infer VCCV posterior", flush=True)
    run_inference(workspace)
    fullchain_validation = _require_fullchain_artifacts(workspace)
    processed_manifest = _write_processed_manifest(workspace)

    lineage = {
        "relationship": {
            "table1_lane": (
                "raw labels -> EviDTI checkpoints -> calibration verifier "
                "-> binding metrics"
            ),
            "posterior_lane": (
                "raw signatures -> ObservedDO; fresh EviDTI probabilities -> "
                "DTI prior -> VirtualDO -> fitted fusion -> alignment -> "
                "VCCV posterior"
            ),
            "status": "Both training and inference lanes completed in this run.",
        },
        "upstream_stats": upstream_stats,
        "evidti_prior": prior_lineage,
        "alignment_reload": align_reload,
        "fullchain_validation": fullchain_validation,
        "processed_data_files": int(len(processed_manifest["files"])),
    }
    _write_json(output_dir / "fullchain_lineage.json", lineage)

    manifest = {
        "status": (
            "PASS"
            if table_manifest["status"] == "PASS"
            else table_manifest["status"]
        ),
        "started_utc": started_at,
        "finished_utc": _utc_now(),
        "elapsed_seconds": float(time.time() - started),
        "package_root": str(package_root),
        "output_dir": str(output_dir),
        "requested_training_device": str(device_request),
        "table1_status": table_manifest["status"],
        "raw_manifest_sha256": input_verification["raw_manifest_sha256"],
        "package_manifest_sha256": input_verification[
            "package_manifest_sha256"
        ],
        "run_files": _hash_tree(
            output_dir, excluded_names={"run_manifest.json", "SUCCESS"}
        ),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    (output_dir / "SUCCESS").write_text(
        "Raw preprocessing, fresh EviDTI training/reload, Table 1 evaluation, "
        "fresh prior construction, VirtualDO training/reload, fitted fusion "
        "reload, alignment training/reload, posterior fit/reload/inference and "
        "processed-data manifest generation completed.\n",
        encoding="utf-8",
    )
    print("[8/8] full chain PASS", flush=True)
    return manifest
