from __future__ import annotations

import hashlib
import inspect
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.do_dictionary.fusion import (
    KEY_COLS as FUSION_KEY_COLS,
    load_fusion_parameters,
    save_fusion_parameters,
    select_fusion_parameters,
)
from src.do_dictionary.virtualdo import (
    KEY_COLS as VIRTUALDO_KEY_COLS,
    VirtualDoBundle,
    VirtualDoNet,
    load_virtualdo_checkpoint,
    predict_virtualdo,
    save_virtualdo_checkpoint,
)
from src.inference import engine as posterior_engine
from src.inference.engine import (
    load_posterior_bundle,
    predict_posterior,
    save_posterior_bundle,
)
from src.preprocessing.parse_pipeline import build_real_dti_labels
from vccv_fullchain.pipeline import reproduce_fullchain


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_manifest_record(path: Path, relative: str) -> dict[str, object]:
    return {
        "path": relative,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def test_raw_manifest_exactly_covers_the_packaged_upstream_files() -> None:
    manifest = json.loads(
        (PACKAGE_ROOT / "RAW_INPUT_MANIFEST.json").read_text(encoding="utf-8")
    )
    declared = manifest["files"]
    actual = {
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in (PACKAGE_ROOT / "data/raw").rglob("*")
        if path.is_file()
    }

    assert manifest["schema_version"] == 1
    assert manifest["algorithm"] == "SHA-256"
    assert set(declared) == actual
    assert len(declared) == 18
    assert all(path.startswith("data/raw/") for path in declared)
    assert all(
        len(record["sha256"]) == 64 and int(record["size"]) > 0
        for record in declared.values()
    )


def _write_tiny_davis_repo(repo_root: Path, *, raw_affinity: bool) -> None:
    configs = repo_root / "configs"
    raw = repo_root / "data/raw/deepdta_davis"
    configs.mkdir(parents=True)
    raw.mkdir(parents=True)
    (configs / "base.yaml").write_text(
        "mode: budget\nseed: 20260219\n",
        encoding="utf-8",
    )
    (configs / "dti_labels.yaml").write_text(
        "\n".join(
            [
                "dataset_preference:",
                "  - davis",
                (
                    "legacy_raw_davis_labels: "
                    f"{'true' if raw_affinity else 'false'}"
                ),
                "binarization:",
                "  davis:",
                "    positive_if_gte: 7.0",
                "    negative_if_lte: 5.0",
                "sampling:",
                "  max_pairs_per_dataset:",
                "    budget: -1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (raw / "ligands_can.txt").write_text(
        json.dumps({"drug": "CCO"}),
        encoding="utf-8",
    )
    (raw / "proteins.txt").write_text(
        json.dumps({"GENE1": "MPEPTIDE"}),
        encoding="utf-8",
    )
    with (raw / "Y").open("wb") as handle:
        pickle.dump(np.asarray([[2.0]], dtype=float), handle)


def test_davis_config_switch_changes_parser_output(tmp_path: Path) -> None:
    transformed_root = tmp_path / "transformed"
    raw_root = tmp_path / "raw"
    _write_tiny_davis_repo(transformed_root, raw_affinity=False)
    _write_tiny_davis_repo(raw_root, raw_affinity=True)

    transformed = build_real_dti_labels(transformed_root)
    raw = build_real_dti_labels(raw_root)

    assert transformed.loc[0, "affinity_scale"] == "pKd"
    assert transformed.loc[0, "y"] == 1
    assert raw.loc[0, "affinity_scale"] == "Kd_nM_legacy_threshold_misuse"
    assert raw.loc[0, "affinity"] == 2.0
    assert raw.loc[0, "y"] == 0


def _virtualdo_key_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target_key": "target-a",
                "context_key": "A549",
                "pert_time": 6.0,
                "pert_dose": 1.0,
                "platform": "L1000",
                "batch": "B1",
                "mode": "LoF",
            },
            {
                "target_key": "target-b",
                "context_key": "A549",
                "pert_time": 24.0,
                "pert_dose": 0.1,
                "platform": "L1000B",
                "batch": "B2",
                "mode": "GoF",
            },
        ],
        columns=VIRTUALDO_KEY_COLS,
    )


def test_virtualdo_safe_npz_json_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(73)
    architecture = {
        "n_targets": 2,
        "n_contexts": 1,
        "embed_dim": 4,
        "hidden_dim": 7,
        "n_genes": 2,
    }
    model = VirtualDoNet(**architecture)
    maps = {
        "target": {"target-a": 0, "target-b": 1},
        "context": {"A549": 0},
        "mode": {"LoF": 0, "GoF": 1},
    }
    gene_order = ["G0", "G1"]
    variance_order = ["V0", "V1"]
    calibration_rows = _virtualdo_key_rows()
    checkpoint_dir = tmp_path / "virtualdo"
    state_path, schema_path = save_virtualdo_checkpoint(
        checkpoint_dir,
        model,
        maps=maps,
        gcols=gene_order,
        vcols=variance_order,
        architecture=architecture,
        calibration_scale=1.25,
        calibration_nll=0.42,
        sigma_min=0.05,
        calibration_rows=calibration_rows,
    )

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["format"] == "vccv.virtualdo.safe-npz"
    assert schema["state"]["sha256"] == _sha256(state_path)
    assert len(schema["calibration"]["holdout_keys"]) == 2
    with np.load(state_path, allow_pickle=False) as archive:
        assert archive.files
        assert all(archive[name].dtype != object for name in archive.files)

    loaded = load_virtualdo_checkpoint(checkpoint_dir, device="cpu")
    assert loaded.model is not model
    assert loaded.maps == maps
    assert loaded.gcols == gene_order
    assert loaded.vcols == variance_order
    assert loaded.calibration_scale == 1.25
    for name, expected in model.state_dict().items():
        assert torch.equal(expected.cpu(), loaded.model.state_dict()[name].cpu())

    in_memory = VirtualDoBundle(
        model=model,
        maps=maps,
        gcols=gene_order,
        vcols=variance_order,
        calibration_scale=1.25,
        sigma_min=0.05,
        architecture=architecture,
        state_sha256=schema["state"]["sha256"],
    )
    before = predict_virtualdo(in_memory, calibration_rows, device="cpu")
    after = predict_virtualdo(loaded, calibration_rows, device="cpu")
    pd.testing.assert_frame_equal(before, after, check_exact=True)


def _fusion_rows() -> tuple[pd.DataFrame, pd.DataFrame, tuple[object, ...]]:
    common = {
        "target_key": "target-a",
        "context_key": "A549",
        "pert_dose": 1.0,
        "platform": "L1000",
        "batch": "B1",
        "mode": "LoF",
    }
    observed = pd.DataFrame(
        [
            {
                **common,
                "pert_time": 6.0,
                "G0": 0.0,
                "G1": 0.0,
                "V0": 0.20,
                "V1": 0.20,
                "qc_do": 0.9,
            },
            {
                **common,
                "pert_time": 24.0,
                "G0": 1.0,
                "G1": -1.0,
                "V0": 0.15,
                "V1": 0.15,
                "qc_do": 0.8,
            },
        ]
    )
    virtual = pd.DataFrame(
        [
            {
                **common,
                "pert_time": 24.0,
                "G0": 0.8,
                "G1": -0.8,
                "V0": 0.25,
                "V1": 0.25,
            }
        ]
    )
    holdout = tuple(observed.loc[1, column] for column in FUSION_KEY_COLS)
    return observed, virtual, holdout


def test_fusion_grid_fit_save_and_load_round_trip(tmp_path: Path) -> None:
    observed, virtual, holdout = _fusion_rows()
    config = {
        "r_grid": {
            "a0": [-1.0, 1.0],
            "a1": [0.5, 2.0],
            "a2": [0.5, 2.0],
        },
        "mapping": {
            "time_bins": [6, 24, 48],
            "dose_bins": [0.1, 1.0, 10.0],
            "lambda_t": 0.3,
            "lambda_d": 1.0,
            "lambda_p": 0.4,
            "lambda_b": 0.2,
        },
    }
    selected, calibration, evaluated = select_fusion_parameters(
        obs=observed,
        virtual=virtual,
        gcols=["G0", "G1"],
        vcols=["V0", "V1"],
        cfg=config,
        holdout_keys={holdout},
    )

    assert len(calibration) == 8
    assert evaluated == 1
    assert calibration["selected"].sum() == 1
    assert selected == {
        key: float(calibration.loc[calibration["selected"].eq(1), key].iloc[0])
        for key in ("a0", "a1", "a2")
    }
    assert np.isfinite(
        calibration[["mean_gaussian_nll", "mean_mse"]].to_numpy()
    ).all()

    parameters_path = tmp_path / "fusion_parameters.json"
    save_fusion_parameters(
        parameters_path,
        parameters=selected,
        mapping=config["mapping"],
        calibration=calibration,
        holdout_source="unit-test-virtualdo-schema",
        declared_holdout_rows=1,
        evaluated_holdout_rows=evaluated,
    )
    loaded = load_fusion_parameters(parameters_path)
    assert loaded["format"] == "vccv.fusion.parameters"
    assert loaded["parameters"] == selected
    assert loaded["selection"]["grid_combinations"] == 8
    assert loaded["holdout"] == {
        "source": "unit-test-virtualdo-schema",
        "declared_rows": 1,
        "evaluated_rows": 1,
        "anchor_policy": "virtualdo_train_rows_only",
    }


def _write_posterior_prediction_inputs(repo_root: Path) -> dict[str, dict[str, object]]:
    paths = {
        "signatures": Path("data/processed/signatures_drug.parquet"),
        "prior": Path("results/predictions_json/dti_prior_scores.parquet"),
        "do_fused": Path("data/processed/do_fused_mu_var.parquet"),
    }
    for relative in paths.values():
        (repo_root / relative).parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "instance_id": "instance-1",
                "drug_key": "drug-a",
                "context_key": "A549",
                "pert_time": 24.0,
                "pert_dose": 1.0,
                "G0": 0.2,
                "G1": -0.2,
            }
        ]
    ).to_parquet(repo_root / paths["signatures"], index=False)
    pd.DataFrame(
        [
            {
                "drug_key": "drug-a",
                "target_key": "target-a",
                "calibrated_prob": 0.9,
            }
        ]
    ).to_parquet(repo_root / paths["prior"], index=False)
    pd.DataFrame(
        [
            {
                "target_key": "target-a",
                "context_key": "A549",
                "pert_time": 24.0,
                "pert_dose": 1.0,
                "mode": mode,
                "G0": sign * 0.2,
                "G1": sign * -0.2,
                "V0": 0.1,
                "V1": 0.1,
            }
            for mode, sign in (("LoF", 1.0), ("GoF", -1.0))
        ]
    ).to_parquet(repo_root / paths["do_fused"], index=False)
    return {
        name: _raw_manifest_record(repo_root / relative, relative.as_posix())
        for name, relative in paths.items()
    }


def _posterior_bundle(input_hashes: dict[str, dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "gene_columns": ["G0", "G1"],
        "eta": 1.5,
        "priors": {
            "single_mass": 0.70,
            "poly_prior": 0.20,
            "null_prior": 0.10,
        },
        "config": {
            "kcand_dti": 1,
            "poly": {"k_poly": 1},
            "decision": {},
        },
        "input_hashes": input_hashes,
        "eta_calibration": [{"eta": 1.5, "mean_nll": 0.3}],
        "null_centers": np.asarray([[0.0, 0.0]], dtype=np.float64),
        "null_weights": np.asarray([1.0], dtype=np.float64),
        "sigma_base": np.asarray([0.20, 0.20], dtype=np.float64),
        "B": np.eye(2, dtype=np.float64),
        "b_lof": np.zeros(2, dtype=np.float64),
        "b_gof": np.zeros(2, dtype=np.float64),
        "beta_lof": np.asarray([1.0], dtype=np.float64),
        "beta_gof": np.asarray([1.0], dtype=np.float64),
    }


def test_posterior_bundle_round_trip_predicts_without_a_truth_file(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "run"
    input_hashes = _write_posterior_prediction_inputs(repo_root)
    fitted = _posterior_bundle(input_hashes)
    metadata_path = save_posterior_bundle(
        fitted,
        repo_root / "results/checkpoints/posterior",
    )
    loaded = load_posterior_bundle(metadata_path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    arrays_path = metadata_path.parent / metadata["arrays_file"]
    assert metadata["format"] == "vccv.posterior_bundle"
    assert metadata["arrays_sha256"] == _sha256(arrays_path)
    with np.load(arrays_path, allow_pickle=False) as archive:
        assert all(archive[name].dtype != object for name in archive.files)
    for name in posterior_engine._BUNDLE_ARRAY_KEYS:
        assert np.array_equal(fitted[name], loaded[name])

    assert not (repo_root / "data/processed/mechanism_truth.parquet").exists()
    before = predict_posterior(repo_root, fitted, write_outputs=False)
    after = predict_posterior(repo_root, loaded, write_outputs=False)
    assert json.dumps(before, sort_keys=True) == json.dumps(after, sort_keys=True)
    assert after[0]["summary"]["instance_id"] == "instance-1"
    assert np.isclose(
        sum(item["posterior"] for item in after[0]["posterior_items"]),
        1.0,
    )


def test_posterior_prediction_interface_has_no_truth_dependency() -> None:
    public_source = inspect.getsource(predict_posterior).lower()
    loader_source = inspect.getsource(
        posterior_engine._load_prediction_inputs
    ).lower()

    assert "read_parquet" not in public_source
    assert "mechanism_truth" not in loader_source
    assert tuple(posterior_engine._PREDICTION_INPUT_KEYS) == (
        "signatures",
        "prior",
        "do_fused",
    )


def test_readme_describes_the_packaged_workflow_in_english() -> None:
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
    runner_source = inspect.getsource(reproduce_fullchain)

    assert "# VCCV Real-Data Reproduction Workflow" in readme
    assert "## Method Summary" in readme
    assert "EviDTI supplies the upstream structural-prior scores" in readme
    assert "Train virtual anchors from observed intervention signatures" in readme
    assert "Score target and warning/null branches" in readme
    assert "Recompute the reference evaluation metrics" in readme
    assert "## Reference Results" not in readme
    assert "VCCV + EviDTI" not in readme
    assert '"table1_lane"' in runner_source
    assert '"posterior_lane"' in runner_source
    assert '"status"' in runner_source
