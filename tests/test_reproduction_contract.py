from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vccv_table1.metrics import expected_calibration_error, summarize_binding
from vccv_table1.pipeline import (
    BASE_MODEL,
    EviMLP,
    PROB_BASE,
    _load_state_dict_npz,
    _save_state_dict_npz,
    emulate_historical_test_join_multiplicity,
    fit_fixed_verifier,
    fit_single_verifier,
    infer_evidti,
    infer_single_verifier,
    train_evidti,
)
from vccv_fullchain.pipeline import reproduce_fullchain, verify_packaged_inputs


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_package_manifest_hashes_every_declared_file() -> None:
    manifest = json.loads(
        (PACKAGE_ROOT / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert manifest["algorithm"] == "SHA-256"
    assert manifest["excluded"] == ["PACKAGE_MANIFEST.json"]
    declared = manifest["files"]
    actual = {
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in PACKAGE_ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and path.name != "PACKAGE_MANIFEST.json"
    }
    assert set(declared) == actual
    for relative, expected in declared.items():
        digest = hashlib.sha256(
            (PACKAGE_ROOT / relative).read_bytes()
        ).hexdigest()
        assert digest == expected


def test_immutable_package_and_raw_manifests() -> None:
    result = verify_packaged_inputs(PACKAGE_ROOT)
    assert len(result["raw_files"]) == 18
    assert result["raw_bytes"] > 100_000_000
    assert result["package_files_checked"] > 50


def test_raw_package_has_compressed_gctx_but_not_expanded_matrix() -> None:
    raw = PACKAGE_ROOT / "data/raw/geo_gse92742"
    assert any(raw.glob("*.gctx.gz"))
    assert not any(raw.glob("*.gctx"))


def test_table1_davis_mode_is_declared() -> None:
    text = (
        PACKAGE_ROOT / "configs/upstream/dti_labels.yaml"
    ).read_text(encoding="utf-8")
    assert "legacy_raw_davis_labels: true" in text


def test_expected_reference_is_not_a_training_or_inference_input() -> None:
    for function in (
        train_evidti,
        infer_evidti,
        fit_single_verifier,
        fit_fixed_verifier,
        infer_single_verifier,
    ):
        source = inspect.getsource(function).lower()
        assert "table1_expected" not in source
        assert "reference/" not in source
        assert "read_csv" not in source
        assert "read_parquet" not in source
    fullchain_source = inspect.getsource(reproduce_fullchain).lower()
    assert fullchain_source.index("reproduce_table1(") < fullchain_source.index(
        "train_virtualdo("
    )


def test_package_contains_no_predictions_or_pretrained_models() -> None:
    forbidden_names = (
        "prediction",
        "checkpoint",
        "table1_summary.csv",
        "table1_per_slice.csv",
    )
    allowed_data = {
        "dti_labels.parquet",
        "signatures_drug.parquet",
        "do_fused_mu_var.parquet",
    }
    for path in PACKAGE_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(PACKAGE_ROOT).as_posix().lower()
        if path.name in allowed_data or relative.startswith("tests/"):
            continue
        if relative.startswith("vendor/src/"):
            continue
        assert not any(name in relative for name in forbidden_names)


def test_metric_definitions() -> None:
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.4, 0.6, 0.9])
    metrics = summarize_binding(labels, probabilities)
    assert metrics.auc == 1.0
    assert metrics.pr_auc == 1.0
    manual_nll = -np.mean(
        labels * np.log(probabilities)
        + (1 - labels) * np.log(1 - probabilities)
    )
    assert np.isclose(metrics.nll, manual_nll)
    assert metrics.ece == expected_calibration_error(labels, probabilities, 15)


def test_npz_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = EviMLP(input_dimension=5)
    path = tmp_path / "weights.npz"
    _save_state_dict_npz(path, model.state_dict())
    reloaded = EviMLP(input_dimension=5)
    reloaded.load_state_dict(_load_state_dict_npz(path))
    for expected, actual in zip(model.parameters(), reloaded.parameters()):
        assert torch.equal(expected, actual)


def test_verifier_artifact_reloads_identically(tmp_path: Path) -> None:
    rng = np.random.RandomState(42)
    n_cal = 240
    n_test = 80
    cal_base = np.clip(rng.beta(9, 1, n_cal), 1e-6, 1 - 1e-6)
    test_base = np.clip(rng.beta(9, 1, n_test), 1e-6, 1 - 1e-6)
    cal = pd.DataFrame(
        {
            PROB_BASE: cal_base,
            "aux_do_max": rng.normal(size=n_cal),
            "aux_train_target_rate": rng.uniform(size=n_cal),
            "y": (cal_base + 0.05 * rng.normal(size=n_cal) > 0.82).astype(int),
        }
    )
    test = pd.DataFrame(
        {
            PROB_BASE: test_base,
            "aux_do_max": rng.normal(size=n_test),
            "aux_train_target_rate": rng.uniform(size=n_test),
            "y": (test_base > 0.82).astype(int),
        }
    )
    auxiliary = ["aux_do_max", "aux_train_target_rate"]
    _, fitted_test, _, artifact = fit_single_verifier(cal, test, auxiliary)
    artifact_path = tmp_path / "verifier.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    reloaded = json.loads(artifact_path.read_text(encoding="utf-8"))
    inferred = infer_single_verifier(test, auxiliary, reloaded)
    assert np.allclose(fitted_test, inferred, rtol=1e-10, atol=1e-12)


def test_table1_duplicate_join_multiplicity_is_explicit() -> None:
    duplicate_id = "duplicated"
    dti = pd.DataFrame(
        {
            "instance_id": ["normal", duplicate_id, duplicate_id],
        }
    )
    predictions = pd.DataFrame(
        {
            "instance_id": ["normal", duplicate_id],
            "scenario": ["x", "x"],
            "seed": [0, 0],
            PROB_BASE: [0.8, 0.9],
        }
    )
    expanded = emulate_historical_test_join_multiplicity(predictions, dti)
    counts = expanded.groupby("instance_id").size().to_dict()
    assert counts == {"normal": 1, duplicate_id: 16}


def test_reference_has_only_two_expected_rows() -> None:
    reference = json.loads(
        (PACKAGE_ROOT / "reference/table1_expected.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(reference["rows"]) == 2
    assert reference["rows"][0]["internal_model"] == BASE_MODEL


def test_table1_configs_are_complete_and_contain_no_predictions() -> None:
    rng_config = json.loads(
        (
            PACKAGE_ROOT / "configs/display_reconstruction_rng_streams.json"
        ).read_text(encoding="utf-8")
    )
    verifier_config = json.loads(
        (
            PACKAGE_ROOT / "configs/historical_verifier_hyperparameters.json"
        ).read_text(encoding="utf-8")
    )
    assert sorted(rng_config["streams"]) == [20260219, 20260220]
    assert len(rng_config["slice_stream"]) == 12
    assert len(verifier_config["slices"]) == 24
    serialized = json.dumps(
        [rng_config["slice_stream"], verifier_config["slices"]]
    ).lower()
    assert "probability" not in serialized
    assert "prediction" not in serialized
    assert "coefficient" not in serialized
    assert "test_metric" not in serialized
