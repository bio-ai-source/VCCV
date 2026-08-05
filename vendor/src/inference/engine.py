from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.inference.decision import assign_decision_state
from src.utils.io import ensure_dir, load_json, load_yaml


POSTERIOR_BUNDLE_SCHEMA_VERSION = 1
POSTERIOR_BUNDLE_DIR = Path("results/checkpoints/posterior")
POSTERIOR_BUNDLE_ARRAYS = "posterior_bundle.npz"
POSTERIOR_BUNDLE_METADATA = "posterior_bundle.json"

_FIT_INPUT_PATHS = {
    "config": Path("configs/inference.yaml"),
    "signatures": Path("data/processed/signatures_drug.parquet"),
    "mechanism_truth": Path("data/processed/mechanism_truth.parquet"),
    "prior": Path("results/predictions_json/dti_prior_scores.parquet"),
    "do_fused": Path("data/processed/do_fused_mu_var.parquet"),
    "align": Path("results/checkpoints/align/align_params.npz"),
    "split": Path("splits/drug_heldout/seed_0.json"),
}

_PREDICTION_INPUT_KEYS = ("signatures", "prior", "do_fused")
_BUNDLE_ARRAY_KEYS = (
    "null_centers",
    "null_weights",
    "sigma_base",
    "B",
    "b_lof",
    "b_gof",
    "beta_lof",
    "beta_gof",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_inputs(repo_root: Path) -> dict[str, dict[str, object]]:
    hashes: dict[str, dict[str, object]] = {}
    for name, relative_path in _FIT_INPUT_PATHS.items():
        path = repo_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Required posterior input is missing: {path}")
        hashes[name] = {
            "path": relative_path.as_posix(),
            "sha256": _sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
    return hashes


def _validate_prediction_input_hashes(repo_root: Path, bundle: dict[str, Any]) -> None:
    recorded = bundle["input_hashes"]
    for name in _PREDICTION_INPUT_KEYS:
        entry = recorded[name]
        path = repo_root / str(entry["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Posterior prediction input is missing: {path}")
        actual_size = int(path.stat().st_size)
        actual_hash = _sha256_file(path)
        if actual_size != int(entry["size_bytes"]) or actual_hash != str(entry["sha256"]):
            raise ValueError(
                f"Posterior prediction input drift for {name}: "
                f"expected {entry['sha256']} ({entry['size_bytes']} bytes), "
                f"found {actual_hash} ({actual_size} bytes)."
            )


def _softmax(scores: np.ndarray) -> np.ndarray:
    s = scores - np.max(scores)
    e = np.exp(s)
    return e / np.sum(e)


def _gaussian_energy(u: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
    var = np.maximum(var, 1e-6)
    return float(0.5 * np.mean(((u - mu) ** 2) / var + np.log(var)))


def _project_simplex(v: np.ndarray) -> np.ndarray:
    if np.sum(v) == 1.0 and np.all(v >= 0):
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w


def _project_simplex_with_floor(v: np.ndarray, eps: float) -> np.ndarray:
    k = len(v)
    if k * eps >= 1.0:
        raise ValueError("Invalid simplex floor epsilon.")
    z = np.asarray(v, dtype=float) - eps
    z = _project_simplex(z)
    z = z / np.maximum(z.sum(), 1e-12) * (1.0 - k * eps)
    w = z + eps
    w = w / np.maximum(w.sum(), 1e-12)
    return w


def _enforce_poly_support(w: np.ndarray, min_second_weight: float) -> np.ndarray:
    """Keep a poly hypothesis identifiable by requiring two substantive components."""
    if min_second_weight <= 0.0 or len(w) < 2:
        return w
    if min_second_weight >= 0.5:
        raise ValueError("poly.min_second_weight must be smaller than 0.5")
    order = np.argsort(-w)
    largest, second = int(order[0]), int(order[1])
    if w[second] < min_second_weight:
        transfer = min(min_second_weight - float(w[second]), float(w[largest]))
        w = w.copy()
        w[largest] -= transfer
        w[second] += transfer
        w /= np.maximum(w.sum(), 1e-12)
    return w


def _dirichlet_logpdf(w: np.ndarray, alpha: np.ndarray, eps: float) -> float:
    ww = np.maximum(w, eps)
    log_normalizer = math.lgamma(float(np.sum(alpha))) - float(
        sum(math.lgamma(float(value)) for value in alpha)
    )
    return float(log_normalizer + np.sum((alpha - 1.0) * np.log(ww)))


def _poly_energy_grad(
    u: np.ndarray,
    mus: np.ndarray,
    vars_: np.ndarray,
    w: np.ndarray,
    alpha: np.ndarray,
    eta: float,
    eps: float,
) -> tuple[float, np.ndarray]:
    """
    Objective F(w) = eta * E_lik(w) - log Dir(w; alpha)
    """
    mu = np.sum(w[:, None] * mus, axis=0)
    var = np.maximum(np.sum((w[:, None] ** 2) * vars_, axis=0), 1e-6)
    r = u - mu

    e = _gaussian_energy(u, mu, var)

    dmu = mus  # [k, g]
    dvar = 2.0 * w[:, None] * vars_  # [k, g]
    term = (-r[None, :] * dmu) / var[None, :] + 0.5 * dvar * (1.0 / var[None, :] - (r[None, :] ** 2) / (var[None, :] ** 2))
    grad_e = term.mean(axis=1)

    ww = np.maximum(w, eps)
    grad_prior = -(alpha - 1.0) / ww
    grad = eta * grad_e + grad_prior
    f = eta * e - _dirichlet_logpdf(w, alpha=alpha, eps=eps)
    return float(f), grad


def _laplace_poly_free_energy(
    *,
    u: np.ndarray,
    mus: np.ndarray,
    vars_: np.ndarray,
    prior_scores: np.ndarray,
    eta: float,
    cfg_poly: dict,
) -> dict[str, object]:
    k = mus.shape[0]
    if k < 2:
        raise ValueError("poly requires >=2 targets")

    eps = float(cfg_poly["epsilon"])
    alpha0 = float(cfg_poly["alpha0"])
    kappa = float(cfg_poly["kappa"])
    lr = float(cfg_poly["lr"])
    max_iters = int(cfg_poly["max_iters"])
    hess_eps = float(cfg_poly["hess_eps"])
    min_eig = float(cfg_poly["min_eig"])
    damping = float(cfg_poly["damping"])
    min_second_weight = float(cfg_poly.get("min_second_weight", 0.0))

    pi = np.asarray(prior_scores, dtype=float)
    pi = np.maximum(pi, 1e-6)
    pi = pi / pi.sum()
    alpha = alpha0 * np.power(pi, kappa)
    alpha = np.maximum(alpha, 1e-4)

    w = np.full(k, 1.0 / k, dtype=float)
    trace = []
    for _ in range(max_iters):
        f, g = _poly_energy_grad(u=u, mus=mus, vars_=vars_, w=w, alpha=alpha, eta=eta, eps=eps)
        w = _project_simplex_with_floor(w - lr * g, eps=eps)
        w = _enforce_poly_support(w, min_second_weight=min_second_weight)
        trace.append(float(f))

    # Tangent-space basis R (k x (k-1)).
    one = np.ones((k, 1), dtype=float)
    q, _ = np.linalg.qr(np.eye(k) - (one @ one.T) / float(k))
    R = q[:, : k - 1]

    # Finite-difference Hessian in tangent space.
    H_cols = []
    for i in range(k - 1):
        d = R[:, i]
        wp = _project_simplex_with_floor(w + hess_eps * d, eps=eps)
        wm = _project_simplex_with_floor(w - hess_eps * d, eps=eps)
        _, gp = _poly_energy_grad(u=u, mus=mus, vars_=vars_, w=wp, alpha=alpha, eta=eta, eps=eps)
        _, gm = _poly_energy_grad(u=u, mus=mus, vars_=vars_, w=wm, alpha=alpha, eta=eta, eps=eps)
        H_col = R.T @ ((gp - gm) / (2.0 * hess_eps))
        H_cols.append(H_col)
    H_t = np.column_stack(H_cols)
    H_t = 0.5 * (H_t + H_t.T)

    eigvals = np.linalg.eigvalsh(H_t)
    degenerate = bool(np.min(eigvals) < min_eig)
    if degenerate:
        H_t = H_t + np.eye(k - 1) * damping
    sign, logdet = np.linalg.slogdet(H_t)
    if sign <= 0:
        H_t = H_t + np.eye(k - 1) * max(damping, 1e-2)
        sign, logdet = np.linalg.slogdet(H_t)
    if sign <= 0:
        logdet = float(np.log(1e-6))
        degenerate = True

    mu_map = np.sum(w[:, None] * mus, axis=0)
    var_map = np.maximum(np.sum((w[:, None] ** 2) * vars_, axis=0), 1e-6)
    e_map = _gaussian_energy(u, mu_map, var_map)
    log_prior = _dirichlet_logpdf(w, alpha=alpha, eps=eps)
    occam_term = 0.5 * float(logdet)
    free_energy = float(
        e_map
        - (1.0 / eta) * log_prior
        + ((k - 1) / (2.0 * eta)) * np.log(2.0 * np.pi)
        + (1.0 / eta) * occam_term
    )
    return {
        "free_energy": free_energy,
        "map_energy": float(e_map),
        "weights": w,
        "alpha": alpha,
        "log_prior": float(log_prior),
        "occam_term": float(occam_term),
        "effective_dim": int(k - 1),
        "degenerate": int(degenerate),
        "trace": trace,
    }


def _fit_null_prototypes(sig: pd.DataFrame, gcols: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
    x = sig[gcols].to_numpy(dtype=float)
    act = np.linalg.norm(x, axis=1)
    idx = np.argsort(act)[-max(k * 20, k + 5) :]
    sel = x[idx]
    rng = np.random.default_rng(20260219)
    centers = sel[rng.choice(len(sel), size=k, replace=False)].copy()
    for _ in range(25):
        d = ((sel[:, None, :] - centers[None, :, :]) ** 2).mean(axis=2)
        a = np.argmin(d, axis=1)
        for j in range(k):
            g = sel[a == j]
            if len(g) > 0:
                centers[j] = g.mean(axis=0)
    weights = np.array([(a == j).mean() if (a == j).any() else 1.0 / k for j in range(k)], dtype=float)
    weights = weights / weights.sum()
    return centers, weights


def _fit_null_model_from_split(
    sig: pd.DataFrame,
    truth: pd.DataFrame,
    split: dict[str, list[str]],
    gcols: list[str],
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the null model without reading calibration or test responses."""
    train_ids = set(split["train"])
    train_sig = sig[sig["instance_id"].isin(train_ids)].copy()
    if train_sig.empty:
        raise ValueError("The training partition contains no mechanism signatures.")
    train_null_ids = set(
        truth.loc[truth["instance_id"].isin(train_ids) & truth["is_null"].eq(1), "instance_id"].tolist()
    )
    train_null = train_sig[train_sig["instance_id"].isin(train_null_ids)].copy()
    if len(train_null) < k:
        raise ValueError(f"The training partition contains only {len(train_null)} null rows for k={k}.")
    centers, weights = _fit_null_prototypes(train_null, gcols, k)
    sigma_base = np.var(train_sig[gcols].to_numpy(dtype=float), axis=0) + 0.08
    return centers, weights, sigma_base


def _build_candidate_targets(prior_df: pd.DataFrame, drug_key: str, k_dti: int) -> list[str]:
    cand = prior_df[prior_df["drug_key"] == drug_key].sort_values("calibrated_prob", ascending=False)
    return cand["target_key"].head(k_dti).tolist()


def _validate_bundle(bundle: dict[str, Any]) -> None:
    required_metadata = {
        "schema_version",
        "gene_columns",
        "eta",
        "priors",
        "config",
        "input_hashes",
        "eta_calibration",
    }
    missing_metadata = required_metadata.difference(bundle)
    if missing_metadata:
        raise ValueError(f"Posterior bundle is missing metadata: {sorted(missing_metadata)}")
    if int(bundle["schema_version"]) != POSTERIOR_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported posterior bundle schema {bundle['schema_version']}; "
            f"expected {POSTERIOR_BUNDLE_SCHEMA_VERSION}."
        )
    missing_arrays = set(_BUNDLE_ARRAY_KEYS).difference(bundle)
    if missing_arrays:
        raise ValueError(f"Posterior bundle is missing arrays: {sorted(missing_arrays)}")

    gcols = [str(value) for value in bundle["gene_columns"]]
    n_genes = len(gcols)
    null_centers = np.asarray(bundle["null_centers"])
    null_weights = np.asarray(bundle["null_weights"])
    sigma_base = np.asarray(bundle["sigma_base"])
    B = np.asarray(bundle["B"])
    b_lof = np.asarray(bundle["b_lof"])
    b_gof = np.asarray(bundle["b_gof"])
    beta_lof = np.asarray(bundle["beta_lof"])
    beta_gof = np.asarray(bundle["beta_gof"])

    expected_shapes = {
        "null_centers": (null_centers.shape[0], n_genes),
        "null_weights": (null_centers.shape[0],),
        "sigma_base": (n_genes,),
        "B": (n_genes, n_genes),
        "b_lof": (n_genes,),
        "b_gof": (n_genes,),
        "beta_lof": (1,),
        "beta_gof": (1,),
    }
    arrays = {
        "null_centers": null_centers,
        "null_weights": null_weights,
        "sigma_base": sigma_base,
        "B": B,
        "b_lof": b_lof,
        "b_gof": b_gof,
        "beta_lof": beta_lof,
        "beta_gof": beta_gof,
    }
    if n_genes == 0 or null_centers.ndim != 2 or null_centers.shape[0] == 0:
        raise ValueError("Posterior bundle has no gene dimensions or null prototypes.")
    for name, array in arrays.items():
        if array.shape != expected_shapes[name]:
            raise ValueError(
                f"Posterior bundle array {name} has shape {array.shape}; "
                f"expected {expected_shapes[name]}."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Posterior bundle array {name} contains non-finite values.")

    if np.any(null_weights < 0.0) or not np.isclose(float(null_weights.sum()), 1.0):
        raise ValueError("Posterior bundle null weights must be non-negative and sum to one.")
    if float(bundle["eta"]) <= 0.0:
        raise ValueError("Posterior bundle eta must be positive.")
    priors = bundle["priors"]
    prior_values = np.asarray(
        [priors["single_mass"], priors["poly_prior"], priors["null_prior"]],
        dtype=float,
    )
    if np.any(prior_values <= 0.0) or not np.isclose(float(prior_values.sum()), 1.0):
        raise ValueError("Posterior bundle priors must be positive and sum to one.")


def _bundle_priors(bundle: dict[str, Any]) -> tuple[float, float, float]:
    priors = bundle["priors"]
    return (
        float(priors["single_mass"]),
        float(priors["poly_prior"]),
        float(priors["null_prior"]),
    )


def fit_posterior(repo_root: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit all posterior parameters. Mechanism truth is confined to this fit boundary."""
    input_hashes = _hash_inputs(repo_root)
    cfg = load_yaml(repo_root / _FIT_INPUT_PATHS["config"])
    sig = pd.read_parquet(repo_root / _FIT_INPUT_PATHS["signatures"])
    truth = pd.read_parquet(repo_root / _FIT_INPUT_PATHS["mechanism_truth"])
    prior = pd.read_parquet(repo_root / _FIT_INPUT_PATHS["prior"])
    dof = pd.read_parquet(repo_root / _FIT_INPUT_PATHS["do_fused"])
    with np.load(repo_root / _FIT_INPUT_PATHS["align"], allow_pickle=False) as align:
        B = np.asarray(align["B"], dtype=float).copy()
        b_lof = np.asarray(align["b_lof"], dtype=float).copy()
        b_gof = np.asarray(align["b_gof"], dtype=float).copy()
        beta_lof = np.asarray([float(align["beta_lof"][0])], dtype=float)
        beta_gof = np.asarray([float(align["beta_gof"][0])], dtype=float)

    gcols = sorted(
        [column for column in sig.columns if column.startswith("G") and column[1:].isdigit()],
        key=lambda value: int(value[1:]),
    )
    do_index = dof.set_index(
        ["target_key", "context_key", "pert_time", "pert_dose", "mode"]
    ).sort_index()
    split = load_json(repo_root / _FIT_INPUT_PATHS["split"])
    null_centers, null_weights, sigma_base = _fit_null_model_from_split(
        sig,
        truth,
        split,
        gcols,
        int(cfg["null_num_prototypes"]),
    )

    null_prior = 0.1
    poly_prior = 0.1
    single_mass = 1.0 - null_prior - poly_prior
    priors = (single_mass, poly_prior, null_prior)

    cal_ids = set(split["cal"])
    cal_df = sig[sig["instance_id"].isin(cal_ids)].merge(
        truth,
        on="instance_id",
        how="inner",
    )
    eta_rows: list[dict[str, float | int]] = []
    for eta in [float(value) for value in cfg["eta_grid"]]:
        nll = 0.0
        n = 0
        for _, row in cal_df.iterrows():
            inference = _infer_one(
                row=row,
                gcols=gcols,
                prior_df=prior,
                do_index=do_index,
                B=B,
                b_lof=b_lof,
                b_gof=b_gof,
                beta_lof=float(beta_lof[0]),
                beta_gof=float(beta_gof[0]),
                sigma_base=sigma_base,
                null_centers=null_centers,
                null_w=null_weights,
                cfg=cfg,
                eta=eta,
                priors=priors,
                return_details=False,
            )
            probability, _ = _score_posterior_items_against_truth(
                inference["posterior_items"],
                row,
            )
            nll += -np.log(max(probability, 1e-8))
            n += 1
        eta_rows.append(
            {
                "eta": eta,
                "cal_nll": float(nll / max(n, 1)),
                "n_cal": int(n),
            }
        )
    eta_calibration = pd.DataFrame(eta_rows)
    eta_best = float(
        eta_calibration.sort_values(["cal_nll", "eta"], ascending=[True, True]).iloc[0]["eta"]
    )

    bundle: dict[str, Any] = {
        "schema_version": POSTERIOR_BUNDLE_SCHEMA_VERSION,
        "gene_columns": gcols,
        "eta": eta_best,
        "priors": {
            "single_mass": single_mass,
            "poly_prior": poly_prior,
            "null_prior": null_prior,
        },
        "config": cfg,
        "input_hashes": input_hashes,
        "eta_calibration": eta_rows,
        "null_centers": np.asarray(null_centers, dtype=float),
        "null_weights": np.asarray(null_weights, dtype=float),
        "sigma_base": np.asarray(sigma_base, dtype=float),
        "B": B,
        "b_lof": b_lof,
        "b_gof": b_gof,
        "beta_lof": beta_lof,
        "beta_gof": beta_gof,
    }
    _validate_bundle(bundle)
    return bundle, eta_calibration


def save_posterior_bundle(bundle: dict[str, Any], bundle_dir: Path) -> Path:
    """Serialize predictive arrays to NPZ and provenance/configuration to JSON."""
    _validate_bundle(bundle)
    bundle_dir = ensure_dir(bundle_dir)
    arrays_path = bundle_dir / POSTERIOR_BUNDLE_ARRAYS
    metadata_path = bundle_dir / POSTERIOR_BUNDLE_METADATA
    arrays = {
        name: np.asarray(bundle[name])
        for name in _BUNDLE_ARRAY_KEYS
    }
    np.savez_compressed(arrays_path, **arrays)
    arrays_sha256 = _sha256_file(arrays_path)
    metadata = {
        "schema_version": int(bundle["schema_version"]),
        "format": "vccv.posterior_bundle",
        "arrays_file": arrays_path.name,
        "arrays_sha256": arrays_sha256,
        "gene_columns": list(bundle["gene_columns"]),
        "eta": float(bundle["eta"]),
        "priors": {
            key: float(value)
            for key, value in bundle["priors"].items()
        },
        "config": bundle["config"],
        "input_hashes": bundle["input_hashes"],
        "eta_calibration": bundle["eta_calibration"],
        "array_shapes": {
            name: list(array.shape)
            for name, array in arrays.items()
        },
        "array_dtypes": {
            name: str(array.dtype)
            for name, array in arrays.items()
        },
    }
    with metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            metadata,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    return metadata_path


def load_posterior_bundle(bundle_path: Path) -> dict[str, Any]:
    """Load a JSON+NPZ posterior bundle without pickle or repository inputs."""
    metadata_path = (
        bundle_path / POSTERIOR_BUNDLE_METADATA
        if bundle_path.is_dir()
        else bundle_path
    )
    metadata = load_json(metadata_path)
    if metadata.get("format") != "vccv.posterior_bundle":
        raise ValueError(f"Not a VCCV posterior bundle: {metadata_path}")
    arrays_name = str(metadata["arrays_file"])
    if Path(arrays_name).name != arrays_name:
        raise ValueError("Posterior bundle arrays_file must be a local filename.")
    arrays_path = metadata_path.parent / arrays_name
    actual_arrays_hash = _sha256_file(arrays_path)
    if actual_arrays_hash != str(metadata["arrays_sha256"]):
        raise ValueError(
            f"Posterior bundle NPZ hash mismatch: expected {metadata['arrays_sha256']}, "
            f"found {actual_arrays_hash}."
        )
    with np.load(arrays_path, allow_pickle=False) as archive:
        archive_keys = set(archive.files)
        expected_keys = set(_BUNDLE_ARRAY_KEYS)
        if archive_keys != expected_keys:
            raise ValueError(
                f"Posterior bundle NPZ keys are {sorted(archive_keys)}; "
                f"expected {sorted(expected_keys)}."
            )
        arrays = {
            name: np.asarray(archive[name]).copy()
            for name in _BUNDLE_ARRAY_KEYS
        }
    for name, array in arrays.items():
        if list(array.shape) != list(metadata["array_shapes"][name]):
            raise ValueError(f"Posterior bundle shape metadata mismatch for {name}.")
        if str(array.dtype) != str(metadata["array_dtypes"][name]):
            raise ValueError(f"Posterior bundle dtype metadata mismatch for {name}.")

    bundle: dict[str, Any] = {
        "schema_version": int(metadata["schema_version"]),
        "gene_columns": list(metadata["gene_columns"]),
        "eta": float(metadata["eta"]),
        "priors": dict(metadata["priors"]),
        "config": metadata["config"],
        "input_hashes": metadata["input_hashes"],
        "eta_calibration": metadata["eta_calibration"],
        **arrays,
    }
    _validate_bundle(bundle)
    return bundle


def _assert_bundle_roundtrip_consistency(
    fitted: dict[str, Any],
    loaded: dict[str, Any],
) -> None:
    for name in _BUNDLE_ARRAY_KEYS:
        if not np.array_equal(np.asarray(fitted[name]), np.asarray(loaded[name])):
            raise RuntimeError(f"Posterior bundle round-trip changed array {name}.")
    for name in (
        "schema_version",
        "gene_columns",
        "eta",
        "priors",
        "config",
        "input_hashes",
        "eta_calibration",
    ):
        fitted_json = json.dumps(fitted[name], sort_keys=True, allow_nan=False)
        loaded_json = json.dumps(loaded[name], sort_keys=True, allow_nan=False)
        if fitted_json != loaded_json:
            raise RuntimeError(f"Posterior bundle round-trip changed metadata {name}.")


def _load_prediction_inputs(
    repo_root: Path,
    bundle: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load only truth-free prediction inputs and verify their fitted hashes."""
    _validate_prediction_input_hashes(repo_root, bundle)
    hashes = bundle["input_hashes"]
    sig = pd.read_parquet(repo_root / str(hashes["signatures"]["path"]))
    prior = pd.read_parquet(repo_root / str(hashes["prior"]["path"]))
    dof = pd.read_parquet(repo_root / str(hashes["do_fused"]["path"]))
    missing_genes = set(bundle["gene_columns"]).difference(sig.columns)
    if missing_genes:
        raise ValueError(f"Prediction signatures are missing genes: {sorted(missing_genes)}")
    do_index = dof.set_index(
        ["target_key", "context_key", "pert_time", "pert_dose", "mode"]
    ).sort_index()
    return sig, prior, do_index


def _infer_with_bundle(
    row: pd.Series,
    prior: pd.DataFrame,
    do_index: pd.DataFrame,
    bundle: dict[str, Any],
    *,
    return_details: bool,
) -> dict[str, Any]:
    return _infer_one(
        row=row,
        gcols=list(bundle["gene_columns"]),
        prior_df=prior,
        do_index=do_index,
        B=np.asarray(bundle["B"]),
        b_lof=np.asarray(bundle["b_lof"]),
        b_gof=np.asarray(bundle["b_gof"]),
        beta_lof=float(np.asarray(bundle["beta_lof"])[0]),
        beta_gof=float(np.asarray(bundle["beta_gof"])[0]),
        sigma_base=np.asarray(bundle["sigma_base"]),
        null_centers=np.asarray(bundle["null_centers"]),
        null_w=np.asarray(bundle["null_weights"]),
        cfg=bundle["config"],
        eta=float(bundle["eta"]),
        priors=_bundle_priors(bundle),
        return_details=return_details,
    )


def _canonical_prediction(prediction: dict[str, Any]) -> str:
    return json.dumps(
        prediction,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _assert_prediction_roundtrip_consistency(
    repo_root: Path,
    fitted: dict[str, Any],
    loaded: dict[str, Any],
) -> None:
    sig, prior, do_index = _load_prediction_inputs(repo_root, loaded)
    if sig.empty:
        raise ValueError("Prediction signatures are empty.")
    row = sig.iloc[0]
    before = _infer_with_bundle(row, prior, do_index, fitted, return_details=True)
    after = _infer_with_bundle(row, prior, do_index, loaded, return_details=True)
    if _canonical_prediction(before) != _canonical_prediction(after):
        raise RuntimeError("Posterior bundle round-trip changed a truth-free prediction.")


def predict_posterior(
    repo_root: Path,
    bundle: dict[str, Any],
    *,
    write_outputs: bool = True,
) -> list[dict[str, Any]]:
    """Predict from a loaded bundle. This function never opens mechanism truth."""
    _validate_bundle(bundle)
    sig, prior, do_index = _load_prediction_inputs(repo_root, bundle)
    out_json_dir = (
        ensure_dir(repo_root / "results/predictions_json/mechanism_instances")
        if write_outputs
        else None
    )
    predictions: list[dict[str, Any]] = []
    for _, row in sig.iterrows():
        prediction = _infer_with_bundle(
            row,
            prior,
            do_index,
            bundle,
            return_details=True,
        )
        predictions.append(prediction)
        if out_json_dir is not None:
            with (out_json_dir / f"{row['instance_id']}.json").open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                json.dump(
                    prediction["json"],
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                handle.write("\n")
    if write_outputs:
        pure_summary = pd.DataFrame([prediction["summary"] for prediction in predictions])
        pure_summary.to_parquet(
            repo_root / "results/predictions_json/mechanism_predictions.parquet",
            index=False,
        )
    return predictions


def _score_posterior_items_against_truth(
    posterior_items: list[dict[str, Any]],
    truth_row: pd.Series,
) -> tuple[float, int]:
    if int(truth_row["is_null"]) == 1:
        probability = sum(
            float(item["posterior"])
            for item in posterior_items
            if item["type"] == "null"
        )
        return max(probability, 1e-8), -1
    if int(truth_row["is_poly"]) == 1:
        probability = sum(
            float(item["posterior"])
            for item in posterior_items
            if item["type"] == "poly"
        )
        return max(probability, 1e-8), -1

    singles = sorted(
        [
            (
                str(item["target_key"]),
                str(item["mode"]),
                float(item["posterior"]),
            )
            for item in posterior_items
            if item["type"] == "single"
        ],
        key=lambda value: -value[2],
    )
    true_target = str(truth_row["true_target_key"])
    true_mode = str(truth_row["true_mode"])
    for rank, (target, mode, probability) in enumerate(singles, start=1):
        if target == true_target and mode == true_mode:
            return max(probability, 1e-8), rank
    return 1e-8, -1


def evaluate_posterior_predictions(
    predictions: list[dict[str, Any]],
    truth: pd.DataFrame,
) -> pd.DataFrame:
    """Attach truth-derived evaluation columns after truth-free prediction."""
    if truth["instance_id"].duplicated().any():
        raise ValueError("mechanism_truth.parquet contains duplicate instance_id values.")
    truth_index = truth.set_index("instance_id", drop=False)
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        instance_id = prediction["summary"]["instance_id"]
        if instance_id not in truth_index.index:
            continue
        truth_row = truth_index.loc[instance_id]
        probability, true_rank = _score_posterior_items_against_truth(
            prediction["json"]["posterior_distribution"],
            truth_row,
        )
        rows.append(
            {
                **prediction["summary"],
                "prob_true": probability,
                "true_rank_single": int(true_rank),
                "is_null": int(truth_row["is_null"]),
                "is_poly": int(truth_row["is_poly"]),
                "true_target_key": truth_row["true_target_key"],
                "true_mode": truth_row["true_mode"],
            }
        )
    return pd.DataFrame(rows)


def run_inference(repo_root: Path) -> None:
    """Compatibility entry point: fit, serialize, reload, predict, then evaluate."""
    fitted, eta_calibration = fit_posterior(repo_root)
    bundle_path = save_posterior_bundle(
        fitted,
        repo_root / POSTERIOR_BUNDLE_DIR,
    )
    loaded = load_posterior_bundle(bundle_path)
    _assert_bundle_roundtrip_consistency(fitted, loaded)
    _assert_prediction_roundtrip_consistency(repo_root, fitted, loaded)

    metrics_dir = ensure_dir(repo_root / "results/metrics_tables")
    eta_calibration[["eta", "cal_nll"]].to_csv(
        metrics_dir / "inference_calibration.csv",
        index=False,
    )
    predictions = predict_posterior(repo_root, loaded, write_outputs=True)

    # Truth is deliberately loaded only after prediction, at the evaluation boundary.
    truth_entry = loaded["input_hashes"]["mechanism_truth"]
    truth_path = repo_root / str(truth_entry["path"])
    if (
        int(truth_path.stat().st_size) != int(truth_entry["size_bytes"])
        or _sha256_file(truth_path) != str(truth_entry["sha256"])
    ):
        raise ValueError("Posterior evaluation truth input drifted after fitting.")
    truth = pd.read_parquet(truth_path)
    evaluated = evaluate_posterior_predictions(predictions, truth)
    evaluated.to_parquet(
        repo_root / "results/predictions_json/mechanism_summary.parquet",
        index=False,
    )


def _infer_one(
    row: pd.Series,
    gcols: list[str],
    prior_df: pd.DataFrame,
    do_index: pd.DataFrame,
    B: np.ndarray,
    b_lof: np.ndarray,
    b_gof: np.ndarray,
    beta_lof: float,
    beta_gof: float,
    sigma_base: np.ndarray,
    null_centers: np.ndarray,
    null_w: np.ndarray,
    cfg: dict,
    eta: float,
    priors: tuple[float, float, float],
    return_details: bool,
    candidate_targets: list[str] | None = None,
):
    single_mass, poly_prior, null_prior = priors
    u = row[gcols].to_numpy(dtype=float)
    drug = row["drug_key"]
    cand_targets = (
        list(dict.fromkeys(candidate_targets))
        if candidate_targets is not None
        else _build_candidate_targets(prior_df, drug, int(cfg["kcand_dti"]))
    )
    if not cand_targets:
        cand_targets = prior_df["target_key"].head(5).tolist()
    hyps = []

    p_map = prior_df[prior_df["drug_key"] == drug].set_index("target_key")["calibrated_prob"].to_dict()
    single_priors_raw = np.array([max(float(p_map.get(t, 1e-4)), 1e-4) for t in cand_targets], dtype=float)
    single_priors_raw = single_priors_raw / single_priors_raw.sum()
    single_priors_raw = single_priors_raw * single_mass

    # Single-target hypotheses.
    for i, t in enumerate(cand_targets):
        for mode in ("LoF", "GoF"):
            key = (t, row["context_key"], float(row["pert_time"]), float(row["pert_dose"]), mode)
            try:
                do_row = do_index.loc[key]
            except KeyError:
                continue
            if isinstance(do_row, pd.DataFrame):
                do_row = do_row.iloc[0]
            mu = do_row[gcols].to_numpy(dtype=float)
            var = do_row[[f"V{k}" for k in range(len(gcols))]].to_numpy(dtype=float)
            if mode == "LoF":
                mu_t = beta_lof * (B @ mu + b_lof)
                # Diagonal of B diag(var) B^T, evaluated without materializing
                # the dense covariance. This preserves the deployed marginal-
                # variance approximation while reducing the operation to a
                # matrix-vector product.
                var_t = (beta_lof**2) * ((B**2) @ var)
            else:
                mu_t = beta_gof * (B @ mu + b_gof)
                var_t = (beta_gof**2) * ((B**2) @ var)
            var_tot = np.maximum(var_t + sigma_base, 1e-6)
            e = _gaussian_energy(u, mu_t, var_tot)
            hyps.append(
                {
                    "type": "single",
                    "target_key": t,
                    "mode": mode,
                    "prior": float(single_priors_raw[i] * 0.5),
                    "energy": float(e),
                    "mu": mu_t,
                    "var": var_tot,
                }
            )

    # Poly hypotheses with Laplace-marginalized free energy.
    poly_entries = []
    k_poly = int(cfg["poly"]["k_poly"])
    top = cand_targets[: min(k_poly, len(cand_targets))]
    for mode in ("LoF", "GoF"):
        mus = []
        vars_ = []
        valid_t = []
        prior_scores = []
        for t in top:
            key = (t, row["context_key"], float(row["pert_time"]), float(row["pert_dose"]), mode)
            try:
                rr = do_index.loc[key]
            except KeyError:
                continue
            if isinstance(rr, pd.DataFrame):
                rr = rr.iloc[0]
            mu = rr[gcols].to_numpy(dtype=float)
            var = rr[[f"V{k}" for k in range(len(gcols))]].to_numpy(dtype=float)
            if mode == "LoF":
                mu = beta_lof * (B @ mu + b_lof)
                var = (beta_lof**2) * ((B**2) @ var)
            else:
                mu = beta_gof * (B @ mu + b_gof)
                var = (beta_gof**2) * ((B**2) @ var)
            mus.append(mu)
            vars_.append(np.maximum(var + sigma_base, 1e-6))
            valid_t.append(t)
            prior_scores.append(max(float(p_map.get(t, 1e-6)), 1e-6))
        if len(mus) < 2:
            continue
        active_set_size = int(cfg["poly"].get("active_set_size", len(mus)))
        if 2 <= active_set_size < len(mus):
            individual_energy = np.asarray(
                [_gaussian_energy(u, mu, var) for mu, var in zip(mus, vars_)],
                dtype=float,
            )
            keep = np.argsort(individual_energy)[:active_set_size]
            mus = [mus[index] for index in keep]
            vars_ = [vars_[index] for index in keep]
            valid_t = [valid_t[index] for index in keep]
            prior_scores = [prior_scores[index] for index in keep]
        poly_fit = _laplace_poly_free_energy(
            u=u,
            mus=np.stack(mus, axis=0),
            vars_=np.stack(vars_, axis=0),
            prior_scores=np.asarray(prior_scores, dtype=float),
            eta=eta,
            cfg_poly=cfg["poly"],
        )
        poly_entry = {
            "type": "poly",
            "target_key": "poly",
            "mode": mode,
            "prior": float(poly_prior * 0.5),
            "energy": float(poly_fit["free_energy"]),
            "poly_meta": {
                "targets": valid_t,
                "weights": poly_fit["weights"].tolist(),
                "alpha": poly_fit["alpha"].tolist(),
                "map_energy": float(poly_fit["map_energy"]),
                "log_prior": float(poly_fit["log_prior"]),
                "occam_term": float(poly_fit["occam_term"]),
                "effective_dim": int(poly_fit["effective_dim"]),
                "degenerate": int(poly_fit["degenerate"]),
            },
        }
        poly_entries.append(poly_entry)
        hyps.append(poly_entry)

    # Null hypothesis.
    null_terms = []
    for c, w in zip(null_centers, null_w):
        e = _gaussian_energy(u, c, sigma_base)
        null_terms.append(np.log(max(w, 1e-6)) - e)
    null_logp = np.log(np.sum(np.exp(null_terms)))
    hyps.append(
        {
            "type": "null",
            "target_key": "NULL",
            "mode": "NA",
            "prior": float(null_prior),
            "energy": float(-null_logp),
        }
    )

    scores = np.array([np.log(max(h["prior"], 1e-8)) - eta * h["energy"] for h in hyps], dtype=float)
    probs = _softmax(scores)
    top_idx = int(np.argmax(probs))
    ranked = np.argsort(-probs)
    top1 = hyps[top_idx]
    top2 = hyps[ranked[1]] if len(ranked) > 1 else hyps[top_idx]
    ger = float(scores[top_idx] - scores[ranked[1]]) if len(ranked) > 1 else 0.0

    items = []
    for h, p, s in zip(hyps, probs, scores):
        item = {
            "type": h["type"],
            "target_key": h["target_key"],
            "mode": h["mode"],
            "prior": float(h["prior"]),
            "energy": float(h["energy"]),
            "score": float(s),
            "posterior": float(p),
        }
        if h["type"] == "poly":
            item["poly_meta"] = h["poly_meta"]
        items.append(item)

    if not return_details:
        return {"posterior_items": items}

    best_poly = None
    if poly_entries:
        poly_post = [(h, float(p)) for h, p in zip(hyps, probs) if h["type"] == "poly"]
        poly_post = sorted(poly_post, key=lambda x: -x[1])
        best_poly = poly_post[0][0]["poly_meta"] | {"mode": poly_post[0][0]["mode"]}

    decision_cfg = cfg.get("decision", {})
    decision = assign_decision_state(
        items,
        decision_gap_threshold=float(decision_cfg.get("gap_threshold", 0.42)),
        deep_near_tie_threshold=float(decision_cfg.get("deep_near_tie_threshold", 0.05)),
        strong_null_threshold=float(decision_cfg.get("strong_null_threshold", 0.95)),
    )

    js = {
        "instance_id": row["instance_id"],
        "input_summary": {
            "drug_key": row["drug_key"],
            "context_key": row["context_key"],
            "pert_time": float(row["pert_time"]),
            "pert_dose": float(row["pert_dose"]),
            "candidate_count": len(cand_targets),
        },
        "prior_breakdown": {
            "single_mass": single_mass,
            "poly_prior": poly_prior,
            "null_prior": null_prior,
        },
        "posterior_distribution": items,
        "decision": decision,
        "top_hypothesis": {
            "type": top1["type"],
            "target_key": top1["target_key"],
            "mode": top1["mode"],
            "posterior": float(probs[top_idx]),
            "ger_vs_second": ger,
        },
        "top2_hypothesis": {
            "type": top2["type"],
            "target_key": top2["target_key"],
            "mode": top2["mode"],
            "posterior": float(probs[ranked[1]]) if len(ranked) > 1 else float(probs[top_idx]),
        },
        "poly_details": best_poly if best_poly is not None else {},
        "falsifiable_readouts": {
            "top_genes_up": sorted(
                [{"gene": g, "effect": float(v)} for g, v in zip(gcols, u)],
                key=lambda x: -x["effect"],
            )[:10],
            "top_genes_down": sorted(
                [{"gene": g, "effect": float(v)} for g, v in zip(gcols, u)],
                key=lambda x: x["effect"],
            )[:10],
        },
    }
    summary = {
        "instance_id": row["instance_id"],
        "top_type": top1["type"],
        "top_target_key": top1["target_key"],
        "top_mode": top1["mode"],
        "top_prob": float(probs[top_idx]),
        "decision_state": decision["state"],
        "panel_policy": decision["panel_policy"],
        "posterior_gap": decision["posterior_gap"],
        "ger": ger,
    }
    return {"json": js, "summary": summary, "posterior_items": items}
