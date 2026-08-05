from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.utils.io import load_yaml
from src.utils.seed import set_global_seed


KEY_COLS = [
    "target_key",
    "context_key",
    "pert_time",
    "pert_dose",
    "platform",
    "batch",
    "mode",
]
CHECKPOINT_VERSION = 1


class DoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, gcols: list[str], vcols: list[str], maps: dict):
        self.target = torch.tensor(df["target_idx"].to_numpy(), dtype=torch.long)
        self.context = torch.tensor(df["context_idx"].to_numpy(), dtype=torch.long)
        self.mode = torch.tensor(df["mode_idx"].to_numpy(), dtype=torch.long)
        self.time = torch.tensor(df["pert_time"].to_numpy(), dtype=torch.float32).unsqueeze(1)
        self.dose = torch.tensor(df["pert_dose"].to_numpy(), dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(df[gcols].to_numpy(), dtype=torch.float32)
        self.v = torch.tensor(df[vcols].to_numpy(), dtype=torch.float32)
        self.maps = maps

    def __len__(self):
        return len(self.target)

    def __getitem__(self, i):
        return (
            self.target[i],
            self.context[i],
            self.mode[i],
            self.time[i],
            self.dose[i],
            self.y[i],
            self.v[i],
        )


class VirtualDoNet(nn.Module):
    def __init__(self, n_targets: int, n_contexts: int, embed_dim: int, hidden_dim: int, n_genes: int):
        super().__init__()
        self.t_emb = nn.Embedding(n_targets, embed_dim)
        self.c_emb = nn.Embedding(n_contexts, embed_dim)
        self.m_emb = nn.Embedding(2, embed_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2 + embed_dim // 2 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head_mu = nn.Linear(hidden_dim, n_genes)
        self.head_lv = nn.Linear(hidden_dim, n_genes)

    def forward(self, t_idx, c_idx, m_idx, time, dose):
        x = torch.cat(
            [self.t_emb(t_idx), self.c_emb(c_idx), self.m_emb(m_idx), time, dose],
            dim=1,
        )
        h = self.net(x)
        mu = self.head_mu(h)
        lv = self.head_lv(h)
        return mu, lv


@dataclass
class VirtualDoBundle:
    """A newly constructed inference model plus its non-tensor schema."""

    model: VirtualDoNet
    maps: dict[str, dict[str, int]]
    gcols: list[str]
    vcols: list[str]
    calibration_scale: float
    sigma_min: float
    architecture: dict[str, int]
    state_sha256: str


def _prep_data(mu_df: pd.DataFrame, var_df: pd.DataFrame):
    df = mu_df.merge(var_df, on=KEY_COLS, how="inner", suffixes=("", "_var"))
    gcols = sorted(
        [c for c in mu_df.columns if c.startswith("G") and c[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )
    vcols = sorted(
        [c for c in var_df.columns if c.startswith("V") and c[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )
    if not gcols or len(gcols) != len(vcols):
        raise ValueError("ObservedDo mean/variance gene schemas are empty or misaligned.")
    t_map = {str(k): i for i, k in enumerate(sorted(df["target_key"].astype(str).unique().tolist()))}
    c_map = {str(k): i for i, k in enumerate(sorted(df["context_key"].astype(str).unique().tolist()))}
    m_map = {"LoF": 0, "GoF": 1}
    df["target_key"] = df["target_key"].astype(str)
    df["context_key"] = df["context_key"].astype(str)
    df["mode"] = df["mode"].astype(str)
    df["target_idx"] = df["target_key"].map(t_map).astype(int)
    df["context_idx"] = df["context_key"].map(c_map).astype(int)
    df["mode_idx"] = df["mode"].map(m_map)
    if df["mode_idx"].isna().any():
        bad = sorted(df.loc[df["mode_idx"].isna(), "mode"].unique().tolist())
        raise ValueError(f"Unsupported intervention modes: {bad}")
    df["mode_idx"] = df["mode_idx"].astype(int)
    maps = {"target": t_map, "context": c_map, "mode": m_map}
    return df, gcols, vcols, maps


def _nll_loss(y, mu, logvar, sigma_min):
    var = torch.clamp(torch.nn.functional.softplus(logvar), min=sigma_min * sigma_min)
    return 0.5 * torch.mean(((y - mu) ** 2) / var + torch.log(var))


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


def _serialise_map(mapping: dict[Any, int]) -> list[dict[str, Any]]:
    return [
        {"key": str(key), "index": int(index)}
        for key, index in sorted(mapping.items(), key=lambda item: int(item[1]))
    ]


def _deserialise_map(records: list[dict[str, Any]]) -> dict[str, int]:
    mapping = {str(item["key"]): int(item["index"]) for item in records}
    expected = list(range(len(mapping)))
    if sorted(mapping.values()) != expected:
        raise ValueError("VirtualDO checkpoint map indices are not contiguous.")
    return mapping


def _key_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame[KEY_COLS].drop_duplicates().itertuples(index=False):
        records.append(
            {
                "target_key": str(row.target_key),
                "context_key": str(row.context_key),
                "pert_time": float(row.pert_time),
                "pert_dose": float(row.pert_dose),
                "platform": str(row.platform),
                "batch": str(row.batch),
                "mode": str(row.mode),
            }
        )
    return records


def save_virtualdo_checkpoint(
    checkpoint_dir: Path,
    model: VirtualDoNet,
    *,
    maps: dict[str, dict[str, int]],
    gcols: list[str],
    vcols: list[str],
    architecture: dict[str, int],
    calibration_scale: float,
    calibration_nll: float,
    sigma_min: float,
    calibration_rows: pd.DataFrame,
) -> tuple[Path, Path]:
    """Save tensors as non-pickle NPZ and all inference schema as JSON."""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint_dir / "virtualdo_state.npz"
    schema_path = checkpoint_dir / "virtualdo_schema.json"

    arrays: dict[str, np.ndarray] = {}
    tensor_schema: list[dict[str, Any]] = []
    for index, (name, tensor) in enumerate(sorted(model.state_dict().items())):
        archive_key = f"tensor_{index:04d}"
        array = tensor.detach().cpu().numpy()
        arrays[archive_key] = array
        tensor_schema.append(
            {
                "name": name,
                "archive_key": archive_key,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        )

    tmp_state = state_path.with_suffix(".npz.tmp")
    with tmp_state.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp_state.replace(state_path)
    state_sha256 = _sha256_file(state_path)

    schema = {
        "format": "vccv.virtualdo.safe-npz",
        "version": CHECKPOINT_VERSION,
        "model_class": "VirtualDoNet",
        "architecture": {key: int(value) for key, value in architecture.items()},
        "maps": {name: _serialise_map(mapping) for name, mapping in maps.items()},
        "gene_order": list(gcols),
        "variance_order": list(vcols),
        "sigma_min": float(sigma_min),
        "calibration": {
            "scale": float(calibration_scale),
            "gaussian_nll": float(calibration_nll),
            "split_seed": 20260219,
            "fraction": 0.2,
            # Fusion consumes only these rows when selecting reliability
            # parameters, so neither the VirtualDO fit nor observed anchors
            # used for that fit can leak into the fusion holdout objective.
            "holdout_keys": _key_records(calibration_rows),
        },
        "state": {
            "file": state_path.name,
            "sha256": state_sha256,
            "tensors": tensor_schema,
        },
    }
    _atomic_json(schema_path, schema)
    return state_path, schema_path


def load_virtualdo_checkpoint(
    checkpoint_dir: Path,
    *,
    device: torch.device | str = "cpu",
) -> VirtualDoBundle:
    """Construct a fresh model instance from the safe NPZ/JSON bundle."""

    schema_path = checkpoint_dir / "virtualdo_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("format") != "vccv.virtualdo.safe-npz":
        raise ValueError("Unsupported VirtualDO checkpoint format.")
    if int(schema.get("version", -1)) != CHECKPOINT_VERSION:
        raise ValueError("Unsupported VirtualDO checkpoint version.")

    state_path = checkpoint_dir / str(schema["state"]["file"])
    state_sha256 = _sha256_file(state_path)
    if state_sha256 != str(schema["state"]["sha256"]):
        raise ValueError("VirtualDO NPZ checkpoint SHA-256 mismatch.")

    architecture = {key: int(value) for key, value in schema["architecture"].items()}
    model = VirtualDoNet(
        n_targets=architecture["n_targets"],
        n_contexts=architecture["n_contexts"],
        embed_dim=architecture["embed_dim"],
        hidden_dim=architecture["hidden_dim"],
        n_genes=architecture["n_genes"],
    )
    expected_state = model.state_dict()
    loaded_state: dict[str, torch.Tensor] = {}
    with np.load(state_path, allow_pickle=False) as archive:
        for item in schema["state"]["tensors"]:
            name = str(item["name"])
            archive_key = str(item["archive_key"])
            if name not in expected_state or archive_key not in archive.files:
                raise ValueError(f"VirtualDO state tensor missing: {name}")
            array = np.asarray(archive[archive_key])
            if list(array.shape) != list(item["shape"]):
                raise ValueError(f"VirtualDO tensor shape mismatch: {name}")
            if str(array.dtype) != str(item["dtype"]):
                raise ValueError(f"VirtualDO tensor dtype mismatch: {name}")
            loaded_state[name] = torch.as_tensor(
                array.copy(),
                dtype=expected_state[name].dtype,
            )
    if set(loaded_state) != set(expected_state):
        missing = sorted(set(expected_state) - set(loaded_state))
        raise ValueError(f"VirtualDO state is incomplete: {missing}")
    model.load_state_dict(loaded_state, strict=True)
    model.to(torch.device(device))
    model.eval()

    maps = {
        name: _deserialise_map(records)
        for name, records in schema["maps"].items()
    }
    gcols = [str(value) for value in schema["gene_order"]]
    vcols = [str(value) for value in schema["variance_order"]]
    if architecture["n_genes"] != len(gcols) or len(gcols) != len(vcols):
        raise ValueError("VirtualDO checkpoint gene schema is inconsistent.")
    return VirtualDoBundle(
        model=model,
        maps=maps,
        gcols=gcols,
        vcols=vcols,
        calibration_scale=float(schema["calibration"]["scale"]),
        sigma_min=float(schema["sigma_min"]),
        architecture=architecture,
        state_sha256=state_sha256,
    )


def _build_prediction_grid(
    observed: pd.DataFrame,
    sig_drug: pd.DataFrame,
    prior: pd.DataFrame,
    maps: dict[str, dict[str, int]],
) -> pd.DataFrame:
    """Build the inference grid and include the held-out observed anchors."""

    top_t = (
        prior.sort_values("calibrated_prob", ascending=False)
        .groupby("drug_key")
        .head(5)
    )
    combos = (
        sig_drug[
            ["drug_key", "context_key", "pert_time", "pert_dose", "platform", "batch"]
        ]
        .drop_duplicates()
        .merge(top_t[["drug_key", "target_key"]], on="drug_key", how="left")
    )
    mode_rows: list[pd.DataFrame] = []
    for mode in ("LoF", "GoF"):
        block = combos.copy()
        block["mode"] = mode
        mode_rows.append(block[KEY_COLS])

    # Calibration of fusion needs out-of-sample VirtualDO predictions for the
    # exact VirtualDO holdout keys. Including all observed keys here also makes
    # the serialized model independently useful beyond the current drug grid.
    pred_grid = pd.concat(
        [*mode_rows, observed[KEY_COLS]],
        ignore_index=True,
    )
    pred_grid = pred_grid.dropna(subset=KEY_COLS).drop_duplicates(KEY_COLS)
    pred_grid["target_key"] = pred_grid["target_key"].astype(str)
    pred_grid["context_key"] = pred_grid["context_key"].astype(str)
    pred_grid["mode"] = pred_grid["mode"].astype(str)
    pred_grid = pred_grid[
        pred_grid["target_key"].isin(maps["target"])
        & pred_grid["context_key"].isin(maps["context"])
        & pred_grid["mode"].isin(maps["mode"])
    ]
    if pred_grid.empty:
        raise ValueError("VirtualDO prediction grid contains no known target/context rows.")
    return pred_grid.sort_values(KEY_COLS, kind="mergesort").reset_index(drop=True)


def predict_virtualdo(
    bundle: VirtualDoBundle,
    pred_grid: pd.DataFrame,
    *,
    device: torch.device | str | None = None,
    batch_size: int = 4096,
) -> pd.DataFrame:
    """Predict means/variances using only a loaded inference bundle."""

    if device is None:
        device_obj = next(bundle.model.parameters()).device
    else:
        device_obj = torch.device(device)
        bundle.model.to(device_obj)
    bundle.model.eval()

    grid = pred_grid[KEY_COLS].copy().reset_index(drop=True)
    grid["target_key"] = grid["target_key"].astype(str)
    grid["context_key"] = grid["context_key"].astype(str)
    grid["mode"] = grid["mode"].astype(str)
    target_idx = grid["target_key"].map(bundle.maps["target"])
    context_idx = grid["context_key"].map(bundle.maps["context"])
    mode_idx = grid["mode"].map(bundle.maps["mode"])
    if target_idx.isna().any() or context_idx.isna().any() or mode_idx.isna().any():
        raise ValueError("Prediction grid contains values absent from the VirtualDO maps.")

    mu_parts: list[np.ndarray] = []
    var_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(grid), batch_size):
            stop = min(start + batch_size, len(grid))
            t = torch.tensor(
                target_idx.iloc[start:stop].to_numpy(),
                dtype=torch.long,
                device=device_obj,
            )
            c = torch.tensor(
                context_idx.iloc[start:stop].to_numpy(),
                dtype=torch.long,
                device=device_obj,
            )
            m = torch.tensor(
                mode_idx.iloc[start:stop].to_numpy(),
                dtype=torch.long,
                device=device_obj,
            )
            tm = torch.tensor(
                grid["pert_time"].iloc[start:stop].to_numpy(),
                dtype=torch.float32,
                device=device_obj,
            ).unsqueeze(1)
            ds = torch.tensor(
                grid["pert_dose"].iloc[start:stop].to_numpy(),
                dtype=torch.float32,
                device=device_obj,
            ).unsqueeze(1)
            mu, logvar = bundle.model(t, c, m, tm, ds)
            var = torch.clamp(
                torch.nn.functional.softplus(logvar) * bundle.calibration_scale,
                min=bundle.sigma_min * bundle.sigma_min,
            )
            mu_parts.append(mu.cpu().numpy())
            var_parts.append(var.cpu().numpy())

    mu_array = np.concatenate(mu_parts, axis=0)
    var_array = np.concatenate(var_parts, axis=0)
    return pd.concat(
        [
            grid,
            pd.DataFrame(mu_array, columns=bundle.gcols),
            pd.DataFrame(var_array, columns=bundle.vcols),
        ],
        axis=1,
    )


def _prediction_reload_deltas(
    in_memory: pd.DataFrame,
    reloaded: pd.DataFrame,
    gcols: list[str],
    vcols: list[str],
) -> tuple[float, float]:
    if not in_memory[KEY_COLS].equals(reloaded[KEY_COLS]):
        raise RuntimeError("VirtualDO reload changed prediction row identity/order.")
    mu_delta = float(
        np.max(
            np.abs(
                in_memory[gcols].to_numpy(dtype=float)
                - reloaded[gcols].to_numpy(dtype=float)
            )
        )
    )
    var_delta = float(
        np.max(
            np.abs(
                in_memory[vcols].to_numpy(dtype=float)
                - reloaded[vcols].to_numpy(dtype=float)
            )
        )
    )
    if not np.allclose(
        in_memory[gcols + vcols].to_numpy(dtype=float),
        reloaded[gcols + vcols].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-7,
    ):
        raise RuntimeError(
            "VirtualDO in-memory and fresh-reload predictions are inconsistent."
        )
    return mu_delta, var_delta


def _resolve_training_device(
    device_request: str | None,
    cuda_id: int,
) -> torch.device:
    requested = "auto" if device_request is None else str(device_request).lower()
    if requested == "auto":
        requested = f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"VirtualDO CUDA device is unavailable: {requested}")
        resolved_index = cuda_id if device.index is None else int(device.index)
        if resolved_index < 0 or resolved_index >= torch.cuda.device_count():
            raise ValueError(f"VirtualDO CUDA device is unavailable: {requested}")
        device = torch.device(f"cuda:{resolved_index}")
    return device


def train_virtualdo(
    repo_root: Path,
    device_request: str | None = None,
) -> None:
    """Fit, serialize, reload into a new model, and predict from that reload."""

    set_global_seed(20260219)
    cfg = load_yaml(repo_root / "configs/virtualdo.yaml")
    mu_path = repo_root / "data/processed/observeddo_mu.parquet"
    var_path = repo_root / "data/processed/observeddo_var_diag.parquet"
    mu_df = pd.read_parquet(mu_path)
    var_df = pd.read_parquet(var_path)
    df, gcols, vcols, maps = _prep_data(mu_df, var_df)
    if len(df) < 2:
        raise ValueError("VirtualDO requires at least two observed rows.")

    idx = np.arange(len(df))
    rng = np.random.default_rng(20260219)
    rng.shuffle(idx)
    n_cal = max(1, int(0.2 * len(df)))
    if n_cal >= len(df):
        n_cal = len(df) - 1
    cal_idx = idx[:n_cal]
    tr_idx = idx[n_cal:]
    tr = df.iloc[tr_idx].reset_index(drop=True)
    cal = df.iloc[cal_idx].reset_index(drop=True)

    train_ds = DoDataset(tr, gcols, vcols, maps)
    cal_ds = DoDataset(cal, gcols, vcols, maps)
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
    )

    cuda_id = int(cfg.get("cuda_id", 0))
    device = _resolve_training_device(device_request, cuda_id)
    architecture = {
        "n_targets": len(maps["target"]),
        "n_contexts": len(maps["context"]),
        "embed_dim": int(cfg["embedding_dim"]),
        "hidden_dim": int(cfg["hidden_dim"]),
        "n_genes": len(gcols),
    }
    model = VirtualDoNet(**architecture).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    sigma_min = float(cfg["sigma_min"])
    model.train()
    for _ in range(int(cfg["epochs"])):
        for t_idx, c_idx, m_idx, time, dose, y, _ in loader:
            t_idx = t_idx.to(device)
            c_idx = c_idx.to(device)
            m_idx = m_idx.to(device)
            time = time.to(device)
            dose = dose.to(device)
            y = y.to(device)
            mu, lv = model(t_idx, c_idx, m_idx, time, dose)
            loss = _nll_loss(y, mu, lv, sigma_min=sigma_min)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        t_idx, c_idx, m_idx, time, dose, y_cal, _ = [
            x.to(device)
            for x in (
                cal_ds.target,
                cal_ds.context,
                cal_ds.mode,
                cal_ds.time,
                cal_ds.dose,
                cal_ds.y,
                cal_ds.v,
            )
        ]
        mu_cal, lv_cal = model(t_idx, c_idx, m_idx, time, dose)
        base_var = torch.clamp(
            torch.nn.functional.softplus(lv_cal),
            min=sigma_min * sigma_min,
        )
        scales = [float(value) for value in cfg["calibration_scales"]]
        best_s = 1.0
        best_nll = float("inf")
        for scale in scales:
            candidate_var = torch.clamp(
                base_var * scale,
                min=sigma_min * sigma_min,
            )
            nll = torch.mean(
                0.5 * ((y_cal - mu_cal) ** 2) / candidate_var
                + 0.5 * torch.log(candidate_var)
            ).item()
            if nll < best_nll:
                best_nll = float(nll)
                best_s = float(scale)

    sig_path = repo_root / "data/processed/signatures_drug.parquet"
    prior_path = repo_root / "results/predictions_json/dti_prior_scores.parquet"
    sig_drug = pd.read_parquet(sig_path)
    prior = pd.read_parquet(prior_path)
    pred_grid = _build_prediction_grid(df, sig_drug, prior, maps)

    in_memory_bundle = VirtualDoBundle(
        model=model,
        maps=maps,
        gcols=gcols,
        vcols=vcols,
        calibration_scale=best_s,
        sigma_min=sigma_min,
        architecture=architecture,
        state_sha256="in-memory",
    )
    in_memory_predictions = predict_virtualdo(
        in_memory_bundle,
        pred_grid,
        device=device,
    )

    checkpoint_dir = repo_root / "results/checkpoints/virtualdo"
    state_path, schema_path = save_virtualdo_checkpoint(
        checkpoint_dir,
        model,
        maps=maps,
        gcols=gcols,
        vcols=vcols,
        architecture=architecture,
        calibration_scale=best_s,
        calibration_nll=best_nll,
        sigma_min=sigma_min,
        calibration_rows=cal,
    )

    # This call constructs a separate VirtualDoNet and loads only safe NPZ/JSON
    # artifacts. The published predictions below come from this new instance.
    reloaded_bundle = load_virtualdo_checkpoint(checkpoint_dir, device=device)
    if reloaded_bundle.model is model:
        raise RuntimeError("VirtualDO reload did not create a new model instance.")
    reloaded_predictions = predict_virtualdo(
        reloaded_bundle,
        pred_grid,
        device=device,
    )
    max_mu_delta, max_var_delta = _prediction_reload_deltas(
        in_memory_predictions,
        reloaded_predictions,
        gcols,
        vcols,
    )

    output_path = repo_root / "data/processed/virtualdo_predictions.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reloaded_predictions.to_parquet(output_path, index=False)

    metrics_path = repo_root / "results/metrics_tables/virtualdo_calibration.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"metric": "cal_nll", "value": best_nll},
            {"metric": "cal_scale", "value": best_s},
            {"metric": "reload_max_abs_mu", "value": max_mu_delta},
            {"metric": "reload_max_abs_var", "value": max_var_delta},
            {"metric": "reload_consistency_pass", "value": 1.0},
        ]
    ).to_csv(metrics_path, index=False)

    lineage = {
        "stage": "virtualdo",
        "device": str(device),
        "fit_rows": int(len(tr)),
        "calibration_rows": int(len(cal)),
        "prediction_rows": int(len(reloaded_predictions)),
        "inputs": {
            str(mu_path.relative_to(repo_root)): _sha256_file(mu_path),
            str(var_path.relative_to(repo_root)): _sha256_file(var_path),
            str(sig_path.relative_to(repo_root)): _sha256_file(sig_path),
            str(prior_path.relative_to(repo_root)): _sha256_file(prior_path),
        },
        "checkpoint": {
            str(state_path.relative_to(repo_root)): _sha256_file(state_path),
            str(schema_path.relative_to(repo_root)): _sha256_file(schema_path),
        },
        "output": {
            str(output_path.relative_to(repo_root)): _sha256_file(output_path),
        },
        "reload_verification": {
            "max_abs_mu": max_mu_delta,
            "max_abs_var": max_var_delta,
            "passed": True,
        },
    }
    _atomic_json(
        repo_root / "results/logs/virtualdo_lineage.json",
        lineage,
    )
