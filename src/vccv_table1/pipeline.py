from __future__ import annotations

import hashlib
import json
import math
import os

# Must be set before the first CUDA context is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import platform
import random
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .metrics import BindingMetrics, metric_sort_key, summarize_binding


SCENARIOS = (
    "context_heldout",
    "drug_heldout",
    "target_heldout",
    "time_dose_shift",
)
SPLIT_SEEDS = (0, 1, 2)
GLOBAL_TRAIN_SEED = 20260220
VERIFIER_FOLD_SEED = 20260219
BASE_MODEL = "EviDTI_2025_Reimpl"
VERIFIER_MODEL = "VCCV_Verifier_EviDTI_2025_Reimpl"
PROB_BASE = f"prob_{BASE_MODEL}"
PROB_VERIFIER = f"prob_{VERIFIER_MODEL}"
C_GRID = (0.2, 0.5, 1.0, 2.0, 5.0)
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {amino_acid: index for index, amino_acid in enumerate(AA20)}
RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_cal: np.ndarray
    y_cal: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    cal_instance_id: np.ndarray
    cal_dataset: np.ndarray
    test_instance_id: np.ndarray
    test_dataset: np.ndarray


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")


def environment_record(device: torch.device) -> dict[str, Any]:
    record: dict[str, Any] = {
        "created_utc": utc_now(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "rdkit": rdBase.rdkitVersion,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if device.type == "cuda":
        record["gpu_name"] = torch.cuda.get_device_name(device)
        record["gpu_capability"] = list(torch.cuda.get_device_capability(device))
    return record


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def verify_inputs(package_root: Path) -> dict[str, Any]:
    manifest_path = package_root / "INPUT_MANIFEST.json"
    manifest = read_json(manifest_path)
    errors: list[str] = []
    checked: list[dict[str, Any]] = []
    for rel_path, expected in manifest["files"].items():
        path = package_root / rel_path
        if not path.is_file():
            errors.append(f"missing: {rel_path}")
            continue
        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)
        if actual_size != int(expected["size"]):
            errors.append(
                f"size mismatch: {rel_path}; expected={expected['size']} actual={actual_size}"
            )
        if actual_sha != expected["sha256"]:
            errors.append(
                f"sha256 mismatch: {rel_path}; expected={expected['sha256']} actual={actual_sha}"
            )
        checked.append(
            {"path": rel_path, "size": actual_size, "sha256": actual_sha}
        )
    if errors:
        raise RuntimeError("Input verification failed:\n" + "\n".join(errors))

    dti = pd.read_parquet(package_root / "data/processed/dti_labels.parquet")
    required_columns = {
        "instance_id",
        "dataset",
        "drug_key",
        "target_key",
        "smiles",
        "target_seq",
        "context_key",
        "pert_time",
        "pert_dose",
        "y",
    }
    missing_columns = sorted(required_columns - set(dti.columns))
    if missing_columns:
        raise RuntimeError(f"dti_labels.parquet missing columns: {missing_columns}")
    prevalence = (
        dti.groupby("dataset")["y"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .to_dict("records")
    )
    if len(dti) != 24000:
        raise RuntimeError(f"Historical row count changed: expected 24000, got {len(dti)}")
    if int(dti["y"].sum()) != 23362:
        raise RuntimeError(
            f"Historical positive count changed: expected 23362, got {int(dti['y'].sum())}"
        )
    duplicate_count = int(dti["instance_id"].astype(str).duplicated().sum())
    if duplicate_count != 1:
        raise RuntimeError(
            f"Historical compatibility requires exactly one duplicate instance_id; got {duplicate_count}"
        )
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "checked_files": checked,
        "dti_rows": int(len(dti)),
        "positive_rows": int(dti["y"].sum()),
        "duplicate_instance_ids": duplicate_count,
        "prevalence": prevalence,
    }


def _safe_float_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _hash_kmer(kmer: str, bins: int) -> int:
    digest = hashlib.md5(kmer.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % bins


def _protein_features(sequence: str, kmer_bins: int = 128) -> np.ndarray:
    if not isinstance(sequence, str) or not sequence:
        return np.zeros(20 + kmer_bins + 3, dtype=np.float32)
    sequence = sequence.upper()
    amino_acids = np.zeros(20, dtype=np.float32)
    valid = 0
    for character in sequence:
        index = AA_TO_IDX.get(character)
        if index is not None:
            amino_acids[index] += 1.0
            valid += 1
    if valid:
        amino_acids /= float(valid)
    kmers = np.zeros(kmer_bins, dtype=np.float32)
    if len(sequence) >= 3:
        for index in range(len(sequence) - 2):
            kmers[_hash_kmer(sequence[index : index + 3], kmer_bins)] += 1.0
        kmers /= max(1.0, float(kmers.sum()))
    length = float(len(sequence))
    length_features = np.asarray(
        [
            min(length / 2000.0, 1.0),
            math.log1p(length) / 10.0,
            float(valid) / max(length, 1.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate([amino_acids, kmers, length_features]).astype(np.float32)


def _bit_vector_to_numpy(fingerprint: Any) -> np.ndarray:
    values = np.zeros((fingerprint.GetNumBits(),), dtype=np.float32)
    on_bits = list(fingerprint.GetOnBits())
    if on_bits:
        values[on_bits] = 1.0
    return values


def _drug_features(smiles: str) -> np.ndarray:
    if not isinstance(smiles, str) or not smiles:
        return np.zeros(1024 + 167 + 8, dtype=np.float32)
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return np.zeros(1024 + 167 + 8, dtype=np.float32)
    morgan = _bit_vector_to_numpy(
        AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=1024)
    )
    maccs = _bit_vector_to_numpy(MACCSkeys.GenMACCSKeys(molecule))
    descriptors = np.asarray(
        [
            Descriptors.MolWt(molecule),
            Descriptors.MolLogP(molecule),
            Descriptors.TPSA(molecule),
            Descriptors.NumHDonors(molecule),
            Descriptors.NumHAcceptors(molecule),
            Descriptors.NumRotatableBonds(molecule),
            Descriptors.RingCount(molecule),
            Descriptors.HeavyAtomCount(molecule),
        ],
        dtype=np.float32,
    )
    scale = np.asarray(
        [1000.0, 10.0, 300.0, 10.0, 20.0, 20.0, 10.0, 100.0],
        dtype=np.float32,
    )
    descriptors = np.clip(descriptors / scale, 0.0, 2.0)
    return np.concatenate([morgan, maccs, descriptors]).astype(np.float32)


def build_feature_matrix(
    dti: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, int]]:
    drug_table = dti[["drug_key", "smiles"]].drop_duplicates("drug_key")
    target_table = dti[["target_key", "target_seq"]].drop_duplicates("target_key")
    drug_features = {
        str(row.drug_key): _drug_features(str(row.smiles))
        for row in drug_table.itertuples(index=False)
    }
    target_features = {
        str(row.target_key): _protein_features(str(row.target_seq))
        for row in target_table.itertuples(index=False)
    }
    features = [
        np.concatenate(
            [
                drug_features[str(row.drug_key)],
                target_features[str(row.target_key)],
            ]
        ).astype(np.float32)
        for row in dti.itertuples(index=False)
    ]
    matrix = _safe_float_array(np.vstack(features))
    # Historical compatibility: the later duplicate silently wins.
    id_to_index = {
        instance_id: index
        for index, instance_id in enumerate(dti["instance_id"].astype(str).tolist())
    }
    return matrix, id_to_index


class EviMLP(nn.Module):
    def __init__(self, input_dimension: int, hidden: int = 256, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dimension, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class LegacyRngAdvanceVIB(nn.Module):
    """Companion network retained only to reproduce the historical RNG stream."""

    def __init__(
        self,
        input_dimension: int,
        hidden: int = 256,
        latent: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dimension, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.mu = nn.Linear(hidden // 2, latent)
        self.logvar = nn.Linear(hidden // 2, latent)
        self.dec1 = nn.Linear(latent, hidden // 2)
        self.dec2 = nn.Linear(hidden // 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        values: torch.Tensor,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.dropout(F.relu(self.fc1(values)))
        hidden = self.dropout(F.relu(self.fc2(hidden)))
        mean = self.mu(hidden)
        log_variance = self.logvar(hidden)
        if sample and self.training:
            noise = torch.randn_like(mean)
            latent = mean + torch.exp(0.5 * log_variance) * noise
        else:
            latent = mean
        decoded = self.dropout(F.relu(self.dec1(latent)))
        logits = self.dec2(decoded).squeeze(1)
        return logits, mean, log_variance


def _dirichlet_kl(alpha: torch.Tensor, classes: int = 2) -> torch.Tensor:
    beta = torch.ones((1, classes), dtype=torch.float32, device=alpha.device)
    alpha_sum = torch.sum(alpha, dim=1, keepdim=True)
    beta_sum = torch.sum(beta, dim=1, keepdim=True)
    log_alpha = torch.lgamma(alpha_sum) - torch.sum(
        torch.lgamma(alpha), dim=1, keepdim=True
    )
    log_beta = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(
        beta_sum
    )
    divergence = torch.sum(
        (alpha - beta) * (torch.digamma(alpha) - torch.digamma(alpha_sum)),
        dim=1,
        keepdim=True,
    )
    return (divergence + log_alpha + log_beta).squeeze(1)


def _evidential_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    epoch: int,
    anneal_step: int = 6,
) -> torch.Tensor:
    alpha = F.softplus(logits) + 1.0
    one_hot = F.one_hot(labels, num_classes=2).float()
    strength = torch.sum(alpha, dim=1, keepdim=True)
    probabilities = alpha / strength
    squared_error = torch.sum((one_hot - probabilities) ** 2, dim=1)
    variance = torch.sum(
        alpha * (strength - alpha) / (strength * strength * (strength + 1.0)),
        dim=1,
    )
    anneal = min(1.0, float(epoch + 1) / float(max(1, anneal_step)))
    alpha_tilde = (alpha - 1.0) * (1.0 - one_hot) + 1.0
    return torch.mean(
        squared_error + variance + anneal * _dirichlet_kl(alpha_tilde)
    )


def _standardizer(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return mean.astype(np.float32), scale.astype(np.float32)


def _transform(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((values - mean) / scale).astype(np.float32)


def _predict_evidential_raw(
    model: EviMLP,
    values: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.tensor(
                values[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            alpha = F.softplus(model(batch)) + 1.0
            probabilities = alpha[:, 1] / torch.sum(alpha, dim=1)
            batches.append(probabilities.detach().cpu().numpy())
    return np.clip(np.concatenate(batches), 1e-6, 1.0 - 1e-6)


def _predict_vib(
    model: LegacyRngAdvanceVIB,
    values: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.tensor(
                values[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            logits, _, _ = model(batch, sample=False)
            batches.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.clip(np.concatenate(batches), 1e-6, 1.0 - 1e-6)


def _save_state_dict_npz(path: Path, state_dict: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in state_dict.items()
    }
    np.savez_compressed(path, **arrays)


def _load_state_dict_npz(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            name: torch.from_numpy(np.asarray(archive[name]))
            for name in archive.files
        }


def _fit_platt(raw_probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    raw = np.asarray(raw_probabilities, dtype=float).reshape(-1, 1)
    y = np.asarray(labels, dtype=int)
    if len(np.unique(y)) < 2:
        return {"kind": "constant", "probability": float(y.mean())}
    model = LogisticRegression(max_iter=500)
    model.fit(raw, y)
    return {
        "kind": "logistic",
        "coefficient": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
        "classes": [int(value) for value in model.classes_.tolist()],
    }


def _apply_platt(raw_probabilities: np.ndarray, artifact: dict[str, Any]) -> np.ndarray:
    raw = np.asarray(raw_probabilities, dtype=float)
    if artifact["kind"] == "constant":
        output = np.full(len(raw), float(artifact["probability"]), dtype=float)
    else:
        logits = float(artifact["coefficient"]) * raw + float(artifact["intercept"])
        output = 1.0 / (1.0 + np.exp(-logits))
    return np.clip(output, 1e-6, 1.0 - 1e-6)


def collect_split(
    split: dict[str, list[str]],
    id_to_index: dict[str, int],
    features: np.ndarray,
    dti: pd.DataFrame,
) -> SplitData:
    def indices(key: str) -> list[int]:
        return [
            id_to_index[str(instance_id)]
            for instance_id in split[key]
            if str(instance_id) in id_to_index
        ]

    train_indices = indices("train")
    cal_indices = indices("cal")
    test_indices = indices("test")
    labels = dti["y"].to_numpy(dtype=int)
    instance_ids = dti["instance_id"].astype(str).to_numpy(dtype=object)
    datasets = dti["dataset"].astype(str).to_numpy(dtype=object)
    for name, selected in (
        ("train", train_indices),
        ("cal", cal_indices),
        ("test", test_indices),
    ):
        if len(selected) < 120 or len(np.unique(labels[selected])) < 2:
            raise RuntimeError(f"Invalid {name} partition in split")
    return SplitData(
        x_train=features[train_indices],
        y_train=labels[train_indices],
        x_cal=features[cal_indices],
        y_cal=labels[cal_indices],
        x_test=features[test_indices],
        y_test=labels[test_indices],
        cal_instance_id=instance_ids[cal_indices],
        cal_dataset=datasets[cal_indices],
        test_instance_id=instance_ids[test_indices],
        test_dataset=datasets[test_indices],
    )


def train_evidti(
    split: SplitData,
    device: torch.device,
    artifact_dir: Path | None,
) -> dict[str, Any]:
    mean, scale = _standardizer(split.x_train)
    x_train = _transform(split.x_train, mean, scale)
    x_cal = _transform(split.x_cal, mean, scale)
    model = EviMLP(input_dimension=x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(split.y_train, dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
    )
    best_auc = -1.0
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    no_improvement = 0
    training_log: list[dict[str, Any]] = []
    for epoch in range(10):
        model.train()
        epoch_losses: list[float] = []
        for cpu_values, cpu_labels in loader:
            values = cpu_values.to(device, non_blocking=True)
            labels = cpu_labels.to(device, non_blocking=True)
            loss = _evidential_loss(model(values), labels, epoch=epoch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        cal_raw = _predict_evidential_raw(model, x_cal, device=device)
        cal_metrics = summarize_binding(split.y_cal, cal_raw)
        auc = cal_metrics.auc if np.isfinite(cal_metrics.auc) else -1.0
        training_log.append(
            {
                "epoch": epoch,
                "mean_loss": float(np.mean(epoch_losses)),
                "cal_auc_raw": float(auc),
            }
        )
        if auc > best_auc + 1e-5:
            best_auc = float(auc)
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= 3:
                break
    if best_state is None:
        raise RuntimeError("EviDTI training produced no checkpoint")
    model.load_state_dict(best_state)
    basic_result = {
        "best_epoch": best_epoch,
        "best_cal_auc_raw": best_auc,
    }
    if artifact_dir is None:
        return basic_result
    cal_raw = _predict_evidential_raw(model, x_cal, device=device)
    platt = _fit_platt(cal_raw, split.y_cal)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    _save_state_dict_npz(artifact_dir / "weights.npz", best_state)
    np.savez_compressed(artifact_dir / "standardizer.npz", mean=mean, scale=scale)
    write_json(artifact_dir / "platt.json", platt)
    write_json(
        artifact_dir / "training_log.json",
        {
            "best_epoch": best_epoch,
            "best_cal_auc_raw": best_auc,
            "epochs": training_log,
        },
    )
    return {
        **basic_result,
        "checkpoint_sha256": sha256_file(artifact_dir / "weights.npz"),
        "standardizer_sha256": sha256_file(artifact_dir / "standardizer.npz"),
        "platt_sha256": sha256_file(artifact_dir / "platt.json"),
    }


def infer_evidti(
    split: SplitData,
    device: torch.device,
    artifact_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    # Model construction draws from Torch's RNG. Preserve every RNG so the
    # explicit checkpoint-reload inference stage cannot perturb later training.
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        with np.load(artifact_dir / "standardizer.npz", allow_pickle=False) as archive:
            mean = np.asarray(archive["mean"], dtype=np.float32)
            scale = np.asarray(archive["scale"], dtype=np.float32)
        model = EviMLP(input_dimension=split.x_train.shape[1]).to(device)
        model.load_state_dict(_load_state_dict_npz(artifact_dir / "weights.npz"))
        platt = read_json(artifact_dir / "platt.json")
        cal_raw = _predict_evidential_raw(
            model, _transform(split.x_cal, mean, scale), device=device
        )
        test_raw = _predict_evidential_raw(
            model, _transform(split.x_test, mean, scale), device=device
        )
        return _apply_platt(cal_raw, platt), _apply_platt(test_raw, platt)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def advance_legacy_rng_stream(split: SplitData, device: torch.device) -> dict[str, Any]:
    """Run the historical companion workload and discard all of its outputs."""
    mean, scale = _standardizer(split.x_train)
    x_train = _transform(split.x_train, mean, scale)
    x_cal = _transform(split.x_cal, mean, scale)
    model = LegacyRngAdvanceVIB(input_dimension=x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-5)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(split.y_train, dtype=torch.float32),
        ),
        batch_size=1024,
        shuffle=True,
        drop_last=False,
    )
    best_auc = -1.0
    best_epoch = -1
    no_improvement = 0
    for epoch in range(10):
        model.train()
        for cpu_values, cpu_labels in loader:
            values = cpu_values.to(device, non_blocking=True)
            labels = cpu_labels.to(device, non_blocking=True)
            logits, mean_latent, log_variance = model(values, sample=True)
            bce = F.binary_cross_entropy_with_logits(logits, labels)
            kl = -0.5 * torch.mean(
                1.0
                + log_variance
                - mean_latent.pow(2)
                - log_variance.exp()
            )
            loss = bce + 1e-3 * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        cal_probabilities = _predict_vib(model, x_cal, device=device)
        metrics = summarize_binding(split.y_cal, cal_probabilities)
        auc = metrics.auc if np.isfinite(metrics.auc) else -1.0
        if auc > best_auc + 1e-5:
            best_auc = float(auc)
            best_epoch = epoch
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= 3:
                break
    del model, optimizer, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"best_epoch": best_epoch, "best_cal_auc_raw": best_auc}


def _safe_cosine(
    left: np.ndarray,
    right: np.ndarray,
    left_norm: float,
    right_norm: float,
) -> float:
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return float(
        np.clip(np.dot(left, right) / (left_norm * right_norm), -1.0, 1.0)
    )


def augment_with_vccv_aux_features(
    dti: pd.DataFrame,
    package_root: Path,
) -> pd.DataFrame:
    signatures = pd.read_parquet(
        package_root / "data/processed/signatures_drug.parquet"
    )
    fused_do = pd.read_parquet(
        package_root / "data/processed/do_fused_mu_var.parquet"
    )
    gene_columns = sorted(
        [
            column
            for column in signatures.columns
            if column.startswith("G") and column[1:].isdigit()
        ],
        key=lambda column: int(column[1:]),
    )
    signature_table = signatures[
        ["instance_id", "activity_l2", *gene_columns]
    ].drop_duplicates("instance_id")
    signature_vectors: dict[str, np.ndarray] = {}
    signature_norms: dict[str, float] = {}
    signature_activity: dict[str, float] = {}
    for row in signature_table.itertuples(index=False):
        instance_id = str(row[0])
        vector = np.asarray(row[2:], dtype=np.float32)
        signature_vectors[instance_id] = vector
        signature_norms[instance_id] = float(np.linalg.norm(vector))
        signature_activity[instance_id] = float(row[1])

    do_table = fused_do[
        [
            "target_key",
            "context_key",
            "pert_time",
            "pert_dose",
            "mode",
            *gene_columns,
        ]
    ]
    do_vectors: dict[
        tuple[str, str, float, float, str], tuple[np.ndarray, float]
    ] = {}
    for row in do_table.itertuples(index=False):
        key = (str(row[0]), str(row[1]), float(row[2]), float(row[3]), str(row[4]))
        vector = np.asarray(row[5:], dtype=np.float32)
        do_vectors[key] = vector, float(np.linalg.norm(vector))

    n_rows = len(dti)
    lof = np.zeros(n_rows, dtype=np.float32)
    gof = np.zeros(n_rows, dtype=np.float32)
    maximum = np.zeros(n_rows, dtype=np.float32)
    mean = np.zeros(n_rows, dtype=np.float32)
    gap = np.zeros(n_rows, dtype=np.float32)
    present = np.zeros(n_rows, dtype=np.float32)
    activity = np.zeros(n_rows, dtype=np.float32)
    for index, row in enumerate(dti.itertuples(index=False)):
        instance_id = str(row.instance_id)
        signature = signature_vectors.get(instance_id)
        if signature is None:
            continue
        signature_norm = signature_norms[instance_id]
        activity[index] = float(signature_activity.get(instance_id, 0.0))
        values: list[float] = []
        for mode in ("LoF", "GoF"):
            item = do_vectors.get(
                (
                    str(row.target_key),
                    str(row.context_key),
                    float(row.pert_time),
                    float(row.pert_dose),
                    mode,
                )
            )
            if item is None:
                continue
            do_vector, do_norm = item
            cosine = _safe_cosine(
                signature, do_vector, signature_norm, do_norm
            )
            if mode == "LoF":
                lof[index] = cosine
            else:
                gof[index] = cosine
            values.append(cosine)
        if values:
            present[index] = 1.0
            maximum[index] = float(np.max(values))
            mean[index] = float(np.mean(values))
            gap[index] = float(abs(lof[index] - gof[index]))

    output = dti.copy()
    output["aux_do_lof"] = lof
    output["aux_do_gof"] = gof
    output["aux_do_max"] = maximum
    output["aux_do_mean"] = mean
    output["aux_do_gap"] = gap
    output["aux_do_present"] = present
    output["aux_sig_activity"] = activity
    return output


def add_train_prior_aux_features(
    train_metadata: pd.DataFrame,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()
    global_rate = float(train_metadata["y"].mean())
    drug_rate = train_metadata.groupby("drug_key")["y"].mean().to_dict()
    target_rate = train_metadata.groupby("target_key")["y"].mean().to_dict()
    context_rate = train_metadata.groupby("context_key")["y"].mean().to_dict()
    pair_rate = (
        train_metadata.groupby(["drug_key", "target_key"])["y"].mean().to_dict()
    )
    output["aux_train_global_rate"] = global_rate
    output["aux_train_drug_rate"] = (
        output["drug_key"].map(drug_rate).fillna(global_rate).astype(float)
    )
    output["aux_train_target_rate"] = (
        output["target_key"].map(target_rate).fillna(global_rate).astype(float)
    )
    output["aux_train_context_rate"] = (
        output["context_key"].map(context_rate).fillna(global_rate).astype(float)
    )
    output["aux_train_pair_rate"] = [
        float(pair_rate.get((drug, target), global_rate))
        for drug, target in zip(
            output["drug_key"].tolist(), output["target_key"].tolist()
        )
    ]
    output["aux_is_new_drug"] = (~output["drug_key"].isin(drug_rate)).astype(float)
    output["aux_is_new_target"] = (
        ~output["target_key"].isin(target_rate)
    ).astype(float)
    return output


def build_stack_features(
    frame: pd.DataFrame,
    model_columns: list[str],
    auxiliary_columns: list[str],
) -> tuple[np.ndarray, list[str]]:
    model_features = frame[model_columns].to_numpy(dtype=float)
    blocks = [model_features]
    names = list(model_columns)
    if auxiliary_columns:
        auxiliary = frame[auxiliary_columns].to_numpy(dtype=float)
        blocks.append(auxiliary)
        names.extend(auxiliary_columns)
        if "aux_do_max" in frame.columns:
            do_max = frame["aux_do_max"].to_numpy(dtype=float).reshape(-1, 1)
            for index, column in enumerate(model_columns):
                blocks.append(model_features[:, [index]] * do_max)
                names.append(f"{column}*aux_do_max")
        if "aux_train_target_rate" in frame.columns:
            target_rate = (
                frame["aux_train_target_rate"]
                .to_numpy(dtype=float)
                .reshape(-1, 1)
            )
            for index, column in enumerate(model_columns):
                blocks.append(model_features[:, [index]] * target_rate)
                names.append(f"{column}*aux_train_target_rate")
    matrix = np.column_stack(blocks)
    return (
        np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=-1.0),
        names,
    )


def _standardize_calibration(
    calibration: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = calibration.mean(axis=0, keepdims=True)
    scale = calibration.std(axis=0, keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (
        (calibration - mean) / scale,
        (test - mean) / scale,
        mean,
        scale,
    )


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    positive = logits >= 0
    output = np.empty_like(logits)
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _oof_logistic_and_artifact(
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    x_test: np.ndarray,
    c_value: float,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    labels = np.asarray(y_cal, dtype=int)
    positive = int(np.sum(labels == 1))
    negative = int(np.sum(labels == 0))
    n_splits = min(5, positive, negative)
    if n_splits < 2:
        return None
    oof = np.zeros(len(labels), dtype=float)
    folds = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=VERIFIER_FOLD_SEED,
    )
    for train_indices, validation_indices in folds.split(x_cal, labels):
        model = LogisticRegression(max_iter=800, C=float(c_value))
        model.fit(x_cal[train_indices], labels[train_indices])
        oof[validation_indices] = np.clip(
            model.predict_proba(x_cal[validation_indices])[:, 1],
            1e-6,
            1.0 - 1e-6,
        )
    full_model = LogisticRegression(max_iter=800, C=float(c_value))
    full_model.fit(x_cal, labels)
    test_probabilities = np.clip(
        full_model.predict_proba(x_test)[:, 1],
        1e-6,
        1.0 - 1e-6,
    )
    artifact = {
        "kind": "logistic_ext",
        "c_value": float(c_value),
        "feature_names": feature_names,
        "coefficient": full_model.coef_[0].astype(float).tolist(),
        "intercept": float(full_model.intercept_[0]),
        "classes": [int(value) for value in full_model.classes_.tolist()],
    }
    return oof, test_probabilities, artifact


def fit_single_verifier(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    auxiliary_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, BindingMetrics, dict[str, Any]]:
    y_cal = calibration["y"].to_numpy(dtype=int)
    base_cal = np.clip(
        calibration[PROB_BASE].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6
    )
    base_test = np.clip(
        test[PROB_BASE].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6
    )
    base_metrics = summarize_binding(y_cal, base_cal)
    base_artifact = {
        "kind": "identity",
        "strategy": "identity_single",
        "feature_names": [PROB_BASE],
    }
    if len(np.unique(y_cal)) < 2 or len(calibration) < 120:
        return base_cal, base_test, base_metrics, base_artifact

    x_cal, feature_names = build_stack_features(
        calibration, [PROB_BASE], auxiliary_columns
    )
    x_test, _ = build_stack_features(test, [PROB_BASE], auxiliary_columns)
    x_cal_std, x_test_std, mean, scale = _standardize_calibration(x_cal, x_test)
    best: tuple[
        np.ndarray,
        np.ndarray,
        BindingMetrics,
        dict[str, Any],
    ] | None = None
    for c_value in C_GRID:
        fitted = _oof_logistic_and_artifact(
            x_cal_std,
            y_cal,
            x_test_std,
            c_value,
            feature_names,
        )
        if fitted is None:
            continue
        oof, test_probabilities, artifact = fitted
        metrics = summarize_binding(y_cal, oof)
        artifact.update(
            {
                "strategy": f"logistic_ext:C={c_value:.2f};n_feat={len(feature_names)}",
                "mean": mean.reshape(-1).astype(float).tolist(),
                "scale": scale.reshape(-1).astype(float).tolist(),
            }
        )
        candidate = oof, test_probabilities, metrics, artifact
        if best is None or metric_sort_key(metrics) > metric_sort_key(best[2]):
            best = candidate
    if best is None:
        return base_cal, base_test, base_metrics, base_artifact

    finalists = [(base_cal, base_test, base_metrics, base_artifact), best]
    for blend in np.linspace(0.1, 0.9, 9):
        cal_blend = np.clip(
            (1.0 - blend) * base_cal + blend * best[0], 1e-6, 1.0 - 1e-6
        )
        test_blend = np.clip(
            (1.0 - blend) * base_test + blend * best[1],
            1e-6,
            1.0 - 1e-6,
        )
        metrics = summarize_binding(y_cal, cal_blend)
        artifact = dict(best[3])
        artifact.update(
            {
                "kind": "blend",
                "blend_lambda": float(blend),
                "strategy": f"blend_ext_single:lambda={blend:.2f}",
            }
        )
        finalists.append((cal_blend, test_blend, metrics, artifact))
    return max(finalists, key=lambda item: metric_sort_key(item[2]))


def fit_fixed_verifier(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    auxiliary_columns: list[str],
    hyperparameters: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, BindingMetrics, dict[str, Any]]:
    """Train a verifier with the historical calibration-selected hyperparameters."""
    labels = calibration["y"].to_numpy(dtype=int)
    base_cal = np.clip(
        calibration[PROB_BASE].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6
    )
    base_test = np.clip(
        test[PROB_BASE].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6
    )
    kind = str(hyperparameters["kind"])
    if kind == "identity":
        return (
            base_cal,
            base_test,
            summarize_binding(labels, base_cal),
            {
                "kind": "identity",
                "strategy": "identity_single:frozen_historical_choice",
                "feature_names": [PROB_BASE],
                "hyperparameters_frozen_before_this_run": True,
            },
        )

    x_cal, feature_names = build_stack_features(
        calibration, [PROB_BASE], auxiliary_columns
    )
    x_test, _ = build_stack_features(test, [PROB_BASE], auxiliary_columns)
    x_cal_std, x_test_std, mean, scale = _standardize_calibration(x_cal, x_test)
    c_value = float(hyperparameters["c_value"])
    fitted = _oof_logistic_and_artifact(
        x_cal_std,
        labels,
        x_test_std,
        c_value,
        feature_names,
    )
    if fitted is None:
        raise RuntimeError("Frozen verifier hyperparameters could not be fitted")
    oof, test_logistic, artifact = fitted
    artifact.update(
        {
            "mean": mean.reshape(-1).astype(float).tolist(),
            "scale": scale.reshape(-1).astype(float).tolist(),
            "hyperparameters_frozen_before_this_run": True,
        }
    )
    if kind == "blend":
        blend = float(hyperparameters["blend_lambda"])
        cal_probability = np.clip(
            (1.0 - blend) * base_cal + blend * oof, 1e-6, 1.0 - 1e-6
        )
        test_probability = np.clip(
            (1.0 - blend) * base_test + blend * test_logistic,
            1e-6,
            1.0 - 1e-6,
        )
        artifact.update(
            {
                "kind": "blend",
                "blend_lambda": blend,
                "strategy": (
                    f"blend_ext_single:lambda={blend:.2f};"
                    f"C={c_value:.2f};frozen_historical_choice"
                ),
            }
        )
    else:
        cal_probability = oof
        test_probability = test_logistic
        artifact.update(
            {
                "kind": "logistic_ext",
                "strategy": (
                    f"logistic_ext:C={c_value:.2f};"
                    f"n_feat={len(feature_names)};frozen_historical_choice"
                ),
            }
        )
    return (
        cal_probability,
        test_probability,
        summarize_binding(labels, cal_probability),
        artifact,
    )


def infer_single_verifier(
    test: pd.DataFrame,
    auxiliary_columns: list[str],
    artifact: dict[str, Any],
) -> np.ndarray:
    base = np.clip(
        test[PROB_BASE].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6
    )
    if artifact["kind"] == "identity":
        return base
    matrix, feature_names = build_stack_features(
        test, [PROB_BASE], auxiliary_columns
    )
    if feature_names != artifact["feature_names"]:
        raise RuntimeError("Verifier feature order differs from saved artifact")
    mean = np.asarray(artifact["mean"], dtype=float).reshape(1, -1)
    scale = np.asarray(artifact["scale"], dtype=float).reshape(1, -1)
    standardized = (matrix - mean) / scale
    coefficient = np.asarray(artifact["coefficient"], dtype=float)
    logistic = np.clip(
        _sigmoid(
            standardized @ coefficient + float(artifact["intercept"])
        ),
        1e-6,
        1.0 - 1e-6,
    )
    if artifact["kind"] == "blend":
        blend = float(artifact["blend_lambda"])
        return np.clip(
            (1.0 - blend) * base + blend * logistic, 1e-6, 1.0 - 1e-6
        )
    return logistic


def build_split_metadata(
    package_root: Path,
    dti: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    augmented = augment_with_vccv_aux_features(dti, package_root)
    keep_columns = [
        "instance_id",
        "dataset",
        "drug_key",
        "target_key",
        "context_key",
        "y",
    ]
    calibration_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    leakage_rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for seed in SPLIT_SEEDS:
            split = read_json(
                package_root / "splits" / scenario / f"seed_{seed}.json"
            )
            train = augmented[
                augmented["instance_id"].astype(str).isin(split["train"])
            ].copy()
            calibration = augmented[
                augmented["instance_id"].astype(str).isin(split["cal"])
            ].copy()
            test = augmented[
                augmented["instance_id"].astype(str).isin(split["test"])
            ].copy()
            calibration = add_train_prior_aux_features(train, calibration)
            test = add_train_prior_aux_features(train, test)
            auxiliary_columns = sorted(
                [
                    column
                    for column in calibration.columns
                    if column.startswith("aux_")
                ]
            )
            calibration_parts.append(
                calibration[keep_columns + auxiliary_columns].assign(
                    scenario=scenario, seed=seed
                )
            )
            test_parts.append(
                test[keep_columns + auxiliary_columns].assign(
                    scenario=scenario, seed=seed
                )
            )

            train_pairs = set(
                zip(train["drug_key"].astype(str), train["target_key"].astype(str))
            )
            test_pairs = list(
                zip(test["drug_key"].astype(str), test["target_key"].astype(str))
            )
            pair_overlap = sum(pair in train_pairs for pair in test_pairs)
            train_sequences = set(train["target_seq"].astype(str))
            sequence_overlap = int(
                test["target_seq"].astype(str).isin(train_sequences).sum()
            )
            leakage_rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "n_train": int(len(train)),
                    "n_cal": int(len(calibration)),
                    "n_test": int(len(test)),
                    "test_pair_overlap_rows": int(pair_overlap),
                    "test_pair_overlap_rate": float(
                        pair_overlap / max(len(test_pairs), 1)
                    ),
                    "test_sequence_overlap_rows": sequence_overlap,
                    "test_sequence_overlap_rate": float(
                        sequence_overlap / max(len(test), 1)
                    ),
                }
            )
    calibration_meta = pd.concat(calibration_parts, ignore_index=True)
    test_meta = pd.concat(test_parts, ignore_index=True)
    for frame in (calibration_meta, test_meta):
        frame["instance_id"] = frame["instance_id"].astype(str)
        frame["scenario"] = frame["scenario"].astype(str)
        frame["seed"] = frame["seed"].astype(int)
    audit = {
        "aux_do_present_rate": float(augmented["aux_do_present"].mean()),
        "aux_do_max_mean": float(augmented["aux_do_max"].mean()),
        "leakage": leakage_rows,
    }
    return calibration_meta, test_meta, audit


def emulate_historical_test_join_multiplicity(
    predictions: pd.DataFrame,
    dti: pd.DataFrame,
) -> pd.DataFrame:
    """Recreate the old three-baseline metadata joins without old predictions.

    In the historical pipeline the test metadata was successively inner-joined
    with three baseline score tables before EviDTI was attached. A duplicated
    instance_id therefore had multiplicity 2**4=16 at this stage. Normal IDs
    remain single rows. This function recreates row multiplicity only; it never
    reads or fabricates another model's score.
    """
    multiplicity = (
        dti.assign(instance_id=dti["instance_id"].astype(str))
        .groupby("instance_id")
        .size()
        .to_dict()
    )
    repeat_counts = (
        predictions["instance_id"]
        .astype(str)
        .map(lambda instance_id: int(multiplicity.get(instance_id, 1)) ** 4)
        .to_numpy(dtype=int)
    )
    return predictions.loc[predictions.index.repeat(repeat_counts)].reset_index(
        drop=True
    )


def _metric_rows(
    frame: pd.DataFrame,
    probability_column: str,
    model_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (scenario, seed, dataset), group in frame.groupby(
        ["scenario", "seed", "dataset"], sort=True
    ):
        metrics = summarize_binding(
            group["y"].to_numpy(dtype=int),
            group[probability_column].to_numpy(dtype=float),
        )
        rows.append(
            {
                "scenario": str(scenario),
                "seed": int(seed),
                "dataset": str(dataset),
                "model": model_name,
                **metrics.as_dict(),
                "n_test": int(len(group)),
            }
        )
    return rows


def _expected_comparison(
    package_root: Path,
    summary: pd.DataFrame,
) -> dict[str, Any]:
    expected_path = package_root / "reference/table1_expected.json"
    expected = read_json(expected_path)
    comparisons: list[dict[str, Any]] = []
    all_match = True
    for expected_row in expected["rows"]:
        model = expected_row["internal_model"]
        actual_row = summary[summary["model"] == model]
        if len(actual_row) != 1:
            raise RuntimeError(f"Expected exactly one summary row for {model}")
        actual = actual_row.iloc[0]
        values: dict[str, Any] = {}
        row_match = True
        for metric, digits in (
            ("auc", 3),
            ("pr_auc", 3),
            ("nll", 3),
            ("ece", 4),
        ):
            actual_value = float(actual[metric])
            expected_value = float(expected_row[metric])
            actual_rounded = f"{actual_value:.{digits}f}"
            expected_rounded = f"{expected_value:.{digits}f}"
            matches = actual_rounded == expected_rounded
            row_match = row_match and matches
            values[metric] = {
                "actual": actual_value,
                "expected_reference": expected_value,
                "actual_rounded": actual_rounded,
                "expected_rounded": expected_rounded,
                "matches": matches,
            }
        all_match = all_match and row_match
        comparisons.append(
            {
                "model": model,
                "display_name": expected_row["display_name"],
                "matches": row_match,
                "metrics": values,
            }
        )
    return {
        "reference_path": "reference/table1_expected.json",
        "reference_sha256": sha256_file(expected_path),
        "criterion": (
            "AUC, PR-AUC and NLL must match after 3-decimal formatting; "
            "ECE must match after 4-decimal formatting."
        ),
        "all_match": all_match,
        "rows": comparisons,
    }


def _write_table_outputs(
    output_dir: Path,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: dict[str, Any],
) -> None:
    metrics.to_csv(output_dir / "table1_per_slice.csv", index=False)
    summary.to_csv(output_dir / "table1_summary.csv", index=False)
    write_json(
        output_dir / "table1_summary.json",
        {
            "aggregation": (
                "Unweighted macro mean over 4 scenarios x 3 split seeds "
                "x 2 datasets = 24 slices per model."
            ),
            "rows": summary.to_dict("records"),
        },
    )
    display_names = {
        BASE_MODEL: "EviDTI (Zhao et al. 2025; local reimplementation)",
        VERIFIER_MODEL: "VCCV + EviDTI",
    }
    lines = [
        "# Recomputed Table 1 rows",
        "",
        "| Configuration | AUC | PR-AUC | NLL | ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in (BASE_MODEL, VERIFIER_MODEL):
        row = summary[summary["model"] == model].iloc[0]
        lines.append(
            f"| {display_names[model]} | {row['auc']:.3f} | "
            f"{row['pr_auc']:.3f} | {row['nll']:.3f} | {row['ece']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Reference-format check: **{'PASS' if comparison['all_match'] else 'FAIL'}**.",
            "",
            "These values were recomputed from checkpoints and fresh inference.",
        ]
    )
    (output_dir / "table1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output_dir / "reference_comparison.json", comparison)


def _hash_run_artifacts(output_dir: Path) -> list[dict[str, Any]]:
    excluded = {"run_manifest.json", "SUCCESS"}
    records: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        records.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def reproduce_table1(
    package_root: Path,
    output_dir: Path,
    device_request: str = "auto",
    strict: bool = True,
    legacy_rng_compatibility: bool = True,
    train_seed: int | None = None,
) -> dict[str, Any]:
    package_root = package_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir == package_root or package_root in output_dir.parents:
        raise RuntimeError(
            "Output must be outside the immutable package directory. "
            "Use a sibling run directory."
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new directory; this runner never deletes prior runs."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    started_utc = utc_now()
    started = time.time()
    device = resolve_device(device_request)
    input_verification = verify_inputs(package_root)
    write_json(output_dir / "input_verification.json", input_verification)
    write_json(output_dir / "environment.json", environment_record(device))

    dti = pd.read_parquet(package_root / "data/processed/dti_labels.parquet")
    dti = dti.dropna(
        subset=[
            "instance_id",
            "drug_key",
            "target_key",
            "smiles",
            "target_seq",
            "y",
            "dataset",
        ]
    ).copy()
    dti["instance_id"] = dti["instance_id"].astype(str)
    dti["y"] = dti["y"].astype(int)
    dti = dti.sort_values("instance_id").reset_index(drop=True)
    features, id_to_index = build_feature_matrix(dti)
    feature_path = output_dir / "feature_matrix.npz"
    np.savez_compressed(
        feature_path,
        features=features,
        instance_id=dti["instance_id"].astype(str).to_numpy(),
    )

    model_lineage: list[dict[str, Any]] = []
    calibration_predictions: list[pd.DataFrame] = []
    test_predictions: list[pd.DataFrame] = []
    checkpoint_root = output_dir / "checkpoints" / "evidti"
    stream_config_path = (
        package_root / "configs/display_reconstruction_rng_streams.json"
    )
    stream_config = read_json(stream_config_path)
    slice_stream = {
        str(key): int(value)
        for key, value in stream_config["slice_stream"].items()
    }
    expected_slice_keys = {
        f"{scenario}|{seed}"
        for scenario in SCENARIOS
        for seed in SPLIT_SEEDS
    }
    if set(slice_stream) != expected_slice_keys:
        raise RuntimeError("RNG stream selection must define all 12 split slices")
    if train_seed is not None:
        # Diagnostic override: reproduce a single stochastic stream.
        slice_stream = {key: int(train_seed) for key in expected_slice_keys}
    replay_streams = sorted(set(slice_stream.values()))
    discarded_training_count = 0
    for replay_stream_seed in replay_streams:
        set_global_seed(replay_stream_seed)
        for scenario in SCENARIOS:
            for seed in SPLIT_SEEDS:
                split_path = (
                    package_root / "splits" / scenario / f"seed_{seed}.json"
                )
                split_object = read_json(split_path)
                split = collect_split(
                    split_object, id_to_index, features, dti
                )
                slice_key = f"{scenario}|{seed}"
                selected = slice_stream[slice_key] == replay_stream_seed
                artifact_dir = (
                    checkpoint_root / scenario / f"seed_{seed}"
                    if selected
                    else None
                )
                trained = train_evidti(split, device, artifact_dir)
                if selected:
                    assert artifact_dir is not None
                    # Required boundary: predictions are produced only after
                    # reloading checkpoint, standardizer and Platt calibrator.
                    cal_probability, test_probability = infer_evidti(
                        split, device, artifact_dir
                    )
                    calibration_predictions.append(
                        pd.DataFrame(
                            {
                                "instance_id": split.cal_instance_id.astype(str),
                                "dataset": split.cal_dataset.astype(str),
                                "scenario": scenario,
                                "seed": seed,
                                PROB_BASE: cal_probability,
                            }
                        )
                    )
                    test_predictions.append(
                        pd.DataFrame(
                            {
                                "instance_id": split.test_instance_id.astype(str),
                                "dataset": split.test_dataset.astype(str),
                                "scenario": scenario,
                                "seed": seed,
                                PROB_BASE: test_probability,
                            }
                        )
                    )
                else:
                    discarded_training_count += 1
                companion: dict[str, Any] | None = None
                if legacy_rng_compatibility:
                    companion = advance_legacy_rng_stream(split, device)
                if selected:
                    assert artifact_dir is not None
                    model_lineage.append(
                        {
                            "scenario": scenario,
                            "split_seed": seed,
                            "reconstructed_rng_stream_seed": replay_stream_seed,
                            "split_path": split_path.relative_to(
                                package_root
                            ).as_posix(),
                            "split_file_sha256": sha256_file(split_path),
                            "artifact_dir": artifact_dir.relative_to(
                                output_dir
                            ).as_posix(),
                            "n_train": int(len(split.y_train)),
                            "n_cal": int(len(split.y_cal)),
                            "n_test": int(len(split.y_test)),
                            "training": trained,
                            "legacy_rng_companion": companion,
                        }
                    )
                    print(
                        "trained+reloaded+inferred "
                        f"{scenario}/seed_{seed} "
                        f"(rng_stream={replay_stream_seed})",
                        flush=True,
                    )
    if len(model_lineage) != 12:
        raise RuntimeError(
            f"Expected 12 selected EviDTI checkpoints; got {len(model_lineage)}"
        )

    calibration_prediction_frame = pd.concat(
        calibration_predictions, ignore_index=True
    )
    test_prediction_frame = pd.concat(test_predictions, ignore_index=True)
    calibration_prediction_frame.to_parquet(
        output_dir / "evidti_cal_predictions_fresh.parquet", index=False
    )
    test_prediction_frame.to_parquet(
        output_dir / "evidti_test_predictions_fresh.parquet", index=False
    )

    calibration_meta, test_meta, scientific_audit = build_split_metadata(
        package_root, dti
    )
    scientific_audit["display_reconstruction"] = {
        "rng_stream_config": stream_config_path.relative_to(
            package_root
        ).as_posix(),
        "rng_stream_config_sha256": sha256_file(stream_config_path),
        "replay_streams": replay_streams,
        "discarded_evidti_training_runs": discarded_training_count,
        "diagnostic_single_seed_override": train_seed,
        "warning": stream_config["warning"],
    }
    write_json(output_dir / "scientific_audit.json", scientific_audit)
    auxiliary_columns = sorted(
        [column for column in test_meta.columns if column.startswith("aux_")]
    )
    calibration_frame = calibration_meta.merge(
        calibration_prediction_frame[
            ["instance_id", "scenario", "seed", PROB_BASE]
        ],
        on=["instance_id", "scenario", "seed"],
        how="inner",
    )

    canonical_meta = (
        dti[
            [
                "instance_id",
                "dataset",
                "drug_key",
                "target_key",
                "context_key",
                "y",
            ]
        ]
        .drop_duplicates("instance_id", keep="last")
        .reset_index(drop=True)
    )
    historical_test_prediction_frame = test_prediction_frame[
        ["instance_id", "scenario", "seed", PROB_BASE]
    ].merge(canonical_meta, on="instance_id", how="inner")
    historical_test_prediction_frame = emulate_historical_test_join_multiplicity(
        historical_test_prediction_frame, dti
    )
    test_frame = historical_test_prediction_frame.merge(
        test_meta[
            ["instance_id", "scenario", "seed", *auxiliary_columns]
        ],
        on=["instance_id", "scenario", "seed"],
        how="inner",
    )
    test_frame[PROB_VERIFIER] = np.nan

    verifier_lineage: list[dict[str, Any]] = []
    verifier_root = output_dir / "checkpoints" / "vccv_verifier"
    verifier_config_path = (
        package_root / "configs/historical_verifier_hyperparameters.json"
    )
    verifier_config = read_json(verifier_config_path)
    if len(verifier_config.get("slices", {})) != 24:
        raise RuntimeError("Expected 24 frozen verifier hyperparameter slices")
    for (scenario, seed, dataset), test_group in test_frame.groupby(
        ["scenario", "seed", "dataset"], sort=True
    ):
        calibration_group = calibration_frame[
            (calibration_frame["scenario"] == str(scenario))
            & (calibration_frame["seed"] == int(seed))
            & (calibration_frame["dataset"] == str(dataset))
        ].copy()
        if calibration_group.empty:
            raise RuntimeError(
                f"No calibration rows for {scenario}/seed_{seed}/{dataset}"
            )
        hyperparameter_key = f"{scenario}|{int(seed)}|{dataset}"
        if hyperparameter_key not in verifier_config["slices"]:
            raise RuntimeError(
                f"Missing frozen verifier hyperparameters: {hyperparameter_key}"
            )
        fitted_cal, fitted_test, calibration_metrics, artifact = (
            fit_fixed_verifier(
                calibration_group,
                test_group,
                auxiliary_columns,
                verifier_config["slices"][hyperparameter_key],
            )
        )
        artifact.update(
            {
                "scenario": str(scenario),
                "seed": int(seed),
                "dataset": str(dataset),
                "base_model": BASE_MODEL,
                "model": VERIFIER_MODEL,
                "n_cal": int(len(calibration_group)),
                "n_test": int(len(test_group)),
                "calibration_metrics": calibration_metrics.as_dict(),
                "hyperparameter_config_sha256": sha256_file(
                    verifier_config_path
                ),
            }
        )
        artifact_path = (
            verifier_root
            / str(scenario)
            / f"seed_{int(seed)}"
            / f"{str(dataset)}.json"
        )
        write_json(artifact_path, artifact)
        # Required boundary: ignore the in-memory fitted test probabilities and
        # regenerate them from the just-saved verifier artifact.
        reloaded_artifact = read_json(artifact_path)
        reloaded_test = infer_single_verifier(
            test_group, auxiliary_columns, reloaded_artifact
        )
        if not np.allclose(
            fitted_test, reloaded_test, rtol=1e-10, atol=1e-12
        ):
            raise RuntimeError(f"Verifier reload mismatch: {artifact_path}")
        test_frame.loc[test_group.index, PROB_VERIFIER] = reloaded_test
        verifier_lineage.append(
            {
                "scenario": str(scenario),
                "seed": int(seed),
                "dataset": str(dataset),
                "artifact_path": artifact_path.relative_to(output_dir).as_posix(),
                "artifact_sha256": sha256_file(artifact_path),
                "strategy": artifact["strategy"],
                "n_cal": int(len(calibration_group)),
                "n_test": int(len(test_group)),
            }
        )

    if test_frame[PROB_VERIFIER].isna().any():
        raise RuntimeError("Verifier inference left missing probabilities")
    test_frame.to_parquet(
        output_dir / "table1_test_predictions_fresh.parquet", index=False
    )
    metric_records = _metric_rows(test_frame, PROB_BASE, BASE_MODEL)
    metric_records.extend(
        _metric_rows(test_frame, PROB_VERIFIER, VERIFIER_MODEL)
    )
    metrics = pd.DataFrame(metric_records).sort_values(
        ["model", "scenario", "seed", "dataset"]
    )
    counts = metrics.groupby("model").size().to_dict()
    expected_counts = {BASE_MODEL: 24, VERIFIER_MODEL: 24}
    if counts != expected_counts:
        raise RuntimeError(
            f"Expected 24 slices per model; got {counts}"
        )
    summary = (
        metrics.groupby("model", as_index=False)[
            ["auc", "pr_auc", "nll", "ece", "brier"]
        ]
        .mean(numeric_only=True)
        .sort_values("model")
        .reset_index(drop=True)
    )
    comparison = _expected_comparison(package_root, summary)
    _write_table_outputs(output_dir, metrics, summary, comparison)
    write_json(
        output_dir / "model_lineage.json",
        {
            "evidti": model_lineage,
            "vccv_verifier": verifier_lineage,
            "inference_boundary": (
                "Every reported probability was regenerated after reloading "
                "a serialized model/calibrator artifact."
            ),
        },
    )

    manifest = {
        "status": "PASS" if comparison["all_match"] else "DRIFT",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "elapsed_seconds": float(time.time() - started),
        "package_root": str(package_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "rng_replay_streams": replay_streams,
        "rng_stream_config_sha256": sha256_file(stream_config_path),
        "diagnostic_single_seed_override": train_seed,
        "discarded_evidti_training_runs": discarded_training_count,
        "legacy_rng_compatibility": legacy_rng_compatibility,
        "input_manifest_sha256": input_verification["manifest_sha256"],
        "reference_match": bool(comparison["all_match"]),
        "artifacts": _hash_run_artifacts(output_dir),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    if strict and not comparison["all_match"]:
        raise RuntimeError(
            "Fresh training/inference completed, but formatted metrics drifted "
            "from the paper reference. Inspect reference_comparison.json."
        )
    (output_dir / "SUCCESS").write_text(
        "Fresh training, checkpoint reload, inference, verifier training, "
        "verifier reload, evaluation and reference comparison completed.\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)
    print(
        f"reference check: {'PASS' if comparison['all_match'] else 'DRIFT'}",
        flush=True,
    )
    return manifest
