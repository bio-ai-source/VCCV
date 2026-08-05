from __future__ import annotations

import gzip
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.hashing import sha256_file, sha512_file
from src.utils.io import ensure_dir, load_yaml, save_yaml


@dataclass
class DownloadRecord:
    source: str
    name: str
    url: str
    path: str
    status: str
    size_bytes: int
    sha256: str
    retries: int


def _session_with_retries(max_retries: int) -> requests.Session:
    retry = Retry(
        total=max_retries,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    s = requests.Session()
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def _download_file(
    session: requests.Session,
    url: str,
    out_path: Path,
    max_retries: int,
) -> tuple[str, int]:
    ensure_dir(out_path.parent)
    if out_path.exists() and out_path.stat().st_size > 0:
        return "cached", 0
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    status = "ok"
    for attempt in range(max_retries):
        try:
            with session.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp.replace(out_path)
            return status, attempt
        except Exception:
            status = "fail"
            if tmp.exists():
                tmp.unlink(missing_ok=True)
    return status, max_retries


def download_all(repo_root: Path) -> list[DownloadRecord]:
    cfg = load_yaml(repo_root / "configs/base.yaml")
    sources = load_yaml(repo_root / "configs/data_sources.yaml")["sources"]
    raw_root = repo_root / "data/raw"
    max_retries = int(cfg["download"]["max_retries"])
    include_lincs_metadata = bool(cfg["download"]["include_lincs_metadata"])
    include_lincs_matrices = bool(cfg["download"]["include_lincs_matrices"])

    session = _session_with_retries(max_retries=max_retries)
    out: list[DownloadRecord] = []
    for source_name, source_cfg in sources.items():
        source_dir = ensure_dir(raw_root / source_name)
        for file_spec in source_cfg["files"]:
            name = file_spec["name"]
            url = file_spec["url"]
            is_lincs_matrix = source_name.startswith("geo_") and name.endswith(".gctx.gz")
            is_lincs_meta = source_name.startswith("geo_") and not is_lincs_matrix
            if is_lincs_matrix and not include_lincs_matrices:
                continue
            if is_lincs_meta and not include_lincs_metadata:
                continue
            out_path = source_dir / name
            status, retries = _download_file(
                session=session,
                url=url,
                out_path=out_path,
                max_retries=max_retries,
            )
            size = out_path.stat().st_size if out_path.exists() else 0
            h = sha256_file(out_path) if out_path.exists() else ""
            out.append(
                DownloadRecord(
                    source=source_name,
                    name=name,
                    url=url,
                    path=str(out_path.relative_to(repo_root)),
                    status=status,
                    size_bytes=size,
                    sha256=h,
                    retries=retries,
                )
            )
    return out


def update_manifest(repo_root: Path, records: list[DownloadRecord]) -> None:
    manifest_path = repo_root / "data/manifests/download_manifest.yaml"
    payload = {
        "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "records": [
            {
                "source": r.source,
                "name": r.name,
                "url": r.url,
                "path": r.path,
                "status": r.status,
                "size_bytes": r.size_bytes,
                "sha256": r.sha256,
                "retries": r.retries,
            }
            for r in records
        ],
    }
    save_yaml(manifest_path, payload)


def verify_geo_sha512(repo_root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"checked": [], "failed": []}
    for source_name in ("geo_gse92742", "geo_gse70138"):
        src_dir = repo_root / "data/raw" / source_name
        if not src_dir.exists():
            continue
        sums = sorted(src_dir.glob("*SHA512SUMS*.gz"))
        if not sums:
            continue
        sum_file = sums[0]
        txt_path = sum_file.with_suffix("")
        with gzip.open(sum_file, "rt", encoding="utf-8", errors="ignore") as f_in:
            with txt_path.open("w", encoding="utf-8") as f_out:
                shutil.copyfileobj(f_in, f_out)
        for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            expected = parts[0].strip()
            fname = parts[-1].strip().lstrip("*")
            target = src_dir / fname
            if not target.exists():
                continue
            actual = sha512_file(target)
            item = {"file": str(target), "expected_sha512": expected, "actual_sha512": actual}
            if expected.lower() != actual.lower():
                report["failed"].append(item)
            report["checked"].append(item)
    return report
