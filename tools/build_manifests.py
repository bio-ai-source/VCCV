#!/usr/bin/env python3
"""Regenerate immutable SHA-256 manifests after assembling the release."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    raw_files = sorted(
        path for path in (ROOT / "data/raw").rglob("*") if path.is_file()
    )
    raw_manifest = {
        "schema_version": 1,
        "algorithm": "SHA-256",
        "scope": (
            "Minimal complete raw dependency set used by the canonical parser: "
            "DeepDTA Davis, DeepDTA KIBA and GEO GSE92742."
        ),
        "files": {
            path.relative_to(ROOT).as_posix(): {
                "size": int(path.stat().st_size),
                "sha256": digest(path),
            }
            for path in raw_files
        },
    }
    write_json(ROOT / "RAW_INPUT_MANIFEST.json", raw_manifest)

    reference_root = ROOT / "data/processed_reference"
    reference_manifest_path = (
        reference_root / "PROCESSED_REFERENCE_MANIFEST.json"
    )
    reference_files = sorted(
        path
        for path in reference_root.rglob("*")
        if path.is_file() and path != reference_manifest_path
    )
    reference_manifest = {
        "schema_version": 1,
        "algorithm": "SHA-256",
        "generated_from_packaged_raw": True,
        "runner_reads_this_snapshot": False,
        "files": {
            path.relative_to(reference_root).as_posix(): {
                "size": int(path.stat().st_size),
                "sha256": digest(path),
            }
            for path in reference_files
        },
    }
    write_json(reference_manifest_path, reference_manifest)

    excluded_names = {"PACKAGE_MANIFEST.json"}
    package_files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name not in excluded_names
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
    )
    package_manifest = {
        "schema_version": 2,
        "algorithm": "SHA-256",
        "excluded": sorted(excluded_names),
        "files": {
            path.relative_to(ROOT).as_posix(): digest(path)
            for path in package_files
        },
    }
    write_json(ROOT / "PACKAGE_MANIFEST.json", package_manifest)
    print(
        f"wrote {len(raw_files)} raw records and "
        f"{len(reference_files)} processed-reference records and "
        f"{len(package_files)} package records"
    )


if __name__ == "__main__":
    main()
