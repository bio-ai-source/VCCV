#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT / "src"
VENDOR_ROOT = PACKAGE_ROOT / "vendor"
for path in (SOURCE_ROOT, VENDOR_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from vccv_fullchain.pipeline import reproduce_fullchain, verify_packaged_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the VCCV upstream data, EviDTI prior, VirtualDO, fitted "
            "fusion, alignment and posterior from packaged raw data, and "
            "reproduce the two EviDTI Table 1 rows."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "New or empty run directory outside this package. "
            "Default: ../vccv_fullchain_run_<UTC timestamp>."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, cuda:0, and so on.",
    )
    parser.add_argument(
        "--verify-inputs-only",
        action="store_true",
        help="Verify package and raw-data hashes, then exit.",
    )
    parser.add_argument(
        "--allow-table1-drift",
        action="store_true",
        help=(
            "Keep the completed run if hardware/library numerical drift changes "
            "paper-formatted Table 1 values. Differences remain recorded."
        ),
    )
    parser.add_argument(
        "--no-legacy-rng-compatibility",
        action="store_true",
        help=(
            "Skip discarded historical companion workloads. This is faster "
            "but normally changes the Table 1 display values."
        ),
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=None,
        help=(
            "Diagnostic single-stream EviDTI seed override. Omit for the "
            "default deterministic stream map."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify_inputs_only:
        result = verify_packaged_inputs(PACKAGE_ROOT)
        print(
            "input verification PASS: "
            f"{len(result['raw_files'])} raw files, "
            f"{result['raw_bytes']} raw bytes, "
            f"{result['package_files_checked']} package files"
        )
        return 0
    output = args.output
    if output is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = PACKAGE_ROOT.parent / f"vccv_fullchain_run_{timestamp}"
    reproduce_fullchain(
        package_root=PACKAGE_ROOT,
        output_dir=output,
        device_request=args.device,
        strict_table1=not args.allow_table1_drift,
        legacy_rng_compatibility=not args.no_legacy_rng_compatibility,
        train_seed=args.train_seed,
    )
    print(f"run directory: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
