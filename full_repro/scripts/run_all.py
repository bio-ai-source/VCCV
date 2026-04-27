from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "01_train_virtual_anchor.py",
    "02_fuse_observed_virtual.py",
    "03_train_alignment.py",
    "04_run_vccv_inference.py",
    "05_evaluate_virtual_experiment.py",
    "06_reproduce_paper_samecell_auc.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"\n==> {script}")
        subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True)


if __name__ == "__main__":
    main()
