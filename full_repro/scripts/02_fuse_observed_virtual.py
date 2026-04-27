from __future__ import annotations

from _demo_paths import DEMO_ROOT
from src.do_dictionary.fusion import fuse_observed_virtual


def main() -> None:
    fuse_observed_virtual(DEMO_ROOT)
    print("stage 02 complete: fused observed and virtual anchors into data/processed/do_fused_mu_var.parquet")


if __name__ == "__main__":
    main()
