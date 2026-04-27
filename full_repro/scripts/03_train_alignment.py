from __future__ import annotations

from _demo_paths import DEMO_ROOT
from src.align.train import train_align


def main() -> None:
    train_align(DEMO_ROOT)
    print("stage 03 complete: trained alignment and wrote results/checkpoints/align/align_params.npz")


if __name__ == "__main__":
    main()
