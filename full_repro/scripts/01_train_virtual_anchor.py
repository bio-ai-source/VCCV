from __future__ import annotations

from _demo_paths import DEMO_ROOT
from src.do_dictionary.virtualdo import train_virtualdo


def main() -> None:
    train_virtualdo(DEMO_ROOT)
    print("stage 01 complete: trained virtual anchor model and wrote data/processed/virtualdo_predictions.parquet")


if __name__ == "__main__":
    main()
