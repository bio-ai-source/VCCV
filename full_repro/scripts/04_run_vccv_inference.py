from __future__ import annotations

from _demo_paths import DEMO_ROOT
from src.inference.engine import run_inference


def main() -> None:
    run_inference(DEMO_ROOT)
    print("stage 04 complete: ran posterior inference and wrote results/predictions_json/mechanism_summary.parquet")


if __name__ == "__main__":
    main()
