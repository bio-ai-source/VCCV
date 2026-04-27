from __future__ import annotations

import shutil

from _demo_paths import DEMO_ROOT, PROJECT_ROOT


COPIES = [
    ("data/processed/signatures_drug.parquet", "data/processed/signatures_drug.parquet"),
    ("data/processed/mechanism_truth.parquet", "data/processed/mechanism_truth.parquet"),
    ("data/processed/observeddo_mu.parquet", "data/processed/observeddo_mu.parquet"),
    ("data/processed/observeddo_var_diag.parquet", "data/processed/observeddo_var_diag.parquet"),
    ("results/predictions_json/dti_prior_scores.parquet", "results/predictions_json/dti_prior_scores.parquet"),
    ("splits/drug_heldout/seed_0.json", "splits/drug_heldout/seed_0.json"),
    (
        "results/revision_round11/linked_transcriptomics/linked_test_predictions_with_shortlists.parquet",
        "results/revision_round11/linked_transcriptomics/linked_test_predictions_with_shortlists.parquet",
    ),
    (
        "results/revision_round11/linked_transcriptomics/tables/linked_transcriptomics_metrics.csv",
        "reference/paper_linked_transcriptomics_metrics.csv",
    ),
    (
        "results/nature_submission/tables/table1_virtual_results.csv",
        "reference/paper_table1_virtual_results.csv",
    ),
]


def main() -> None:
    for src_rel, dst_rel in COPIES:
        src = PROJECT_ROOT / src_rel
        dst = DEMO_ROOT / dst_rel
        if not src.exists():
            raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"copied {src_rel} -> {dst_rel}")


if __name__ == "__main__":
    main()
