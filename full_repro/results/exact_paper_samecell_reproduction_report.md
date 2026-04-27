# Exact Paper Same-Cell Result Reproduction

Endpoint: main-text Section 4.3 / Figure 3d expression-supported same-cell VCCV result.

- prediction rows: `results/revision_round11/linked_transcriptomics/linked_test_predictions_with_shortlists.parquet`
- recomputed table: `results/metrics_tables/recomputed_linked_transcriptomics_metrics.csv`
- check table: `results/metrics_tables/paper_samecell_auc_reproduction_check.csv`
- VCCV ROC-AUC recomputed: 0.697230824917715
- VCCV ROC-AUC paper: 0.697230824917715
- VCCV PR-AUC recomputed: 0.993208270405652
- VCCV PR-AUC paper: 0.993208270405652
- max endpoint absolute error: 1.11e-16

This final check recomputes the paper metric from held-out linked prediction rows, not from the summary table.