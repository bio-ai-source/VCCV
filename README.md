# VCCV Real-Data Reproduction Workflow

This package contains a compact real-data workflow for VCCV. The workflow starts from processed perturbational signatures and frozen structural-prior scores, trains the virtual anchor model, fits the alignment layer, runs posterior inference, and recomputes evaluation metrics.

Each stage writes intermediate artifacts to the local `data/`, `results/`, and `model_cards/` directories. The workflow can be run as a single command or stage by stage.

## Method Summary

VCCV evaluates a compound-response query against a structural target prior and a transcriptomic evidence model. For each candidate target, a reference intervention signature is obtained from an observed anchor when available and from a trained virtual anchor otherwise. The reference signature is aligned into compound-response space, scored with a Gaussian energy, and combined with the calibrated structural prior. An explicit null branch is scored from stress-like response prototypes so that globally unsupported or stress-dominated profiles are not forced into a target call.

The computational graph implemented here is:

1. Train virtual anchors from observed intervention signatures.
2. Predict virtual anchors for candidate target/context/time/dose combinations.
3. Fuse observed and virtual anchors by match quality.
4. Fit the intervention-to-compound alignment map.
5. Score target and null branches and normalize a joint posterior.
6. Recompute held-out evaluation metrics.

## Inputs

`data/processed/` contains processed VCCV inputs:

- `observeddo_mu.parquet`
- `observeddo_var_diag.parquet`
- `signatures_drug.parquet`
- `mechanism_truth.parquet`

Additional inputs are:

- `results/predictions_json/dti_prior_scores.parquet`
- `splits/drug_heldout/seed_0.json`
- `results/revision_round11/linked_transcriptomics/linked_test_predictions_with_shortlists.parquet`

The structural prior is provided as a frozen upstream input, consistent with the VCCV design as a transcriptomic corroboration layer over pre-trained structural DTI predictions.

## Environment

Required Python packages:

- `numpy`
- `pandas`
- `pyarrow`
- `torch`
- `scikit-learn`

## Run All

```bash
py scripts/run_all.py
```

## Stage Commands

```bash
py scripts/01_train_virtual_anchor.py
py scripts/02_fuse_observed_virtual.py
py scripts/03_train_alignment.py
py scripts/04_run_vccv_inference.py
py scripts/05_evaluate_virtual_experiment.py
py scripts/06_reproduce_paper_samecell_auc.py
```

Optional input refresh, when the parent repository data are available:

```bash
py scripts/00_refresh_inputs_from_parent.py
```

## Pipeline Outputs

Virtual-anchor and inference outputs:

- `data/processed/virtualdo_predictions.parquet`
- `data/processed/do_fused_mu_var.parquet`
- `results/checkpoints/align/align_params.npz`
- `results/predictions_json/mechanism_summary.parquet`

Virtual mechanism experiment outputs:

- `results/metrics_tables/mechanism_metrics.csv`
- `results/metrics_tables/paper_virtual_reproduction_check.csv`
- `results/reproduction_report.md`

Same-cell endpoint outputs:

- `results/metrics_tables/recomputed_linked_transcriptomics_metrics.csv`
- `results/exact_paper_samecell_reproduction_report.md`

## Reproduction Endpoint

The final endpoint recomputes the expression-supported same-cell VCCV metrics from held-out linked prediction rows. The generated report is `results/exact_paper_samecell_reproduction_report.md`.
