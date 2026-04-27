from __future__ import annotations

from _demo_paths import DEMO_ROOT
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


MODEL_SCORE_COLUMNS = {
    "Structural-prior control": "p_structural_prior_only",
    "FRoGS-shortlist": "p_frogs_shortlist",
    "GMPTI-shortlist": "p_gmpti_shortlist",
    "Context-aware fusion control": "p_context_expression_fusion",
    "VCCV": "p_vccv_r2_linked",
}


SUBSET_FILTERS = {
    "expression_supported_same_cell": lambda df: (
        df["expression_supported_flag"].astype(int).eq(1)
        & df["time_diff_lof"].astype(float).le(72.0)
        & df["log10_dose_diff_lof"].astype(float).le(2.0)
    ),
    "expression_supported_same_cell_power_augmented": lambda df: (
        df["expression_supported_flag"].astype(int).eq(1)
    ),
    "strict_same_cell_power_augmented": lambda df: (
        df["expression_supported_flag"].astype(int).eq(1)
        & df["time_diff_lof"].astype(float).le(96.0)
        & df["log10_dose_diff_lof"].astype(float).le(1.5)
    ),
    "dose_proximal_power_augmented": lambda df: (
        df["expression_supported_flag"].astype(int).eq(1)
        & df["time_diff_lof"].astype(float).le(96.0)
        & df["log10_dose_diff_lof"].astype(float).le(1.0)
    ),
}


def main() -> None:
    pred_path = DEMO_ROOT / "results/revision_round11/linked_transcriptomics/linked_test_predictions_with_shortlists.parquet"
    ref_path = DEMO_ROOT / "reference/paper_linked_transcriptomics_metrics.csv"
    pred = pd.read_parquet(pred_path)
    ref = pd.read_csv(ref_path)
    rows = []
    for subset, filt in SUBSET_FILTERS.items():
        sub = pred[filt(pred)].copy()
        y = sub["y"].astype(int).to_numpy()
        prevalence = float(y.mean())
        for model, col in MODEL_SCORE_COLUMNS.items():
            score = sub[col].astype(float).to_numpy()
            pr = float(average_precision_score(y, score))
            rows.append(
                {
                    "subset": subset,
                    "model": model,
                    "auc": float(roc_auc_score(y, score)),
                    "pr_auc": pr,
                    "excess_pr_over_prevalence": pr - prevalence,
                    "prevalence": prevalence,
                    "n_rows": int(len(sub)),
                    "negatives": int((1 - y).sum()),
                }
            )
    out = pd.DataFrame(rows)
    out_path = DEMO_ROOT / "results/metrics_tables/recomputed_linked_transcriptomics_metrics.csv"
    out.to_csv(out_path, index=False)
    check = out.merge(ref, on=["subset", "model"], suffixes=("_recomputed", "_paper"), how="left")
    for metric in ["auc", "pr_auc", "excess_pr_over_prevalence", "prevalence"]:
        check[f"{metric}_abs_error"] = (check[f"{metric}_recomputed"] - check[f"{metric}_paper"]).abs()
    check_path = DEMO_ROOT / "results/metrics_tables/paper_samecell_auc_reproduction_check.csv"
    check.to_csv(check_path, index=False)

    target = check[
        check["subset"].eq("expression_supported_same_cell") & check["model"].eq("VCCV")
    ].iloc[0]
    max_err = max(float(target["auc_abs_error"]), float(target["pr_auc_abs_error"]), float(target["excess_pr_over_prevalence_abs_error"]))
    lines = [
        "# Exact Paper Same-Cell Result Reproduction",
        "",
        "Endpoint: main-text Section 4.3 / Figure 3d expression-supported same-cell VCCV result.",
        "",
        f"- prediction rows: `{pred_path.relative_to(DEMO_ROOT).as_posix()}`",
        f"- recomputed table: `{out_path.relative_to(DEMO_ROOT).as_posix()}`",
        f"- check table: `{check_path.relative_to(DEMO_ROOT).as_posix()}`",
        f"- VCCV ROC-AUC recomputed: {float(target['auc_recomputed']):.15f}",
        f"- VCCV ROC-AUC paper: {float(target['auc_paper']):.15f}",
        f"- VCCV PR-AUC recomputed: {float(target['pr_auc_recomputed']):.15f}",
        f"- VCCV PR-AUC paper: {float(target['pr_auc_paper']):.15f}",
        f"- max endpoint absolute error: {max_err:.3g}",
        "",
        "This final check recomputes the paper metric from held-out linked prediction rows, not from the summary table.",
    ]
    report = DEMO_ROOT / "results/exact_paper_samecell_reproduction_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    if max_err > 1e-12:
        raise RuntimeError(f"paper same-cell endpoint mismatch: {max_err}")


if __name__ == "__main__":
    main()
