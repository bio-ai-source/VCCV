from __future__ import annotations

import math

from _demo_paths import DEMO_ROOT
import pandas as pd

from src.eval.evaluate import evaluate_mechanism


def main() -> None:
    metrics = evaluate_mechanism(DEMO_ROOT)
    out_path = DEMO_ROOT / "results/metrics_tables/mechanism_metrics.csv"
    ref = pd.read_csv(DEMO_ROOT / "reference/paper_table1_virtual_results.csv")
    got = metrics[metrics["scenario"].eq("virtual")].copy()
    merged = got.merge(
        ref[ref["scenario"].eq("virtual")],
        on=["scenario", "model"],
        how="left",
        suffixes=("_recomputed", "_paper"),
    )
    for metric in ["mrr", "hits1", "hits3", "nll"]:
        merged[f"{metric}_abs_error"] = (merged[f"{metric}_recomputed"] - merged[f"{metric}_paper"]).abs()
    merged.to_csv(DEMO_ROOT / "results/metrics_tables/paper_virtual_reproduction_check.csv", index=False)

    vccv = merged[merged["model"].eq("VCCV_Full")].iloc[0]
    max_err = max(
        float(vccv["mrr_abs_error"]),
        float(vccv["hits1_abs_error"]),
        float(vccv["hits3_abs_error"]),
    )
    lines = [
        "# Full Real-Data Demo Reproduction Report",
        "",
        "Endpoint: paper virtual mechanism experiment (`reference/paper_table1_virtual_results.csv`).",
        "",
        f"- recomputed metrics: `{out_path.relative_to(DEMO_ROOT).as_posix()}`",
        f"- check table: `results/metrics_tables/paper_virtual_reproduction_check.csv`",
        f"- VCCV_Full MRR recomputed: {float(vccv['mrr_recomputed']):.12f}",
        f"- VCCV_Full MRR paper: {float(vccv['mrr_paper']):.12f}",
        f"- VCCV_Full Hits@1 recomputed: {float(vccv['hits1_recomputed']):.12f}",
        f"- VCCV_Full Hits@1 paper: {float(vccv['hits1_paper']):.12f}",
        f"- VCCV_Full Hits@3 recomputed: {float(vccv['hits3_recomputed']):.12f}",
        f"- VCCV_Full Hits@3 paper: {float(vccv['hits3_paper']):.12f}",
        f"- max rank-metric absolute error: {max_err:.12g}",
        "",
        "The run includes virtual anchor training, anchor fusion, alignment training, posterior inference, and metric recomputation.",
    ]
    (DEMO_ROOT / "results/reproduction_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    if not math.isfinite(max_err):
        raise RuntimeError("reproduction check produced non-finite error")


if __name__ == "__main__":
    main()
