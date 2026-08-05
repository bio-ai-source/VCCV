from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import ensure_dir


def build_observed_dictionary(repo_root: Path) -> None:
    do_df = pd.read_parquet(repo_root / "data/processed/signatures_do.parquet")
    gcols = sorted([c for c in do_df.columns if c.startswith("G") and c[1:].isdigit()], key=lambda x: int(x[1:]))
    vcols = sorted([c for c in do_df.columns if c.startswith("V") and c[1:].isdigit()], key=lambda x: int(x[1:]))

    group_cols = ["target_key", "context_key", "pert_time", "pert_dose", "platform", "batch", "mode"]
    # Preserve the observed-anchor quality used by the reliability function.
    # Earlier aggregation retained only gene means, which silently forced the
    # downstream fallback q_obs=0.8 for every observed anchor.
    agg_mu = do_df.groupby(group_cols, as_index=False).agg(
        {**{column: "mean" for column in gcols}, "qc_do": "mean", "n_rep": "sum"}
    )
    agg_var = do_df.groupby(group_cols, as_index=False)[vcols].mean()

    processed = ensure_dir(repo_root / "data/processed")
    agg_mu.to_parquet(processed / "observeddo_mu.parquet", index=False)
    agg_var.to_parquet(processed / "observeddo_var_diag.parquet", index=False)

    cover = do_df.groupby(["target_key", "context_key", "mode"]).size().reset_index(name="n")
    lines = [
        "# Observeddo coverage",
        f"- total_rows: {len(do_df)}",
        f"- unique_targets: {cover['target_key'].nunique()}",
        f"- unique_contexts: {cover['context_key'].nunique()}",
        f"- unique_target_context_mode: {len(cover)}",
    ]
    (repo_root / "results/logs/observeddo_coverage.md").write_text("\n".join(lines), encoding="utf-8")
