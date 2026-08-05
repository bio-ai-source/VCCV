from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.io import ensure_dir, load_yaml


def run_linked_mfs_audit(repo_root: Path) -> None:
    cfg = load_yaml(repo_root / "configs/linked_benchmark.yaml")
    out_root = ensure_dir(repo_root / str(cfg["mfs_audit"]["output_root"]))
    tables_dir = ensure_dir(out_root / "tables")
    logs_dir = ensure_dir(out_root / "logs")

    gene_map_path = repo_root / str(cfg["output_root"]) / str(cfg["tables_dir"]) / "lincs_gene_provenance.csv"
    if not gene_map_path.exists():
        raise FileNotFoundError(f"Missing gene provenance table: {gene_map_path}")
    gene_map = pd.read_csv(gene_map_path)
    gene_map["g_index"] = gene_map["g_index"].astype(str)
    gene_lookup = gene_map.set_index("g_index").to_dict(orient="index")

    legacy_dim = int(cfg["mfs_audit"]["legacy_gene_dim"])
    gene_map[gene_map["legacy_256_flag"] == 1].to_csv(tables_dir / "legacy_256_gene_map.csv", index=False)

    panel_dir = repo_root / str(cfg["mfs_audit"]["legacy_panel_dir"])
    panel_rows = []
    summary_rows = []
    for js_path in sorted(panel_dir.glob("*.json")):
        obj = json.loads(js_path.read_text(encoding="utf-8"))
        panel = [str(x) for x in obj.get("panel", [])]
        landmark_hits = 0
        legacy_hits = 0
        for rank, gene in enumerate(panel, start=1):
            meta = gene_lookup.get(gene, {})
            landmark_flag = int(meta.get("landmark_flag", 0))
            legacy_flag = int(meta.get("legacy_256_flag", 0))
            landmark_hits += landmark_flag
            legacy_hits += legacy_flag
            panel_rows.append(
                {
                    "instance_id": str(obj.get("instance_id", js_path.stem)),
                    "panel_rank": rank,
                    "g_index": gene,
                    "row_id": meta.get("row_id", ""),
                    "gene_symbol": meta.get("pr_gene_symbol", ""),
                    "landmark_flag": landmark_flag,
                    "legacy_256_flag": legacy_flag,
                }
            )
        summary_rows.append(
            {
                "instance_id": str(obj.get("instance_id", js_path.stem)),
                "panel_size": len(panel),
                "landmark_share": float(landmark_hits / max(len(panel), 1)),
                "legacy_256_share": float(legacy_hits / max(len(panel), 1)),
                "all_in_legacy_256": int(
                    all(str(g).startswith("G") and int(str(g)[1:]) < legacy_dim for g in panel if str(g)[1:].isdigit())
                ),
            }
        )

    pd.DataFrame(panel_rows).to_csv(tables_dir / "legacy_panel_gene_provenance.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(tables_dir / "legacy_panel_landmark_summary.csv", index=False)

    lines = [
        "# Linked MFS provenance audit",
        f"- gene_map_rows: {len(gene_map)}",
        f"- legacy_gene_dim: {legacy_dim}",
        f"- audited_panels: {len(summary_df)}",
    ]
    if not summary_df.empty:
        lines.append(f"- mean_landmark_share: {summary_df['landmark_share'].mean():.6f}")
        lines.append(f"- mean_legacy_256_share: {summary_df['legacy_256_share'].mean():.6f}")
    (logs_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
