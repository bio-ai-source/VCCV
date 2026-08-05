# Processed Data Reference Snapshot

These files were generated in the reference environment by this package's raw-data parser, quality-control workflow, and ObservedDO aggregation.

They support manual checks of schemas, row counts, the gene axis, and byte-level hashes. They are not precomputed model predictions.

- `interim/` contains GEO metadata, the entity manifest, and `gene_axis.parquet`.
- `processed/` contains DTI labels, raw and quality-controlled signatures, mechanism and alignment truth, and ObservedDO means and variances.

This directory does not contain EviDTI probabilities, DTI priors, VirtualDO predictions, fused DO values, alignment checkpoints, posterior predictions, or metric results.

By default, `reproduce.py` does not read this directory. Every run regenerates all training data from `data/raw/`.

Per-file sizes and SHA-256 values are listed in `PROCESSED_REFERENCE_MANIFEST.json` in this directory.
