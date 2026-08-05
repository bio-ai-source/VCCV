# Virtual-to-Cellular Corroboration and Validation (VCCV) Real-Data Reproduction Workflow

This package contains a compact, self-contained real-data workflow for VCCV. It starts from bundled interaction data and LINCS perturbational profiles, then builds transcriptomic evidence and runs posterior inference.

Intermediate data, fitted artifacts, predictions, and evaluation files are written to a user-selected output directory. The complete workflow is controlled by one runner.

## Method Summary

VCCV evaluates a compound-response query against a structural target prior and a transcriptomic evidence model.

For each candidate target, a reference intervention signature is obtained from an observed anchor when available and from a trained virtual anchor otherwise.

The reference is aligned into compound-response space, scored with a covariance-aware Gaussian energy, and combined with the calibrated structural prior.

An explicit warning/null branch represents globally unsupported or stress-associated responses, preventing every query from being forced into a target call.

For the included benchmark, EviDTI supplies the upstream structural-prior scores. It is used as the prior source, while VCCV remains the focus of the downstream evidence workflow.

The computational graph implemented here is:

1. Validate and parse the packaged raw inputs.
2. Build quality-controlled interaction labels and perturbational signatures.
3. Train and reload the upstream structural-prior model.
4. Train virtual anchors from observed intervention signatures.
5. Fuse observed and virtual anchors by match quality.
6. Fit the intervention-to-compound alignment map.
7. Score target and warning/null branches and normalize the posterior.
8. Recompute the reference evaluation metrics.

## Data

`data/raw/` contains the packaged Davis, KIBA, and LINCS GSE92742 inputs. Davis and KIBA provide drug-target interaction data; GSE92742 provides perturbational transcriptomic profiles and metadata.

File sizes and cryptographic hashes are recorded in `RAW_INPUT_MANIFEST.json`. Frozen evaluation splits are stored under `splits/`.

`data/processed_reference/` contains audit snapshots produced from the raw inputs. The runner does not use these snapshots as training inputs and regenerates processed data for every new run.

## Environment

The reference environment uses Python 3.12 and PyTorch 2.5.1. CUDA is recommended for the complete workflow, while CPU execution is available for compatibility checks.

```bash
python -m venv .venv
```

Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
```

## Run All

Verify the source, configuration, split, and data hashes:

```bash
python reproduce.py --verify-inputs-only
```

Run the complete workflow in a new output directory:

```bash
python reproduce.py --device cuda:0 --output ../vccv_fullchain_run
```

CPU execution:

```bash
python reproduce.py --device cpu --output ../vccv_fullchain_run_cpu
```

The output directory must be empty. On different hardware, `--allow-table1-drift` can be used for execution checks; observed differences are retained in the comparison output.

## Pipeline Outputs

The main outputs are:

- `table1/checkpoints/` for the upstream model and VCCV verifier artifacts.
- `table1/table1_summary.csv` for the recomputed benchmark metrics.
- `workspace/data/` for regenerated intermediate and processed data.
- `workspace/results/checkpoints/` for virtual-anchor, fusion, alignment, and posterior artifacts.
- `workspace/results/predictions_json/` for posterior prediction records.
- `fullchain_lineage.json` and `run_manifest.json` for provenance and run metadata.

Saved models use NPZ or JSON artifacts. Final probabilities are produced after those artifacts are loaded into newly created model objects.

## Tests

```bash
python -m pytest -q
```

## Reproduction Endpoint

The final endpoint recomputes the packaged benchmark metrics from freshly trained and reloaded artifacts. A successful run writes `SUCCESS` markers at the top level and inside `table1/`.

Additional implementation and validation details are provided in `METHOD.md`, `PROVENANCE.md`, and `VALIDATION.md`.
