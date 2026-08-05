# 数据与模型溯源

## 不可变输入

`RAW_INPUT_MANIFEST.json` 记录原始数据的相对路径、字节数与 SHA-256。
`PACKAGE_MANIFEST.json` 覆盖除自身以外的全部包文件。

`reproduce.py` 在解析或训练前验证两份 manifest。

## DeepDTA Davis

数据来自 DeepDTA GitHub commit
`a546a8433a6822e958f36171c4356ad6f414d623` 的 `data/davis`。

快照包含 68 个 drug、442 个 protein，以及 30,056 个非空 affinity。

随包文件为 `ligands_can.txt`、`proteins.txt`、`Y` 和 train/test fold。

## DeepDTA KIBA

数据来自同一固定 commit 的 `data/kiba`。

快照包含 2,111 个 drug、229 个 protein，以及 118,254 个非空 affinity。

## GEO GSE92742

数据包括 Level-2 delta GCTX、signature info/metrics、gene/cell/pert info 和
GEO checksum 文件。

GCTX 以 `.gctx.gz` 保存。runner 在 workspace 中解压后解析。

## 生成数据

每次运行从 raw 生成：

- `data/interim/*.parquet`
- `dti_labels.parquet`
- `signatures_do_raw.parquet`
- `signatures_drug_raw.parquet`
- `mechanism_truth.parquet`
- `align_ground_truth.npz`
- QC 后的 drug/DO signature
- ObservedDO mean/variance
- VirtualDO prediction
- fused DO

运行目录中的 `PROCESSED_DATA_MANIFEST.json` 保存这些文件的大小与 SHA-256。

包内 `processed_reference` 用于逐字节核对，runner 不读取。

## 模型父子关系

```text
raw Davis/KIBA
  └─ dti_labels
      └─ EviDTI checkpoints
          ├─ Table 1 verifier artifacts
          └─ fresh DTI prior
                 └─ VirtualDO
                        └─ fusion parameters
                               └─ fused DO

raw GSE92742
  └─ DO signatures
       └─ ObservedDO ───────────────┘

drug signatures + fused DO
  └─ alignment
       └─ posterior bundle
            └─ predictions
                 └─ evaluation
```

## Lineage 文件

- `table1/model_lineage.json`
- `workspace/results/predictions_json/dti_prior_lineage.json`
- `workspace/results/logs/virtualdo_lineage.json`
- `workspace/results/logs/fusion_lineage.json`
- `workspace/results/checkpoints/align/reload_validation.json`
- `workspace/results/checkpoints/posterior/posterior_bundle.json`
- `fullchain_lineage.json`

这些文件记录各阶段输入、artifact、输出与 SHA-256。
