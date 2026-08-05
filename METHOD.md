# 方法与可执行 DAG

## 运行隔离

`reproduce.py` 要求新的或空的输出目录。它先核对 package 与 raw manifest，
再把原始数据和配置复制到 `<run>/workspace`。

解析、解压、训练和推理全部发生在 workspace。包内
`processed_reference` 仅供人工核对，不是 runner 输入。

## 数据重建

解析器从 DeepDTA Davis/KIBA 的 ligand、protein 与 affinity matrix 构造
24,000 个 DTI 行，并生成统一的 context、time、dose、platform 和 batch 字段。

GSE92742 Level-2 GCTX 提供 genetic perturbation signature。GCTX row ID 与
`G*`/`V*` 列的映射保存为 `data/interim/gene_axis.parquet`。

ObservedDO 聚合相同 target、context、time、dose、platform、batch、mode 下的
表达均值与对角方差。

药物 signature、mechanism truth 与 alignment truth 由
`synthetic_mechanism.py` 生成。QC、阈值选择、winsorize 与 split 均保存日志。

## EviDTI 与 Table 1

`EviDTI_2025_Reimpl` 使用 RDKit Morgan/MACCS/描述符、protein 3-mer 和
evidential MLP。

每个 split 执行训练集标准化、模型训练、early stopping 和 Platt calibration。

Weights、standardizer 与 Platt 参数分别保存为 NPZ/JSON。随后创建新模型实例，
重新加载全部 artifact，并生成 calibration/test probability。

Verifier 使用 calibration 分区拟合，保存 scaler、logistic coefficients、
blend 和 feature schema，再由重载 artifact 生成 test probability。

指标在 4 个 scenario × 3 个 seed × 2 个 dataset 的 24 个 slice 上计算后等权平均。

## EviDTI prior

runner 不复制旧 prior。它把本次重载 EviDTI checkpoint 产生的 test probability
按 drug-target 求均值，生成 `dti_prior_scores.parquet`。

运行会验证该 prior 覆盖全部 mechanism drug，并记录 prediction、DTI 数据和
prior 的父哈希。

## VirtualDO

VirtualDO 使用 target/context/mode embedding 与连续 time/dose，输出每个表达
维度的均值和异方差。

ObservedDO 做确定性 train/calibration 划分。Calibration grid 用于选择方差
scale。

训练后保存：

- `virtualdo_state.npz`：模型 tensor；
- `virtualdo_schema.json`：网络尺寸、映射、gene 顺序、方差顺序和 calibration。

随后创建全新 `VirtualDoNet`，重载 NPZ/JSON，并生成最终 prediction。内存模型与
重载模型的输出差超过容差时运行失败。

## Fusion

Fusion 枚举 `fusion.yaml` 中全部 `(a0,a1,a2)`：

```text
r = sigmoid(a0 + a1 × q_observed - a2 × mapping_distance)
```

参数通过 VirtualDO calibration holdout 选择。排序键为 Gaussian NLL、MSE 和
参数值。

选中参数保存到 `fusion_params.json`，重新加载后生成
`do_fused_mu_var.parquet`。

## Alignment

Alignment 搜索 rank 与 regularization，保存 `B`、LoF/GoF bias 和 beta 到
`align_params.npz`。

runner 重新加载全部数组并逐元素验证，然后进入 posterior。

## Posterior

Posterior fit 在 mechanism train/calibration split 上拟合 null prototype、
基础噪声和 eta。

`posterior_bundle.npz` 保存 null、noise 和 alignment arrays。
`posterior_bundle.json` 保存 eta、priors、配置、schema 与父输入哈希。

`predict_posterior` 从新加载 bundle，读取 drug signature、DTI prior 和 fused
DO，生成 `mechanism_predictions.parquet` 与 per-instance JSON。

随后评价函数生成 `mechanism_summary.parquet`。runner 还会验证 candidate、
非 NULL hypothesis、NULL top-rate 与 fusion holdout。
