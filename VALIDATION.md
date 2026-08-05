# 验证记录

验证日期：2026-07-26  
参考硬件：NVIDIA RTX 4090  
参考软件：Python 3.12、PyTorch 2.5.1+cu121、CUDA 12.1

## 静态与单元验证

```text
python reproduce.py --verify-inputs-only
PASS: 18 raw files, 134,593,644 raw bytes

python -m pytest -q
19 passed
```

测试覆盖 closed-world package/raw manifest、Davis 数据解析、指标定义、
EviDTI/Verifier artifact reload、VirtualDO 安全 NPZ+JSON roundtrip、fusion
grid fit/save/load、posterior bundle roundtrip、无 truth prediction 接口和双链路
声明。

## 完整冷运行

从两个不同空输出目录执行：

```text
python reproduce.py --device cuda:0 \
  --output <new-empty-dir>
```

两个运行都从包内压缩 GCTX 与 DeepDTA raw 文件开始，没有读取结果文件或
预训练 checkpoint。验证目录：

```text
H:\idea\DTI虚拟细胞\vccvdti\dist\_validation_fullchain_v2
H:\idea\DTI虚拟细胞\vccvdti\dist\_validation_fullchain_v2_final
```

最终源码运行耗时 524.91 秒，顶层和 `table1/` 均生成 `SUCCESS`。

## 表 1

24 个 slice 等权平均的新结果：

| Internal model | AUC | PR-AUC | NLL | ECE |
|---|---:|---:|---:|---:|
| EviDTI_2025_Reimpl | 0.843879 | 0.991524 | 0.080390 | 0.014935 |
| VCCV_Verifier_EviDTI_2025_Reimpl | 0.875425 | 0.993555 | 0.074366 | 0.013562 |

论文显示精度比较为：

| Configuration | AUC | PR-AUC | NLL | ECE |
|---|---:|---:|---:|---:|
| EviDTI | 0.844 | 0.992 | 0.080 | 0.0149 |
| VCCV + EviDTI | 0.875 | 0.994 | 0.074 | 0.0136 |

`reference_comparison.json`：`all_match=true`。

## 数据与 fullchain 规模

```text
dti_labels                         24,000 × 22
signatures_drug_raw                 4,697 × 267
signatures_do_raw                   1,377 × 522
signatures_drug (QC)                4,697 × 268
signatures_do (QC)                  1,377 × 523
observeddo_mu                       1,377 × 265
virtualdo_predictions               6,281 × 519
do_fused_mu_var                     6,281 × 519
fresh EviDTI prior                 21,356 × 3
posterior predictions               4,697 × 9
posterior evaluated summary         4,697 × 15
```

VirtualDO 从新加载 checkpoint 产生最终预测，均值与方差最大绝对 reload 差均为
0。Fusion 在 275 个 leakage-safe holdout 上比较全部 27 组参数，选中
`a0=1.0, a1=2.0, a2=0.5`。

Posterior 的 4,697 个实例全部具有非 NULL hypothesis：最少 4 个、中位数
18 个、最多 30 个；最小 candidate count 为 40。NULL 为 top hypothesis 的
实例是 843 个，比例 0.179476，未发生 NULL 塌缩。纯 prediction 在 truth 文件
被读取前完成。

## 两次运行的确定性

以下关键文件在两次完整运行间 SHA-256 全部相同：

- gene axis、DTI labels、raw/QC drug 与 DO signatures、ObservedDO；
- fresh EviDTI prior；
- VirtualDO state 与 prediction；
- fusion parameter JSON 与 fused DO；
- alignment NPZ；
- posterior NPZ、truth-free predictions 与 evaluated summary；
- EviDTI test predictions、verifier predictions 与 Table 1 summary。

包内 16 个 binary/parquet processed reference 文件也与最终 full run 的相应
raw-derived 文件逐字节一致。

## ZIP 验收

ZIP 在与源目录不同的临时目录解压后，通过 SHA-256、input manifest 和
19 个测试。完整 fullchain 已在本版源包上通过。最终 ZIP 哈希记录在同目录
`.sha256` sidecar。
