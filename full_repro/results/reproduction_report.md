# Full Real-Data Demo Reproduction Report

Endpoint: paper virtual mechanism experiment (`reference/paper_table1_virtual_results.csv`).

- recomputed metrics: `results/metrics_tables/mechanism_metrics.csv`
- check table: `results/metrics_tables/paper_virtual_reproduction_check.csv`
- VCCV_Full MRR recomputed: 0.625102880658
- VCCV_Full MRR paper: 0.603268526345
- VCCV_Full Hits@1 recomputed: 0.401234567901
- VCCV_Full Hits@1 paper: 0.360946745562
- VCCV_Full Hits@3 recomputed: 0.839506172840
- VCCV_Full Hits@3 paper: 0.810650887574
- max rank-metric absolute error: 0.0402878223391

The run includes virtual anchor training, anchor fusion, alignment training, posterior inference, and metric recomputation.