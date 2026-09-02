# TSB-AD 部分评测结果（Eva 划分）

此文件已过时。当前完整汇总见 [`benchmark-tsb-ad.md`](benchmark-tsb-ad.md)。

快照时间：2026-09-01（Time-RCD 双侧已齐）。结果来自 `results/benchmark/`，只汇总**该划分下单元集合已经齐全**的方法。

## 纳入规则

- 评测划分：Eva。有效单元共 69 个：单变量 **TSB-AD-U 23** 个、多变量 **TSB-AD-M 46** 个（Exathlon-05 / Exathlon-22 无 Eva 序列，不计入）。
- **单变量表**只收齐了全部 23 个 TSB-AD-U 单元的方法；**多变量表**只收齐了全部 46 个 TSB-AD-M 单元的方法。一侧跑完即可进入对应表。
- 平均方式：该划分下所有单元的 `metrics.csv` 按**序列**宏平均（每条序列等权，NaN 跳过），与 `EvaluationResult.summary(stat="mean")` 一致。
- 指标为 `DEFAULT_METRICS` 的 11 项。表内按 **VUS-PR** 降序；每列最优值加粗（3 位小数并列则同时加粗）。

## 方法覆盖

| 方法 | 类型 | TSB-AD-U | TSB-AD-M | 纳入 |
| --- | --- | ---: | ---: | --- |
| ClusterLocalOutlierFactor | baseline | 23/23 | 45/46 | 单变量 |
| ConnectivityOutlierFactor | baseline | 23/23 | 46/46 | 单变量、多变量 |
| Copula | baseline | 0/23 | 3/46 | 未纳入 |
| ExtendedIsolationForest | baseline | 0/23 | 38/46 | 未纳入 |
| Fourier | baseline | 23/23 | 46/46 | 单变量、多变量 |
| Histogram | baseline | 23/23 | 46/46 | 单变量、多变量 |
| IsolationForest | baseline | 23/23 | 46/46 | 单变量、多变量 |
| KMeansDistance | baseline | 0/23 | 3/46 | 未纳入 |
| LocalOutlierFactor | baseline | 23/23 | 45/46 | 单变量 |
| LocalPolynomial | baseline | 0/23 | 4/46 | 未纳入 |
| MatrixProfile | baseline | 23/23 | 46/46 | 单变量、多变量 |
| MinimumCovarianceDeterminant | baseline | 0/23 | 4/46 | 未纳入 |
| NearestNeighbors | baseline | 23/23 | 45/46 | 单变量 |
| OneClassSVM | baseline | 0/23 | 3/46 | 未纳入 |
| PrincipalComponent | baseline | 23/23 | 45/46 | 单变量 |
| RobustPCA | baseline | 23/23 | 46/46 | 单变量、多变量 |
| SpectralResidual | baseline | 23/23 | 46/46 | 单变量、多变量 |
| StatisticalFeatures | baseline | 23/23 | 46/46 | 单变量、多变量 |
| StreamingMatrixProfile | baseline | 23/23 | 46/46 | 单变量、多变量 |
| DADA (zeroshot) | pretrained | 23/23 | 46/46 | 单变量、多变量 |
| Time-RCD (zeroshot) | pretrained | 23/23 | 46/46 | 单变量、多变量 |
| TSPulse-ZS (zeroshot) | pretrained | 0/23 | 38/46 | 未纳入 |
| TSPulse-FT (finetune) | pretrained | 0/23 | 1/46 | 未纳入 |

未纳入原因（单元缺失，不参与平均）：

- IsolationForest：单变量缺 SWaT、Stock、TODS、UCR、WSD、YAHOO。
- ClusterLocalOutlierFactor / LocalOutlierFactor / NearestNeighbors：多变量缺 LTDB-ECG1-ECG2-ECG3。
- PrincipalComponent：多变量缺 OPPORTUNITY。
- Copula、ExtendedIsolationForest、KMeansDistance、LocalPolynomial、MinimumCovarianceDeterminant、OneClassSVM、TSPulse-ZS、TSPulse-FT：两侧均未齐。

## 单变量平均（TSB-AD-U）

23 个单元、**350 条序列**。纳入 15 个方法。

### 阈值无关 / 体积指标

| 方法 | AUC-PR | AUC-ROC | R-AUC-PR | R-AUC-ROC | VUS-PR | VUS-ROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NearestNeighbors | 0.439 | 0.822 | **0.584** | 0.867 | **0.517** | 0.848 |
| StatisticalFeatures | 0.459 | **0.860** | 0.536 | **0.903** | 0.498 | **0.887** |
| Time-RCD (zeroshot) | **0.472** | 0.853 | 0.518 | 0.897 | 0.491 | 0.879 |
| PrincipalComponent | 0.375 | 0.714 | 0.454 | 0.797 | 0.414 | 0.758 |
| IsolationForest | 0.353 | 0.748 | 0.413 | 0.835 | 0.385 | 0.801 |
| Histogram | 0.313 | 0.709 | 0.379 | 0.785 | 0.349 | 0.747 |
| ClusterLocalOutlierFactor | 0.314 | 0.702 | 0.338 | 0.801 | 0.320 | 0.767 |
| MatrixProfile | 0.233 | 0.725 | 0.374 | 0.787 | 0.318 | 0.753 |
| DADA (zeroshot) | 0.309 | 0.712 | 0.331 | 0.802 | 0.314 | 0.770 |
| Fourier | 0.239 | 0.650 | 0.339 | 0.723 | 0.300 | 0.693 |
| RobustPCA | 0.283 | 0.641 | 0.307 | 0.756 | 0.287 | 0.715 |
| SpectralResidual | 0.279 | 0.704 | 0.305 | 0.802 | 0.287 | 0.769 |
| StreamingMatrixProfile | 0.158 | 0.677 | 0.263 | 0.755 | 0.227 | 0.722 |
| LocalOutlierFactor | 0.140 | 0.585 | 0.189 | 0.711 | 0.167 | 0.672 |
| ConnectivityOutlierFactor | 0.136 | 0.586 | 0.186 | 0.712 | 0.165 | 0.675 |

### F1 族指标

| 方法 | Standard-F1 | PA-F1 | Event-based-F1 | R-based-F1 | Affiliation-F |
| --- | ---: | ---: | ---: | ---: | ---: |
| NearestNeighbors | 0.505 | 0.713 | 0.623 | **0.466** | 0.904 |
| StatisticalFeatures | 0.499 | 0.829 | **0.681** | 0.407 | **0.906** |
| Time-RCD (zeroshot) | **0.528** | 0.752 | 0.645 | 0.443 | 0.890 |
| PrincipalComponent | 0.414 | 0.570 | 0.502 | 0.406 | 0.852 |
| IsolationForest | 0.392 | 0.580 | 0.485 | 0.370 | 0.847 |
| Histogram | 0.369 | 0.532 | 0.446 | 0.347 | 0.827 |
| ClusterLocalOutlierFactor | 0.359 | 0.731 | 0.599 | 0.333 | 0.859 |
| MatrixProfile | 0.314 | 0.593 | 0.413 | 0.295 | 0.822 |
| DADA (zeroshot) | 0.349 | 0.772 | 0.578 | 0.324 | 0.866 |
| Fourier | 0.315 | 0.578 | 0.439 | 0.269 | 0.833 |
| RobustPCA | 0.333 | 0.650 | 0.513 | 0.310 | 0.833 |
| SpectralResidual | 0.334 | **0.872** | 0.651 | 0.349 | 0.882 |
| StreamingMatrixProfile | 0.229 | 0.524 | 0.304 | 0.232 | 0.778 |
| LocalOutlierFactor | 0.213 | 0.626 | 0.405 | 0.221 | 0.787 |
| ConnectivityOutlierFactor | 0.203 | 0.729 | 0.491 | 0.237 | 0.807 |

## 多变量平均（TSB-AD-M）

46 个单元、**180 条序列**。纳入 11 个方法。

### 阈值无关 / 体积指标

| 方法 | AUC-PR | AUC-ROC | R-AUC-PR | R-AUC-ROC | VUS-PR | VUS-ROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| StatisticalFeatures | **0.388** | **0.765** | **0.410** | **0.810** | **0.390** | **0.791** |
| Time-RCD (zeroshot) | 0.313 | 0.726 | 0.379 | 0.792 | 0.355 | 0.765 |
| Fourier | 0.292 | 0.673 | 0.344 | 0.718 | 0.326 | 0.700 |
| DADA (zeroshot) | 0.298 | 0.676 | 0.312 | 0.732 | 0.300 | 0.707 |
| SpectralResidual | 0.281 | 0.667 | 0.306 | 0.715 | 0.293 | 0.691 |
| RobustPCA | 0.273 | 0.673 | 0.279 | 0.730 | 0.269 | 0.702 |
| IsolationForest | 0.205 | 0.703 | 0.275 | 0.773 | 0.246 | 0.745 |
| MatrixProfile | 0.173 | 0.592 | 0.235 | 0.662 | 0.217 | 0.624 |
| Histogram | 0.161 | 0.633 | 0.203 | 0.693 | 0.186 | 0.669 |
| StreamingMatrixProfile | 0.128 | 0.629 | 0.199 | 0.711 | 0.179 | 0.672 |
| ConnectivityOutlierFactor | 0.079 | 0.536 | 0.143 | 0.646 | 0.123 | 0.616 |

### F1 族指标

| 方法 | Standard-F1 | PA-F1 | Event-based-F1 | R-based-F1 | Affiliation-F |
| --- | ---: | ---: | ---: | ---: | ---: |
| StatisticalFeatures | **0.425** | **0.787** | 0.550 | **0.334** | **0.855** |
| Time-RCD (zeroshot) | 0.367 | 0.674 | 0.468 | 0.257 | 0.821 |
| Fourier | 0.353 | 0.743 | 0.511 | 0.301 | 0.817 |
| DADA (zeroshot) | 0.339 | 0.754 | 0.492 | 0.242 | 0.823 |
| SpectralResidual | 0.326 | **0.787** | 0.471 | 0.250 | 0.814 |
| RobustPCA | 0.324 | 0.744 | **0.557** | 0.325 | 0.836 |
| IsolationForest | 0.284 | 0.643 | 0.372 | 0.239 | 0.792 |
| MatrixProfile | 0.238 | 0.533 | 0.337 | 0.283 | 0.779 |
| Histogram | 0.244 | 0.662 | 0.397 | 0.243 | 0.792 |
| StreamingMatrixProfile | 0.201 | 0.483 | 0.259 | 0.223 | 0.752 |
| ConnectivityOutlierFactor | 0.140 | 0.670 | 0.322 | 0.175 | 0.763 |

## 简要观察

- 单变量：`Time-RCD (zeroshot)` 在 AUC-PR / Standard-F1 上最高，和 `StatisticalFeatures`、`NearestNeighbors` 同一档；`DADA (zeroshot)` 明显落后。
- 多变量：`StatisticalFeatures` 仍全面领先。`Time-RCD (zeroshot)` 收齐后 VUS-PR 0.355，高于 DADA（0.300）但仍低于 StatisticalFeatures（0.390）。预训练模型没有系统性超过简单统计特征。
- TSPulse zeroshot / finetune 尚未跑完，未进入表。

