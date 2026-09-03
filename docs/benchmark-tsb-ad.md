# TSB-AD 评测结果（Eva 划分）

> **范围说明（2026-09-03）：** 以下数值来自 AgentAD 本地实现，并不等于当前公开
> TSB-AD 榜单的严格复现。公开榜单混合了 2024 原始 benchmark 和后续社区提交，
> 其中部分方法有提交专用协议。需要与公开榜单比较时，请使用固定官方源码版本的
> [严格复现流程](tsb-ad-leaderboard-reproduction.md)，不要把本页数值当作官方 parity。

快照时间：2026-09-02。结果来自 `results/benchmark/`。已完成配置 **28 / 33**（19 baseline + 4 预训练 + ScatterAD / CrossAD / Left / MMPAD / PaAno），共 1985 张 `metrics.csv`。表内只收录该划分下单元已齐的方法（单变量 23/23 且多变量 46/46）。

## 评测设定

- 划分：Eva。有效单元 69 个：单变量 **TSB-AD-U 23**、多变量 **TSB-AD-M 46**（Exathlon-05 / Exathlon-22 无 Eva 序列，不计入）。
- 平均：该划分下所有单元的 `metrics.csv` 按**序列**宏平均（每条序列等权，NaN 跳过），与 `EvaluationResult.summary(stat="mean")` 一致。
- 指标：`DEFAULT_METRICS` 11 项。表内按 **VUS-PR** 降序；每列最优值加粗（三位小数并列则同时加粗）。

## 方法覆盖

| 方法 | 类型 | TSB-AD-U | TSB-AD-M |
| --- | --- | ---: | ---: |
| ClusterLocalOutlierFactor | baseline | 23/23 | 46/46 |
| ConnectivityOutlierFactor | baseline | 23/23 | 46/46 |
| Copula | baseline | 23/23 | 46/46 |
| ExtendedIsolationForest | baseline | 23/23 | 46/46 |
| Fourier | baseline | 23/23 | 46/46 |
| Histogram | baseline | 23/23 | 46/46 |
| IsolationForest | baseline | 23/23 | 46/46 |
| KMeansDistance | baseline | 23/23 | 46/46 |
| LocalOutlierFactor | baseline | 23/23 | 46/46 |
| LocalPolynomial | baseline | 23/23 | 46/46 |
| MatrixProfile | baseline | 23/23 | 46/46 |
| MinimumCovarianceDeterminant | baseline | 23/23 | 46/46 |
| NearestNeighbors | baseline | 23/23 | 46/46 |
| OneClassSVM | baseline | 23/23 | 46/46 |
| PrincipalComponent | baseline | 23/23 | 46/46 |
| RobustPCA | baseline | 23/23 | 46/46 |
| SpectralResidual | baseline | 23/23 | 46/46 |
| StatisticalFeatures | baseline | 23/23 | 46/46 |
| StreamingMatrixProfile | baseline | 23/23 | 46/46 |
| DADA (zeroshot) | pretrained | 23/23 | 46/46 |
| Time-RCD (zeroshot) | pretrained | 23/23 | 46/46 |
| TSPulse (zeroshot) | pretrained | 23/23 | 46/46 |
| TSPulse (finetune) | pretrained | 23/23 | 46/46 |
| ScatterAD | method | 23/23 | 46/46 |
| CrossAD | method | 23/23 | 46/46 |
| Left | method | 23/23 | 46/46 |
| MMPAD | method | 23/23 | 46/46 |
| PaAno | method | 23/23 | 46/46 |
| AxonAD | method | 0/23 | 41/46 |
| xLSTMAD | method | 0/23 | 12/46 |
| CARLA | method | 0/23 | 0/46 |
| KanAD | method | 0/23 | 0/46 |
| AERCA | method | 0/23 | 0/46 |

## 单变量平均（TSB-AD-U）

23 个单元、**350 条序列**。

### 阈值无关 / 体积指标

| 方法 | AUC-PR | AUC-ROC | R-AUC-PR | R-AUC-ROC | VUS-PR | VUS-ROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NearestNeighbors | 0.439 | 0.822 | **0.584** | 0.867 | **0.517** | 0.848 |
| MinimumCovarianceDeterminant | 0.417 | 0.833 | 0.572 | 0.886 | 0.502 | 0.863 |
| StatisticalFeatures | 0.459 | **0.860** | 0.536 | **0.903** | 0.498 | **0.887** |
| Time-RCD (zeroshot) | **0.472** | 0.853 | 0.518 | 0.897 | 0.491 | 0.879 |
| TSPulse (finetune) | 0.380 | 0.833 | 0.565 | 0.886 | 0.485 | 0.866 |
| PaAno | 0.421 | 0.831 | 0.521 | 0.877 | 0.473 | 0.857 |
| MMPAD | 0.413 | 0.786 | 0.490 | 0.808 | 0.448 | 0.789 |
| OneClassSVM | 0.397 | 0.773 | 0.490 | 0.854 | 0.445 | 0.819 |
| CrossAD | 0.424 | 0.771 | 0.462 | 0.854 | 0.438 | 0.825 |
| PrincipalComponent | 0.375 | 0.714 | 0.454 | 0.797 | 0.414 | 0.758 |
| TSPulse (zeroshot) | 0.324 | 0.742 | 0.478 | 0.818 | 0.411 | 0.786 |
| Left | 0.392 | 0.762 | 0.434 | 0.843 | 0.406 | 0.814 |
| IsolationForest | 0.353 | 0.748 | 0.413 | 0.835 | 0.385 | 0.801 |
| Histogram | 0.313 | 0.709 | 0.379 | 0.785 | 0.349 | 0.747 |
| KMeansDistance | 0.309 | 0.728 | 0.383 | 0.771 | 0.348 | 0.746 |
| ClusterLocalOutlierFactor | 0.314 | 0.702 | 0.338 | 0.801 | 0.320 | 0.767 |
| MatrixProfile | 0.233 | 0.725 | 0.374 | 0.787 | 0.318 | 0.753 |
| Copula | 0.308 | 0.701 | 0.334 | 0.804 | 0.316 | 0.769 |
| DADA (zeroshot) | 0.309 | 0.712 | 0.331 | 0.802 | 0.314 | 0.770 |
| Fourier | 0.239 | 0.650 | 0.339 | 0.723 | 0.300 | 0.693 |
| ExtendedIsolationForest | 0.289 | 0.709 | 0.313 | 0.806 | 0.295 | 0.771 |
| RobustPCA | 0.283 | 0.641 | 0.307 | 0.756 | 0.287 | 0.715 |
| SpectralResidual | 0.279 | 0.704 | 0.305 | 0.802 | 0.287 | 0.769 |
| ScatterAD | 0.213 | 0.586 | 0.262 | 0.695 | 0.242 | 0.654 |
| StreamingMatrixProfile | 0.158 | 0.677 | 0.263 | 0.755 | 0.227 | 0.722 |
| LocalPolynomial | 0.144 | 0.655 | 0.211 | 0.688 | 0.181 | 0.657 |
| LocalOutlierFactor | 0.140 | 0.585 | 0.189 | 0.711 | 0.167 | 0.672 |
| ConnectivityOutlierFactor | 0.136 | 0.586 | 0.186 | 0.712 | 0.165 | 0.675 |

### F1 族指标

| 方法 | Standard-F1 | PA-F1 | Event-based-F1 | R-based-F1 | Affiliation-F |
| --- | ---: | ---: | ---: | ---: | ---: |
| NearestNeighbors | 0.505 | 0.713 | 0.623 | **0.466** | 0.904 |
| MinimumCovarianceDeterminant | 0.485 | 0.656 | 0.561 | 0.464 | 0.899 |
| StatisticalFeatures | 0.499 | 0.829 | **0.681** | 0.407 | 0.906 |
| Time-RCD (zeroshot) | **0.528** | 0.752 | 0.645 | 0.443 | 0.890 |
| TSPulse (finetune) | 0.456 | 0.668 | 0.562 | 0.428 | **0.915** |
| PaAno | 0.473 | 0.688 | 0.569 | 0.458 | 0.881 |
| MMPAD | 0.454 | 0.600 | 0.509 | 0.442 | 0.851 |
| OneClassSVM | 0.441 | 0.583 | 0.503 | 0.397 | 0.862 |
| CrossAD | 0.451 | 0.764 | 0.655 | 0.401 | 0.880 |
| PrincipalComponent | 0.414 | 0.570 | 0.502 | 0.406 | 0.852 |
| TSPulse (zeroshot) | 0.390 | 0.621 | 0.502 | 0.395 | 0.887 |
| Left | 0.426 | 0.773 | 0.639 | 0.407 | 0.880 |
| IsolationForest | 0.392 | 0.580 | 0.485 | 0.370 | 0.847 |
| Histogram | 0.369 | 0.532 | 0.446 | 0.347 | 0.827 |
| KMeansDistance | 0.358 | 0.565 | 0.430 | 0.359 | 0.822 |
| ClusterLocalOutlierFactor | 0.359 | 0.731 | 0.599 | 0.333 | 0.859 |
| MatrixProfile | 0.314 | 0.593 | 0.413 | 0.295 | 0.822 |
| Copula | 0.353 | 0.726 | 0.592 | 0.310 | 0.863 |
| DADA (zeroshot) | 0.349 | 0.772 | 0.578 | 0.324 | 0.866 |
| Fourier | 0.315 | 0.578 | 0.439 | 0.269 | 0.833 |
| ExtendedIsolationForest | 0.349 | 0.732 | 0.561 | 0.301 | 0.844 |
| RobustPCA | 0.333 | 0.650 | 0.513 | 0.310 | 0.833 |
| SpectralResidual | 0.334 | **0.872** | 0.651 | 0.349 | 0.882 |
| ScatterAD | 0.284 | 0.705 | 0.513 | 0.353 | 0.830 |
| StreamingMatrixProfile | 0.229 | 0.524 | 0.304 | 0.232 | 0.778 |
| LocalPolynomial | 0.203 | 0.320 | 0.251 | 0.161 | 0.784 |
| LocalOutlierFactor | 0.213 | 0.626 | 0.405 | 0.221 | 0.787 |
| ConnectivityOutlierFactor | 0.203 | 0.729 | 0.491 | 0.237 | 0.807 |

## 多变量平均（TSB-AD-M）

46 个单元、**180 条序列**。

### 阈值无关 / 体积指标

| 方法 | AUC-PR | AUC-ROC | R-AUC-PR | R-AUC-ROC | VUS-PR | VUS-ROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TSPulse (finetune) | 0.359 | 0.699 | **0.457** | 0.763 | **0.420** | 0.738 |
| MMPAD | 0.372 | **0.773** | 0.444 | 0.813 | 0.415 | 0.789 |
| StatisticalFeatures | **0.388** | 0.765 | 0.410 | 0.810 | 0.390 | 0.791 |
| TSPulse (zeroshot) | 0.334 | 0.661 | 0.423 | 0.732 | 0.389 | 0.703 |
| Left | 0.379 | 0.771 | 0.391 | **0.824** | 0.378 | **0.801** |
| PaAno | 0.334 | 0.735 | 0.407 | 0.794 | 0.376 | 0.768 |
| Fourier | 0.292 | 0.673 | 0.344 | 0.718 | 0.326 | 0.700 |
| CrossAD | 0.317 | 0.694 | 0.332 | 0.747 | 0.320 | 0.724 |
| Time-RCD (zeroshot) | 0.267 | 0.677 | 0.330 | 0.743 | 0.307 | 0.717 |
| DADA (zeroshot) | 0.298 | 0.676 | 0.312 | 0.732 | 0.300 | 0.707 |
| SpectralResidual | 0.281 | 0.667 | 0.306 | 0.715 | 0.293 | 0.691 |
| KMeansDistance | 0.249 | 0.691 | 0.312 | 0.760 | 0.289 | 0.730 |
| PrincipalComponent | 0.241 | 0.675 | 0.298 | 0.740 | 0.269 | 0.705 |
| RobustPCA | 0.273 | 0.673 | 0.279 | 0.730 | 0.269 | 0.702 |
| ClusterLocalOutlierFactor | 0.270 | 0.666 | 0.277 | 0.729 | 0.268 | 0.699 |
| MinimumCovarianceDeterminant | 0.235 | 0.646 | 0.285 | 0.707 | 0.265 | 0.681 |
| OneClassSVM | 0.234 | 0.579 | 0.279 | 0.669 | 0.261 | 0.639 |
| IsolationForest | 0.205 | 0.703 | 0.275 | 0.773 | 0.246 | 0.745 |
| ScatterAD | 0.191 | 0.554 | 0.232 | 0.645 | 0.220 | 0.608 |
| ExtendedIsolationForest | 0.217 | 0.672 | 0.231 | 0.733 | 0.220 | 0.704 |
| MatrixProfile | 0.173 | 0.592 | 0.235 | 0.662 | 0.217 | 0.624 |
| Copula | 0.203 | 0.652 | 0.209 | 0.711 | 0.199 | 0.682 |
| Histogram | 0.161 | 0.633 | 0.203 | 0.693 | 0.186 | 0.669 |
| LocalPolynomial | 0.153 | 0.615 | 0.205 | 0.619 | 0.180 | 0.594 |
| StreamingMatrixProfile | 0.128 | 0.629 | 0.199 | 0.711 | 0.179 | 0.672 |
| NearestNeighbors | 0.132 | 0.502 | 0.191 | 0.592 | 0.171 | 0.569 |
| LocalOutlierFactor | 0.093 | 0.532 | 0.149 | 0.601 | 0.130 | 0.581 |
| ConnectivityOutlierFactor | 0.079 | 0.536 | 0.143 | 0.646 | 0.123 | 0.616 |

### F1 族指标

| 方法 | Standard-F1 | PA-F1 | Event-based-F1 | R-based-F1 | Affiliation-F |
| --- | ---: | ---: | ---: | ---: | ---: |
| TSPulse (finetune) | 0.419 | 0.718 | 0.546 | 0.408 | 0.862 |
| MMPAD | 0.414 | 0.593 | 0.501 | 0.403 | 0.829 |
| StatisticalFeatures | **0.425** | 0.787 | 0.550 | 0.334 | 0.855 |
| TSPulse (zeroshot) | 0.394 | 0.719 | 0.529 | **0.409** | 0.852 |
| Left | 0.417 | **0.856** | **0.631** | 0.382 | **0.864** |
| PaAno | 0.384 | 0.639 | 0.493 | 0.370 | 0.843 |
| Fourier | 0.353 | 0.743 | 0.511 | 0.301 | 0.817 |
| CrossAD | 0.365 | 0.806 | 0.580 | 0.286 | 0.840 |
| Time-RCD (zeroshot) | 0.319 | 0.680 | 0.449 | 0.249 | 0.821 |
| DADA (zeroshot) | 0.339 | 0.754 | 0.492 | 0.242 | 0.823 |
| SpectralResidual | 0.326 | 0.787 | 0.471 | 0.250 | 0.814 |
| KMeansDistance | 0.307 | 0.668 | 0.475 | 0.333 | 0.816 |
| PrincipalComponent | 0.312 | 0.505 | 0.369 | 0.331 | 0.788 |
| RobustPCA | 0.324 | 0.744 | 0.557 | 0.325 | 0.836 |
| ClusterLocalOutlierFactor | 0.314 | 0.656 | 0.463 | 0.309 | 0.818 |
| MinimumCovarianceDeterminant | 0.302 | 0.493 | 0.412 | 0.260 | 0.810 |
| OneClassSVM | 0.282 | 0.481 | 0.408 | 0.276 | 0.801 |
| IsolationForest | 0.284 | 0.643 | 0.372 | 0.239 | 0.792 |
| ScatterAD | 0.248 | 0.772 | 0.430 | 0.263 | 0.808 |
| ExtendedIsolationForest | 0.282 | 0.730 | 0.419 | 0.255 | 0.805 |
| MatrixProfile | 0.238 | 0.533 | 0.337 | 0.283 | 0.779 |
| Copula | 0.268 | 0.711 | 0.411 | 0.244 | 0.799 |
| Histogram | 0.244 | 0.662 | 0.397 | 0.243 | 0.792 |
| LocalPolynomial | 0.220 | 0.329 | 0.257 | 0.194 | 0.757 |
| StreamingMatrixProfile | 0.201 | 0.483 | 0.259 | 0.223 | 0.752 |
| NearestNeighbors | 0.184 | 0.523 | 0.398 | 0.181 | 0.624 |
| LocalOutlierFactor | 0.147 | 0.453 | 0.287 | 0.129 | 0.597 |
| ConnectivityOutlierFactor | 0.140 | 0.670 | 0.322 | 0.175 | 0.763 |

## 简要观察

- **单变量**：第一档仍是简单方法。VUS-PR 前三是 `NearestNeighbors`（0.517）、`MinimumCovarianceDeterminant`（0.502）、`StatisticalFeatures`（0.498）。新齐的方法里 `PaAno`（0.473）最接近这一档，`MMPAD`（0.448）和 `CrossAD`（0.438）随后；`Left`（0.406）弱一截，`ScatterAD`（0.242）明显落后。预训练里 `Time-RCD (zeroshot)`（0.491）仍能打平，`TSPulse` 微调（0.485）接近，DADA 落后。
- **多变量**：`TSPulse (finetune)` 仍最高（VUS-PR 0.420）。新齐的 **`MMPAD`（0.415）超过 `StatisticalFeatures`（0.390）**，是目前除 TSPulse 微调外最强的非简单方法。`Left`（0.378）和 `PaAno`（0.376）接近统计特征；`CrossAD`（0.320）与 Time-RCD / DADA 一档；`ScatterAD`（0.220）落后。
- 尚未进表：`AxonAD`（多变量 41/46，单变量 0/23）、`xLSTMAD`（多变量 12/46）、`CARLA` / `KanAD` / `AERCA`（尚未开工）。

