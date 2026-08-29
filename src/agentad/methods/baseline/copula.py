"""Copula-based outlier detector.

Each feature column contributes left- and right-tail depths from its
empirical CDF — ties receive the probability of the largest equal value —
combined by the skewness-aware rule ``U_skew = -U_l*sign(skew-1) +
U_r*sign(skew+1)`` and reduced to ``max(U_skew, (U_l+U_r)/2)``; the point
score sums the per-feature depths. Univariate input is standardized along
time; multivariate input is standardized per point across features.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ._common import zscore
from .._utils import validate_series


@dataclass(frozen=True, slots=True)
class CopulaConfig:
    normalize: bool = True


def _column_ecdf(values: Tensor) -> Tensor:
    """ECDF per column with ties mapped to the largest equal probability."""
    count = values.shape[0]
    probabilities = torch.linspace(
        1.0 / count, 1.0, count, device=values.device, dtype=values.dtype
    )
    columns = []
    for column in values.T:
        _, inverse, counts = torch.unique(
            column, return_inverse=True, return_counts=True
        )
        last = counts.cumsum(0) - 1
        columns.append(probabilities[last][inverse])
    return torch.stack(columns, dim=1)


def _score_item(matrix: Tensor, config: CopulaConfig) -> Tensor:
    if config.normalize:
        if matrix.shape[1] == 1:
            matrix = zscore(matrix, dim=0, ddof=0)
        else:
            matrix = zscore(matrix, dim=1, ddof=1)
    left = -torch.log(_column_ecdf(matrix))
    right = -torch.log(_column_ecdf(-matrix))
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    moment2 = centered.square().mean(dim=0)
    moment3 = centered.pow(3).mean(dim=0)
    skewness = torch.where(
        moment2 > 0, moment3 / moment2.pow(1.5), torch.zeros_like(moment3)
    ).sign()
    u_skew = -left * (skewness - 1).sign() + right * (skewness + 1).sign()
    return torch.maximum(u_skew, (left + right) / 2).sum(dim=1)


@torch.inference_mode()
def score(series: Tensor, config: CopulaConfig) -> Tensor:
    series = validate_series(series)
    return torch.stack([_score_item(item, config) for item in series])
