"""Robust-PCA detector via principal component pursuit.

Decomposes the ``[time, feature]`` matrix into a low-rank part and a sparse
part with the inexact augmented Lagrangian method, then scores each point
by its sparse residual magnitude summed over features: sustained
cross-feature structure lands in the low-rank part while point-wise
deviations survive as sparse corrections. All-zero feature columns are
pruned before the decomposition; the iteration stops once the
reconstruction error falls below ``tol`` — ``1e-7`` times the Frobenius
norm by default — or after ``max_iter`` rounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .._utils import validate_series


@dataclass(frozen=True, slots=True)
class RobustPCAConfig:
    max_iter: int = 1000
    tol: float | None = None
    zero_pruning: bool = True

    def __post_init__(self) -> None:
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        if self.tol is not None and self.tol <= 0:
            raise ValueError("tol must be positive when provided")


def _shrink(matrix: Tensor, threshold: Tensor) -> Tensor:
    return matrix.sign() * (matrix.abs() - threshold).clamp_min(0.0)


def _score_item(matrix: Tensor, config: RobustPCAConfig) -> Tensor:
    if config.zero_pruning:
        keep = matrix.any(dim=0)
        if not keep.any():
            return matrix.new_zeros(matrix.shape[0])
        matrix = matrix[:, keep]
    norm = torch.linalg.norm(matrix, ord=1)
    if norm == 0:
        return matrix.new_zeros(matrix.shape[0])

    rows, columns = matrix.shape
    mu = rows * columns / (4 * norm)
    lam = 1 / max(rows, columns) ** 0.5
    tolerance = config.tol or 1e-7 * torch.linalg.norm(matrix, ord="fro")

    sparse = torch.zeros_like(matrix)
    dual = torch.zeros_like(matrix)
    for _ in range(config.max_iter):
        left, singular, right = torch.linalg.svd(
            matrix - sparse + dual / mu, full_matrices=False
        )
        low_rank = (left * _shrink(singular, 1 / mu)) @ right
        sparse = _shrink(matrix - low_rank + dual / mu, lam / mu)
        dual = dual + mu * (matrix - low_rank - sparse)
        if torch.linalg.norm(matrix - low_rank - sparse, ord="fro") <= tolerance:
            break
    return sparse.abs().sum(dim=1)


@torch.inference_mode()
def score(series: Tensor, config: RobustPCAConfig) -> Tensor:
    series = validate_series(series)
    # The decomposition iterates SVDs; float64 keeps it stable over up to
    # max_iter rounds.
    items = series.to(torch.float64)
    scores = torch.stack([_score_item(item, config) for item in items])
    return scores.to(series.dtype)
