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


# ---------------------------------------------------------------------------
# TSB-AD evaluation entry point (spec form C: baseline evaluate).

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd

from ...benchmark import (
    has_score,
    load_split,
    save_score,
    unit_dir,
    write_metrics,
)

# TSB-AD optimal HP mapped to this config: max_iter -> max_iter.
# TSB-AD-M entry Optimal_Multi_algo_HP_dict['RobustPCA'] = {'max_iter':
# 1000}, equal to run_RobustPCA's default. TSB-AD-U does not benchmark
# RobustPCA, so univariate keeps the config defaults. run_RobustPCA has
# no window parameter: the decomposition scores points directly.
DEFAULT_HP: Mapping[str, Any] = {"max_iter": 1000}
_UNI_HP: Mapping[str, Any] = {}


def _config(
    full: np.ndarray, hp: Mapping[str, Any] | None
) -> RobustPCAConfig:
    """Config with the channel-selected TSB-AD optimal HP, overridden by ``hp``."""
    base = _UNI_HP if full.shape[1] == 1 else DEFAULT_HP
    return replace(RobustPCAConfig(), **{**base, **(hp or {})})


def evaluate(
    dataset_dir: str | Path,
    output_root: str | Path = "benchmarks/RobustPCA",
    *,
    artifact: str | None = None,
    partition: str | None = "Eva",
    seed: int = 2024,
    hp: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Score every series of one artifact on the concatenated train+test
    full length, store per-series scores, and return the per-series
    metric table; ``resume`` skips series with a stored score."""
    split = load_split(dataset_dir, artifact, partition=partition)
    unit = unit_dir(output_root, "RobustPCA", split)
    for series_id in split.test.ids:
        if resume and has_score(unit, series_id):
            continue
        L.seed_everything(seed)
        train_item = split.train[series_id] if split.train is not None else None
        test_item = split.test[series_id]
        full = (
            np.concatenate([train_item.data, test_item.data])
            if train_item is not None
            else np.asarray(test_item.data, dtype=np.float64)
        )
        scores = score(torch.from_numpy(full)[None], _config(full, hp))[0].numpy()
        if scores.shape != (full.shape[0],) or not np.isfinite(scores).all():
            raise RuntimeError(f"RobustPCA: invalid scores for {series_id}")
        save_score(unit, series_id, scores)
    return write_metrics(split, unit)
