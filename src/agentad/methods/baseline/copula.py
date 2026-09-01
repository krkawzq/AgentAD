"""Copula-based outlier detector.

Each feature column contributes left- and right-tail depths from its
empirical CDF — ties receive the probability of the largest equal value —
combined by the skewness-aware rule ``U_skew = -U_l*sign(skew-1) +
U_r*sign(skew+1)`` and reduced to ``max(U_skew, (U_l+U_r)/2)``; the point
score sums the per-feature depths. Univariate input is standardized along
time; multivariate input is standardized per point across features.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from ...benchmark import (
    has_score,
    load_split,
    prepare_run,
    save_score,
    unit_dir,
    write_metrics,
)
from .._utils import validate_series
from ._common import zscore


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
        # Univariate: normalize along time like the other univariate
        # baselines; upstream COPOD always uses axis=1, which collapses a
        # [T, 1] window to all-zero scores (and TSB-AD has no uni entry).
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


# ---------------------------------------------------------------------------
# TSB-AD baseline evaluation (spec: tmp/method_train_eval_spec.md §4 shape C)
# ---------------------------------------------------------------------------


METHOD_NAME = "Copula"

# COPOD is parameter-free: its only tuned entry, Optimal_Multi_algo_HP_dict
# ["COPOD"] = {"n_jobs": 1} (forks/TSB-AD/TSB_AD/HP_list.py), is execution
# parallelism with no counterpart in CopulaConfig, and the univariate HP
# dicts hold no COPOD entry. normalize=True reproduces the COPOD
# constructor default; identical HPs apply to uni and multi.
DEFAULT_HP: Mapping[str, Any] = {"normalize": True}


def _check_scores(scores: np.ndarray, length: int) -> None:
    if scores.shape != (length,):
        raise ValueError(f"score length {scores.shape[0]} != series length {length}")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")


def evaluate(
    dataset_dir: str | Path,
    output_root: str | Path = "outputs/benchmark",
    *,
    artifact: str | None = None,
    results_root: str | Path = "results/benchmark",
    partition: str | None = "Eva",
    seed: int = 2024,
    hp: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Score every series of one artifact on its train+test full length.

    ``hp`` overrides :data:`DEFAULT_HP` for the frozen CopulaConfig.
    Scores are stored under ``output_root`` (already-scored series are
    skipped when ``resume``) and the per-series metric table is returned.
    """
    split = load_split(dataset_dir, artifact, partition=partition)
    unit = unit_dir(output_root, f"baseline/{METHOD_NAME}", split)
    results = unit_dir(results_root, f"baseline/{METHOD_NAME}", split)
    prepare_run(
        unit,
        split,
        method=f"baseline/{METHOD_NAME}",
        partition=partition,
        seed=seed,
        hp=hp,
        resume=resume,
        implementation_file=__file__,
        device="cpu",
    )
    config = CopulaConfig(**{**DEFAULT_HP, **(hp or {})})
    for series_id in split.test.ids:
        if resume and has_score(unit, series_id):
            continue
        L.seed_everything(seed)
        train_item = split.train[series_id] if split.train is not None else None
        test_item = split.test[series_id]
        full = (
            np.concatenate([train_item.data, test_item.data])
            if train_item is not None
            else test_item.data
        )
        scores = score(torch.from_numpy(full)[None], config)[0].numpy()
        _check_scores(scores, len(full))
        save_score(unit, series_id, scores)
    return write_metrics(split, unit, results)
