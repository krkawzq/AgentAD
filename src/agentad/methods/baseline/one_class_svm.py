"""One-class support-vector detector.

Converts the series into sliding-window feature vectors and fits a
one-class SVM with an RBF kernel on the training prefix — or the whole
series when no prefix is given — by sequential minimal optimization of
the dual objective ``sum(a) - 0.5 a'Ka`` under ``0 <= a_i <= 1/(nu*n)``
and ``sum(a) = 1``: every step moves mass between the lowest-margin point
below the cap and the highest-margin point above zero by the exact segment
optimum. A window scores as the negative margin
``rho - sum(a_i K(x_i, x))``, so points beyond the boundary score positive.
The kernel matrix is quadratic, so fitting subsamples the training prefix
to ``max_train_points`` rows; scoring stays exact for every window. Window
scores center-align to points.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
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
    save_score,
    unit_dir,
    write_metrics,
)
from ...evaluation.period import find_period
from .._utils import validate_series
from ._common import infer_period, pad_window_scores, windowed_features


@dataclass(frozen=True, slots=True)
class OneClassSVMConfig:
    window: int | None = 1
    nu: float = 0.5
    gamma: float | None = None
    train_points: int | None = None
    max_train_points: int = 5000
    steps: int = 2000
    seed: int = 0
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.window is not None and self.window < 1:
            raise ValueError("window must be positive when provided")
        if not 0.0 < self.nu <= 1.0:
            raise ValueError("nu must be in (0, 1]")
        if self.gamma is not None and self.gamma <= 0:
            raise ValueError("gamma must be positive when provided")
        if self.train_points is not None and self.train_points < 1:
            raise ValueError("train_points must be positive when provided")
        if self.max_train_points < 2:
            raise ValueError("max_train_points must be at least two")
        if self.steps < 1:
            raise ValueError("steps must be positive")


def _kernel(rows: Tensor, gamma: float) -> Tensor:
    squared = torch.cdist(rows, rows).square()
    return torch.exp(-gamma * squared)


def _fit(
    rows: Tensor, config: OneClassSVMConfig, gamma: float
) -> tuple[Tensor, Tensor, float]:
    """SMO dual fit; returns support rows, weights, and rho.

    Starting from the cap on the most isolated points — already a sparse
    feasible point — each step transfers the exact segment optimum between
    the pair, so the coefficients reach the KKT solution where interior
    support vectors sit at margin one.
    """
    count = rows.shape[0]
    cap = 1.0 / (config.nu * count)
    kernel = _kernel(rows.to(torch.float64), gamma)
    order = kernel.sum(dim=1).argsort()
    coefficients = torch.zeros(count, dtype=kernel.dtype)
    take = min(int(1.0 / cap), count)
    coefficients[order[:take]] = cap
    if take < count:
        coefficients[order[take]] = 1.0 - take * cap
    margins = kernel @ coefficients
    for _ in range(config.steps):
        raisable = torch.where(
            coefficients < cap, margins, margins.new_full((), -torch.inf)
        )
        lowerable = torch.where(
            coefficients > 0, margins, margins.new_full((), torch.inf)
        )
        i, j = int(raisable.argmin()), int(lowerable.argmax())
        gain = margins[j] - margins[i]
        room = min(cap - coefficients[i], coefficients[j])
        denominator = kernel[i, i] + kernel[j, j] - 2.0 * kernel[i, j]
        step = min(gain / denominator, room) if denominator > 0 else room
        if gain <= 1e-9 or step <= 0:
            break
        coefficients[i] += step
        coefficients[j] -= step
        margins += step * (kernel[:, i] - kernel[:, j])
    margin = coefficients @ kernel
    # Interior support vectors sit at margin one; when every vector rests
    # on a bound, the mean margin over the support set stands in.
    free = (coefficients > 0) & (coefficients < cap)
    if bool(free.any()):
        rho = margin[free].mean()
    else:
        rho = margin[coefficients > 0].mean()
    support = coefficients > 0
    return rows[support], coefficients[support].to(rows.dtype), float(rho)


def _score_item(
    features: Tensor,
    config: OneClassSVMConfig,
    window: int,
    length: int,
) -> Tensor:
    rows, width = features.shape
    if rows == 0:
        return features.new_zeros(length)
    # Windows fully inside the training prefix fit the boundary.
    if config.train_points is not None:
        limit = max(2, config.train_points - window + 1)
        limit = min(limit, rows)
    else:
        limit = rows
    train = features[:limit]
    if train.shape[0] > config.max_train_points:
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        picked = (
            torch.randperm(train.shape[0], generator=generator)[
                : config.max_train_points
            ]
            .sort()
            .values
        )
        train = train[picked]
    gamma = config.gamma if config.gamma is not None else 1.0 / width
    support, weights, rho = _fit(train, config, gamma)
    squared = torch.cdist(features, support).square()
    margin = (torch.exp(-gamma * squared) * weights).sum(dim=1)
    window_scores = rho - margin
    return pad_window_scores(window_scores[None], window, length)[0]


@torch.inference_mode()
def score(series: Tensor, config: OneClassSVMConfig) -> Tensor:
    window = config.window or infer_period(series[0, :, 0])
    series = validate_series(series, min_length=window)
    features = windowed_features(series, window, config.normalize)
    return torch.stack(
        [_score_item(item, config, window, series.shape[1]) for item in features]
    )


# ---------------------------------------------------------------------------
# TSB-AD evaluation entry point (spec form C: baseline evaluate).


# TSB-AD optimal HP mapped to this config: nu -> nu; 'rbf' is this
# implementation's only kernel, and gamma keeps the sklearn-'auto'
# equivalent 1/n_features (config gamma=None). TSB-AD-M entry
# Optimal_Multi_algo_HP_dict['OCSVM'] = {'kernel': 'rbf', 'nu': 0.1} on
# run_OCSVM's slidingWindow=1 points. TSB-AD-U benchmarks the Sub_OCSVM
# variant, Optimal_Uni_algo_HP_dict['Sub_OCSVM'] = {'periodicity': 2,
# 'kernel': 'rbf'}: window = find_length_rank(data, rank=2) and the
# wrapper-default nu=0.5 (config default). OCSVM sits in TSB-AD's
# Semisupervise pool, so fitting is restricted to the normal train
# prefix (train_points).
DEFAULT_HP: Mapping[str, Any] = {"window": 1, "nu": 0.1}
_UNI_HP: Mapping[str, Any] = {}


def _config(
    full: np.ndarray,
    train_points: int | None,
    hp: Mapping[str, Any] | None,
) -> OneClassSVMConfig:
    """Config with the channel-selected TSB-AD optimal HP, overridden by ``hp``.

    Univariate runs take the Sub_OCSVM window (periodicity 2, i.e. the
    rank-2 period of the first channel); ``train_points`` restricts
    fitting to the normal train prefix.
    """
    base = dict(_UNI_HP if full.shape[1] == 1 else DEFAULT_HP)
    if full.shape[1] == 1:
        # TSB-AD's run_Sub_OCSVM estimates this window from data_test;
        # this evaluate feeds the concatenated train+test full length
        # instead.
        base["window"] = find_period(full[:, 0], rank=2)
    base["train_points"] = train_points
    return replace(OneClassSVMConfig(), **{**base, **(hp or {})})


def evaluate(
    dataset_dir: str | Path,
    output_root: str | Path = "benchmarks",
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
    unit = unit_dir(output_root, "OneClassSVM", split)
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
        train_points = len(train_item) if train_item is not None else None
        scores = score(torch.from_numpy(full)[None], _config(full, train_points, hp))[
            0
        ].numpy()
        if scores.shape != (full.shape[0],) or not np.isfinite(scores).all():
            raise RuntimeError(f"OneClassSVM: invalid scores for {series_id}")
        save_score(unit, series_id, scores)
    return write_metrics(split, unit)
