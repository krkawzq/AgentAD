"""Nearest-neighbor distance detector.

Converts the series into sliding-window feature vectors — window 1 scores
raw points — and scores each window by the distance to its ``neighbors``
nearest other windows: the largest (k-th), mean, or median of those
distances. A query never counts as its own neighbor, so isolated windows
score high. Window scores center-align to points with boundary replication.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping

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
from ._common import infer_period, knn_search, pad_window_scores, windowed_features


@dataclass(frozen=True, slots=True)
class NearestNeighborsConfig:
    window: int | None = 1
    neighbors: int = 10
    method: Literal["largest", "mean", "median"] = "largest"
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.window is not None and self.window < 1:
            raise ValueError("window must be positive when provided")
        if self.neighbors < 1:
            raise ValueError("neighbors must be positive")


def _score_item(
    features: Tensor, config: NearestNeighborsConfig, window: int, length: int
) -> Tensor:
    if features.shape[0] <= 1:
        return features.new_zeros(length)
    distances, _ = knn_search(features, config.neighbors, include_self=False)
    if config.method == "largest":
        window_scores = distances[:, -1]
    elif config.method == "mean":
        window_scores = distances.mean(dim=1)
    else:
        window_scores = distances.median(dim=1).values
    return pad_window_scores(window_scores[None], window, length)[0]


@torch.inference_mode()
def score(series: Tensor, config: NearestNeighborsConfig) -> Tensor:
    window = config.window or infer_period(series[0, :, 0])
    series = validate_series(series, min_length=window)
    features = windowed_features(series, window, config.normalize)
    return torch.stack(
        [_score_item(item, config, window, series.shape[1]) for item in features]
    )


# TSB-AD evaluation entry (form C in tmp/method_train_eval_spec.md); the
# detector above is frozen and reused as-is.


# TSB-AD optimal HP (HP_list.py), selected per channel count in evaluate:
# multivariate runs 'KNN' = {'n_neighbors': 50, 'method': 'mean'} at window 1
# (run_KNN slidingWindow default); the univariate track runs the windowed
# 'Sub_KNN' variant = {'periodicity': 2, 'n_neighbors': 50} with the method
# left at the run_Sub_KNN default 'largest', so the window is
# find_length_rank(data, rank=2), injected per series below.
DEFAULT_HP: Mapping[str, Any] = {"neighbors": 50, "method": "mean"}


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
    """Score every series of one artifact pair over its full train+test
    length and write the per-series metric table.

    Existing score files are kept when ``resume``; returns the
    ``write_metrics`` frame.
    """
    split = load_split(dataset_dir, artifact, partition=partition)
    unit = unit_dir(output_root, "baseline/NearestNeighbors", split)
    results = unit_dir(results_root, "baseline/NearestNeighbors", split)
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
        overrides = dict(DEFAULT_HP)
        if full.shape[1] == 1:
            # period.py picks rank>=2 as the rank-th strongest ACF peak,
            # while TSB-AD's find_length_rank takes the strongest peak past
            # the previous rank's peak; only rank=1 is exactly equivalent
            # (period.py docstring), so rank=2 approximates the TSB-AD rule.
            overrides.update(window=find_period(full[:, 0], rank=2), method="largest")
        else:
            overrides["window"] = 1
        overrides.update(hp or {})
        config = replace(NearestNeighborsConfig(), **overrides)
        scores = score(torch.from_numpy(full)[None], config)[0].numpy()
        if scores.shape != (len(full),) or not np.isfinite(scores).all():
            raise ValueError(f"{series_id}: scores must be finite and full-length")
        save_score(unit, series_id, scores)
    return write_metrics(split, unit, results)
