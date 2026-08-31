"""Local outlier factor detector.

LOF over the same windowed features as the nearest-neighbor baseline: the
local reachability density of a window is the inverse of its mean reach
distance, ``reach(i, j) = max(d(i, j), k-distance of j)``, and the score is
the mean neighbor-density ratio — windows sparser than their neighborhoods
score above one. The density adds a 1e-10 guard so duplicate windows stay
finite.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

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
from .._utils import validate_series
from ._common import infer_period, knn_search, pad_window_scores, windowed_features


@dataclass(frozen=True, slots=True)
class LocalOutlierFactorConfig:
    window: int | None = 1
    neighbors: int = 20
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.window is not None and self.window < 1:
            raise ValueError("window must be positive when provided")
        if self.neighbors < 1:
            raise ValueError("neighbors must be positive")


def _score_item(
    features: Tensor, config: LocalOutlierFactorConfig, window: int, length: int
) -> Tensor:
    count = features.shape[0]
    if count <= 1:
        return features.new_ones(length)
    k = min(config.neighbors, count - 1)
    distances, indices = knn_search(features, k, include_self=False)
    # The k-th column of a window's own neighbor distances is exactly its
    # k-distance; reach(i, j) takes the max with the k-distance of j.
    reach = torch.maximum(distances, distances[indices, k - 1])
    density = 1.0 / (reach.mean(dim=1) + 1e-10)
    factor = density[indices].mean(dim=1) / density
    return pad_window_scores(factor[None], window, length)[0]


@torch.inference_mode()
def score(series: Tensor, config: LocalOutlierFactorConfig) -> Tensor:
    window = config.window or infer_period(series[0, :, 0])
    series = validate_series(series, min_length=window)
    features = windowed_features(series, window, config.normalize)
    return torch.stack(
        [_score_item(item, config, window, series.shape[1]) for item in features]
    )


# TSB-AD evaluation entry (form C in tmp/method_train_eval_spec.md); the
# detector above is frozen and reused as-is.


# TSB-AD scores LOF on raw points at window 1 (run_LOF slidingWindow default)
# with n_neighbors=50 optimal on both tracks (HP_list.py
# Optimal_Uni_algo_HP_dict['LOF'], Optimal_Multi_algo_HP_dict['LOF']); the
# multivariate entry pins metric='euclidean', which the torch.cdist search
# already implements, and normalize stays at the Config default matching the
# TSB-AD zscore protocol.
DEFAULT_HP: Mapping[str, Any] = {"window": 1, "neighbors": 50}


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
    """Score every series of one artifact pair over its full train+test
    length and write the per-series metric table.

    Existing score files are kept when ``resume``; returns the
    ``write_metrics`` frame.
    """
    split = load_split(dataset_dir, artifact, partition=partition)
    unit = unit_dir(output_root, "LocalOutlierFactor", split)
    overrides = {**DEFAULT_HP, **(hp or {})}
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
        config = replace(LocalOutlierFactorConfig(), **overrides)
        scores = score(torch.from_numpy(full)[None], config)[0].numpy()
        if scores.shape != (len(full),) or not np.isfinite(scores).all():
            raise ValueError(f"{series_id}: scores must be finite and full-length")
        save_score(unit, series_id, scores)
    return write_metrics(split, unit)
