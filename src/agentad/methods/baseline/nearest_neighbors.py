"""Nearest-neighbor distance detector.

Converts the series into sliding-window feature vectors — window 1 scores
raw points — and scores each window by the distance to its ``neighbors``
nearest other windows: the largest (k-th), mean, or median of those
distances. A query never counts as its own neighbor, so isolated windows
score high. Window scores center-align to points with boundary replication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from ._common import infer_period, knn_search, pad_window_scores, windowed_features
from .._utils import validate_series


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
        [
            _score_item(item, config, window, series.shape[1])
            for item in features
        ]
    )
