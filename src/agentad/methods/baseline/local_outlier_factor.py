"""Local outlier factor detector.

LOF over the same windowed features as the nearest-neighbor baseline: the
local reachability density of a window is the inverse of its mean reach
distance, ``reach(i, j) = max(d(i, j), k-distance of j)``, and the score is
the mean neighbor-density ratio — windows sparser than their neighborhoods
score above one. The density adds a 1e-10 guard so duplicate windows stay
finite.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ._common import infer_period, knn_search, pad_window_scores, windowed_features
from .._utils import validate_series


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
        [
            _score_item(item, config, window, series.shape[1])
            for item in features
        ]
    )
