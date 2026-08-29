"""Histogram-density outlier detector.

Builds a fixed-bin density histogram per windowed feature column and
scores each window by the negative sum of ``log2(density + alpha)`` over
features: dense regions contribute large (near-zero) log densities, while
any feature landing in an empty or thin bin dominates the sum. Window
scores center-align to points with boundary replication.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ._common import infer_period, pad_window_scores, windowed_features
from .._utils import validate_series


@dataclass(frozen=True, slots=True)
class HistogramConfig:
    window: int | None = 1
    bins: int = 10
    alpha: float = 0.1
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.window is not None and self.window < 1:
            raise ValueError("window must be positive when provided")
        if self.bins < 2:
            raise ValueError("bins must be at least two")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")


def _histograms(features: Tensor, bins: int, alpha: float) -> tuple[Tensor, Tensor]:
    """Per-column log2 density scores and per-sample bin assignments."""
    count, width = features.shape
    minimum = features.min(dim=0).values
    maximum = features.max(dim=0).values
    # A zero-range column widens to (min - 0.5, max + 0.5) so every bin
    # stays non-degenerate.
    low = torch.where(minimum == maximum, minimum - 0.5, minimum)
    high = torch.where(minimum == maximum, maximum + 0.5, maximum)
    edges = (
        torch.linspace(
            0.0, 1.0, bins + 1, device=features.device, dtype=features.dtype
        )[:, None]
        * (high - low)[None, :]
        + low[None, :]
    )  # [bins+1, width]
    widths = (edges[1:] - edges[:-1]).clamp_min(torch.finfo(features.dtype).tiny)

    # Values on the lower edge fall into bin 0 and on the upper edge into
    # the last bin; interior values take the bin below the edge they hit.
    indices = torch.searchsorted(
        edges.T.contiguous(), features.T.contiguous()
    ).T  # [count, width]
    assignment = (indices - 1).clamp(0, bins - 1)
    counts = (
        assignment[None, :, :]
        == torch.arange(bins, device=features.device)[:, None, None]
    ).sum(dim=1).to(features.dtype)  # [bins, width]
    density = counts / (count * widths)
    return torch.log2(density + alpha), assignment


def _score_item(
    features: Tensor, config: HistogramConfig, window: int, length: int
) -> Tensor:
    bin_scores, assignment = _histograms(features, config.bins, config.alpha)
    window_scores = -torch.gather(bin_scores, 0, assignment).sum(dim=1)
    if features.shape[0] == length:
        return window_scores
    return pad_window_scores(window_scores[None], window, length)[0]


@torch.inference_mode()
def score(series: Tensor, config: HistogramConfig) -> Tensor:
    window = config.window or infer_period(series[0, :, 0])
    series = validate_series(series, min_length=window)
    features = windowed_features(series, window, config.normalize)
    return torch.stack(
        [
            _score_item(item, config, window, series.shape[1])
            for item in features
        ]
    )
