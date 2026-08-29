"""K-means distance detector.

Converts the series into sliding-window feature vectors, standardizes every
window across its own features — single-feature windows keep their raw
scale — clusters the windows into ``clusters`` centers with Lloyd
iterations from k-means++ seeds, and scores each window by its euclidean
distance to the assigned center. Window scores map back to points by
averaging every window that covers the point, matching the original
reverse-windowing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ._common import infer_period, zscore
from .._utils import sliding_windows, validate_series


@dataclass(frozen=True, slots=True)
class KMeansDistanceConfig:
    window: int | None = None
    clusters: int = 20
    iterations: int = 50
    seed: int = 0
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.window is not None and self.window < 1:
            raise ValueError("window must be positive when provided")
        if self.clusters < 1:
            raise ValueError("clusters must be positive")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")


def _windows(series: Tensor, window: int) -> Tensor:
    """Flatten ``[batch, time, feature]`` into window rows ``[count, window*feature]``."""
    stacked = sliding_windows(series, window)
    batch, count = stacked.shape[0], stacked.shape[1]
    return stacked.permute(0, 1, 3, 2).reshape(batch * count, -1)


def _kmeans_centers(
    rows: Tensor, *, clusters: int, iterations: int, seed: int
) -> Tensor:
    """Lloyd iterations from k-means++ seeds; ties keep the earlier center."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    count, width = rows.shape
    k = min(clusters, count)
    centers = torch.empty((k, width), dtype=rows.dtype, device=rows.device)
    first = torch.randint(count, (1,), generator=generator).item()
    centers[0] = rows[first]
    closest = (rows - centers[0]).square().sum(dim=1)
    for index in range(1, k):
        total = closest.sum()
        if not total > 0:
            centers[index] = rows[torch.randint(count, (1,), generator=generator).item()]
        else:
            pick = torch.multinomial(closest, 1, generator=generator).item()
            centers[index] = rows[pick]
        closest = torch.minimum(closest, (rows - centers[index]).square().sum(dim=1))
    for _ in range(iterations):
        distances = torch.cdist(rows, centers)
        assignment = distances.argmin(dim=1)
        updated = centers.clone()
        for index in range(k):
            members = rows[assignment == index]
            if members.numel():
                updated[index] = members.mean(dim=0)
        if torch.equal(updated, centers):
            break
        centers = updated
    return centers


def _overlap_average(
    window_scores: Tensor, window: int, length: int
) -> Tensor:
    """Average the scores of all windows covering each point; the tail padding
    beyond the last window start takes the last window's score."""
    scores = window_scores
    tail = length - scores.numel()
    if tail > 0:
        scores = torch.cat((scores, scores[-1:].expand(tail)))
    padded = torch.cat((scores, scores[-1:].expand(window - 1)))
    windows = padded.unfold(0, window, 1)
    return windows.mean(dim=1)[:length]


@torch.inference_mode()
def score(series: Tensor, config: KMeansDistanceConfig) -> Tensor:
    window = config.window or infer_period(series[0, :, 0])
    series = validate_series(series, min_length=window)
    batch = series.shape[0]
    outputs: list[Tensor] = []
    for item in series:
        rows = _windows(item[None], window)
        # A one-column row — window 1 on univariate input — has no spread
        # to standardize and keeps its raw scale.
        if config.normalize and rows.shape[1] > 1:
            rows = zscore(rows, dim=1, ddof=1)
        if rows.shape[0] == 0:
            outputs.append(rows.new_zeros(series.shape[1]))
            continue
        centers = _kmeans_centers(
            rows,
            clusters=config.clusters,
            iterations=config.iterations,
            seed=config.seed,
        )
        assignment = torch.cdist(rows, centers).argmin(dim=1)
        window_scores = (rows - centers[assignment]).square().sum(dim=1).sqrt()
        outputs.append(
            _overlap_average(window_scores, window, series.shape[1])
        )
    return torch.stack(outputs)
