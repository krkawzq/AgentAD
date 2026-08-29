"""Cluster-based local outlier factor detector.

Standardizes the points — across features when multivariate, along time
when univariate — and clusters them with Lloyd k-means. Clusters are split
into large and small sets by the original's rule — sorting clusters by
size, the boundary is the smallest ``k`` with ``k*cluster_k >= alpha * n``
or a size jump of at least ``beta`` times the previous cluster — then a
point in a large cluster scores its distance to its own center, and a
point in a small cluster scores its distance to the nearest large-cluster
center. The original keeps points unwindowed, so each series point is one
sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .kmeans_distance import _kmeans_centers
from ._common import standardize_points
from .._utils import validate_series


@dataclass(frozen=True, slots=True)
class ClusterLocalOutlierFactorConfig:
    clusters: int = 8
    alpha: float = 0.9
    beta: float = 5.0
    weighted: bool = False
    seed: int = 0
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.clusters < 1:
            raise ValueError("clusters must be positive")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if self.beta <= 1:
            raise ValueError("beta must be greater than one")


def _split(
    sizes: Tensor, config: ClusterLocalOutlierFactorConfig
) -> tuple[Tensor, Tensor]:
    """Return large- and small-cluster indices following the original rule.

    Boundary candidates are split positions ``i`` where the leading ``i``
    clusters hold at least ``alpha`` of the points (alpha candidates) or a
    cluster is at least ``beta`` times larger than the next (beta
    candidates); the alpha∩beta intersection wins, then alpha, then beta.
    """
    count = sizes.numel()
    order = torch.argsort(sizes, descending=True)
    sorted_sizes = sizes[order]
    total = float(sizes.sum())
    alpha_list: list[int] = []
    beta_list: list[int] = []
    for position in range(1, count):
        if float(sorted_sizes[:position].sum()) >= config.alpha * total:
            alpha_list.append(position)
        if float(sorted_sizes[position - 1]) >= config.beta * max(
            float(sorted_sizes[position]), 1e-12
        ):
            beta_list.append(position)
    intersection = [i for i in alpha_list if i in beta_list]
    if intersection:
        boundary = intersection[0]
    elif alpha_list:
        boundary = alpha_list[0]
    elif beta_list:
        boundary = beta_list[0]
    else:
        # No cluster qualifies as large; the caller scores every point
        # against the nearest center, matching the degenerate branch.
        return order.new_empty(0), order
    return order[:boundary], order[boundary:]


def _score_item(
    points: Tensor, config: ClusterLocalOutlierFactorConfig
) -> Tensor:
    count = points.shape[0]
    k = min(config.clusters, count)
    centers = _kmeans_centers(
        points, clusters=k, iterations=50, seed=config.seed
    )
    distances = torch.cdist(points, centers)
    assignment = distances.argmin(dim=1)
    sizes = torch.bincount(assignment, minlength=k).to(points.dtype)
    large, _small = _split(sizes, config)
    scores = points.new_zeros(count)
    if large.numel():
        large_mask = torch.isin(assignment, large)
        own = centers[assignment[large_mask]]
        scores[large_mask] = (points[large_mask] - own).square().sum(dim=1).sqrt()
        small_mask = ~large_mask
        if small_mask.any():
            to_large = torch.cdist(points[small_mask], centers[large])
            scores[small_mask] = to_large.min(dim=1).values
    else:
        # No cluster qualifies as large; every point scores against the
        # nearest center, matching the degenerate branch of the original.
        scores = distances.min(dim=1).values
    if config.weighted:
        scores = scores * sizes[assignment]
    return scores


@torch.inference_mode()
def score(series: Tensor, config: ClusterLocalOutlierFactorConfig) -> Tensor:
    series = validate_series(series)
    outputs = []
    for item in series:
        points = standardize_points(item, normalize=config.normalize)
        outputs.append(_score_item(points, config))
    return torch.stack(outputs)
