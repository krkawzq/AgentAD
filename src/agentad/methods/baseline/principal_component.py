"""Principal-component outlier detector.

Converts the series into sliding-window feature vectors, standardizes them
twice — per the original pipeline, first over time (univariate) or over the
window (multivariate), then per feature — prunes all-zero feature columns,
and fits a full-rank PCA basis. A window scores as the sum of its euclidean
distances to every eigenvector, each divided by the eigenvector's
explained-variance share, so deviation along low-variance directions
dominates the score. Window scores center-align to points.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ._common import infer_period, pad_window_scores, windowed_features
from .._utils import validate_series


@dataclass(frozen=True, slots=True)
class PrincipalComponentConfig:
    window: int | None = None
    components: int | float | None = None
    normalize: bool = True
    standardize: bool = True
    zero_pruning: bool = True

    def __post_init__(self) -> None:
        if self.window is not None and self.window < 1:
            raise ValueError("window must be positive when provided")
        if isinstance(self.components, int) and self.components < 1:
            raise ValueError("components must be positive when integral")
        if (
            isinstance(self.components, float)
            and not 0.0 < self.components <= 1.0
        ):
            raise ValueError("fractional components must be in (0, 1]")


def _basis(
    features: Tensor, config: PrincipalComponentConfig
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Standardize ``features`` and return the trailing PCA basis.

    Returns the standardized matrix (with zero columns pruned), the fitted
    per-feature standardization, the eigenvectors in descending-variance
    order, and their explained-variance shares.
    """
    rows, width = features.shape
    mean = features.mean(dim=0, keepdim=True)
    scale = features.std(dim=0, correction=0, keepdim=True)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    standardized = (features - mean) / scale if config.standardize else features
    keep = torch.ones(width, dtype=torch.bool, device=features.device)
    if config.zero_pruning:
        keep = standardized.abs().any(dim=0)
    standardized = standardized[:, keep]
    centered = standardized - standardized.mean(dim=0, keepdim=True)
    _, _, vectors = torch.linalg.svd(centered, full_matrices=False)
    variance = centered.square().sum(dim=0) / max(rows - 1, 1)
    total = variance.sum()
    shares = variance / total.clamp_min(1e-12)
    return standardized, keep, vectors, shares


def _component_count(
    config: PrincipalComponentConfig, width: int, shares: Tensor
) -> int:
    if config.components is None:
        return width
    if isinstance(config.components, int):
        return min(config.components, width)
    # sklearn semantics: the smallest count whose cumulative explained
    # variance reaches the requested fraction.
    cumulative = shares.cumsum(0)
    count = int((cumulative < config.components).sum()) + 1
    return max(1, min(count, width))


def _score_item(
    features: Tensor, config: PrincipalComponentConfig, window: int, length: int
) -> Tensor:
    standardized, keep, vectors, shares = _basis(features, config)
    width = standardized.shape[1]
    kept = int(keep.sum())
    count = min(_component_count(config, kept, shares), width)
    if count <= 0 or width == 0:
        return standardized.new_zeros(length)
    # Only the leading ``count`` eigenvectors define the fitted basis; the
    # score keeps every fitted eigenvector, matching the original, which
    # weights each by its explained-variance share.
    basis = vectors[:count]
    weights = shares[:count].clamp_min(1e-12)
    distances = torch.cdist(standardized, basis)
    window_scores = (distances / weights).sum(dim=1)
    return pad_window_scores(window_scores[None], window, length)[0]


@torch.inference_mode()
def score(series: Tensor, config: PrincipalComponentConfig) -> Tensor:
    window = config.window or infer_period(series[0, :, 0])
    series = validate_series(series, min_length=window)
    features = windowed_features(series, window, config.normalize)
    return torch.stack(
        [
            _score_item(item, config, window, series.shape[1])
            for item in features
        ]
    )
