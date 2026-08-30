"""Minimum-covariance-determinant detector.

Converts the series into sliding-window feature vectors and fits a robust
location-scatter pair on the training prefix — or the whole series when no
prefix is given — by concentration steps: starting from random subsets of
``support_fraction`` of the rows, each step recomputes the subset mean and
covariance, rescores every row by its Mahalanobis distance, and keeps the
closest subset until the subset stops changing. The run with the smallest
determinant wins, and every row scores by its Mahalanobis distance to the
fitted pair; the covariance carries a scale-relative ridge because window
standardization leaves it rank-deficient. Window scores center-align to
points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ._common import infer_period, pad_window_scores, windowed_features
from .._utils import validate_series


@dataclass(frozen=True, slots=True)
class MinimumCovarianceDeterminantConfig:
    window: int | None = 1
    support_fraction: float | None = None
    train_points: int | None = None
    starts: int = 4
    seed: int = 0
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.window is not None and self.window < 1:
            raise ValueError("window must be positive when provided")
        if self.support_fraction is not None and not (
            0.0 < self.support_fraction <= 1.0
        ):
            raise ValueError("support_fraction must be in (0, 1]")
        if self.train_points is not None and self.train_points < 1:
            raise ValueError("train_points must be positive when provided")
        if self.starts < 1:
            raise ValueError("starts must be positive")


def _mahalanobis(
    rows: Tensor, mean: Tensor, precision: Tensor
) -> Tensor:
    centered = rows - mean
    return (centered @ precision * centered).sum(dim=1).sqrt()


def _precision(covariance: Tensor) -> Tensor:
    """Inverse with a scale-relative ridge.

    Window standardization leaves every row summing to zero, so the
    covariance is rank-deficient by construction and an absolute ridge
    vanishes next to its O(1) entries. Identical subset rows zero the
    covariance; every distance is then zero, which a zero precision
    reports directly.
    """
    scale = covariance.diagonal().sum()
    if not scale > 0:
        return torch.zeros_like(covariance)
    width = covariance.shape[0]
    eye = torch.eye(width, dtype=covariance.dtype, device=covariance.device)
    return torch.linalg.inv(covariance + eye * (1e-10 * scale / width))


def _fit_robust(
    rows: Tensor, config: MinimumCovarianceDeterminantConfig, h: int
) -> tuple[Tensor, Tensor]:
    """Concentration steps from random subsets; the smallest determinant wins."""
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    count = rows.shape[0]
    best: tuple[Tensor, Tensor, Tensor] | None = None
    for _ in range(config.starts):
        subset = torch.randperm(count, generator=generator)[:h].sort().values
        for _ in range(100):
            block = rows[subset]
            mean = block.mean(dim=0)
            centered = block - mean
            covariance = centered.T @ centered / max(h - 1, 1)
            distances = _mahalanobis(rows, mean, _precision(covariance))
            updated = distances.argsort()[:h].sort().values
            if torch.equal(updated, subset):
                break
            subset = updated
        block = rows[subset]
        mean = block.mean(dim=0)
        centered = block - mean
        covariance = centered.T @ centered / max(h - 1, 1)
        determinant = torch.linalg.det(covariance)
        if best is None or determinant < best[0]:
            best = (determinant, mean, covariance)
    if best is None:
        raise RuntimeError("MCD refinement produced no candidate subset")
    _, mean, covariance = best
    return mean, _precision(covariance)


def _score_item(
    features: Tensor,
    config: MinimumCovarianceDeterminantConfig,
    window: int,
    length: int,
) -> Tensor:
    rows, width = features.shape
    if rows == 0:
        return features.new_zeros(length)
    if config.support_fraction is None:
        h = math.ceil((rows + width + 1) / 2)
    else:
        h = math.ceil(config.support_fraction * rows)
    h = max(2, min(h, rows))
    # Windows fully inside the training prefix build the robust pair.
    if config.train_points is not None:
        limit = max(2, config.train_points - window + 1)
        limit = min(limit, rows)
    else:
        limit = rows
    mean, precision = _fit_robust(features[:limit], config, min(h, limit))
    window_scores = _mahalanobis(features, mean, precision)
    return pad_window_scores(window_scores[None], window, length)[0]


@torch.inference_mode()
def score(
    series: Tensor, config: MinimumCovarianceDeterminantConfig
) -> Tensor:
    window = config.window or infer_period(series[0, :, 0])
    series = validate_series(series, min_length=window)
    # Concentration steps iterate covariance inverses; float64 keeps them
    # stable and leaves the ridge in control of the rank-deficient
    # direction.
    features = windowed_features(series.to(torch.float64), window, config.normalize)
    scores = torch.stack(
        [
            _score_item(item, config, window, series.shape[1])
            for item in features
        ]
    )
    return scores.to(series.dtype)
