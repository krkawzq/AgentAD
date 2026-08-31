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
        if isinstance(self.components, float) and not 0.0 < self.components <= 1.0:
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
    # full_matrices=False keeps only min(rows, width) directions; a count
    # derived from the width-long shares spectrum can exceed that on
    # wide-short window matrices (rows < width), where the excess shares
    # are all zero.
    count = min(_component_count(config, kept, shares), width, vectors.shape[0])
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
        [_score_item(item, config, window, series.shape[1]) for item in features]
    )


# ---------------------------------------------------------------------------
# TSB-AD evaluation entry point (spec form C: baseline evaluate).


# TSB-AD optimal HP mapped to this config: n_components -> components.
# TSB-AD-M entry Optimal_Multi_algo_HP_dict['PCA'] = {'n_components':
# 0.25} on run_PCA's slidingWindow=100 windows. TSB-AD-U benchmarks the
# Sub_PCA variant, Optimal_Uni_algo_HP_dict['Sub_PCA'] = {'periodicity':
# 1, 'n_components': None}: window = the rank-1 period (periodicity 1)
# and the config-default component count.
DEFAULT_HP: Mapping[str, Any] = {"window": 100, "components": 0.25}
_UNI_HP: Mapping[str, Any] = {"components": None}


def _config(full: np.ndarray, hp: Mapping[str, Any] | None) -> PrincipalComponentConfig:
    """Config with the channel-selected TSB-AD optimal HP, overridden by ``hp``.

    Univariate runs take the Sub_PCA window: the rank-1 period of the
    first channel.
    """
    base = dict(_UNI_HP if full.shape[1] == 1 else DEFAULT_HP)
    if full.shape[1] == 1:
        base["window"] = find_period(full[:, 0])
    return replace(PrincipalComponentConfig(), **{**base, **(hp or {})})


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
    """Score every series of one artifact on the concatenated train+test
    full length, store per-series scores, and return the per-series
    metric table; ``resume`` skips series with a stored score."""
    split = load_split(dataset_dir, artifact, partition=partition)
    unit = unit_dir(output_root, "baseline/PrincipalComponent", split)
    results = unit_dir(results_root, "baseline/PrincipalComponent", split)
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
        scores = score(torch.from_numpy(full)[None], _config(full, hp))[0].numpy()
        if scores.shape != (full.shape[0],) or not np.isfinite(scores).all():
            raise RuntimeError(f"PrincipalComponent: invalid scores for {series_id}")
        save_score(unit, series_id, scores)
    return write_metrics(split, unit, results)
