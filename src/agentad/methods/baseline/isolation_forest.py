"""Isolation-forest detector.

Converts the series into sliding-window feature vectors and grows an
ensemble of isolation trees on subsamples of at most ``sample_size``
windows: every internal node picks a random feature — each tree sees only
a ``max_features`` share of the columns — and a uniform split threshold
between the observed min and max. A window's score is the average
isolation depth normalized by the expected unsuccessful-search length of a
binary search tree, so windows that isolate near the root score close to
one. Window scores center-align to points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ._common import (
    expected_search_depth,
    infer_period,
    pad_window_scores,
    windowed_features,
)
from .._utils import validate_series


@dataclass(frozen=True, slots=True)
class IsolationForestConfig:
    window: int | None = None
    trees: int = 100
    sample_size: int = 256
    max_features: float = 1.0
    seed: int = 0
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.window is not None and self.window < 1:
            raise ValueError("window must be positive when provided")
        if self.trees < 1:
            raise ValueError("trees must be positive")
        if self.sample_size < 2:
            raise ValueError("sample_size must be at least two")
        if not 0.0 < self.max_features <= 1.0:
            raise ValueError("max_features must be in (0, 1]")


def _grow_tree(
    rows: np.ndarray,
    limit: int,
    columns: np.ndarray,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Grow one axis-aligned tree; returns parallel node arrays.

    ``feature < 0`` marks a leaf whose depth and subtree size carry the
    expected isolation depth of samples routed into it.
    """
    feature: list[int] = []
    threshold: list[float] = []
    left: list[int] = []
    right: list[int] = []
    depth: list[int] = []
    size: list[int] = []

    def build(indices: np.ndarray, level: int) -> int:
        node = len(feature)
        feature.append(-1)
        threshold.append(0.0)
        left.append(-1)
        right.append(-1)
        depth.append(level)
        size.append(indices.size)
        if level >= limit or indices.size <= 1:
            return node
        column = int(columns[generator.integers(columns.size)])
        values = rows[indices, column]
        low, high = float(values.min()), float(values.max())
        if not high > low:
            return node
        cut = float(generator.uniform(low, high))
        feature[node] = column
        threshold[node] = cut
        mask = values < cut
        left[node] = build(indices[mask], level + 1)
        right[node] = build(indices[~mask], level + 1)
        return node

    build(np.arange(rows.shape[0]), 0)
    return (
        np.array(feature),
        np.array(threshold),
        np.array(left),
        np.array(right),
        np.array(depth),
        np.array(size),
    )


def _expected_leaf_depth(leaf_size: Tensor) -> Tensor:
    """c(n) = 2H(n-1) - 2(n-1)/n with H approximated by ln + Euler's constant."""
    shifted = (leaf_size - 1).clamp_min(1)
    harmonic = torch.log(shifted.to(torch.float64)) + 0.5772156649
    return (2.0 * harmonic - 2.0 * shifted / leaf_size.clamp_min(1)).to(leaf_size.dtype)


def _tree_depths(
    rows: Tensor,
    feature: np.ndarray,
    threshold: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    depth: np.ndarray,
    size: np.ndarray,
) -> Tensor:
    """Route every row to its leaf and add the expected leaf depth."""
    count = rows.shape[0]
    current = torch.zeros(count, dtype=torch.long, device=rows.device)
    feature_t = torch.from_numpy(feature).to(rows.device)
    threshold_t = torch.from_numpy(threshold).to(rows.device)
    left_t = torch.from_numpy(left).to(rows.device)
    right_t = torch.from_numpy(right).to(rows.device)
    while True:
        active = feature_t[current] >= 0
        if not bool(active.any()):
            break
        safe = torch.where(active, current, torch.zeros_like(current))
        column = feature_t[safe]
        cut = threshold_t[safe]
        go_left = rows.gather(1, column[:, None]).squeeze(1) < cut
        current = torch.where(
            active,
            torch.where(go_left, left_t[safe], right_t[safe]),
            current,
        )
    leaf_depth = torch.from_numpy(depth).to(rows.device)[current]
    leaf_size = torch.from_numpy(size.astype(np.float64)).to(rows.device)[current]
    singleton = leaf_size <= 1
    expected = leaf_depth + _expected_leaf_depth(leaf_size)
    return torch.where(singleton, leaf_depth.to(rows.dtype), expected).to(rows.dtype)


def _score_item(
    features: Tensor, config: IsolationForestConfig, window: int, length: int
) -> Tensor:
    rows, width = features.shape
    if rows == 0:
        return features.new_zeros(length)
    sample = min(config.sample_size, rows)
    limit = int(math.ceil(math.log2(sample)))
    denominator = expected_search_depth(sample)
    generator = np.random.default_rng(config.seed)
    total = features.new_zeros(rows)
    for _ in range(config.trees):
        chosen = generator.choice(rows, size=sample, replace=False)
        columns = np.arange(width)
        if config.max_features < 1.0:
            kept = max(1, int(math.ceil(config.max_features * width)))
            columns = np.sort(generator.choice(width, size=kept, replace=False))
        tree = _grow_tree(features[chosen].cpu().numpy(), limit, columns, generator)
        depths = _tree_depths(features, *tree)
        total = total + depths
    average = total / config.trees
    window_scores = torch.pow(
        2.0, -average / max(denominator, 1e-12)
    )
    return pad_window_scores(window_scores[None], window, length)[0]


@torch.inference_mode()
def score(series: Tensor, config: IsolationForestConfig) -> Tensor:
    window = config.window or infer_period(series[0, :, 0])
    series = validate_series(series, min_length=window)
    features = windowed_features(series, window, config.normalize)
    return torch.stack(
        [
            _score_item(item, config, window, series.shape[1])
            for item in features
        ]
    )
