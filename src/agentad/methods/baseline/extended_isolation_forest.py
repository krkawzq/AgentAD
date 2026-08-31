"""Extended isolation-forest detector.

Scores every series point directly — no windows, matching the original —
after standardizing the points (across features when multivariate, along
time when univariate). Each tree subsamples at most ``sample_size`` points
and splits with a random hyperplane: a normal vector with ``extension_level
+ 1`` random components, passing through a point whose coordinates are
uniform between the per-dimension extrema. A point's score is the mean
traversal depth — with the average unsuccessful-search depth of the
samples stranded in its leaf added, as in the standard forest —
normalized by the expected unsuccessful-search depth of a
binary search tree, so points isolated near the root score close to one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ._common import expected_search_depth, standardize_points
from .._utils import validate_series


@dataclass(frozen=True, slots=True)
class ExtendedIsolationForestConfig:
    trees: int = 100
    sample_size: int = 256
    extension_level: int | None = None
    seed: int = 0
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.trees < 1:
            raise ValueError("trees must be positive")
        if self.sample_size < 2:
            raise ValueError("sample_size must be at least two")
        if self.extension_level is not None and self.extension_level < 0:
            raise ValueError("extension_level must be non-negative when provided")


def _grow_tree(
    rows: np.ndarray, limit: int, extension: int, generator: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Grow one oblique tree; returns node arrays of normals, points,
    children, depths, and subtree sizes, with ``left < 0`` marking a leaf."""
    dim = rows.shape[1]
    normals: list[np.ndarray] = []
    points: list[np.ndarray] = []
    left: list[int] = []
    right: list[int] = []
    depth: list[int] = []
    sizes: list[int] = []

    def build(indices: np.ndarray, level: int) -> int:
        node = len(left)
        normals.append(np.zeros(dim))
        points.append(np.zeros(dim))
        left.append(-1)
        right.append(-1)
        depth.append(level)
        sizes.append(int(indices.size))
        if level >= limit or indices.size <= 1:
            return node
        subset = rows[indices]
        normal = np.asarray(generator.standard_normal(dim))
        zeroed = generator.choice(dim, dim - extension - 1, replace=False)
        normal[zeroed] = 0.0
        point = generator.uniform(subset.min(axis=0), subset.max(axis=0))
        normals[node] = normal
        points[node] = point
        projection = (subset - point) @ normal
        mask = projection < 0
        if mask.all() or not mask.any():
            return node
        left[node] = build(indices[mask], level + 1)
        right[node] = build(indices[~mask], level + 1)
        return node

    build(np.arange(rows.shape[0]), 0)
    return (
        np.stack(normals),
        np.stack(points),
        np.array(left),
        np.array(right),
        np.array(depth),
        np.array(sizes),
    )


def _tree_depths(
    rows: Tensor,
    normals: np.ndarray,
    points: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    depth: np.ndarray,
    sizes: np.ndarray,
) -> Tensor:
    """Route every point to its leaf; the path length is the traversal depth
    plus the average unsuccessful-search depth of the leaf's samples."""
    count = rows.shape[0]
    current = torch.zeros(count, dtype=torch.long, device=rows.device)
    normals_t = torch.from_numpy(normals).to(rows.device, rows.dtype)
    points_t = torch.from_numpy(points).to(rows.device, rows.dtype)
    left_t = torch.from_numpy(left).to(rows.device)
    right_t = torch.from_numpy(right).to(rows.device)
    while True:
        active = left_t[current] >= 0
        if not bool(active.any()):
            break
        safe = torch.where(active, current, torch.zeros_like(current))
        projection = ((rows - points_t[safe]) * normals_t[safe]).sum(dim=1)
        current = torch.where(
            active,
            torch.where(projection < 0, left_t[safe], right_t[safe]),
            current,
        )
    depths = torch.from_numpy(depth).to(rows.device, rows.dtype)[current]
    leaf_sizes = torch.from_numpy(sizes).to(rows.device)[current]
    table = torch.tensor(
        [expected_search_depth(size) for size in range(int(sizes.max()) + 1)],
        dtype=rows.dtype,
        device=rows.device,
    )
    return depths + table[leaf_sizes]


def _score_item(
    points: Tensor, config: ExtendedIsolationForestConfig
) -> Tensor:
    rows, width = points.shape
    if rows == 0:
        return points.new_zeros(0)
    sample = min(config.sample_size, rows)
    limit = int(math.ceil(math.log2(sample)))
    extension = (
        width - 1 if config.extension_level is None else config.extension_level
    )
    extension = min(extension, width - 1)
    denominator = expected_search_depth(sample)
    generator = np.random.default_rng(config.seed)
    total = points.new_zeros(rows)
    for _ in range(config.trees):
        chosen = generator.choice(rows, size=sample, replace=False)
        tree = _grow_tree(
            points[chosen].cpu().numpy(), limit, extension, generator
        )
        total = total + _tree_depths(points, *tree)
    average = total / config.trees
    return torch.pow(2.0, -average / max(denominator, 1e-12))


@torch.inference_mode()
def score(series: Tensor, config: ExtendedIsolationForestConfig) -> Tensor:
    series = validate_series(series)
    outputs = []
    for item in series:
        points = standardize_points(item, normalize=config.normalize)
        outputs.append(_score_item(points, config))
    return torch.stack(outputs)


# ---------------------------------------------------------------------------
# TSB-AD evaluation entry point (spec form C: baseline evaluate).

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import lightning as L
import pandas as pd

from ...benchmark import (
    has_score,
    load_split,
    save_score,
    unit_dir,
    write_metrics,
)

# TSB-AD optimal HP mapped to this config: n_trees -> trees. TSB-AD-M
# entry Optimal_Multi_algo_HP_dict['EIF'] = {'n_trees': 50}. TSB-AD-U does
# not benchmark EIF, so univariate keeps the config defaults
# (run_EIF-default n_trees=100, full extension level). run_EIF has no
# window parameter: points are scored directly.
DEFAULT_HP: Mapping[str, Any] = {"trees": 50}
_UNI_HP: Mapping[str, Any] = {}


def _config(
    full: np.ndarray, hp: Mapping[str, Any] | None
) -> ExtendedIsolationForestConfig:
    """Config with the channel-selected TSB-AD optimal HP, overridden by ``hp``."""
    base = _UNI_HP if full.shape[1] == 1 else DEFAULT_HP
    return replace(ExtendedIsolationForestConfig(), **{**base, **(hp or {})})


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
    """Score every series of one artifact on the concatenated train+test
    full length, store per-series scores, and return the per-series
    metric table; ``resume`` skips series with a stored score."""
    split = load_split(dataset_dir, artifact, partition=partition)
    unit = unit_dir(output_root, "ExtendedIsolationForest", split)
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
            raise RuntimeError(
                f"ExtendedIsolationForest: invalid scores for {series_id}"
            )
        save_score(unit, series_id, scores)
    return write_metrics(split, unit)
