"""Connectivity-based outlier factor detector.

Runs on the raw ``[time, feature]`` points. The k nearest neighbors of a
point — including itself at distance zero — form a chaining path; each new
path member pays the cost of linking to the nearest earlier member, costs
are weighted by ``2(k-j)/(k(k+1))`` and summed into an average chaining
distance. The score compares a point's chaining distance with the mean of
its neighbors', so points on poorly connected paths stand out.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ._common import knn_search
from .._utils import validate_series


@dataclass(frozen=True, slots=True)
class ConnectivityOutlierFactorConfig:
    neighbors: int = 30

    def __post_init__(self) -> None:
        if self.neighbors < 1:
            raise ValueError("neighbors must be positive")


def _score_item(features: Tensor, config: ConnectivityOutlierFactorConfig) -> Tensor:
    count = features.shape[0]
    k = min(config.neighbors, count - 1)
    if count <= 1 or k < 1:
        return torch.zeros(count, device=features.device, dtype=features.dtype)

    # The path holds k+1 members including the query itself.
    _, path = knn_search(features, k + 1, include_self=True)
    member = torch.arange(k + 1, device=features.device)
    target = torch.arange(k, device=features.device)
    weight = 2.0 * (k - target) / (k * (k + 1))

    chunk = max(1, 2**22 // max(k * (k + 1), 1))
    chaining = features.new_empty(count)
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        neighbors = features[path[start:stop]]  # [c, k+1, d]
        targets = neighbors[:, 1:]  # [c, k, d]: the chained members
        # Link cost of target j: distance to the nearest of path members
        # 0..j, computed through a gram matrix to avoid a [c, k, k+1, d]
        # tensor.
        cross = torch.bmm(targets, neighbors.transpose(1, 2))
        squared = (
            targets.square().sum(-1)[:, :, None]
            + neighbors.square().sum(-1)[:, None, :]
            - 2.0 * cross
        ).clamp_min(0.0)
        pair = squared.sqrt().masked_fill(
            member[None, None, :] > target[None, :, None], torch.inf
        )
        chaining[start:stop] = (pair.min(dim=2).values * weight).sum(dim=1)

    neighbor_sum = chaining[path[:, 1:]].sum(dim=1)
    factor = torch.where(
        neighbor_sum > 0, chaining * k / neighbor_sum, torch.zeros_like(chaining)
    )
    return torch.nan_to_num(factor)


@torch.inference_mode()
def score(series: Tensor, config: ConnectivityOutlierFactorConfig) -> Tensor:
    series = validate_series(series)
    return torch.stack([_score_item(item, config) for item in series])


# TSB-AD evaluation entry (form C in tmp/method_train_eval_spec.md); the
# detector above is frozen and reused as-is.

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import lightning as L
import numpy as np
import pandas as pd
import torch

from ...benchmark import (
    has_score,
    load_split,
    save_score,
    unit_dir,
    write_metrics,
)

# TSB-AD lists COF in neither Optimal dict; the run_COF wrapper default
# n_neighbors=30 coincides with this Config's default. COF scores raw
# unstandardized points, so there is no window or normalization knob.
DEFAULT_HP: Mapping[str, Any] = {"neighbors": 30}


def evaluate(
    dataset_dir: str | Path,
    output_root: str | Path = "benchmarks/ConnectivityOutlierFactor",
    *,
    artifact: str | None = None,
    partition: str | None = "Eva",
    seed: int = 2024,
    hp: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Score every series of one artifact pair over its full train+test
    length and write the per-series metric table.

    Existing score files are kept when ``resume``; returns the
    ``write_metrics`` frame.
    """
    split = load_split(dataset_dir, artifact, partition=partition)
    unit = unit_dir(output_root, "ConnectivityOutlierFactor", split)
    overrides = {**DEFAULT_HP, **(hp or {})}
    for series_id in split.test.ids:
        if resume and has_score(unit, series_id):
            continue
        L.seed_everything(seed)
        train_item = split.train[series_id] if split.train is not None else None
        test_item = split.test[series_id]
        full = (
            np.concatenate([train_item.data, test_item.data])
            if train_item is not None
            else test_item.data
        )
        config = replace(ConnectivityOutlierFactorConfig(), **overrides)
        scores = score(torch.from_numpy(full)[None], config)[0].numpy()
        if scores.shape != (len(full),) or not np.isfinite(scores).all():
            raise ValueError(f"{series_id}: scores must be finite and full-length")
        save_score(unit, series_id, scores)
    return write_metrics(split, unit)
