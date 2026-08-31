"""K-means distance detector.

Converts the series into sliding-window feature vectors, standardizes every
window across its own features — single-feature windows keep their raw
scale — clusters the windows into ``clusters`` centers with Lloyd
iterations from k-means++ seeds, and scores each window by its euclidean
distance to the assigned center. Window scores map back to points by
averaging every window that covers the point; edge points average only
the windows that actually cover them, matching the original
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
    first = int(torch.randint(count, (1,), generator=generator).item())
    centers[0] = rows[first]
    closest = (rows - centers[0]).square().sum(dim=1)
    for index in range(1, k):
        total = closest.sum()
        if not total > 0:
            centers[index] = rows[
                int(torch.randint(count, (1,), generator=generator).item())
            ]
        else:
            pick = int(torch.multinomial(closest, 1, generator=generator).item())
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
    """Average the scores of all windows covering each point.

    Window ``j`` covers points ``[j, j + window - 1]``, so point ``p`` takes
    the mean over window starts ``[p - window + 1, p]`` clipped to the
    available windows; the sequence edges average fewer windows instead of
    borrowing scores from beyond the boundary.
    """
    count = window_scores.numel()
    prefix = torch.cat(
        (window_scores.new_zeros(1), window_scores.double().cumsum(0))
    )
    positions = torch.arange(length, device=window_scores.device)
    start = (positions - (window - 1)).clamp_min(0)
    end = positions.clamp_max(count - 1)
    totals = prefix[end + 1] - prefix[start]
    sizes = (end - start + 1).clamp_min(1)
    return (totals / sizes).to(window_scores.dtype)


@torch.inference_mode()
def score(series: Tensor, config: KMeansDistanceConfig) -> Tensor:
    window = config.window or infer_period(series[0, :, 0])
    series = validate_series(series, min_length=window)
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
from ...evaluation.period import find_period

# TSB-AD optimal HP (HP_list.py), selected per channel count in evaluate:
# multivariate 'KMeansAD' = {'n_clusters': 10, 'window_size': 40}; the
# univariate track runs 'KMeansAD_U' = {'periodicity': 2, 'n_clusters': 10},
# whose window is find_length_rank(data, rank=2), injected per series below.
# TSB-AD leaves the sklearn KMeans random_state unset and seeds the process
# once (2024), so the evaluate seed feeds config.seed here.
DEFAULT_HP: Mapping[str, Any] = {"clusters": 10}


def evaluate(
    dataset_dir: str | Path,
    output_root: str | Path = "benchmarks/KMeansDistance",
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
    unit = unit_dir(output_root, "KMeansDistance", split)
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
        overrides: dict[str, Any] = {**DEFAULT_HP, "seed": seed}
        if full.shape[1] == 1:
            # period.py picks rank>=2 as the rank-th strongest ACF peak,
            # while TSB-AD's find_length_rank takes the strongest peak past
            # the previous rank's peak; only rank=1 is exactly equivalent
            # (period.py docstring), so rank=2 approximates the TSB-AD rule.
            overrides["window"] = find_period(full[:, 0], rank=2)
        else:
            overrides["window"] = 40
        overrides.update(hp or {})
        config = replace(KMeansDistanceConfig(), **overrides)
        scores = score(torch.from_numpy(full)[None], config)[0].numpy()
        if scores.shape != (len(full),) or not np.isfinite(scores).all():
            raise ValueError(f"{series_id}: scores must be finite and full-length")
        save_score(unit, series_id, scores)
    return write_metrics(split, unit)
