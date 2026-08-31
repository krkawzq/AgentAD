"""Matrix-profile discord detector.

Scores every subsequence by its z-normalized euclidean distance to the
nearest other subsequence outside a trivial-match exclusion zone of
``ceil(window / 4)``, the matrix-profile convention that forbids a window
from matching its overlapping neighbors. Uncovered edge points take the
minimum profile value. Multivariate input is scored per channel and
averaged; ``window=None`` estimates the dominant period from the
autocorrelation.
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

from ._common import (
    combine_channels,
    infer_period,
    pad_window_scores,
    per_channel,
    znormalized_window_min,
)
from .._utils import sliding_windows
from ...benchmark import (
    has_score,
    load_split,
    save_score,
    unit_dir,
    write_metrics,
)
from ...evaluation.period import find_period


@dataclass(frozen=True, slots=True)
class MatrixProfileConfig:
    window: int | None = None

    def __post_init__(self) -> None:
        if self.window is not None and self.window < 2:
            raise ValueError("window must be at least two when provided")


def _profile(channel: Tensor, window: int) -> Tensor:
    windows = sliding_windows(channel[None, :, None], window)[0][..., 0]
    mean = windows.mean(dim=1, keepdim=True)
    scale = windows.var(dim=1, keepdim=True, unbiased=False).sqrt()
    normalized = (windows - mean) / scale.clamp_min(1e-12)

    exclusion = -(-window // 4)  # ceil(window / 4)
    if normalized.shape[0] < exclusion + 2:
        raise ValueError(
            f"series too short for window {window}: every window pair falls "
            "inside the exclusion zone"
        )

    def keep(query: Tensor, reference: Tensor) -> Tensor:
        return (query - reference).abs() > exclusion

    profile = znormalized_window_min(normalized, keep)
    return pad_window_scores(
        profile[None], window, channel.numel(), fill=profile.min()[None]
    )[0]


@torch.inference_mode()
def score(series: Tensor, config: MatrixProfileConfig) -> Tensor:
    channels, batch, features = per_channel(series)
    window = config.window or infer_period(channels[0])
    scores = torch.stack([_profile(channel, window) for channel in channels])
    return combine_channels(scores, batch, features)


# TSB-AD tunes MatrixProfile only through the period ("periodicity": 1 in
# Optimal_Uni_algo_HP_dict; no Optimal_Multi entry), and run_MatrixProfile sets
# window = find_length_rank(data, rank=1). The data-derived window therefore
# stays out of DEFAULT_HP and is computed per series from the first channel's
# rank-1 autocorrelation period; hp may pin it explicitly via {"window": int}.
DEFAULT_HP: Mapping[str, Any] = {}


def evaluate(
    dataset_dir: str | Path,
    output_root: str | Path = "benchmarks/MatrixProfile",
    *,
    artifact: str | None = None,
    partition: str | None = "Eva",
    seed: int = 2024,
    hp: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Evaluate the baseline on one dataset artifact.

    Scores each train+test concatenated full length, stores one score array
    per series under the method unit (``resume`` skips stored ones), and
    returns the per-series metric table. The window defaults to the first
    channel's rank-1 period unless ``hp`` provides ``window``; the algorithm
    is deterministic, and ``seed`` only re-seeds the global RNG state for
    protocol parity.
    """
    split = load_split(dataset_dir, artifact, partition=partition)
    unit = unit_dir(output_root, "MatrixProfile", split)
    overrides = {**DEFAULT_HP, **(hp or {})}
    window_override = overrides.pop("window", None)
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
        window = window_override or find_period(full[:, 0])
        config = replace(MatrixProfileConfig(), **overrides, window=window)
        scores = score(torch.from_numpy(full)[None], config)[0].numpy()
        if scores.shape != (len(full),) or not np.isfinite(scores).all():
            raise ValueError(f"invalid scores for series {series_id!r}")
        save_score(unit, series_id, scores)
    return write_metrics(split, unit)
