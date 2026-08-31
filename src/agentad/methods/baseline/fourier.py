"""Low-frequency reconstruction detector.

Reconstructs each z-scored channel from its lowest ``frequencies`` Fourier
coefficients and marks points whose reconstruction error exceeds the mean
error as outlier candidates. Candidates are z-scored against their
deviation from a local neighborhood mean, thresholded, and merged into
regions whenever consecutive candidates stay within ``region_gap`` points;
every point of a region shares the region's mean absolute z-score.
Multivariate input is scored per channel and averaged.
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
from ._common import combine_channels, per_channel, zscore


@dataclass(frozen=True, slots=True)
class FourierConfig:
    frequencies: int = 5
    neighbor_window: int = 21
    outlier_threshold: float = 0.6
    region_gap: int = 10
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.frequencies < 1 or self.neighbor_window < 1:
            raise ValueError("frequencies and neighbor_window must be positive")
        if self.outlier_threshold < 0:
            raise ValueError("outlier_threshold must be non-negative")
        if self.region_gap < 1:
            raise ValueError("region_gap must be positive")


def _score_channel(channel: Tensor, config: FourierConfig) -> Tensor:
    length = channel.numel()
    values = zscore(channel, dim=0, ddof=0) if config.normalize else channel
    scores = torch.zeros(length, device=channel.device, dtype=channel.dtype)

    keep = min(config.frequencies, length)
    coefficients = torch.fft.fft(values)
    coefficients[keep:] = 0
    error = (torch.fft.ifft(coefficients).real - values).abs()

    candidate = (error > error.mean()).nonzero().squeeze(1)
    if candidate.numel() == 0:
        return scores

    # Local neighborhood means from a prefix sum.
    half = config.neighbor_window // 2
    prefix = torch.cat((values.new_zeros(1), values.cumsum(0)))
    low = (candidate - half).clamp_min(0)
    high = (candidate + half + 1).clamp_max(length)
    deviation = values[candidate] - (prefix[high] - prefix[low]) / (high - low)
    z = (deviation - deviation.mean()) / (deviation.std(correction=0) + 1e-6)
    selected = z.abs() > config.outlier_threshold
    candidate, z = candidate[selected], z[selected]
    if candidate.numel() == 0:
        return scores

    # Candidates whose indices stay within region_gap merge into one
    # region; lone candidates stay unscored.
    boundary = candidate.diff() > config.region_gap
    region = torch.cat((torch.zeros_like(boundary[:1]), boundary.cumsum(0)))
    for group in torch.unique(region):
        member = region == group
        if int(member.sum()) < 2:
            continue
        indices = candidate[member]
        scores[indices[0] : indices[-1] + 1] = z[member].abs().mean()
    return scores


@torch.inference_mode()
def score(series: Tensor, config: FourierConfig) -> Tensor:
    channels, batch, features = per_channel(series)
    scores = torch.stack([_score_channel(channel, config) for channel in channels])
    return combine_channels(scores, batch, features)


# TSB-AD leaves FFT untuned (Optimal_Uni_algo_HP_dict["FFT"] is empty and there
# is no Optimal_Multi entry), so DEFAULT_HP spells out the algorithm defaults
# of TSB_AD/models/FFT.py: ifft_parameters=5, local_neighbor_window=21,
# local_outlier_threshold=0.6, max_sign_change_distance=10. The original
# max_region_size=50 has no counterpart in FourierConfig, whose region merging
# is bounded by region_gap alone.
DEFAULT_HP: Mapping[str, Any] = {
    "frequencies": 5,
    "neighbor_window": 21,
    "outlier_threshold": 0.6,
    "region_gap": 10,
}


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
    """Evaluate the baseline on one dataset artifact.

    Scores each train+test concatenated full length, stores one score array
    per series under the method unit (``resume`` skips stored ones), and
    returns the per-series metric table. The algorithm is deterministic;
    ``seed`` re-seeds the global RNG state for protocol parity.
    """
    split = load_split(dataset_dir, artifact, partition=partition)
    unit = unit_dir(output_root, "Fourier", split)
    config = replace(FourierConfig(), **{**DEFAULT_HP, **(hp or {})})
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
        scores = score(torch.from_numpy(full)[None], config)[0].numpy()
        if scores.shape != (len(full),) or not np.isfinite(scores).all():
            raise ValueError(f"invalid scores for series {series_id!r}")
        save_score(unit, series_id, scores)
    return write_metrics(split, unit)
