"""Causal matrix-profile detector.

Scores every window by its z-normalized distance to the nearest *earlier*
window outside the trivial-match exclusion zone of ``ceil(window / 4)``, so
a window is never explained by future or concurrent data. Windows starting
inside ``warmup`` — a trusted training prefix — score zero, and window
scores center-align to points. Multivariate input is scored per channel and
averaged.
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
from .._utils import sliding_windows
from ._common import (
    combine_channels,
    pad_window_scores,
    per_channel,
    znormalized_window_min,
    zscore,
)


@dataclass(frozen=True, slots=True)
class StreamingMatrixProfileConfig:
    warmup: int = 100
    window: int = 50
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.warmup < 3:
            raise ValueError("warmup must be at least three")
        if self.window < 3:
            raise ValueError("window must be at least three")
        if self.window > self.warmup:
            # A window cannot reach beyond the trusted prefix.
            object.__setattr__(self, "window", self.warmup)


def _profile(channel: Tensor, config: StreamingMatrixProfileConfig) -> Tensor:
    values = zscore(channel, dim=0, ddof=0) if config.normalize else channel
    windows = sliding_windows(values[None, :, None], config.window)[0][..., 0]
    mean = windows.mean(dim=1, keepdim=True)
    scale = windows.var(dim=1, keepdim=True, unbiased=False).sqrt()
    normalized = (windows - mean) / scale.clamp_min(1e-12)
    exclusion = -(-config.window // 4)  # ceil(window / 4)

    def keep(query: Tensor, reference: Tensor) -> Tensor:
        return reference <= query - exclusion - 1

    profile = znormalized_window_min(normalized, keep)
    profile[: min(config.warmup, profile.numel())] = 0.0
    return pad_window_scores(profile[None], config.window, channel.numel())[0]


@torch.inference_mode()
def score(series: Tensor, config: StreamingMatrixProfileConfig) -> Tensor:
    channels, batch, features = per_channel(series, min_length=config.window)
    scores = torch.stack([_profile(channel, config) for channel in channels])
    return combine_channels(scores, batch, features)


# TSB-AD leaves Left_STAMPi untuned (empty Optimal_Uni entry, no Optimal_Multi
# entry) and runs it in a Semisupervise streaming protocol: stumpi warms up on
# the train prefix (n_init_train=len(data_train), wrapper-forced
# window_size=100) and emits scores point by point. This evaluate instead
# scores the full series in one causal pass with the original class defaults
# (n_init_train=100, window_size=50, normalize=True); both variants leave the
# warmup prefix at score zero.
DEFAULT_HP: Mapping[str, Any] = {"warmup": 100, "window": 50}


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
    unit = unit_dir(output_root, "StreamingMatrixProfile", split)
    config = replace(StreamingMatrixProfileConfig(), **{**DEFAULT_HP, **(hp or {})})
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
