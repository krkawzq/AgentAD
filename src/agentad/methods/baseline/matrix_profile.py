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

from dataclasses import dataclass

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
