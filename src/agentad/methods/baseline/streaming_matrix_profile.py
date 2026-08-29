"""Causal matrix-profile detector.

Scores every window by its z-normalized distance to the nearest *earlier*
window outside the trivial-match exclusion zone of ``ceil(window / 4)``, so
a window is never explained by future or concurrent data. Windows starting
inside ``warmup`` — a trusted training prefix — score zero, and window
scores center-align to points. Multivariate input is scored per channel and
averaged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ._common import (
    combine_channels,
    pad_window_scores,
    per_channel,
    znormalized_window_min,
    zscore,
)
from .._utils import sliding_windows


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
