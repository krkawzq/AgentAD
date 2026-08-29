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

from dataclasses import dataclass

import torch
from torch import Tensor

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
    region = torch.cat(
        (torch.zeros_like(boundary[:1]), boundary.cumsum(0))
    )
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
