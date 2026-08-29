"""Local polynomial reconstruction detector.

Slides non-overlapping windows of length ``window`` across each channel and
fits a degree-``degree`` polynomial on the surrounding context, excluding
the window itself. A window scores by the L2 norm of the Fourier-transform
difference between the series and the polynomial estimate, divided by the
window length, so sharp deviations that survive a smooth local fit stand
out. The initial segment (10% of the series, at most 500 points) only
establishes context and is not scored. Multivariate input is scored per
channel and averaged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ._common import combine_channels, infer_period, per_channel


@dataclass(frozen=True, slots=True)
class LocalPolynomialConfig:
    degree: int = 1
    window: int | None = 200
    neighborhood: int | None = None
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.degree < 0:
            raise ValueError("degree must be non-negative")
        if self.window is not None and self.window < 1:
            raise ValueError("window must be positive when provided")
        if self.neighborhood is not None and self.neighborhood < 1:
            raise ValueError("neighborhood must be positive when provided")


def _fit(positions: Tensor, values: Tensor, degree: int) -> Tensor | None:
    """Least-squares coefficients on a centered and scaled basis.

    Centering keeps the Vandermonde matrix well conditioned for the large
    index positions of long series.
    """
    if positions.numel() == 0:
        return None
    positions = positions.to(torch.float64)
    spread = (positions.max() - positions.min()) / 2
    if not spread > 0:
        spread = positions.new_ones(())
    scaled = (positions - positions.mean()) / spread
    design = scaled[:, None] ** torch.arange(
        degree + 1, dtype=torch.float64, device=positions.device
    )
    return torch.linalg.lstsq(design, values.to(torch.float64)).solution


def _evaluate(coefficients: Tensor, positions: Tensor) -> Tensor:
    """Horner evaluation from the highest degree down."""
    scaled = positions.to(torch.float64)
    result = torch.zeros_like(scaled)
    for coefficient in coefficients.flip(0):
        result = result * scaled + coefficient
    return result


def _score_channel(
    channel: Tensor, config: LocalPolynomialConfig, window: int
) -> Tensor:
    length = channel.numel()
    minimum = channel.min()
    span = channel.max() - minimum
    if config.normalize:
        if not span > 0:
            return torch.zeros_like(channel)
        values = (channel - minimum) / span
    else:
        values = channel

    neighborhood = min(config.neighborhood or max(10 * window, 100), length)
    half = neighborhood // 2
    initial = min(500, length // 10)
    first = -(-initial // window)  # ceil
    last = length // window

    scores = torch.zeros(length, dtype=torch.float64, device=channel.device)
    for step in range(first, last + 1):
        index = step * window
        end = min(index + window, length)
        if index >= length:
            continue
        # Context: everything within ``half`` of the window, minus the
        # window itself, clamped at the series boundaries.
        positions = torch.cat(
            (
                torch.arange(max(index - half, 0), index),
                torch.arange(end, min(end + half, length)),
            )
        )
        coefficients = _fit(positions, values[positions], config.degree)
        if coefficients is None:
            continue
        estimate = _evaluate(
            coefficients,
            torch.arange(index, end, dtype=torch.float64, device=channel.device),
        )
        residual = torch.fft.fft(values[index:end].to(torch.float64)) - torch.fft.fft(
            estimate
        )
        scores[index:end] = residual.abs().square().sum().sqrt() / (end - index)
    return scores.to(channel.dtype)


@torch.inference_mode()
def score(series: Tensor, config: LocalPolynomialConfig) -> Tensor:
    channels, batch, features = per_channel(series)
    window = config.window or infer_period(channels[0])
    scores = torch.stack(
        [_score_channel(channel, config, window) for channel in channels]
    )
    return combine_channels(scores, batch, features)
