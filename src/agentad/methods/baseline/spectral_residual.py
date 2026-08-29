"""Spectral-residual salience detector.

Min-max normalizes each channel, takes the log magnitude spectrum, and
subtracts a moving average over the spectrum so the periodic background
cancels; mapping the residual spectrum back to the time domain yields a
salience whose peaks mark spectrally novel points. Multivariate input is
scored per channel and averaged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from ._common import combine_channels, per_channel


@dataclass(frozen=True, slots=True)
class SpectralResidualConfig:
    smoothing_window: int = 100

    def __post_init__(self) -> None:
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window must be positive")


@torch.inference_mode()
def score(series: Tensor, config: SpectralResidualConfig) -> Tensor:
    channels, batch, features = per_channel(series, min_length=2)
    minimum = channels.min(dim=1, keepdim=True).values
    span = channels.max(dim=1, keepdim=True).values - minimum
    scale = torch.where(span > 0, span, torch.ones_like(span))
    normalized = torch.where(span > 0, (channels - minimum) / scale, channels)
    # A constant channel has no spectral background to deviate from.
    constant = span <= 0

    spectrum = torch.fft.fft(normalized, dim=1)
    amplitude = spectrum.abs().clamp_min(torch.finfo(spectrum.abs().dtype).tiny)
    log_amplitude = amplitude.log()
    phase = spectrum.angle()

    # Average the log amplitude of the positive-frequency half spectrum,
    # then mirror it around the bias term to rebuild a full spectrum.
    bias, symmetric = log_amplitude[:, :1], log_amplitude[:, 1:]
    half = symmetric[:, : (symmetric.shape[1] + 1) // 2]
    pad_left = (config.smoothing_window - 1) // 2
    pad_right = config.smoothing_window - pad_left - 1
    padded = torch.cat(
        (
            half[:, :1].expand(-1, pad_left),
            half,
            half[:, -1:].expand(-1, pad_right),
        ),
        dim=1,
    )
    kernel = torch.full(
        (config.smoothing_window,),
        1.0 / config.smoothing_window,
        device=channels.device,
        dtype=channels.dtype,
    )
    averaged = F.conv1d(padded.unsqueeze(1), kernel.view(1, 1, -1)).squeeze(1)
    tail = averaged[:, :-1] if symmetric.shape[1] % 2 else averaged
    averaged_spectrum = torch.cat((bias, averaged, tail.flip(1)), dim=1)

    residual = log_amplitude - averaged_spectrum
    salience = torch.fft.ifft(torch.polar(residual.exp(), phase), dim=1).abs()
    salience = torch.where(constant, torch.zeros_like(salience), salience)
    return combine_channels(salience, batch, features)
