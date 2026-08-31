"""Spectral-residual salience detector.

Min-max normalizes each channel, takes the log magnitude spectrum, and
subtracts a moving average over the spectrum so the periodic background
cancels; mapping the residual spectrum back to the time domain yields a
salience whose peaks mark spectrally novel points. Multivariate input is
scored per channel and averaged.
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
from torch.nn import functional as F

from ._common import combine_channels, per_channel
from ...benchmark import (
    has_score,
    load_split,
    save_score,
    unit_dir,
    write_metrics,
)


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
    # The smoothing pads with the spectrum's own edges; upstream pads with
    # the first signal value, which mismatches the spectrum's scale.
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


# The original SR hardcodes its spectrum smoothing window (window_amp=100).
# TSB-AD's Optimal_Uni_algo_HP_dict["SR"] = {"periodicity": 1} only feeds
# find_length_rank into a window_size parameter that the original SR
# implementation ignores, and there is no Optimal_Multi entry, so the original
# constant is the one effective hyperparameter.
DEFAULT_HP: Mapping[str, Any] = {"smoothing_window": 100}


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
    unit = unit_dir(output_root, "SpectralResidual", split)
    config = replace(SpectralResidualConfig(), **{**DEFAULT_HP, **(hp or {})})
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
