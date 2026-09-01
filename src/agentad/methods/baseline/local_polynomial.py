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
    prepare_run,
    save_score,
    unit_dir,
    write_metrics,
)
from ...evaluation.period import find_period
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


# TSB-AD's Optimal_Uni_algo_HP_dict["POLY"] = {"periodicity": 1, "power": 4}
# (no Optimal_Multi entry); run_POLY maps power to the polynomial degree and
# window to find_length_rank(data, rank=1), leaving neighborhood on the
# original max(10*window, 100) rule. "power" maps to "degree" here, and the
# data-derived window is injected per series from the first channel's rank-1
# period, so it stays out of DEFAULT_HP.
DEFAULT_HP: Mapping[str, Any] = {"degree": 4}


def evaluate(
    dataset_dir: str | Path,
    output_root: str | Path = "outputs/benchmark",
    *,
    artifact: str | None = None,
    results_root: str | Path = "results/benchmark",
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
    unit = unit_dir(output_root, "baseline/LocalPolynomial", split)
    results = unit_dir(results_root, "baseline/LocalPolynomial", split)
    prepare_run(
        unit,
        split,
        method="baseline/LocalPolynomial",
        partition=partition,
        seed=seed,
        hp=hp,
        resume=resume,
        implementation_file=__file__,
        device="cpu",
    )
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
        config = replace(LocalPolynomialConfig(), **overrides, window=window)
        scores = score(torch.from_numpy(full)[None], config)[0].numpy()
        if scores.shape != (len(full),) or not np.isfinite(scores).all():
            raise ValueError(f"invalid scores for series {series_id!r}")
        save_score(unit, series_id, scores)
    return write_metrics(split, unit, results)
