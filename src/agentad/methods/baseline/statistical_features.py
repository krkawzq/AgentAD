"""Robust statistical feature-ensemble detector.

Combines five groups of univariate proxy features, each robustly scaled by
median/MAD, clipped to positive z-scores, and averaged inside its group:

1. local dynamics — absolute first differences, jerk, residual magnitude
   against a local mean, and short-window residual/difference energies;
2. lag-1 coupling — deviation and instability of the rolling lag-1
   autocorrelation plus the residual-to-signal energy ratio;
3. spectral signature — rolling three-band power-distribution deviation,
   entropy deviation, and high-band share against whole-series medians;
4. recovery — delayed-impulse recovery error, residual persistence,
   long-lag autocorrelation, and slow-minus-fast residual energy;
5. predictability — AR(1), exponentially weighted, and periodicity
   prediction residuals.

The score is the configured weighted sum of the five groups. Rolling
statistics run in float64 with centered windows and pairwise-complete
correlations. Multivariate input is scored per channel and averaged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as F

from ...benchmark import (
    has_score,
    load_split,
    save_score,
    unit_dir,
    write_metrics,
)
from ._common import combine_channels, per_channel

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class StatisticalFeaturesConfig:
    window: int | None = None
    short_window: int | None = None
    long_window: int | None = None
    spectral_window: int | None = None
    spectral_stride: int | None = None
    max_period_lag: int = 256
    weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        for name in ("window", "short_window", "long_window", "spectral_window"):
            if getattr(self, name) is not None and getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive when provided")
        if self.spectral_stride is not None and self.spectral_stride < 1:
            raise ValueError("spectral_stride must be positive when provided")
        if self.max_period_lag < 2:
            raise ValueError("max_period_lag must be at least two")
        if len(self.weights) != 5 or not all(
            math.isfinite(weight) for weight in self.weights
        ):
            raise ValueError("weights must hold five finite values")


def _clean(x: Tensor) -> Tensor:
    """Linearly interpolate non-finite entries between nearest finite ones."""
    finite = torch.isfinite(x)
    if finite.all():
        return x
    if not finite.any():
        return torch.zeros_like(x)
    n = x.numel()
    index = torch.arange(n, device=x.device)
    previous = torch.where(finite, index, torch.full_like(index, -1)).cummax(0).values
    following = torch.where(finite, index, torch.full_like(index, n))
    following = following.flip(0).cummin(0).values.flip(0)
    safe = torch.where(finite, x, torch.zeros_like(x))
    left = torch.where(previous >= 0, safe[previous.clamp_min(0)], torch.zeros_like(x))
    right = torch.where(
        following < n, safe[following.clamp_max(n - 1)], torch.zeros_like(x)
    )
    bad = ~finite
    both = bad & (previous >= 0) & (following < n)
    alpha = ((index - previous) / (following - previous).clamp_min(1)).to(x.dtype)
    out = torch.where(both, left * (1 - alpha) + right * alpha, x)
    out = torch.where(bad & (previous < 0), right, out)
    out = torch.where(bad & (following >= n), left, out)
    return out


def _shift(x: Tensor, periods: int, fill: float) -> Tensor:
    if periods == 0 or x.numel() == 0:
        return x.clone()
    out = (
        F.pad(x, (periods, 0), value=fill)
        if periods > 0
        else F.pad(x, (0, -periods), value=fill)
    )
    if periods > 0:
        return out[: x.numel()]
    return out[-x.numel() :]


def _window_sums(x: Tensor, window: int, left: int) -> Tensor:
    """Sums of clamped centered windows; edges see fewer elements."""
    n = x.numel()
    padded = F.pad(x, (left, window - 1 - left))
    prefix = torch.cat((x.new_zeros(1), padded.cumsum(0)))
    start = torch.arange(n, device=x.device)
    return prefix[start + window] - prefix[start]


def _rolling_mean(x: Tensor, window: int) -> Tensor:
    sums = _window_sums(x, window, (window - 1) // 2)
    counts = _window_sums(torch.ones_like(x), window, (window - 1) // 2)
    return sums / counts.clamp_min(1)


def _rolling_std(x: Tensor, window: int) -> Tensor:
    left = (window - 1) // 2
    counts = _window_sums(torch.ones_like(x), window, left)
    mean = _window_sums(x, window, left) / counts.clamp_min(1)
    square = _window_sums(x.square(), window, left) / counts.clamp_min(1)
    std = (square - mean.square()).clamp_min(0).sqrt()
    # Windows with fewer than two observations stay undefined.
    return _clean(torch.where(counts >= 2, std, torch.full_like(std, torch.nan)))


def _rolling_corr_lag(x: Tensor, lag: int, window: int) -> Tensor:
    n = x.numel()
    if n <= 2 or lag <= 0 or lag >= n:
        return torch.zeros(n, dtype=x.dtype, device=x.device)
    lagged = _shift(x, lag, torch.nan)
    valid = torch.isfinite(lagged)
    masked_x = torch.where(valid, x, torch.zeros_like(x))
    masked_y = torch.where(valid, lagged, torch.zeros_like(lagged))
    left = (window - 1) // 2
    count = _window_sums(valid.to(x.dtype), window, left)
    sum_x = _window_sums(masked_x, window, left)
    sum_y = _window_sums(masked_y, window, left)
    sum_xy = _window_sums(masked_x * masked_y, window, left)
    sum_xx = _window_sums(masked_x.square(), window, left)
    sum_yy = _window_sums(masked_y.square(), window, left)
    numerator = count * sum_xy - sum_x * sum_y
    denominator = (
        (count * sum_xx - sum_x.square()).clamp_min(0)
        * (count * sum_yy - sum_y.square()).clamp_min(0)
    ).sqrt()
    correlation = torch.where(
        (count >= 3) & (denominator > _EPS),
        numerator / denominator.clamp_min(_EPS),
        torch.full_like(numerator, torch.nan),
    )
    return torch.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)


def _ewma(x: Tensor, span: int) -> Tensor:
    # Closed form of the recursive average y_t = (1-a)*y_{t-1} + a*x_t with
    # y_0 = x_0: the extra (1-a)^(t+1)*x_0 term restores the full weight of
    # the first sample, which the plain convolution undercounts by a.
    alpha = 2.0 / (span + 1.0)
    n = x.numel()
    decay = (1.0 - alpha) ** torch.arange(n + 1, dtype=x.dtype, device=x.device)
    size = 1
    while size < 2 * n:
        size <<= 1
    convolution = torch.fft.irfft(
        torch.fft.rfft(x, n=size) * torch.fft.rfft(decay[:n], n=size), n=size
    )[:n]
    return alpha * convolution + decay[1:] * x[0]


def _robust_positive(x: Tensor) -> Tensor:
    median = torch.quantile(x, 0.5)
    mad = torch.quantile((x - median).abs(), 0.5)
    scale = 1.4826 * mad
    if not torch.isfinite(scale) or scale < _EPS:
        scale = x.std(correction=0)
    if not torch.isfinite(scale) or scale < _EPS:
        return torch.zeros_like(x)
    z = torch.nan_to_num((x - median) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    return z.clamp_min(0.0).clamp(max=50.0)


def _combine(features: list[Tensor]) -> Tensor:
    return _clean(
        torch.stack([_robust_positive(feature) for feature in features]).mean(0)
    )


def _spectral_samples(
    x: Tensor, window: int, stride: int
) -> tuple[Tensor, Tensor, Tensor]:
    n = x.numel()
    if n < 4:
        empty = torch.zeros(0, 3, dtype=x.dtype, device=x.device)
        return empty, empty[:, 0], torch.zeros(0, dtype=torch.long, device=x.device)
    window = max(1, min(window, n))
    stride = max(1, stride)
    centers = torch.arange(0, n, stride, dtype=torch.long, device=x.device)
    if centers[-1] != n - 1:
        centers = torch.cat((centers, centers.new_tensor([n - 1])))
    half = max(1, window // 2)
    distributions: list[Tensor] = []
    entropies: list[Tensor] = []
    for center in centers.tolist():
        segment = x[max(center - half, 0) : min(center + half + 1, n)]
        if segment.numel() < 4 or segment.std(correction=0) < _EPS:
            distributions.append(torch.zeros(3, dtype=x.dtype, device=x.device))
            entropies.append(torch.zeros((), dtype=x.dtype, device=x.device))
            continue
        segment = segment - segment.mean()
        tapered = segment * torch.hann_window(
            segment.numel(), periodic=False, dtype=x.dtype, device=x.device
        )
        power = torch.fft.rfft(tapered).abs().square()[1:]
        total = power.sum()
        if power.numel() == 0 or total < _EPS:
            distributions.append(torch.zeros(3, dtype=x.dtype, device=x.device))
            entropies.append(torch.zeros((), dtype=x.dtype, device=x.device))
            continue
        # Earlier bands absorb the remainder when the spectrum does not
        # divide evenly into three.
        base, remainder = power.numel() // 3, power.numel() % 3
        bounds = [0]
        for part in range(3):
            bounds.append(bounds[-1] + base + (1 if part < remainder else 0))
        band = torch.stack(
            [power[bounds[part] : bounds[part + 1]].sum() for part in range(3)]
        )
        distribution = band / band.sum().clamp_min(_EPS)
        density = power / total
        entropy = -(density * (density + _EPS).log()).sum() / torch.log(
            torch.tensor(max(power.numel(), 2), dtype=x.dtype, device=x.device)
        )
        distributions.append(distribution)
        entropies.append(entropy)
    return torch.stack(distributions), torch.stack(entropies), centers


def _interp_samples(values: Tensor, centers: Tensor, n: int) -> Tensor:
    if values.numel() == 0:
        return values.new_zeros(n)
    if values.numel() == 1:
        return values.new_full((n,), float(values[0]))
    position = torch.arange(n, device=values.device).to(values.dtype)
    upper = torch.searchsorted(centers.to(values.dtype), position, right=True)
    lower = (upper - 1).clamp(0, values.numel() - 1)
    upper = upper.clamp(0, values.numel() - 1)
    alpha = (position - centers[lower].to(values.dtype)) / (
        centers[upper].to(values.dtype) - centers[lower].to(values.dtype)
    ).clamp_min(1)
    interpolated = values[lower] + (values[upper] - values[lower]) * alpha
    return torch.where(
        position < float(centers[0]),
        values[0],
        torch.where(position > float(centers[-1]), values[-1], interpolated),
    )


def _estimate_ar1_phi(x: Tensor) -> float:
    if x.numel() < 3 or x.std(correction=0) < _EPS:
        return 0.0
    centered = x - torch.quantile(x, 0.5)
    previous = centered[:-1]
    current = centered[1:]
    denominator = float(previous @ previous)
    if denominator < _EPS:
        return 0.0
    phi = float(previous @ current) / denominator
    if phi != phi:
        return 0.0
    return min(max(phi, -0.99), 0.99)


def _estimate_period(x: Tensor, max_lag: int, threshold: float = 0.35) -> int | None:
    n = x.numel()
    if n < 12 or x.std(correction=0) < _EPS:
        return None
    centered = x - torch.quantile(x, 0.5)
    upper = min(max_lag, max(2, n // 2))
    if upper <= 2:
        return None
    best_lag: int | None = None
    best_correlation = float("-inf")
    for lag in range(2, upper + 1):
        a = centered[:-lag]
        b = centered[lag:]
        if a.numel() < 3 or a.std(correction=0) < _EPS or b.std(correction=0) < _EPS:
            continue
        a = a - a.mean()
        b = b - b.mean()
        correlation = float(a @ b) / math.sqrt(float(a @ a) * float(b @ b))
        if correlation == correlation and correlation > best_correlation:
            best_lag, best_correlation = lag, correlation
    if best_lag is None or best_correlation < threshold:
        return None
    return best_lag


def _score_channel(x: Tensor, config: StatisticalFeaturesConfig) -> Tensor:
    n = x.numel()
    if n == 0:
        return x
    if x.std(correction=0) < _EPS:
        return torch.zeros_like(x)

    def resolve(value: int | None, default: int) -> int:
        return max(1, min(value if value is not None else default, n))

    default_window = min(256, max(16, n // 20)) if n >= 16 else n
    window = resolve(config.window, default_window)
    short_window = resolve(config.short_window, max(5, window // 4))
    long_window = resolve(config.long_window, max(window, 2 * window))
    spectral_window = resolve(config.spectral_window, max(16, window))
    spectral_stride = config.spectral_stride or max(1, spectral_window // 4)
    weights = torch.tensor(config.weights, dtype=x.dtype, device=x.device)

    center = torch.quantile(x, 0.5)
    phi = _estimate_ar1_phi(x)
    period = _estimate_period(x, config.max_period_lag)
    acorr_baseline = torch.quantile(_rolling_corr_lag(x, 1, window), 0.5)
    distributions, entropies, centers = _spectral_samples(
        x, spectral_window, spectral_stride
    )
    if distributions.numel():
        signature = torch.quantile(distributions, 0.5, dim=0)
        total = signature.sum()
        signature = signature / total if total > _EPS else torch.zeros_like(signature)
        spectral_entropy = torch.quantile(entropies, 0.5)
    else:
        signature = torch.zeros(3, dtype=x.dtype, device=x.device)
        spectral_entropy = torch.zeros((), dtype=x.dtype, device=x.device)

    difference = (x - _shift(x, 1, float(x[0]))).abs()
    first_difference = x - _shift(x, 1, float(x[0]))
    jerk = (
        first_difference - _shift(first_difference, 1, float(first_difference[0]))
    ).abs()
    local_mean = _rolling_mean(x, short_window)
    residual = x - local_mean
    residual_abs = residual.abs()
    k1 = _combine(
        [
            difference,
            jerk,
            residual_abs,
            _rolling_mean(residual.square(), short_window),
            _rolling_mean(difference.square(), short_window),
        ]
    )

    acorr1 = _rolling_corr_lag(x, 1, window)
    signal_center = x - _rolling_mean(x, long_window)
    signal_energy = _rolling_mean(signal_center.square(), window)
    residual_energy = _rolling_mean(residual.square(), window)
    energy_ratio = residual_energy / (signal_energy + _EPS)
    energy_ratio = torch.where(
        (signal_energy < _EPS) & (residual_energy < _EPS),
        torch.zeros_like(energy_ratio),
        energy_ratio,
    )
    k2 = _combine(
        [
            (acorr1 - acorr_baseline).abs(),
            (acorr1 - _shift(acorr1, 1, float(acorr1[0]))).abs(),
            energy_ratio,
        ]
    )

    if distributions.numel():
        signature_deviation = _interp_samples(
            (distributions - signature).abs().sum(dim=1),
            centers,
            n,
        )
        entropy_deviation = _interp_samples(
            (entropies - spectral_entropy).abs(),
            centers,
            n,
        )
        high_band = _interp_samples(
            distributions[:, -1],
            centers,
            n,
        )
    else:
        zeros = x.new_zeros(n)
        signature_deviation = entropy_deviation = high_band = zeros
    k3 = _combine([signature_deviation, entropy_deviation, high_band])

    impulse = difference + jerk
    recovery_error = _shift(_ewma(impulse, short_window), 1, 0.0) * residual_abs
    decay_lag = min(max(2, short_window), max(2, n - 1))
    fast_residual = _ewma(residual.square(), short_window)
    slow_residual = _ewma(residual.square(), long_window)
    k4 = _combine(
        [
            recovery_error,
            _rolling_mean(residual_abs, long_window),
            _rolling_corr_lag(x, decay_lag, long_window).abs(),
            (slow_residual - fast_residual).clamp_min(0.0),
        ]
    )

    previous = _shift(x, 1, float(center))
    ar_residual = (x - (center + phi * (previous - center))).abs()
    ewma_residual = (x - _shift(_ewma(x, short_window), 1, float(center))).abs()
    if period is not None and period < n:
        periodic_error = (x - _shift(x, period, float(center))).abs()
        repeatability_error = (impulse - _shift(impulse, period, 0.0)).abs()
    else:
        periodic_error = ewma_residual
        repeatability_error = _rolling_std(impulse, window) / (
            _rolling_mean(impulse.abs(), window) + _EPS
        )
    k5 = _combine(
        [
            ar_residual,
            ewma_residual,
            _rolling_mean(ar_residual.square(), window),
            periodic_error,
            repeatability_error,
        ]
    )

    score = weights @ torch.stack([k1, k2, k3, k4, k5])
    return _clean(score)


@torch.inference_mode()
def score(series: Tensor, config: StatisticalFeaturesConfig) -> Tensor:
    channels, batch, features = per_channel(series)
    channels = channels.to(torch.float64)
    scores = torch.stack([_score_channel(channel, config) for channel in channels])
    return combine_channels(scores, batch, features).to(series.dtype)


# ---------------------------------------------------------------------------
# TSB-AD baseline evaluation (spec: tmp/method_train_eval_spec.md §4 shape C)
# ---------------------------------------------------------------------------


METHOD_NAME = "StatisticalFeatures"

# TSB-AD tunes no hyperparameters for the HSF family: the univariate grid
# Uni_algo_HP_dict["HSF_U"] only lists window ∈ {64, 128, 256} without a
# selected optimum, Optimal_Uni_algo_HP_dict["HSF_U"] = {}, and HSF has no
# Optimal_Multi entry at all (forks/TSB-AD/TSB_AD/HP_list.py). Empty HP is
# therefore the protocol for uni and multi alike; it matches HSF_AD's own
# defaults, which the frozen Config reproduces (window/short/long/spectral
# sizes resolve from the series length at scoring time, weights all 1.0).
DEFAULT_HP: Mapping[str, Any] = {}


def _check_scores(scores: np.ndarray, length: int) -> None:
    if scores.shape != (length,):
        raise ValueError(f"score length {scores.shape[0]} != series length {length}")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")


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
    """Score every series of one artifact on its train+test full length.

    ``hp`` overrides :data:`DEFAULT_HP` for the frozen
    StatisticalFeaturesConfig. Scores are stored under ``output_root``
    (already-scored series are skipped when ``resume``) and the per-series
    metric table is returned.
    """
    split = load_split(dataset_dir, artifact, partition=partition)
    unit = unit_dir(output_root, METHOD_NAME, split)
    config = StatisticalFeaturesConfig(**{**DEFAULT_HP, **(hp or {})})
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
        _check_scores(scores, len(full))
        save_score(unit, series_id, scores)
    return write_metrics(split, unit)
