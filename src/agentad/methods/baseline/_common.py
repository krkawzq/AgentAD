"""Shared helpers for the training-free baselines.

Every baseline consumes ``[batch, time, feature]`` tensors and returns
``[batch, time]`` scores through a frozen config plus a ``score``
function; whatever fitting a method does happens inside the call.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from .._utils import sliding_windows, validate_series


def zscore(values: Tensor, dim: int, ddof: int = 0) -> Tensor:
    """Standardize along ``dim``; constant slices map to zero."""
    mean = values.mean(dim, keepdim=True)
    std = values.std(dim, keepdim=True, correction=ddof)
    limit = torch.finfo(values.dtype).max
    return torch.nan_to_num((values - mean) / std, nan=0.0, posinf=limit, neginf=-limit)


def expected_search_depth(size: int) -> float:
    """Average unsuccessful-search depth c(n) = 2H(n-1) - 2(n-1)/n of a
    binary search tree, used to normalize isolation depths."""
    if size <= 1:
        return 0.0
    harmonic = sum(1.0 / i for i in range(1, size))
    return 2.0 * harmonic - 2.0 * (size - 1) / size


def per_channel(
    series: Tensor, *, min_length: int = 1
) -> tuple[Tensor, int, int]:
    """Flatten ``[batch, time, feature]`` to ``[batch*feature, time]`` rows.

    Univariate-family baselines score every channel independently; the
    caller recombines with :func:`combine_channels`.
    """
    series = validate_series(series, min_length=min_length)
    batch, time, features = series.shape
    rows = series.permute(0, 2, 1).reshape(batch * features, time)
    return rows, batch, features


def combine_channels(scores: Tensor, batch: int, features: int) -> Tensor:
    """Average per-channel scores back to ``[batch, time]``."""
    return scores.reshape(batch, features, -1).mean(dim=1)


def window_features(series: Tensor, window: int) -> Tensor:
    """Sliding-window feature matrix ``[batch, time-window+1, window*feature]``."""
    windows = sliding_windows(series, window)
    batch, count = windows.shape[0], windows.shape[1]
    return windows.permute(0, 1, 3, 2).reshape(batch, count, -1)


def windowed_features(series: Tensor, window: int, normalize: bool) -> Tensor:
    """Windowed feature matrix with the pointwise-baseline normalization.

    Univariate input standardizes every feature column over time;
    multivariate input standardizes each window row across its features.
    """
    features = window_features(series, window)
    if not normalize:
        return features
    if series.shape[2] == 1:
        return torch.stack([zscore(item, dim=0, ddof=0) for item in features])
    return torch.stack([zscore(item, dim=1, ddof=1) for item in features])


def standardize_points(points: Tensor, *, normalize: bool) -> Tensor:
    """Standardize raw ``[time, feature]`` points for point-family baselines.

    Univariate input standardizes along time and multivariate input across
    each point's features, mirroring the window normalization of
    :func:`windowed_features`.
    """
    if not normalize:
        return points
    if points.shape[1] == 1:
        return zscore(points, dim=0, ddof=0)
    return zscore(points, dim=1, ddof=1)


def pad_window_scores(
    scores: Tensor, window: int, length: int, fill: Tensor | None = None
) -> Tensor:
    """Center-align window scores to per-point scores.

    Window ``i`` lands on point ``i + window // 2``; the uncovered edges
    replicate the boundary scores, or take ``fill`` (``[batch]``) when given.
    """
    left, right = window // 2, (window - 1) // 2
    if fill is None:
        head = scores[:, :1].expand(-1, left)
        tail = scores[:, -1:].expand(-1, right)
    else:
        head = fill[:, None].expand(-1, left)
        tail = fill[:, None].expand(-1, right)
    return torch.cat((head, scores, tail), dim=1)[:, :length]


def infer_period(values: Tensor, *, fallback: int = 125, cap: int = 300) -> int:
    """Estimate the dominant period from the autocorrelation.

    Scans lags 3..400 of the biased ACF over the first 20k points, takes
    the strongest local maximum, and falls back to ``fallback`` when no
    peak exists or the peak exceeds ``cap``.
    """
    values = values[:20_000].to(torch.float64)
    count = values.numel()
    base = 3
    if count < base + 3:
        return fallback
    centered = values - values.mean()
    size = 1
    while size < 2 * count:
        size <<= 1
    spectrum = torch.fft.rfft(centered, n=size)
    covariance = torch.fft.irfft(spectrum * spectrum.conj(), n=size)[:count]
    if not covariance[0] > 0:
        return fallback
    correlations = covariance / covariance[0]
    lags = min(400, count - 1)
    window = correlations[base : lags + 1]
    if window.numel() < 3:
        return fallback
    peaks = (
        (window[1:-1] > window[:-2]) & (window[1:-1] > window[2:])
    ).nonzero().squeeze(1) + 1
    if peaks.numel() == 0:
        return fallback
    period = int(peaks[window[peaks].argmax()]) + base
    return fallback if period > cap else period


def knn_search(
    features: Tensor, k: int, *, include_self: bool, chunk_entries: int = 2**22
) -> tuple[Tensor, Tensor]:
    """Chunked exact euclidean k-NN of a ``[n, d]`` matrix against itself.

    ``include_self=False`` never returns the query point as its own
    neighbor; ``True`` keeps the zero self-distance in the candidate set,
    which the chaining-distance baseline relies on.
    """
    count = features.shape[0]
    limit = count if include_self else count - 1
    k = min(k, limit)
    if k <= 0:
        return features.new_empty((count, 0)), torch.empty(
            (count, 0), dtype=torch.long, device=features.device
        )
    chunk = max(1, chunk_entries // count)
    distances = features.new_empty((count, k))
    indices = torch.empty((count, k), dtype=torch.long, device=features.device)
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        block = torch.cdist(features[start:stop], features)
        if not include_self:
            rows = torch.arange(stop - start, device=features.device)
            block[rows, torch.arange(start, stop, device=features.device)] = torch.inf
        top = block.topk(k, dim=1, largest=False, sorted=True)
        distances[start:stop] = top.values
        indices[start:stop] = top.indices
    return distances, indices


def znormalized_window_min(
    windows: Tensor,
    keep: Callable[[Tensor, Tensor], Tensor],
    *,
    empty_value: float = 0.0,
    chunk_entries: int = 2**22,
) -> Tensor:
    """Minimum z-normalized distance from each window to allowed neighbors.

    ``windows`` is ``[count, length]`` z-normalized by the caller; constant
    windows normalize to zero vectors and land at ``sqrt(2*length)`` from
    everything else, keeping every distance finite.
    ``keep(query_positions, reference_positions)`` marks admissible pairs,
    so exclusion zones apply without materializing the full distance
    matrix. Rows without an admissible neighbor get ``empty_value``.
    """
    count, length = windows.shape
    positions = torch.arange(count, device=windows.device)
    chunk = max(1, chunk_entries // max(count, 1))
    profile = windows.new_full((count,), empty_value)
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        correlation = (windows[start:stop] @ windows.T) / length
        squared = (2.0 * length * (1.0 - correlation)).clamp_min(0.0)
        distances = squared.sqrt().masked_fill(
            ~keep(positions[start:stop, None], positions[None, :]), torch.inf
        )
        # Real z-normalized distances are bounded by 2*sqrt(length), so inf
        # only marks rows whose neighbors are all excluded.
        minimum = distances.min(dim=1).values
        profile[start:stop] = torch.where(
            torch.isinf(minimum), profile.new_tensor(empty_value), minimum
        )
    return profile
