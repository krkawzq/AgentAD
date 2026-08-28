"""ACF-based period estimation used as the VUS window size."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import argrelextrema
from statsmodels.tsa.stattools import acf

__all__ = ["find_length_rank", "find_period"]


def find_period(data: ArrayLike, *, rank: int = 1, fallback: int = 125) -> int:
    """Estimate a dominant period from autocorrelation local maxima.

    At most 20,000 points and 400 lags are inspected. ``rank=1`` matches the
    TSB-AD benchmark rule; ``rank=0`` explicitly selects a one-point window.
    """
    values = np.asarray(data).squeeze()
    if values.ndim != 1:
        raise ValueError(f"data must reduce to one dimension, got shape {values.shape}")
    if values.size == 0:
        raise ValueError("data must not be empty")
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
        raise TypeError("rank must be an integer")
    if rank < 0:
        raise ValueError("rank must be non-negative")
    if rank == 0:
        return 1
    if fallback <= 0:
        raise ValueError("fallback must be positive")

    try:
        values = np.asarray(values[:20_000], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError("data must be numeric") from error
    if not np.all(np.isfinite(values)):
        raise ValueError("data contains non-finite values")
    if values.size < 4 or np.ptp(values) == 0:
        return int(fallback)
    base = 3
    correlation = acf(values, nlags=min(400, values.size - 1), fft=True)[base:]
    maxima = argrelextrema(correlation, np.greater)[0]
    finite = maxima[np.isfinite(correlation[maxima])]
    if finite.size == 0:
        return int(fallback)
    by_strength = finite[np.argsort(correlation[finite])[::-1]]
    selected = by_strength[min(rank - 1, by_strength.size - 1)]
    period = int(selected + base)
    return int(fallback) if period > 300 else period


def find_length_rank(data: ArrayLike, rank: int = 1) -> int:
    """Compatibility alias for TSB-AD's function name."""
    return find_period(data, rank=rank)
