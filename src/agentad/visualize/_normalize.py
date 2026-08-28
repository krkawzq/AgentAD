"""Numerically stable normalization for time-series visualization."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Normalization = Literal[
    "none",
    "zscore",
    "minmax",
    "robust",
    "maxabs",
    "l1",
    "l2",
]
NormalizationScope = Literal["feature", "global"]

NORMALIZATIONS: tuple[dict[str, str], ...] = (
    {
        "id": "none",
        "name": "原始值",
        "description": "不改变数值尺度",
    },
    {
        "id": "zscore",
        "name": "Z-score",
        "description": "减去均值并除以标准差",
    },
    {
        "id": "minmax",
        "name": "Min-max",
        "description": "线性缩放到 [0, 1]",
    },
    {
        "id": "robust",
        "name": "Robust",
        "description": "减去中位数并除以四分位距",
    },
    {
        "id": "maxabs",
        "name": "Max-abs",
        "description": "除以最大绝对值，保留零点",
    },
    {
        "id": "l1",
        "name": "L1",
        "description": "除以绝对值之和",
    },
    {
        "id": "l2",
        "name": "L2",
        "description": "除以欧氏范数",
    },
)

_METHOD_IDS = frozenset(item["id"] for item in NORMALIZATIONS)
_SCOPE_IDS = frozenset(("feature", "global"))


def _normalize_finite(values: NDArray[np.float64], method: str) -> None:
    """Normalize one flat view without allowing intermediate overflow."""
    finite = np.isfinite(values)
    if not finite.any():
        return
    selected = values[finite]

    # Work on a max-abs-scaled copy. Direct operations such as ``max - min``,
    # ``sum(abs(x))`` and ``std(x)`` can overflow even when every input is
    # finite. Scaling first keeps all fitting operations in a compact range.
    magnitude = float(np.max(np.abs(selected)))
    if magnitude == 0.0:
        values[finite] = 0.0
        return

    # Preserve the natural arithmetic path when it is safe. Besides avoiding
    # unnecessary rounding, this keeps familiar values such as a three-point
    # min-max transform exactly at ``0, 0.5, 1``.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if method == "zscore":
            raw_center = float(np.mean(selected))
            raw_denominator = float(np.std(selected))
        elif method == "minmax":
            raw_center = float(np.min(selected))
            raw_denominator = float(np.max(selected) - raw_center)
        elif method == "robust":
            raw_center = float(np.median(selected))
            raw_denominator = float(
                np.quantile(selected, 0.75) - np.quantile(selected, 0.25)
            )
        elif method == "maxabs":
            raw_center = 0.0
            raw_denominator = magnitude
        elif method == "l1":
            raw_center = 0.0
            raw_denominator = float(np.sum(np.abs(selected)))
        elif method == "l2":
            raw_center = 0.0
            raw_denominator = float(np.linalg.norm(selected))
        else:  # Guarded by ``_validate_options``.
            raise AssertionError(f"unhandled normalization method: {method!r}")

        if raw_denominator == 0.0:
            values[finite] = 0.0
            return
        if np.isfinite(raw_center) and np.isfinite(raw_denominator):
            direct = (selected - raw_center) / raw_denominator
            if np.isfinite(direct).all():
                values[finite] = direct
                return

    scaled = selected / magnitude

    if method == "zscore":
        center = float(np.mean(scaled))
        denominator = float(np.std(scaled))
    elif method == "minmax":
        center = float(np.min(scaled))
        denominator = float(np.max(scaled) - center)
    elif method == "robust":
        center = float(np.median(scaled))
        denominator = float(np.quantile(scaled, 0.75) - np.quantile(scaled, 0.25))
    elif method == "maxabs":
        center = 0.0
        denominator = 1.0
    elif method == "l1":
        center = 0.0
        denominator = float(np.sum(np.abs(scaled)))
    elif method == "l2":
        center = 0.0
        denominator = float(np.linalg.norm(scaled))
    else:  # Guarded by ``_validate_options``.
        raise AssertionError(f"unhandled normalization method: {method!r}")

    if not np.isfinite(denominator) or denominator == 0.0:
        values[finite] = 0.0
    else:
        values[finite] = (scaled - center) / denominator


def _validate_options(method: str, scope: str) -> None:
    if method not in _METHOD_IDS:
        raise ValueError(f"unknown normalization method: {method!r}")
    if scope not in _SCOPE_IDS:
        raise ValueError(f"unknown normalization scope: {scope!r}")


def _normalize_inplace(
    values: NDArray[np.float64],
    method: str,
    scope: str,
) -> None:
    """Normalize a two-dimensional float64 array in place."""
    _validate_options(method, scope)
    if values.ndim != 2:
        raise ValueError(f"values must be 2-D, got shape {values.shape}")
    if method == "none" or values.size == 0:
        return
    if scope == "global":
        _normalize_finite(values.reshape(-1), method)
        return
    for column in range(values.shape[1]):
        _normalize_finite(values[:, column], method)


def normalize(
    values: ArrayLike,
    method: Normalization = "none",
    *,
    scope: NormalizationScope = "feature",
) -> NDArray[np.float64]:
    """Return a float64 normalized copy of one- or two-dimensional values.

    Non-finite values remain non-finite and do not contribute to fitted
    statistics. Constant finite slices map to zero. With ``scope="feature"``
    each column is fitted independently; ``scope="global"`` fits all values
    together.
    """
    _validate_options(method, scope)

    source = np.asarray(values)
    if np.iscomplexobj(source):
        raise TypeError("complex values cannot be normalized for visualization")
    result = np.array(source, dtype=np.float64, copy=True, order="C")
    original_ndim = result.ndim
    if original_ndim == 1:
        result = result.reshape(-1, 1)
    elif original_ndim != 2:
        raise ValueError(f"values must be 1-D or 2-D, got shape {result.shape}")

    _normalize_inplace(result, method, scope)

    if original_ndim == 1:
        return result[:, 0]
    return result
