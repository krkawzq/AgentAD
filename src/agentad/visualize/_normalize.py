"""Numerically stable normalization for time-series visualization."""

from __future__ import annotations

from collections.abc import Callable
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


def _scale_finite(
    values: NDArray[np.float64],
    transform: Callable[[NDArray[np.float64]], tuple[float, float]],
) -> None:
    finite = np.isfinite(values)
    if not finite.any():
        return
    selected = values[finite]
    center, scale = transform(selected)
    if not np.isfinite(scale) or scale == 0.0:
        values[finite] = 0.0
        return
    values[finite] = (selected - center) / scale


def _parameters(
    method: str,
) -> Callable[[NDArray[np.float64]], tuple[float, float]]:
    if method == "zscore":
        return lambda x: (float(np.mean(x)), float(np.std(x)))
    if method == "minmax":
        return lambda x: (float(np.min(x)), float(np.max(x) - np.min(x)))
    if method == "robust":
        return lambda x: (
            float(np.median(x)),
            float(np.quantile(x, 0.75) - np.quantile(x, 0.25)),
        )
    if method == "maxabs":
        return lambda x: (0.0, float(np.max(np.abs(x))))
    if method == "l1":
        return lambda x: (0.0, float(np.sum(np.abs(x))))
    if method == "l2":
        return lambda x: (0.0, float(np.linalg.norm(x)))
    raise ValueError(f"unknown normalization method: {method!r}")


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
    if method not in _METHOD_IDS:
        raise ValueError(f"unknown normalization method: {method!r}")
    if scope not in _SCOPE_IDS:
        raise ValueError(f"unknown normalization scope: {scope!r}")

    result = np.array(values, dtype=np.float64, copy=True, order="C")
    original_ndim = result.ndim
    if original_ndim == 1:
        result = result.reshape(-1, 1)
    elif original_ndim != 2:
        raise ValueError(f"values must be 1-D or 2-D, got shape {result.shape}")

    if method != "none" and result.size:
        transform = _parameters(method)
        if scope == "global":
            _scale_finite(result.reshape(-1), transform)
        else:
            for column in range(result.shape[1]):
                _scale_finite(result[:, column], transform)

    if original_ndim == 1:
        return result[:, 0]
    return result

