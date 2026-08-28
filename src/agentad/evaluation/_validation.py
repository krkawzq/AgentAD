"""Array normalization shared by the evaluation APIs."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "binary_array",
    "non_negative_integer",
    "score_array",
    "threshold_count",
    "threshold_grid",
    "validate_pair",
]


def _one_dimensional(values: ArrayLike, *, name: str, dtype: Any) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a numeric one-dimensional array") from error
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


def binary_array(values: ArrayLike, *, name: str) -> NDArray[np.uint8]:
    """Return a contiguous 0/1 vector, rejecting implicit thresholding."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.issubdtype(array.dtype, np.bool_):
        return np.ascontiguousarray(array, dtype=np.uint8)
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real numeric values")
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"{name} must be a numeric one-dimensional array"
            ) from error
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    valid = (array == 0) | (array == 1)
    if not np.all(valid):
        bad = np.unique(array[~valid])[:5]
        raise ValueError(f"{name} must contain only 0 and 1, got {bad.tolist()}")
    return np.ascontiguousarray(array, dtype=np.uint8)


def score_array(values: ArrayLike, *, name: str = "y_score") -> NDArray[np.float64]:
    """Return a contiguous finite float64 score vector."""
    return _one_dimensional(values, name=name, dtype=np.float64)


def validate_pair(
    y_true: ArrayLike,
    y_score: ArrayLike,
) -> tuple[NDArray[np.uint8], NDArray[np.float64]]:
    labels = binary_array(y_true, name="y_true")
    scores = score_array(y_score)
    if labels.size != scores.size:
        raise ValueError(
            f"y_true and y_score must have the same length: {labels.size} != {scores.size}"
        )
    return labels, scores


def non_negative_integer(value: int, *, name: str) -> int:
    """Validate a non-negative integer parameter without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(value)


def threshold_count(value: int, *, name: str = "threshold_count") -> int:
    """Validate the number of samples used by a threshold grid."""
    count = non_negative_integer(value, name=name)
    if count < 2:
        raise ValueError(f"{name} must be at least 2")
    return count


def threshold_grid(scores: NDArray[np.float64], count: int) -> NDArray[np.float64]:
    """TSB-AD's deterministic evenly spaced oracle-threshold grid."""
    count = threshold_count(count)
    return np.linspace(float(scores.min()), float(scores.max()), count)
