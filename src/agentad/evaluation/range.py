"""Point-adjusted, event, range and affiliation metrics.

Adapted from TheDatumOrg/TSB-AD (Apache License 2.0).

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._kernels import (
    best_event_f1_kernel,
    best_point_adjusted_f1_kernel,
    best_range_f1_kernel,
    event_prf_kernel,
    point_adjust_kernel,
    range_prf_kernel,
)
from ._types import PRF1
from ._validation import binary_array, threshold_grid, validate_pair
from .affiliation._kernels import affiliation_prf_kernel, best_affiliation_f1_kernel
from .point import _point_metrics_prepared

__all__ = [
    "PRF1",
    "affiliation_prf",
    "best_affiliation_f1",
    "best_event_f1",
    "best_point_adjusted_f1",
    "best_range_f1",
    "event_prf",
    "point_adjusted_prf",
    "range_prf",
]


def _fixed_pair(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    labels = binary_array(y_true, name="y_true")
    predictions = binary_array(y_pred, name="y_pred")
    if labels.size != predictions.size:
        raise ValueError(
            f"y_true and y_pred must have the same length: {labels.size} != {predictions.size}"
        )
    return labels, predictions


def _triplet(values: tuple[float, float, float]) -> PRF1:
    return PRF1(*(float(value) for value in values))


def point_adjusted_prf(y_true: ArrayLike, y_pred: ArrayLike) -> PRF1:
    """Point precision/recall/F1 after point adjustment."""
    labels, predictions = _fixed_pair(y_true, y_pred)
    adjusted = point_adjust_kernel(labels, predictions)
    result = _point_metrics_prepared(labels, adjusted)
    return PRF1(result.precision, result.recall, result.f1)


def event_prf(y_true: ArrayLike, y_pred: ArrayLike) -> PRF1:
    """Composite event score: point precision, event recall and F1."""
    labels, predictions = _fixed_pair(y_true, y_pred)
    return _triplet(event_prf_kernel(labels, predictions))


def range_prf(y_true: ArrayLike, y_pred: ArrayLike, *, alpha: float = 0.2) -> PRF1:
    """Tatbul-style range precision/recall/F1 with flat positional bias."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    labels, predictions = _fixed_pair(y_true, y_pred)
    return _triplet(range_prf_kernel(labels, predictions, alpha))


def _affiliation_prf_prepared(
    labels: NDArray[np.uint8], predictions: NDArray[np.uint8]
) -> PRF1:
    return _triplet(affiliation_prf_kernel(labels, predictions))


def affiliation_prf(y_true: ArrayLike, y_pred: ArrayLike) -> PRF1:
    """Affiliation precision/recall/F1 for a fixed prediction."""
    labels, predictions = _fixed_pair(y_true, y_pred)
    return _affiliation_prf_prepared(labels, predictions)


def _oracle_inputs(
    y_true: ArrayLike, y_score: ArrayLike, threshold_count: int
) -> tuple[NDArray[np.uint8], NDArray[np.float64], NDArray[np.float64]]:
    labels, scores = validate_pair(y_true, y_score)
    return labels, scores, threshold_grid(scores, threshold_count)


def best_point_adjusted_f1(
    y_true: ArrayLike, y_score: ArrayLike, *, threshold_count: int = 100
) -> float:
    """Best point-adjusted F1 on TSB-AD's deterministic threshold grid."""
    labels, scores, thresholds = _oracle_inputs(y_true, y_score, threshold_count)
    return float(best_point_adjusted_f1_kernel(labels, scores, thresholds))


def best_event_f1(
    y_true: ArrayLike, y_score: ArrayLike, *, threshold_count: int = 100
) -> float:
    """Best composite event F1 on the deterministic threshold grid."""
    labels, scores, thresholds = _oracle_inputs(y_true, y_score, threshold_count)
    return float(best_event_f1_kernel(labels, scores, thresholds))


def best_range_f1(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    threshold_count: int = 100,
    alpha: float = 0.2,
) -> float:
    """Best range F1 on the deterministic threshold grid."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    labels, scores, thresholds = _oracle_inputs(y_true, y_score, threshold_count)
    return float(best_range_f1_kernel(labels, scores, thresholds, alpha))


def best_affiliation_f1(
    y_true: ArrayLike, y_score: ArrayLike, *, threshold_count: int = 100
) -> float:
    """Best affiliation F1 on the deterministic threshold grid."""
    labels, scores, thresholds = _oracle_inputs(y_true, y_score, threshold_count)
    return _best_affiliation_f1_prepared(labels, scores, thresholds)


def _best_affiliation_f1_prepared(
    labels: NDArray[np.uint8],
    scores: NDArray[np.float64],
    thresholds: NDArray[np.float64],
) -> float:
    return float(best_affiliation_f1_kernel(labels, scores, thresholds))
