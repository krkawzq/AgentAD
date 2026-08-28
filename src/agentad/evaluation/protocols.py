"""Extended point, interval, delay and early-detection protocols."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._kernels import best_point_adjusted_f1_kernel, point_adjust_kernel
from ._protocol_kernels import (
    adjust_event_scores_kernel,
    best_interval_prf_kernel,
    delay_adjust_prediction_kernel,
    dilate_labels_kernel,
    event_compress_kernel,
    interval_prf_kernel,
    pate_auc_kernel,
    pate_f1_kernel,
    pate_pr_kernel,
)
from ._types import FirstHit, PRF1
from ._validation import (
    binary_array,
    non_negative_integer,
    threshold_count as validate_threshold_count,
    threshold_grid,
    validate_pair,
)
from .point import _best_point_metrics_prepared, _point_metrics_prepared

EventScale = Literal["squeeze", "log", "sqrt", "raw"]

__all__ = [
    "EventScale",
    "best_event_adjusted_f1",
    "best_interval_f1",
    "best_k_delay_point_adjusted_f1",
    "best_tolerance_point_adjusted_f1",
    "event_adjusted_prf",
    "first_hit",
    "interval_prf",
    "k_delay_point_adjusted_prf",
    "pate",
    "pate_f1",
    "pate_prf",
    "tolerance_point_adjusted_prf",
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


def _event_scale_code(scale: EventScale, base: int) -> tuple[int, int]:
    modes = {"squeeze": 0, "log": 1, "sqrt": 2, "raw": 3}
    if scale not in modes:
        raise ValueError(f"invalid event_scale: {scale!r}")
    if isinstance(base, bool) or not isinstance(base, (int, np.integer)):
        raise TypeError("event_base must be an integer")
    if scale == "log" and base < 2:
        raise ValueError("event_base must be at least 2 for logarithmic scaling")
    return modes[scale], int(base)


def _event_adjusted_prepared(
    labels: NDArray[np.uint8],
    scores: NDArray[np.float64],
    predictions: NDArray[np.uint8] | None,
    *,
    scale: EventScale,
    base: int,
) -> PRF1:
    mode, base = _event_scale_code(scale, base)
    if predictions is None:
        compressed_labels, compressed_scores = event_compress_kernel(
            labels, scores, mode, base
        )
        result = _best_point_metrics_prepared(
            compressed_labels, compressed_scores, smoothing=1e-15
        )
    else:
        compressed_labels, compressed_predictions = event_compress_kernel(
            labels, predictions, mode, base
        )
        result = _point_metrics_prepared(
            compressed_labels,
            np.ascontiguousarray(compressed_predictions > 0, dtype=np.uint8),
        )
    return PRF1(result.precision, result.recall, result.f1)


def event_adjusted_prf(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    scale: EventScale = "squeeze",
    base: int = 3,
) -> PRF1:
    """Event-scaled point-adjusted precision, recall and F1."""
    labels, predictions = _fixed_pair(y_true, y_pred)
    return _event_adjusted_prepared(
        labels,
        predictions.astype(np.float64),
        predictions,
        scale=scale,
        base=base,
    )


def best_event_adjusted_f1(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    scale: EventScale = "squeeze",
    base: int = 3,
) -> float:
    labels, scores = validate_pair(y_true, y_score)
    return _event_adjusted_prepared(
        labels, scores, None, scale=scale, base=base
    ).f1


def _k_delay_prepared(
    labels: NDArray[np.uint8],
    scores: NDArray[np.float64],
    predictions: NDArray[np.uint8] | None,
    *,
    delay: int,
) -> PRF1:
    delay = non_negative_integer(delay, name="k_delay")
    if delay < 1:
        raise ValueError("k_delay must be at least 1")
    if predictions is None:
        adjusted_scores = adjust_event_scores_kernel(labels, scores, delay)
        result = _best_point_metrics_prepared(labels, adjusted_scores, smoothing=1e-15)
    else:
        adjusted = delay_adjust_prediction_kernel(labels, predictions, delay)
        result = _point_metrics_prepared(labels, adjusted)
    return PRF1(result.precision, result.recall, result.f1)


def k_delay_point_adjusted_prf(
    y_true: ArrayLike, y_pred: ArrayLike, *, delay: int = 5
) -> PRF1:
    """Point-adjusted PRF when an event must be hit in its first ``delay`` points."""
    labels, predictions = _fixed_pair(y_true, y_pred)
    return _k_delay_prepared(
        labels, predictions.astype(np.float64), predictions, delay=delay
    )


def best_k_delay_point_adjusted_f1(
    y_true: ArrayLike, y_score: ArrayLike, *, delay: int = 5
) -> float:
    labels, scores = validate_pair(y_true, y_score)
    return _k_delay_prepared(labels, scores, None, delay=delay).f1


def _tolerance_labels(labels: NDArray[np.uint8], tolerance: int) -> NDArray[np.uint8]:
    tolerance = non_negative_integer(tolerance, name="tolerance")
    return dilate_labels_kernel(labels, tolerance)


def tolerance_point_adjusted_prf(
    y_true: ArrayLike, y_pred: ArrayLike, *, tolerance: int = 5
) -> PRF1:
    """Point-adjusted PRF after symmetric ground-truth tolerance dilation."""
    labels, predictions = _fixed_pair(y_true, y_pred)
    tolerant = _tolerance_labels(labels, tolerance)
    adjusted = point_adjust_kernel(tolerant, predictions)
    result = _point_metrics_prepared(tolerant, adjusted)
    return PRF1(result.precision, result.recall, result.f1)


def best_tolerance_point_adjusted_f1(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    tolerance: int = 5,
    threshold_count: int = 100,
) -> float:
    labels, scores = validate_pair(y_true, y_score)
    tolerant = _tolerance_labels(labels, tolerance)
    thresholds = threshold_grid(scores, threshold_count)
    return float(best_point_adjusted_f1_kernel(tolerant, scores, thresholds))


def interval_prf(y_true: ArrayLike, y_pred: ArrayLike) -> PRF1:
    """Overlap-count precision, recall and F1 over contiguous intervals."""
    labels, predictions = _fixed_pair(y_true, y_pred)
    return PRF1(*(float(value) for value in interval_prf_kernel(labels, predictions)))


def best_interval_f1(
    y_true: ArrayLike, y_score: ArrayLike, *, threshold_count: int = 100
) -> float:
    labels, scores = validate_pair(y_true, y_score)
    thresholds = threshold_grid(scores, threshold_count)
    return float(best_interval_prf_kernel(labels, scores, thresholds)[2])


def _buffer_points(maximum: int, splits: int, include_zero: bool) -> NDArray[np.int64]:
    maximum = non_negative_integer(maximum, name="buffer")
    splits = non_negative_integer(splits, name="buffer_splits")
    if splits < 1:
        raise ValueError("buffer_splits must be at least 1")
    if not isinstance(include_zero, (bool, np.bool_)):
        raise TypeError("include_zero must be a boolean")
    start = 0.0 if include_zero else maximum / splits
    count = splits + 1 if include_zero else splits
    return np.ascontiguousarray(
        np.linspace(start, maximum, num=count, dtype=np.int64)
    )


def _rank_thresholds(
    scores: NDArray[np.float64], count: int
) -> NDArray[np.float64]:
    count = validate_threshold_count(count, name="pate_threshold_count")
    descending = np.sort(scores)[::-1]
    indices = np.linspace(0, scores.size - 1, min(count, scores.size)).astype(
        np.int64
    )
    sampled = descending[indices]
    keep = np.concatenate(([True], sampled[1:] != sampled[:-1]))
    return np.ascontiguousarray(sampled[keep])


def pate_prf(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    early_buffer: int = 100,
    delayed_buffer: int = 100,
) -> PRF1:
    """Early/delayed weighted PATE precision, recall and F1."""
    labels, predictions = _fixed_pair(y_true, y_pred)
    early_buffer = non_negative_integer(early_buffer, name="early_buffer")
    delayed_buffer = non_negative_integer(delayed_buffer, name="delayed_buffer")
    precision, recall = pate_pr_kernel(
        labels, predictions, early_buffer, delayed_buffer
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return PRF1(float(precision), float(recall), float(f1))


def pate(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    early_buffer: int = 100,
    delayed_buffer: int = 100,
    buffer_splits: int = 1,
    include_zero: bool = False,
    threshold_count: int = 250,
) -> float:
    """Area under the early/delayed weighted PATE precision-recall curve."""
    labels, scores = validate_pair(y_true, y_score)
    thresholds = _rank_thresholds(scores, threshold_count)
    early = _buffer_points(early_buffer, buffer_splits, include_zero)
    delayed = _buffer_points(delayed_buffer, buffer_splits, include_zero)
    return float(pate_auc_kernel(labels, scores, thresholds, early, delayed)[0])


def pate_f1(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    y_pred: ArrayLike | None = None,
    early_buffer: int = 100,
    delayed_buffer: int = 100,
    buffer_splits: int = 1,
    include_zero: bool = False,
    threshold_count: int = 250,
) -> float:
    """PATE F1 averaged across the configured buffer pairs."""
    labels, scores = validate_pair(y_true, y_score)
    early = _buffer_points(early_buffer, buffer_splits, include_zero)
    delayed = _buffer_points(delayed_buffer, buffer_splits, include_zero)
    if y_pred is not None:
        predictions = binary_array(y_pred, name="y_pred")
        if predictions.size != labels.size:
            raise ValueError(
                "y_true and y_pred must have the same length: "
                f"{labels.size} != {predictions.size}"
            )
        return float(pate_f1_kernel(labels, predictions, early, delayed))
    thresholds = _rank_thresholds(scores, threshold_count)
    return float(pate_auc_kernel(labels, scores, thresholds, early, delayed)[1])


def first_hit(y_true: ArrayLike, y_score: ArrayLike) -> FirstHit:
    """Return the first anomalous point's one-based score rank."""
    labels, scores = validate_pair(y_true, y_score)
    if not labels.any():
        return FirstHit(0, math.nan, False, False)
    order = np.argsort(-scores, kind="stable")
    rank = int(np.flatnonzero(labels[order])[0]) + 1
    fraction = rank / labels.size
    return FirstHit(rank, fraction, fraction < 0.03, fraction < 0.10)
