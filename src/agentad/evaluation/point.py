"""Point-wise anomaly-detection metrics."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._kernels import confusion_counts_kernel, point_adjust_kernel
from ._types import PointMetrics
from ._validation import binary_array, validate_pair

__all__ = [
    "PointMetrics",
    "average_precision",
    "best_point_f1",
    "point_adjust",
    "point_metrics",
    "roc_auc",
]


def _same_length(y_true: NDArray[np.uint8], y_pred: NDArray[np.uint8]) -> None:
    if y_true.size != y_pred.size:
        raise ValueError(
            f"y_true and y_pred must have the same length: {y_true.size} != {y_pred.size}"
        )


def _point_metrics_prepared(
    y_true: NDArray[np.uint8], y_pred: NDArray[np.uint8]
) -> PointMetrics:
    tp, fp, fn, tn = confusion_counts_kernel(y_true, y_pred)
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / math.sqrt(denominator) if denominator else 0.0
    return PointMetrics(accuracy, precision, recall, f1, mcc)


def point_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> PointMetrics:
    """Evaluate a fixed binary prediction without threshold selection."""
    labels = binary_array(y_true, name="y_true")
    predictions = binary_array(y_pred, name="y_pred")
    _same_length(labels, predictions)
    return _point_metrics_prepared(labels, predictions)


def point_adjust(y_true: ArrayLike, y_pred: ArrayLike) -> NDArray[np.bool_]:
    """Fill a ground-truth event when at least one of its points was detected."""
    labels = binary_array(y_true, name="y_true")
    predictions = binary_array(y_pred, name="y_pred")
    _same_length(labels, predictions)
    return point_adjust_kernel(labels, predictions).astype(bool, copy=False)


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Point-wise ROC AUC, or NaN when only one label class is present."""
    labels, scores = validate_pair(y_true, y_score)
    _, value, _ = _score_metrics_prepared(labels, scores, smoothing=0.0)
    return value


def average_precision(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Point-wise average precision, or NaN when no anomaly exists."""
    labels, scores = validate_pair(y_true, y_score)
    value, _, _ = _score_metrics_prepared(labels, scores, smoothing=0.0)
    return value


def _best_point_f1_prepared(
    labels: NDArray[np.uint8],
    scores: NDArray[np.float64],
    *,
    smoothing: float,
) -> float:
    _, _, value = _score_metrics_prepared(labels, scores, smoothing=smoothing)
    return value


def _score_metrics_prepared(
    labels: NDArray[np.uint8],
    scores: NDArray[np.float64],
    *,
    smoothing: float,
) -> tuple[float, float, float]:
    """Average precision, ROC AUC and best F1 from one stable score ordering."""
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    distinct = np.flatnonzero(np.diff(sorted_scores))
    threshold_indices = np.concatenate((distinct, [labels.size - 1]))
    true_positives = np.cumsum(sorted_labels, dtype=np.float64)[threshold_indices]
    false_positives = 1 + threshold_indices - true_positives

    positives = int(labels.sum())
    if positives:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / positives
        average_precision_value = float(
            np.dot(np.diff(np.concatenate(([0.0], recall))), precision)
        )
        denominator = precision + recall + smoothing
        f1 = np.divide(
            2.0 * precision * recall,
            denominator,
            out=np.zeros_like(precision),
            where=denominator != 0,
        )
        best_f1 = float(f1.max(initial=0.0))
    else:
        average_precision_value = math.nan
        best_f1 = 0.0

    negatives = labels.size - positives
    if positives and negatives:
        true_positive_rate = true_positives / positives
        false_positive_rate = false_positives / negatives
        roc_auc_value = float(
            np.trapezoid(
                np.concatenate(([0.0], true_positive_rate)),
                np.concatenate(([0.0], false_positive_rate)),
            )
        )
    else:
        roc_auc_value = math.nan
    return average_precision_value, roc_auc_value, best_f1


def best_point_f1(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    smoothing: float = 1e-5,
) -> float:
    """Best point F1 over the exact PR curve.

    The default smoothing matches TSB-AD's ``Standard-F1`` definition.
    Passing ``smoothing=0`` gives the ordinary harmonic mean.
    """
    labels, scores = validate_pair(y_true, y_score)
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")
    return _best_point_f1_prepared(labels, scores, smoothing=smoothing)
