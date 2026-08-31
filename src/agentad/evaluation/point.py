"""Point-wise anomaly-detection metrics.

Parts are adapted from TheDatumOrg/TSB-AD under Apache-2.0; see
``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._types import FirstHit, PRF1, PointMetrics
from ._validation import (
    binary_array,
    non_negative_integer,
    threshold_grid,
    validate_pair,
)

EventScale = Literal["squeeze", "log", "sqrt", "raw"]

__all__ = [
    "PointMetrics",
    "EventScale",
    "average_precision",
    "best_event_adjusted_f1",
    "best_k_delay_point_adjusted_f1",
    "best_point_f1",
    "best_tolerance_point_adjusted_f1",
    "event_adjusted_prf",
    "first_hit",
    "k_delay_point_adjusted_prf",
    "point_adjusted_average_precision",
    "point_adjust",
    "point_metrics",
    "precision_at_k",
    "roc_auc",
    "tolerance_point_adjusted_prf",
]


@njit(cache=True, nogil=True)
def _confusion_counts(y_true, y_pred):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for index in range(y_true.size):
        actual = y_true[index] != 0
        predicted = y_pred[index] != 0
        if actual:
            if predicted:
                tp += 1
            else:
                fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    return tp, fp, fn, tn


@njit(cache=True, nogil=True)
def _binary_runs(values):
    """Return start and exclusive-stop arrays for positive runs."""
    starts = np.empty(values.size, dtype=np.int64)
    stops = np.empty(values.size, dtype=np.int64)
    count = 0
    index = 0
    while index < values.size:
        if values[index] == 0:
            index += 1
            continue
        start = index
        index += 1
        while index < values.size and values[index] != 0:
            index += 1
        starts[count] = start
        stops[count] = index
        count += 1
    return starts[:count], stops[:count]


@njit(cache=True, nogil=True)
def _point_adjust_from_runs(y_pred, starts, stops):
    adjusted = y_pred.copy()
    for run in range(starts.size):
        hit = False
        for index in range(starts[run], stops[run]):
            if y_pred[index] != 0:
                hit = True
        if hit:
            for position in range(starts[run], stops[run]):
                adjusted[position] = 1
    return adjusted


@njit(cache=True, nogil=True)
def _point_adjust(y_true, y_pred):
    starts, stops = _binary_runs(y_true)
    return _point_adjust_from_runs(y_pred, starts, stops)


@njit(cache=True, nogil=True)
def _prf_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


@njit(cache=True, nogil=True)
def _best_point_adjusted_f1(y_true, scores, thresholds):
    prediction = np.empty(y_true.size, dtype=np.uint8)
    starts, stops = _binary_runs(y_true)
    best = 0.0
    for threshold in thresholds:
        for index in range(scores.size):
            prediction[index] = 1 if scores[index] > threshold else 0
        adjusted = _point_adjust_from_runs(prediction, starts, stops)
        tp, fp, fn, _ = _confusion_counts(y_true, adjusted)
        _, _, f1 = _prf_from_counts(tp, fp, fn)
        if f1 > best:
            best = f1
    return best


@njit(cache=True, nogil=True)
def _best_point_adjusted_prf(y_true, scores, thresholds):
    prediction = np.empty(y_true.size, dtype=np.uint8)
    starts, stops = _binary_runs(y_true)
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    for threshold in thresholds:
        for index in range(scores.size):
            prediction[index] = 1 if scores[index] > threshold else 0
        adjusted = _point_adjust_from_runs(prediction, starts, stops)
        tp, fp, fn, _ = _confusion_counts(y_true, adjusted)
        precision, recall, f1 = _prf_from_counts(tp, fp, fn)
        if f1 > best_f1:
            best_precision = precision
            best_recall = recall
            best_f1 = f1
    return best_precision, best_recall, best_f1


@njit(cache=True, nogil=True)
def _best_point_adjusted_counts(y_true, scores, thresholds):
    prediction = np.empty(y_true.size, dtype=np.uint8)
    starts, stops = _binary_runs(y_true)
    best_f1 = -1.0
    best_tp = 0
    best_fp = y_true.size + 1
    best_fn = 0
    best_tn = 0
    for threshold in thresholds:
        for index in range(scores.size):
            prediction[index] = 1 if scores[index] > threshold else 0
        adjusted = _point_adjust_from_runs(prediction, starts, stops)
        tp, fp, fn, tn = _confusion_counts(y_true, adjusted)
        _, _, f1 = _prf_from_counts(tp, fp, fn)
        if f1 > best_f1 or (f1 == best_f1 and fp < best_fp):
            best_f1 = f1
            best_tp = tp
            best_fp = fp
            best_fn = fn
            best_tn = tn
    return best_tp, best_fp, best_fn, best_tn


@njit(cache=True, nogil=True)
def _adjust_event_scores(labels, scores, max_points):
    adjusted = scores.copy()
    starts, stops = _binary_runs(labels)
    for run in range(starts.size):
        start = starts[run]
        stop = stops[run]
        eligible_stop = min(stop, start + max_points)
        maximum = -np.inf
        for index in range(start, eligible_stop):
            maximum = max(maximum, scores[index])
        for index in range(start, stop):
            adjusted[index] = maximum
    return adjusted


@njit(cache=True, nogil=True)
def _delay_adjust_prediction(labels, predictions, max_points):
    adjusted = predictions.copy()
    starts, stops = _binary_runs(labels)
    for run in range(starts.size):
        start = starts[run]
        stop = stops[run]
        eligible_stop = min(stop, start + max_points)
        hit = False
        for index in range(start, eligible_stop):
            if predictions[index] != 0:
                hit = True
                break
        if hit:
            for index in range(start, stop):
                adjusted[index] = 1
    return adjusted


@njit(cache=True, nogil=True)
def _dilate_labels(labels, tolerance):
    dilated = np.zeros(labels.size, dtype=np.uint8)
    starts, stops = _binary_runs(labels)
    for run in range(starts.size):
        start = max(0, starts[run] - tolerance)
        stop = min(labels.size, stops[run] + tolerance)
        for index in range(start, stop):
            dilated[index] = 1
    return dilated


@njit(cache=True, nogil=True)
def _event_weight(length, mode, base):
    if mode == 0:
        return 1
    if mode == 1:
        return max(1, int(math.floor(math.log(length + base) / math.log(base))))
    if mode == 2:
        return max(1, int(math.floor(math.sqrt(length))))
    return length


@njit(cache=True, nogil=True)
def _event_compress(labels, values, mode, base):
    compressed_labels = np.empty(labels.size, dtype=np.uint8)
    compressed_values = np.empty(values.size, dtype=np.float64)
    source = 0
    target = 0
    while source < labels.size:
        if labels[source] == 0:
            compressed_labels[target] = 0
            compressed_values[target] = values[source]
            source += 1
            target += 1
            continue
        start = source
        maximum = values[source]
        source += 1
        while source < labels.size and labels[source] != 0:
            maximum = max(maximum, values[source])
            source += 1
        weight = _event_weight(source - start, mode, base)
        for _ in range(weight):
            compressed_labels[target] = 1
            compressed_values[target] = maximum
            target += 1
    return compressed_labels[:target], compressed_values[:target]


def _same_length(y_true: NDArray[np.uint8], y_pred: NDArray[np.uint8]) -> None:
    if y_true.size != y_pred.size:
        raise ValueError(
            f"y_true and y_pred must have the same length: {y_true.size} != {y_pred.size}"
        )


def _fixed_pair(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    labels = binary_array(y_true, name="y_true")
    predictions = binary_array(y_pred, name="y_pred")
    _same_length(labels, predictions)
    return labels, predictions


def _point_metrics_prepared(
    y_true: NDArray[np.uint8], y_pred: NDArray[np.uint8]
) -> PointMetrics:
    tp, fp, fn, tn = _confusion_counts(y_true, y_pred)
    return _point_metrics_from_counts(tp, fp, fn, tn)


def _point_metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> PointMetrics:
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    f05_denominator = 0.25 * precision + recall
    f05 = 1.25 * precision * recall / f05_denominator if f05_denominator else 0.0
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / math.sqrt(denominator) if denominator else 0.0
    return PointMetrics(accuracy, precision, recall, f1, mcc, f05, tp, fp, fn, tn)


def point_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> PointMetrics:
    """Evaluate a fixed binary prediction without threshold selection.

    ``y_true`` and ``y_pred`` must be equally sized, non-empty, one-dimensional
    arrays containing only ``0`` and ``1``. Values are interpreted in sequence
    order; timestamps and sampling intervals are not used. Inputs are never
    mutated.
    """
    labels, predictions = _fixed_pair(y_true, y_pred)
    return _point_metrics_prepared(labels, predictions)


def point_adjust(y_true: ArrayLike, y_pred: ArrayLike) -> NDArray[np.bool_]:
    """Fill a ground-truth event when at least one of its points was detected.

    Both inputs must satisfy the binary-vector contract documented by
    :func:`point_metrics`. A ground-truth event is one contiguous run of ones;
    the returned boolean array is newly allocated.
    """
    labels, predictions = _fixed_pair(y_true, y_pred)
    return _point_adjust(labels, predictions).astype(bool, copy=False)


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Return point-wise ROC AUC, or NaN when only one class is present.

    ``y_true`` must be a non-empty one-dimensional binary vector. ``y_score``
    must be an equally sized, finite numeric vector where larger values mean
    more anomalous points. Tied scores retain input order.
    """
    labels, scores = validate_pair(y_true, y_score)
    _, value, _ = _score_metrics_prepared(labels, scores, smoothing=0.0)
    return value


def average_precision(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Return point-wise average precision, or NaN when no anomaly exists.

    Labels and scores must satisfy the contract of :func:`roc_auc`; the score
    scale itself is unrestricted because only ranking is used.
    """
    labels, scores = validate_pair(y_true, y_score)
    value, _, _ = _score_metrics_prepared(labels, scores, smoothing=0.0)
    return value


def point_adjusted_average_precision(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Average precision after replacing each event by its maximum score.

    Labels and scores must satisfy the contract of :func:`roc_auc`.
    Contiguous positive label runs define events. Each event receives the
    maximum score observed anywhere inside that event before ranking.
    """
    labels, scores = validate_pair(y_true, y_score)
    return _point_adjusted_average_precision_prepared(labels, scores)


def _point_adjusted_average_precision_prepared(
    labels: NDArray[np.uint8], scores: NDArray[np.float64]
) -> float:
    adjusted = _adjust_event_scores(labels, scores, labels.size)
    value, _, _ = _score_metrics_prepared(labels, adjusted, smoothing=0.0)
    return value


def precision_at_k(
    y_true: ArrayLike, y_score: ArrayLike, *, k: int | None = None
) -> float:
    """Precision among the ``k`` highest scores.

    ``y_true`` and ``y_score`` follow the contract of :func:`roc_auc`. ``k``
    must be a non-negative integer no larger than the series length.
    By default ``k`` is the number of anomalous points, which is the TSAD
    benchmark convention. Ties are resolved by the stable score ordering.
    """
    labels, scores = validate_pair(y_true, y_score)
    if k is None:
        k = int(labels.sum())
    else:
        k = non_negative_integer(k, name="k")
    return _precision_at_k_prepared(labels, scores, k)


def _precision_at_k_prepared(
    labels: NDArray[np.uint8], scores: NDArray[np.float64], k: int
) -> float:
    if k == 0:
        return math.nan
    if k > labels.size:
        raise ValueError(f"k must not exceed the series length: {k} > {labels.size}")
    order = np.argsort(-scores, kind="stable")
    return float(labels[order[:k]].sum() / k)


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
    order = np.argsort(-scores, kind="stable")
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


def _best_point_metrics_prepared(
    labels: NDArray[np.uint8],
    scores: NDArray[np.float64],
    *,
    smoothing: float,
) -> PointMetrics:
    """Point metrics at the exact threshold that maximizes point F1."""
    positives = int(labels.sum())
    if positives == 0:
        return _point_metrics_from_counts(0, 0, 0, labels.size)
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    distinct = np.flatnonzero(np.diff(sorted_scores))
    threshold_indices = np.concatenate((distinct, [labels.size - 1]))
    true_positives = np.cumsum(sorted_labels, dtype=np.int64)[threshold_indices]
    predicted = threshold_indices + 1
    false_positives = predicted - true_positives
    precision = np.divide(
        true_positives,
        predicted,
        out=np.zeros_like(true_positives, dtype=np.float64),
        where=predicted != 0,
    )
    recall = (
        true_positives / positives
        if positives
        else np.zeros_like(precision, dtype=np.float64)
    )
    denominator = precision + recall + smoothing
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(precision),
        where=denominator != 0,
    )
    best = int(np.argmax(f1))
    tp = int(true_positives[best])
    fp = int(false_positives[best])
    fn = positives - tp
    tn = labels.size - positives - fp
    result = _point_metrics_from_counts(tp, fp, fn, tn)
    return PointMetrics(
        result.accuracy,
        result.precision,
        result.recall,
        float(f1[best]),
        result.mcc,
        result.f05,
        result.tp,
        result.fp,
        result.fn,
        result.tn,
    )


def best_point_f1(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    smoothing: float = 1e-5,
) -> float:
    """Best point F1 over the exact PR curve.

    ``y_true`` and ``y_score`` follow the contract of :func:`roc_auc`;
    ``smoothing`` must be a non-negative scalar. No deployment threshold is
    returned: this is an oracle ranking score.

    The default smoothing matches TSB-AD's ``Standard-F1`` definition.
    Passing ``smoothing=0`` gives the ordinary harmonic mean.
    """
    labels, scores = validate_pair(y_true, y_score)
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")
    return _best_point_f1_prepared(labels, scores, smoothing=smoothing)


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
        compressed_labels, compressed_scores = _event_compress(
            labels, scores, mode, base
        )
        result = _best_point_metrics_prepared(
            compressed_labels, compressed_scores, smoothing=1e-15
        )
    else:
        compressed_labels, compressed_predictions = _event_compress(
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
    """Return event-scaled point-adjusted precision, recall and F1.

    ``y_true`` and ``y_pred`` must be equal-length binary vectors. Contiguous
    positive labels form events; each event is represented by its maximum
    prediction and receives a duration weight selected by ``scale``. ``base``
    must be at least two when logarithmic scaling is used.
    """
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
    """Return the oracle-best event-scaled point-adjusted F1.

    ``y_true`` and ``y_score`` follow the contract of :func:`roc_auc`.
    Positive label runs define events, and ``scale``/``base`` have the same
    meaning and constraints as in :func:`event_adjusted_prf`.
    """
    labels, scores = validate_pair(y_true, y_score)
    return _event_adjusted_prepared(labels, scores, None, scale=scale, base=base).f1


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
        adjusted_scores = _adjust_event_scores(labels, scores, delay)
        result = _best_point_metrics_prepared(labels, adjusted_scores, smoothing=1e-15)
    else:
        adjusted = _delay_adjust_prediction(labels, predictions, delay)
        result = _point_metrics_prepared(labels, adjusted)
    return PRF1(result.precision, result.recall, result.f1)


def k_delay_point_adjusted_prf(
    y_true: ArrayLike, y_pred: ArrayLike, *, delay: int = 5
) -> PRF1:
    """Return point-adjusted PRF with a limited event-detection delay.

    ``y_true`` and ``y_pred`` must be equal-length binary vectors. ``delay`` is
    a positive number of points, counted from each event's first point. An
    event is filled only when it is detected within that prefix.
    """
    labels, predictions = _fixed_pair(y_true, y_pred)
    return _k_delay_prepared(
        labels, predictions.astype(np.float64), predictions, delay=delay
    )


def best_k_delay_point_adjusted_f1(
    y_true: ArrayLike, y_score: ArrayLike, *, delay: int = 5
) -> float:
    """Return oracle-best delay-limited point-adjusted F1.

    ``y_true`` and ``y_score`` follow the contract of :func:`roc_auc`, and
    ``delay`` follows :func:`k_delay_point_adjusted_prf`. Threshold selection
    uses the exact stable score ordering.
    """
    labels, scores = validate_pair(y_true, y_score)
    return _k_delay_prepared(labels, scores, None, delay=delay).f1


def _tolerance_labels(labels: NDArray[np.uint8], tolerance: int) -> NDArray[np.uint8]:
    tolerance = non_negative_integer(tolerance, name="tolerance")
    return _dilate_labels(labels, tolerance)


def tolerance_point_adjusted_prf(
    y_true: ArrayLike, y_pred: ArrayLike, *, tolerance: int = 5
) -> PRF1:
    """Return point-adjusted PRF after symmetric label dilation.

    ``y_true`` and ``y_pred`` must be equal-length binary vectors. ``tolerance``
    is a non-negative point count added to both sides of every labelled event;
    overlapping dilated events merge before point adjustment.
    """
    labels, predictions = _fixed_pair(y_true, y_pred)
    tolerant = _tolerance_labels(labels, tolerance)
    adjusted = _point_adjust(tolerant, predictions)
    result = _point_metrics_prepared(tolerant, adjusted)
    return PRF1(result.precision, result.recall, result.f1)


def best_tolerance_point_adjusted_f1(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    tolerance: int = 5,
    threshold_count: int = 100,
) -> float:
    """Return oracle-best F1 after symmetric label dilation.

    ``y_true`` and ``y_score`` follow the contract of :func:`roc_auc`.
    ``tolerance`` is a non-negative number of points and ``threshold_count``
    must be at least two; thresholds use the deterministic TSB-AD grid.
    """
    labels, scores = validate_pair(y_true, y_score)
    tolerant = _tolerance_labels(labels, tolerance)
    thresholds = threshold_grid(scores, threshold_count)
    return float(_best_point_adjusted_f1(tolerant, scores, thresholds))


def first_hit(y_true: ArrayLike, y_score: ArrayLike) -> FirstHit:
    """Return the first anomalous point's one-based score rank.

    ``y_true`` and ``y_score`` follow the contract of :func:`roc_auc`. Ranking
    is descending and stable for ties. With no positive label, ``rank`` is
    zero, ``fraction`` is NaN, and both hit flags are false.
    """
    labels, scores = validate_pair(y_true, y_score)
    if not labels.any():
        return FirstHit(0, math.nan, False, False)
    order = np.argsort(-scores, kind="stable")
    rank = int(np.flatnonzero(labels[order])[0]) + 1
    fraction = rank / labels.size
    return FirstHit(rank, fraction, fraction < 0.03, fraction < 0.10)
