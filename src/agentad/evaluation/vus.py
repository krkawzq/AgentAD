"""Range-AUC volume-under-surface metrics.

This implements the TSB-AD VUS definition while replacing its nested
``window x threshold x point`` Python loops with a compiled rank/prefix-sum
kernel. The algorithm remains serial and uses O(n + windows x thresholds)
working memory.

Adapted from TheDatumOrg/TSB-AD under Apache-2.0; see
``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._kernels import binary_runs_kernel
from ._types import AUCResult, PRF1
from ._validation import (
    binary_array,
    non_negative_integer,
    threshold_count as validate_threshold_count,
    validate_pair,
)

__all__ = [
    "VUSResult",
    "fixed_vus_prf",
    "range_auc",
    "volume_under_surface",
]


@dataclass(frozen=True, slots=True)
class VUSResult:
    """VUS scalars and their per-window threshold surfaces."""

    roc: float
    pr: float
    roc_by_window: NDArray[np.float64]
    pr_by_window: NDArray[np.float64]
    tpr: NDArray[np.float64]
    fpr: NDArray[np.float64]
    precision: NDArray[np.float64]
    windows: NDArray[np.int64]


@njit(cache=True, nogil=True)
def _padded_runs(starts, stops, length, window):
    padded_starts = np.empty(starts.size, dtype=np.int64)
    padded_stops = np.empty(stops.size, dtype=np.int64)
    half = window // 2
    count = 0
    for index in range(starts.size):
        start = starts[index] - half
        if start < 0:
            start = 0
        stop = stops[index] + half
        if stop > length:
            stop = length
        if count and padded_stops[count - 1] > start:
            if stop > padded_stops[count - 1]:
                padded_stops[count - 1] = stop
        else:
            padded_starts[count] = start
            padded_stops[count] = stop
            count += 1
    return padded_starts[:count], padded_stops[:count]


@njit(cache=True, nogil=True)
def _mask_runs_into(mask, starts, stops):
    mask.fill(0)
    for run in range(starts.size):
        for index in range(starts[run], stops[run]):
            mask[index] = 1


@njit(cache=True, nogil=True)
def _mask_runs(starts, stops, length):
    mask = np.empty(length, dtype=np.uint8)
    _mask_runs_into(mask, starts, stops)
    return mask


@njit(cache=True, nogil=True)
def _extend_labels_into(extended, labels, starts, stops, window):
    for index in range(labels.size):
        extended[index] = labels[index]
    if window == 0:
        return
    half = window // 2
    for run in range(starts.size):
        start = starts[run]
        last = stops[run] - 1
        right_stop = last + half + 1
        if right_stop > labels.size:
            right_stop = labels.size
        for index in range(last + 1, right_stop):
            extended[index] += math.sqrt(1.0 - (index - last) / window)
        left_start = start - half
        if left_start < 0:
            left_start = 0
        for index in range(left_start, start):
            extended[index] += math.sqrt(1.0 - (start - index) / window)
    for index in range(extended.size):
        if extended[index] > 1.0:
            extended[index] = 1.0


@njit(cache=True, nogil=True)
def _vus_kernel(
    labels,
    scores,
    thresholds,
    descending_order,
    sorted_scores,
    window_size,
    store_surface,
):
    length = labels.size
    threshold_count = thresholds.size
    starts, stops = binary_runs_kernel(labels)
    max_starts, max_stops = _padded_runs(starts, stops, length, window_size)
    max_mask = _mask_runs(max_starts, max_stops, length)

    prediction_counts = np.empty(threshold_count, dtype=np.int64)
    end = 0
    for threshold_index in range(threshold_count):
        threshold = thresholds[threshold_index]
        while end < length and sorted_scores[end] >= threshold:
            end += 1
        prediction_counts[threshold_index] = end

    gt_prefix = np.empty(threshold_count, dtype=np.float64)
    running = 0.0
    query = 0
    for rank in range(length):
        running += labels[descending_order[rank]]
        count = rank + 1
        while query < threshold_count and prediction_counts[query] == count:
            gt_prefix[query] = running
            query += 1

    if store_surface:
        tpr = np.zeros((window_size + 1, threshold_count + 2), dtype=np.float64)
        fpr = np.zeros((window_size + 1, threshold_count + 2), dtype=np.float64)
        precision = np.ones((window_size + 1, threshold_count + 1), dtype=np.float64)
    else:
        tpr = np.empty((0, 0), dtype=np.float64)
        fpr = np.empty((0, 0), dtype=np.float64)
        precision = np.empty((0, 0), dtype=np.float64)
    roc_by_window = np.empty(window_size + 1, dtype=np.float64)
    pr_by_window = np.empty(window_size + 1, dtype=np.float64)
    positives = float(labels.sum())
    extended = np.empty(length, dtype=np.float64)
    current_mask = np.empty(length, dtype=np.uint8)
    max_prefix = np.empty(threshold_count, dtype=np.float64)
    current_prefix = np.empty(threshold_count, dtype=np.float64)
    event_max_scores = np.empty(starts.size, dtype=np.float64)

    for window in range(window_size + 1):
        _extend_labels_into(extended, labels, starts, stops, window)
        current_starts, current_stops = _padded_runs(starts, stops, length, window)
        _mask_runs_into(current_mask, current_starts, current_stops)

        sum_max = 0.0
        sum_current = 0.0
        for index in range(length):
            if max_mask[index]:
                sum_max += extended[index]
            if current_mask[index]:
                sum_current += extended[index]

        running_max = 0.0
        running_current = 0.0
        query = 0
        for rank in range(length):
            index = descending_order[rank]
            if max_mask[index]:
                running_max += extended[index]
            if current_mask[index]:
                running_current += extended[index]
            count = rank + 1
            while query < threshold_count and prediction_counts[query] == count:
                max_prefix[query] = running_max
                current_prefix[query] = running_current
                query += 1

        for run in range(current_starts.size):
            maximum = -np.inf
            for index in range(current_starts[run], current_stops[run]):
                if scores[index] > maximum:
                    maximum = scores[index]
            event_max_scores[run] = maximum

        auc = 0.0
        average_precision = 0.0
        previous_tpr = 0.0
        previous_fpr = 0.0
        for threshold_index in range(threshold_count):
            count = prediction_counts[threshold_index]
            true_positive = max_prefix[threshold_index]
            predicted_current = current_prefix[threshold_index]
            predicted_gt = gt_prefix[threshold_index]
            label_mass = (
                sum_max - sum_current + predicted_current - predicted_gt + positives
            )
            effective_positives = (positives + label_mass) / 2.0
            recall = true_positive / effective_positives
            if recall > 1.0:
                recall = 1.0

            detected = 0
            threshold = thresholds[threshold_index]
            for run in range(current_starts.size):
                if event_max_scores[run] >= threshold:
                    detected += 1
            existence_ratio = detected / current_starts.size
            true_positive_rate = recall * existence_ratio
            false_positive = count - true_positive
            effective_negatives = length - effective_positives
            false_positive_rate = false_positive / effective_negatives
            current_precision = true_positive / count

            auc += (
                (false_positive_rate - previous_fpr)
                * (true_positive_rate + previous_tpr)
                / 2.0
            )
            average_precision += (true_positive_rate - previous_tpr) * current_precision
            previous_tpr = true_positive_rate
            previous_fpr = false_positive_rate

            if store_surface:
                curve_index = threshold_index + 1
                tpr[window, curve_index] = true_positive_rate
                fpr[window, curve_index] = false_positive_rate
                precision[window, curve_index] = current_precision

        auc += (1.0 - previous_fpr) * (1.0 + previous_tpr) / 2.0
        roc_by_window[window] = auc
        pr_by_window[window] = average_precision
        if store_surface:
            tpr[window, threshold_count + 1] = 1.0
            fpr[window, threshold_count + 1] = 1.0

    return tpr, fpr, precision, roc_by_window, pr_by_window


@njit(cache=True, nogil=True)
def _fixed_vus_prf_kernel(labels, predictions, window_size):
    starts, stops = binary_runs_kernel(labels)
    if starts.size == 0:
        return np.nan, np.nan, np.nan
    length = labels.size
    positives = float(labels.sum())
    prediction_count = float(predictions.sum())
    max_starts, max_stops = _padded_runs(starts, stops, length, window_size)
    max_mask = _mask_runs(max_starts, max_stops, length)
    current_mask = np.empty(length, dtype=np.uint8)
    extended = np.empty(length, dtype=np.float64)
    precision_total = 0.0
    recall_total = 0.0
    f1_total = 0.0

    for window in range(window_size + 1):
        _extend_labels_into(extended, labels, starts, stops, window)
        current_starts, current_stops = _padded_runs(starts, stops, length, window)
        _mask_runs_into(current_mask, current_starts, current_stops)
        true_positive = 0.0
        label_mass = 0.0
        for index in range(length):
            if not max_mask[index]:
                continue
            if labels[index]:
                label_mass += 1.0
                if predictions[index]:
                    true_positive += 1.0
            elif current_mask[index] and predictions[index]:
                label_mass += extended[index]
                true_positive += extended[index]
        effective_positives = (positives + label_mass) / 2.0
        recall = true_positive / effective_positives if effective_positives else 0.0
        if recall > 1.0:
            recall = 1.0
        precision = true_positive / prediction_count if prediction_count else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        precision_total += precision
        recall_total += recall
        f1_total += f1

    count = window_size + 1
    return precision_total / count, recall_total / count, f1_total / count


def volume_under_surface(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    window_size: int = 100,
    threshold_count: int = 250,
) -> VUSResult:
    """Compute VUS-ROC/VUS-PR and their underlying surfaces."""
    labels, scores = validate_pair(y_true, y_score)
    window_size = non_negative_integer(window_size, name="window_size")
    threshold_count = validate_threshold_count(threshold_count)

    return _volume_under_surface_prepared(
        labels,
        scores,
        window_size=window_size,
        threshold_count=threshold_count,
    )


def range_auc(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    window_size: int = 100,
    threshold_count: int = 250,
) -> AUCResult:
    """Range-AUC-ROC and Range-AUC-PR at one tolerance window."""
    result = volume_under_surface(
        y_true,
        y_score,
        window_size=window_size,
        threshold_count=threshold_count,
    )
    return AUCResult(float(result.roc_by_window[-1]), float(result.pr_by_window[-1]))


def fixed_vus_prf(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    window_size: int = 100,
) -> PRF1:
    """VUS precision, recall and F1 for one fixed binary decision."""
    labels = binary_array(y_true, name="y_true")
    predictions = binary_array(y_pred, name="y_pred")
    if labels.size != predictions.size:
        raise ValueError(
            f"y_true and y_pred must have the same length: {labels.size} != {predictions.size}"
        )
    window_size = non_negative_integer(window_size, name="window_size")
    precision, recall, f1 = _fixed_vus_prf_kernel(
        labels, predictions, window_size
    )
    return PRF1(float(precision), float(recall), float(f1))


def _volume_under_surface_prepared(
    labels: NDArray[np.uint8],
    scores: NDArray[np.float64],
    *,
    window_size: int,
    threshold_count: int,
    return_surface: bool = True,
) -> VUSResult:
    windows = np.arange(window_size + 1, dtype=np.int64)
    positives = int(labels.sum())
    if positives == 0 or positives == labels.size:
        shape = (window_size + 1, threshold_count + 2) if return_surface else (0, 0)
        precision_shape = (
            (window_size + 1, threshold_count + 1) if return_surface else (0, 0)
        )
        per_window = np.full(window_size + 1, math.nan)
        return VUSResult(
            math.nan,
            math.nan,
            per_window,
            per_window.copy(),
            np.full(shape, math.nan),
            np.full(shape, math.nan),
            np.full(precision_shape, math.nan),
            windows,
        )

    descending_order = np.ascontiguousarray(np.argsort(scores)[::-1], dtype=np.int64)
    sorted_scores = np.ascontiguousarray(scores[descending_order])
    indices = np.linspace(0, scores.size - 1, threshold_count).astype(np.int64)
    thresholds = np.ascontiguousarray(sorted_scores[indices])
    tpr, fpr, precision, roc_by_window, pr_by_window = _vus_kernel(
        labels,
        scores,
        thresholds,
        descending_order,
        sorted_scores,
        window_size,
        return_surface,
    )
    return VUSResult(
        float(roc_by_window.mean()),
        float(pr_by_window.mean()),
        roc_by_window,
        pr_by_window,
        tpr,
        fpr,
        precision,
        windows,
    )
