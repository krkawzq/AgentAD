"""Numba kernels for repeated binary and range-metric scans.

The kernels are deliberately serial. Evaluation commonly processes many short
series concurrently, where implicit per-series parallelism would oversubscribe
CPUs and cost more than the arithmetic it saves.

The point-adjust, event and range definitions are adapted from TheDatumOrg's
TSB-AD implementation (Apache License 2.0).

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = [
    "best_event_f1_kernel",
    "best_oracle_f1s_kernel",
    "best_point_adjusted_f1_kernel",
    "best_range_f1_kernel",
    "binary_runs_kernel",
    "confusion_counts_kernel",
    "event_prf_kernel",
    "point_adjust_kernel",
    "range_prf_kernel",
]


@njit(cache=True, nogil=True)
def confusion_counts_kernel(y_true, y_pred):
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
def binary_runs_kernel(values):
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
def point_adjust_kernel(y_true, y_pred):
    """Promote every detected ground-truth event to a fully positive run."""
    starts, stops = binary_runs_kernel(y_true)
    return _point_adjust_from_runs(y_pred, starts, stops)


@njit(cache=True, nogil=True)
def _prf_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


@njit(cache=True, nogil=True)
def _event_prf_from_runs(y_true, y_pred, starts, stops):
    tp, fp, _, _ = confusion_counts_kernel(y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    detected = 0
    for run in range(starts.size):
        hit = False
        for index in range(starts[run], stops[run]):
            if y_pred[index] != 0:
                hit = True
        if hit:
            detected += 1
    recall = detected / starts.size if starts.size else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


@njit(cache=True, nogil=True)
def event_prf_kernel(y_true, y_pred):
    """Composite event metric: point precision and event recall."""
    starts, stops = binary_runs_kernel(y_true)
    return _event_prf_from_runs(y_true, y_pred, starts, stops)


@njit(cache=True, nogil=True)
def _range_reward_runs(ref_starts, ref_stops, cand_starts, cand_stops, alpha):
    if ref_starts.size == 0:
        return 0.0

    existence = 0.0
    overlap_reward = 0.0
    candidate_index = 0
    for ref_index in range(ref_starts.size):
        start = ref_starts[ref_index]
        stop = ref_stops[ref_index]
        while (
            candidate_index < cand_starts.size and cand_stops[candidate_index] <= start
        ):
            candidate_index += 1
        scan = candidate_index
        overlap_points = 0
        cardinality = 0
        while scan < cand_starts.size and cand_starts[scan] < stop:
            left = start if start > cand_starts[scan] else cand_starts[scan]
            right = stop if stop < cand_stops[scan] else cand_stops[scan]
            if right > left:
                overlap_points += right - left
                cardinality += 1
            scan += 1
        if overlap_points:
            existence += 1.0
            overlap_reward += (overlap_points / (stop - start)) / cardinality

    count = ref_starts.size
    return (alpha * existence + (1.0 - alpha) * overlap_reward) / count


@njit(cache=True, nogil=True)
def range_prf_kernel(y_true, y_pred, alpha=0.2):
    true_starts, true_stops = binary_runs_kernel(y_true)
    pred_starts, pred_stops = binary_runs_kernel(y_pred)
    recall = _range_reward_runs(true_starts, true_stops, pred_starts, pred_stops, alpha)
    precision = _range_reward_runs(
        pred_starts, pred_stops, true_starts, true_stops, 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


@njit(cache=True, nogil=True)
def best_point_adjusted_f1_kernel(y_true, scores, thresholds):
    prediction = np.empty(y_true.size, dtype=np.uint8)
    starts, stops = binary_runs_kernel(y_true)
    best = 0.0
    for threshold in thresholds:
        for index in range(scores.size):
            prediction[index] = 1 if scores[index] > threshold else 0
        adjusted = _point_adjust_from_runs(prediction, starts, stops)
        tp, fp, fn, _ = confusion_counts_kernel(y_true, adjusted)
        _, _, f1 = _prf_from_counts(tp, fp, fn)
        if f1 > best:
            best = f1
    return best


@njit(cache=True, nogil=True)
def best_event_f1_kernel(y_true, scores, thresholds):
    prediction = np.empty(y_true.size, dtype=np.uint8)
    starts, stops = binary_runs_kernel(y_true)
    best = 0.0
    for threshold in thresholds:
        for index in range(scores.size):
            prediction[index] = 1 if scores[index] > threshold else 0
        _, _, f1 = _event_prf_from_runs(y_true, prediction, starts, stops)
        if f1 > best:
            best = f1
    return best


@njit(cache=True, nogil=True)
def best_range_f1_kernel(y_true, scores, thresholds, alpha=0.2):
    prediction = np.empty(y_true.size, dtype=np.uint8)
    true_starts, true_stops = binary_runs_kernel(y_true)
    best = 0.0
    for threshold in thresholds:
        for index in range(scores.size):
            prediction[index] = 1 if scores[index] > threshold else 0
        pred_starts, pred_stops = binary_runs_kernel(prediction)
        recall = _range_reward_runs(
            true_starts, true_stops, pred_starts, pred_stops, alpha
        )
        precision = _range_reward_runs(
            pred_starts, pred_stops, true_starts, true_stops, 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        if f1 > best:
            best = f1
    return best


@njit(cache=True, nogil=True)
def best_oracle_f1s_kernel(y_true, scores, thresholds, alpha=0.2):
    """Best PA, event and range F1 values in one prediction sweep."""
    prediction = np.empty(y_true.size, dtype=np.uint8)
    true_starts, true_stops = binary_runs_kernel(y_true)
    best_pa = 0.0
    best_event = 0.0
    best_range = 0.0
    for threshold in thresholds:
        for index in range(scores.size):
            prediction[index] = 1 if scores[index] > threshold else 0

        adjusted = _point_adjust_from_runs(prediction, true_starts, true_stops)
        tp, fp, fn, _ = confusion_counts_kernel(y_true, adjusted)
        _, _, pa_f1 = _prf_from_counts(tp, fp, fn)
        if pa_f1 > best_pa:
            best_pa = pa_f1

        _, _, event_f1 = _event_prf_from_runs(
            y_true, prediction, true_starts, true_stops
        )
        if event_f1 > best_event:
            best_event = event_f1

        pred_starts, pred_stops = binary_runs_kernel(prediction)
        recall = _range_reward_runs(
            true_starts, true_stops, pred_starts, pred_stops, alpha
        )
        precision = _range_reward_runs(
            pred_starts, pred_stops, true_starts, true_stops, 0.0
        )
        range_f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        if range_f1 > best_range:
            best_range = range_f1
    return best_pa, best_event, best_range
