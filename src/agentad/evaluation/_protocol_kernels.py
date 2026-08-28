"""Compiled numeric kernels for extended TSAD evaluation protocols.

All kernels operate on contiguous one-dimensional arrays and run encodings.
They deliberately avoid Python event objects so threshold scans stay inside
compiled code.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from ._kernels import binary_runs_kernel

__all__ = [
    "adjust_event_scores_kernel",
    "delay_adjust_prediction_kernel",
    "dilate_labels_kernel",
    "event_compress_kernel",
    "interval_prf_kernel",
    "best_interval_prf_kernel",
    "pate_auc_kernel",
    "pate_f1_kernel",
    "pate_pr_kernel",
]


@njit(cache=True, nogil=True)
def adjust_event_scores_kernel(labels, scores, max_points):
    """Fill each labelled event with its maximum eligible anomaly score."""
    adjusted = scores.copy()
    starts, stops = binary_runs_kernel(labels)
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
def delay_adjust_prediction_kernel(labels, predictions, max_points):
    """Apply point adjustment only when an event is hit within its delay limit."""
    adjusted = predictions.copy()
    starts, stops = binary_runs_kernel(labels)
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
def dilate_labels_kernel(labels, tolerance):
    """Dilate every labelled event on both sides by a fixed point margin."""
    dilated = np.zeros(labels.size, dtype=np.uint8)
    starts, stops = binary_runs_kernel(labels)
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
def event_compress_kernel(labels, values, mode, base):
    """Compress labelled runs while retaining every normal point in order."""
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


@njit(cache=True, nogil=True)
def interval_prf_kernel(labels, predictions):
    """Overlap-count interval precision, recall and F1."""
    true_starts, true_stops = binary_runs_kernel(labels)
    pred_starts, pred_stops = binary_runs_kernel(predictions)

    true_positive = 0
    false_positive = 0
    true_index = 0
    for pred_index in range(pred_starts.size):
        while (
            true_index < true_starts.size
            and true_stops[true_index] <= pred_starts[pred_index]
        ):
            true_index += 1
        scan = true_index
        overlaps = 0
        while scan < true_starts.size and true_starts[scan] < pred_stops[pred_index]:
            overlaps += 1
            scan += 1
        if overlaps:
            true_positive += overlaps
        else:
            false_positive += 1

    false_negative = 0
    pred_index = 0
    for true_index in range(true_starts.size):
        while (
            pred_index < pred_starts.size
            and pred_stops[pred_index] <= true_starts[true_index]
        ):
            pred_index += 1
        if (
            pred_index >= pred_starts.size
            or pred_starts[pred_index] >= true_stops[true_index]
        ):
            false_negative += 1

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


@njit(cache=True, nogil=True)
def best_interval_prf_kernel(labels, scores, thresholds):
    prediction = np.empty(labels.size, dtype=np.uint8)
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    for threshold in thresholds:
        for index in range(scores.size):
            prediction[index] = 1 if scores[index] > threshold else 0
        precision, recall, f1 = interval_prf_kernel(labels, prediction)
        if f1 > best_f1:
            best_precision = precision
            best_recall = recall
            best_f1 = f1
    return best_precision, best_recall, best_f1


@njit(cache=True, nogil=True)
def _pate_zones(starts, stops, length, early_buffer, delayed_buffer):
    zone = np.zeros(length, dtype=np.uint8)
    event_of = np.full(length, -1, dtype=np.int64)
    pre_starts = np.empty(starts.size, dtype=np.int64)
    post_stops = np.empty(starts.size, dtype=np.int64)
    previous_post_stop = 0
    for event in range(starts.size):
        start = starts[event]
        stop = stops[event]
        next_start = starts[event + 1] if event + 1 < starts.size else length
        pre_start = max(previous_post_stop, max(0, start - early_buffer))
        post_stop = min(next_start, stop + delayed_buffer)
        pre_starts[event] = pre_start
        post_stops[event] = post_stop
        for index in range(pre_start, start):
            zone[index] = 1
            event_of[index] = event
        for index in range(start, stop):
            zone[index] = 2
            event_of[index] = event
        for index in range(stop, post_stop):
            zone[index] = 3
            event_of[index] = event
        previous_post_stop = post_stop
    return zone, event_of, pre_starts, post_stops


@njit(cache=True, nogil=True)
def _clamp_unit(value):
    return min(1.0, max(0.0, value))


@njit(cache=True, nogil=True)
def _pate_pr_from_zones(
    predictions,
    starts,
    stops,
    zone,
    event_of,
    pre_starts,
    post_stops,
):
    detected = np.zeros(starts.size, dtype=np.int64)
    for index in range(predictions.size):
        event = event_of[index]
        if predictions[index] and event >= 0 and zone[index] == 2:
            detected[event] += 1

    true_positive = 0.0
    false_positive = 0.0
    false_negative = 0.0
    for index in range(predictions.size):
        event = event_of[index]
        current_zone = zone[index]
        if predictions[index]:
            if current_zone == 2:
                true_positive += 1.0
            elif current_zone == 1:
                if detected[event] == 0:
                    false_positive += 1.0
                else:
                    start = starts[event]
                    stop = stops[event]
                    length = stop - start
                    mean = (start + stop - 1) / 2.0
                    denominator = length * (mean - pre_starts[event])
                    distance = length * (mean - index)
                    weight = 1.0 if denominator <= 0 else 1.0 - distance / denominator
                    weight = _clamp_unit(weight)
                    true_positive += weight
                    false_positive += 1.0 - weight
            elif current_zone == 3:
                start = starts[event]
                stop = stops[event]
                length = stop - start
                mean = (start + stop - 1) / 2.0
                last = post_stops[event] - 1
                denominator = length * (last - mean)
                distance = length * (index - mean)
                weight = 1.0 if denominator <= 0 else 1.0 - distance / denominator
                weight = _clamp_unit(weight)
                true_positive += weight
                false_positive += 1.0 - weight
            else:
                false_positive += 1.0
        elif current_zone == 2:
            if detected[event] == 0:
                false_negative += 1.0
            else:
                start = starts[event]
                stop = stops[event]
                length = stop - start
                buffer_end = min(stop - 1, start + detected[event])
                if index <= buffer_end:
                    weight = 1.0
                else:
                    count = buffer_end - start + 1
                    distance = count * index - count * (start + buffer_end) / 2.0
                    max_distance = length * (length - 1) / 2.0
                    weight = 1.0 if max_distance <= 0 else 1.0 - distance / max_distance
                    weight = _clamp_unit(weight)
                false_negative += weight

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    return precision, recall


@njit(cache=True, nogil=True)
def pate_pr_kernel(labels, predictions, early_buffer, delayed_buffer):
    starts, stops = binary_runs_kernel(labels)
    if starts.size == 0:
        return np.nan, np.nan
    zone, event_of, pre_starts, post_stops = _pate_zones(
        starts, stops, labels.size, early_buffer, delayed_buffer
    )
    return _pate_pr_from_zones(
        predictions, starts, stops, zone, event_of, pre_starts, post_stops
    )


@njit(cache=True, nogil=True)
def pate_auc_kernel(labels, scores, thresholds, early_buffers, delayed_buffers):
    starts, stops = binary_runs_kernel(labels)
    if starts.size == 0:
        return np.nan, np.nan
    prediction = np.empty(labels.size, dtype=np.uint8)
    auc_total = 0.0
    best_f1_total = 0.0
    combinations = early_buffers.size * delayed_buffers.size
    for early_buffer in early_buffers:
        for delayed_buffer in delayed_buffers:
            zone, event_of, pre_starts, post_stops = _pate_zones(
                starts, stops, labels.size, early_buffer, delayed_buffer
            )
            previous_precision = 1.0
            previous_recall = 0.0
            area = 0.0
            best_f1 = 0.0
            for threshold in thresholds:
                for index in range(scores.size):
                    prediction[index] = 1 if scores[index] >= threshold else 0
                precision, recall = _pate_pr_from_zones(
                    prediction,
                    starts,
                    stops,
                    zone,
                    event_of,
                    pre_starts,
                    post_stops,
                )
                if recall >= previous_recall:
                    area += (recall - previous_recall) * (
                        precision + previous_precision
                    ) / 2.0
                    previous_precision = precision
                    previous_recall = recall
                if precision + recall:
                    f1 = 2.0 * precision * recall / (precision + recall)
                    best_f1 = max(best_f1, f1)
            auc_total += area
            best_f1_total += best_f1
    return auc_total / combinations, best_f1_total / combinations


@njit(cache=True, nogil=True)
def pate_f1_kernel(labels, predictions, early_buffers, delayed_buffers):
    starts, stops = binary_runs_kernel(labels)
    if starts.size == 0:
        return np.nan
    total = 0.0
    combinations = early_buffers.size * delayed_buffers.size
    for early_buffer in early_buffers:
        for delayed_buffer in delayed_buffers:
            zone, event_of, pre_starts, post_stops = _pate_zones(
                starts, stops, labels.size, early_buffer, delayed_buffer
            )
            precision, recall = _pate_pr_from_zones(
                predictions,
                starts,
                stops,
                zone,
                event_of,
                pre_starts,
                post_stops,
            )
            if precision + recall:
                total += 2.0 * precision * recall / (precision + recall)
    return total / combinations
