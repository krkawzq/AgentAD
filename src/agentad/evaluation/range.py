"""Point-adjusted, event, range and affiliation metrics.

Parts are adapted from TheDatumOrg/TSB-AD under Apache-2.0; see
``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._types import PRF1
from ._validation import (
    binary_array,
    non_negative_integer,
    threshold_count as validate_threshold_count,
    threshold_grid,
    validate_pair,
)
from .point import (
    _best_point_adjusted_f1,
    _best_point_adjusted_prf,
    _binary_runs,
    _confusion_counts,
    _fixed_pair,
    _point_adjust,
    _point_adjust_from_runs,
    _point_metrics_prepared,
    _prf_from_counts,
)

__all__ = [
    "PRF1",
    "affiliation_prf",
    "best_affiliation_f1",
    "best_affiliation_prf",
    "best_event_f1",
    "best_event_prf",
    "best_point_adjusted_f1",
    "best_point_adjusted_prf",
    "best_range_f1",
    "best_range_prf",
    "best_interval_f1",
    "event_prf",
    "interval_prf",
    "pate",
    "pate_f1",
    "pate_prf",
    "point_adjusted_prf",
    "range_prf",
]


@njit(cache=True, nogil=True)
def _event_prf_from_runs(y_true, y_pred, starts, stops):
    tp, fp, _, _ = _confusion_counts(y_true, y_pred)
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
def _event_prf(y_true, y_pred):
    starts, stops = _binary_runs(y_true)
    return _event_prf_from_runs(y_true, y_pred, starts, stops)


@njit(cache=True, nogil=True)
def _range_reward(ref_starts, ref_stops, cand_starts, cand_stops, alpha):
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
def _range_prf(y_true, y_pred, alpha=0.2):
    true_starts, true_stops = _binary_runs(y_true)
    pred_starts, pred_stops = _binary_runs(y_pred)
    recall = _range_reward(true_starts, true_stops, pred_starts, pred_stops, alpha)
    precision = _range_reward(pred_starts, pred_stops, true_starts, true_stops, 0.0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


@njit(cache=True, nogil=True)
def _best_event_f1(y_true, scores, thresholds):
    prediction = np.empty(y_true.size, dtype=np.uint8)
    starts, stops = _binary_runs(y_true)
    best = 0.0
    for threshold in thresholds:
        for index in range(scores.size):
            prediction[index] = 1 if scores[index] > threshold else 0
        _, _, f1 = _event_prf_from_runs(y_true, prediction, starts, stops)
        if f1 > best:
            best = f1
    return best


@njit(cache=True, nogil=True)
def _best_event_prf(y_true, scores, thresholds):
    prediction = np.empty(y_true.size, dtype=np.uint8)
    starts, stops = _binary_runs(y_true)
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    for threshold in thresholds:
        for index in range(scores.size):
            prediction[index] = 1 if scores[index] > threshold else 0
        precision, recall, f1 = _event_prf_from_runs(
            y_true, prediction, starts, stops
        )
        if f1 > best_f1:
            best_precision = precision
            best_recall = recall
            best_f1 = f1
    return best_precision, best_recall, best_f1


@njit(cache=True, nogil=True)
def _best_range_f1(y_true, scores, thresholds, alpha=0.2):
    prediction = np.empty(y_true.size, dtype=np.uint8)
    true_starts, true_stops = _binary_runs(y_true)
    best = 0.0
    for threshold in thresholds:
        for index in range(scores.size):
            prediction[index] = 1 if scores[index] > threshold else 0
        pred_starts, pred_stops = _binary_runs(prediction)
        recall = _range_reward(
            true_starts, true_stops, pred_starts, pred_stops, alpha
        )
        precision = _range_reward(
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
def _best_range_prf(y_true, scores, thresholds, alpha=0.2):
    prediction = np.empty(y_true.size, dtype=np.uint8)
    true_starts, true_stops = _binary_runs(y_true)
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    for threshold in thresholds:
        for index in range(scores.size):
            prediction[index] = 1 if scores[index] > threshold else 0
        pred_starts, pred_stops = _binary_runs(prediction)
        recall = _range_reward(
            true_starts, true_stops, pred_starts, pred_stops, alpha
        )
        precision = _range_reward(
            pred_starts, pred_stops, true_starts, true_stops, 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        if f1 > best_f1:
            best_precision = precision
            best_recall = recall
            best_f1 = f1
    return best_precision, best_recall, best_f1


@njit(cache=True, nogil=True)
def _best_oracle_prfs(y_true, scores, thresholds, alpha=0.2):
    prediction = np.empty(y_true.size, dtype=np.uint8)
    true_starts, true_stops = _binary_runs(y_true)
    best_pa_p = 0.0
    best_pa_r = 0.0
    best_pa_f = 0.0
    best_event_p = 0.0
    best_event_r = 0.0
    best_event_f = 0.0
    best_range_p = 0.0
    best_range_r = 0.0
    best_range_f = 0.0
    for threshold in thresholds:
        for index in range(scores.size):
            prediction[index] = 1 if scores[index] > threshold else 0

        adjusted = _point_adjust_from_runs(prediction, true_starts, true_stops)
        tp, fp, fn, _ = _confusion_counts(y_true, adjusted)
        pa_p, pa_r, pa_f = _prf_from_counts(tp, fp, fn)
        if pa_f > best_pa_f:
            best_pa_p = pa_p
            best_pa_r = pa_r
            best_pa_f = pa_f

        event_p, event_r, event_f = _event_prf_from_runs(
            y_true, prediction, true_starts, true_stops
        )
        if event_f > best_event_f:
            best_event_p = event_p
            best_event_r = event_r
            best_event_f = event_f

        pred_starts, pred_stops = _binary_runs(prediction)
        range_r = _range_reward(
            true_starts, true_stops, pred_starts, pred_stops, alpha
        )
        range_p = _range_reward(
            pred_starts, pred_stops, true_starts, true_stops, 0.0
        )
        range_f = (
            2.0 * range_p * range_r / (range_p + range_r)
            if range_p + range_r
            else 0.0
        )
        if range_f > best_range_f:
            best_range_p = range_p
            best_range_r = range_r
            best_range_f = range_f
    return (
        best_pa_p,
        best_pa_r,
        best_pa_f,
        best_event_p,
        best_event_r,
        best_event_f,
        best_range_p,
        best_range_r,
        best_range_f,
    )


@njit(cache=True, nogil=True)
def _precision_outside(i0, i1, j0, j1, e0, e1):
    d_min = max(i0 - j1, j0 - i1)
    d_max = max(i1 - j1, j0 - i0)
    margin = min(j0 - e0, e1 - j1)
    minimum_piece = 0.5 * (
        min(d_max, margin) ** 2 - min(d_min, margin) ** 2
    ) + margin * (max(d_max, margin) - max(d_min, margin))
    linear_piece = 0.5 * (d_max**2 - d_min**2)
    remaining_piece = (j1 - j0) * (i1 - i0)
    return (i1 - i0) - (minimum_piece + linear_piece + remaining_piece) / (e1 - e0)


@njit(cache=True, nogil=True)
def _precision_integral(i0, i1, j0, j1, e0, e1):
    total = 0.0
    left_stop = min(i1, j0)
    if left_stop > i0:
        total += _precision_outside(i0, left_stop, j0, j1, e0, e1)
    middle_start = max(i0, j0)
    middle_stop = min(i1, j1)
    if middle_stop > middle_start:
        total += middle_stop - middle_start
    right_start = max(i0, j1)
    if i1 > right_start:
        total += _precision_outside(right_start, i1, j0, j1, e0, e1)
    return total


@njit(cache=True, nogil=True)
def _square_difference(a, b):
    return b * b - a * a


@njit(cache=True, nogil=True)
def _recall_outside(i0, i1, j0, j1, e0, e1):
    if j1 <= i0:
        pivot = i0
    else:
        pivot = i1
    if pivot <= e0 or pivot >= e1:
        return 0.0

    e_mean = (e0 + e1) / 2.0
    left_mean = (e0 + pivot) / 2.0
    right_mean = (e1 + pivot) / 2.0

    before0 = j0
    before1 = min(j1, e_mean)
    after0 = max(j0, e_mean)
    after1 = j1

    bbe0 = before0
    bbe1 = min(before1, left_mean)
    bbi0 = max(before0, left_mean)
    bbi1 = before1
    abi0 = after0
    abi1 = min(after1, right_mean)
    abe0 = max(after0, right_mean)
    abe1 = after1

    part1 = 0.0
    part2 = 0.0
    part3 = 0.0
    part4 = 0.0
    if pivot >= j1:
        if bbe1 > bbe0:
            part1 = (pivot - e0) * (bbe1 - bbe0)
        if bbi1 > bbi0:
            part2 = 2.0 * pivot * (bbi1 - bbi0) - _square_difference(bbi0, bbi1)
        if abi1 > abi0:
            part3 = 2.0 * pivot * (abi1 - abi0) - _square_difference(abi0, abi1)
        if abe1 > abe0:
            part4 = (e1 + pivot) * (abe1 - abe0) - _square_difference(abe0, abe1)
    else:
        if bbe1 > bbe0:
            part1 = _square_difference(bbe0, bbe1) - (e0 + pivot) * (bbe1 - bbe0)
        if bbi1 > bbi0:
            part2 = _square_difference(bbi0, bbi1) - 2.0 * pivot * (bbi1 - bbi0)
        if abi1 > abi0:
            part3 = _square_difference(abi0, abi1) - 2.0 * pivot * (abi1 - abi0)
        if abe1 > abe0:
            part4 = (e1 - pivot) * (abe1 - abe0)
    integral = part1 + part2 + part3 + part4
    return (j1 - j0) - integral / (e1 - e0)


@njit(cache=True, nogil=True)
def _recall_integral(i0, i1, j0, j1, e0, e1):
    total = 0.0
    left_stop = min(j1, i0)
    if left_stop > j0:
        total += _recall_outside(i0, i1, j0, left_stop, e0, e1)
    middle_start = max(j0, i0)
    middle_stop = min(j1, i1)
    if middle_stop > middle_start:
        total += middle_stop - middle_start
    right_start = max(j0, i1)
    if j1 > right_start:
        total += _recall_outside(i0, i1, right_start, j1, e0, e1)
    return total


@njit(cache=True, nogil=True)
def _affiliation_pr_from_runs(pred_starts, pred_stops, gt_starts, gt_stops, length):
    if gt_starts.size == 0:
        return np.nan, np.nan
    if pred_starts.size == 0:
        return 0.0, 0.0

    precision_sum = 0.0
    precision_count = 0
    recall_sum = 0.0
    first_candidate = 0
    for gt_index in range(gt_starts.size):
        j0 = float(gt_starts[gt_index])
        j1 = float(gt_stops[gt_index])
        e0 = (
            0.0
            if gt_index == 0
            else (gt_stops[gt_index - 1] + gt_starts[gt_index]) / 2.0
        )
        e1 = (
            float(length)
            if gt_index + 1 == gt_starts.size
            else (gt_stops[gt_index] + gt_starts[gt_index + 1]) / 2.0
        )

        while first_candidate < pred_starts.size and pred_stops[first_candidate] <= e0:
            first_candidate += 1
        candidate_stop = first_candidate
        while candidate_stop < pred_starts.size and pred_starts[candidate_stop] < e1:
            candidate_stop += 1

        precision_numerator = 0.0
        precision_denominator = 0.0
        recall_numerator = 0.0
        for pred_index in range(first_candidate, candidate_stop):
            i0 = max(float(pred_starts[pred_index]), e0)
            i1 = min(float(pred_stops[pred_index]), e1)
            if i1 <= i0:
                continue
            precision_numerator += _precision_integral(i0, i1, j0, j1, e0, e1)
            precision_denominator += i1 - i0

            if pred_index == first_candidate:
                pred_zone_start = e0
            else:
                previous_stop = min(float(pred_stops[pred_index - 1]), e1)
                pred_zone_start = (previous_stop + i0) / 2.0
            if pred_index + 1 == candidate_stop:
                pred_zone_stop = e1
            else:
                next_start = max(float(pred_starts[pred_index + 1]), e0)
                pred_zone_stop = (i1 + next_start) / 2.0

            piece_start = max(j0, pred_zone_start)
            piece_stop = min(j1, pred_zone_stop)
            if piece_stop > piece_start:
                recall_numerator += _recall_integral(
                    i0, i1, piece_start, piece_stop, e0, e1
                )

        if precision_denominator > 0.0:
            precision_sum += precision_numerator / precision_denominator
            precision_count += 1
        recall_sum += recall_numerator / (j1 - j0)

    precision = precision_sum / precision_count if precision_count else np.nan
    return precision, recall_sum / gt_starts.size


@njit(cache=True, nogil=True)
def _affiliation_prf(labels, predictions):
    gt_starts, gt_stops = _binary_runs(labels)
    pred_starts, pred_stops = _binary_runs(predictions)
    precision, recall = _affiliation_pr_from_runs(
        pred_starts, pred_stops, gt_starts, gt_stops, labels.size
    )
    if np.isnan(precision) or np.isnan(recall):
        return precision, recall, np.nan
    f1 = 2.0 * precision * recall / (precision + recall + 1e-15)
    return precision, recall, f1


@njit(cache=True, nogil=True)
def _best_affiliation_f1(labels, scores, thresholds):
    gt_starts, gt_stops = _binary_runs(labels)
    if gt_starts.size == 0:
        return np.nan
    prediction = np.empty(labels.size, dtype=np.uint8)
    best = 0.0
    for threshold in thresholds:
        for index in range(scores.size):
            prediction[index] = 1 if scores[index] > threshold else 0
        pred_starts, pred_stops = _binary_runs(prediction)
        precision, recall = _affiliation_pr_from_runs(
            pred_starts, pred_stops, gt_starts, gt_stops, labels.size
        )
        value = 2.0 * precision * recall / (precision + recall + 1e-15)
        if not np.isnan(value) and value > best:
            best = value
    return best


@njit(cache=True, nogil=True)
def _best_affiliation_prf(labels, scores, thresholds):
    gt_starts, gt_stops = _binary_runs(labels)
    if gt_starts.size == 0:
        return np.nan, np.nan, np.nan
    prediction = np.empty(labels.size, dtype=np.uint8)
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    for threshold in thresholds:
        for index in range(scores.size):
            prediction[index] = 1 if scores[index] > threshold else 0
        pred_starts, pred_stops = _binary_runs(prediction)
        precision, recall = _affiliation_pr_from_runs(
            pred_starts, pred_stops, gt_starts, gt_stops, labels.size
        )
        f1 = 2.0 * precision * recall / (precision + recall + 1e-15)
        if not np.isnan(f1) and f1 > best_f1:
            best_precision = precision
            best_recall = recall
            best_f1 = f1
    return best_precision, best_recall, best_f1


@njit(cache=True, nogil=True)
def _interval_prf(labels, predictions):
    true_starts, true_stops = _binary_runs(labels)
    pred_starts, pred_stops = _binary_runs(predictions)

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
def _best_interval_prf(labels, scores, thresholds):
    prediction = np.empty(labels.size, dtype=np.uint8)
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    for threshold in thresholds:
        for index in range(scores.size):
            prediction[index] = 1 if scores[index] > threshold else 0
        precision, recall, f1 = _interval_prf(labels, prediction)
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
def _pate_pr(labels, predictions, early_buffer, delayed_buffer):
    starts, stops = _binary_runs(labels)
    if starts.size == 0:
        return np.nan, np.nan
    zone, event_of, pre_starts, post_stops = _pate_zones(
        starts, stops, labels.size, early_buffer, delayed_buffer
    )
    return _pate_pr_from_zones(
        predictions, starts, stops, zone, event_of, pre_starts, post_stops
    )


@njit(cache=True, nogil=True)
def _pate_auc(labels, scores, thresholds, early_buffers, delayed_buffers):
    starts, stops = _binary_runs(labels)
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
def _pate_f1(labels, predictions, early_buffers, delayed_buffers):
    starts, stops = _binary_runs(labels)
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


def _triplet(values: tuple[float, float, float]) -> PRF1:
    return PRF1(*(float(value) for value in values))


def point_adjusted_prf(y_true: ArrayLike, y_pred: ArrayLike) -> PRF1:
    """Return point precision, recall and F1 after point adjustment.

    ``y_true`` and ``y_pred`` must be equally sized, non-empty,
    one-dimensional binary vectors. Contiguous positive labels form events;
    an event is filled when any of its points is predicted positive.
    """
    labels, predictions = _fixed_pair(y_true, y_pred)
    adjusted = _point_adjust(labels, predictions)
    result = _point_metrics_prepared(labels, adjusted)
    return PRF1(result.precision, result.recall, result.f1)


def event_prf(y_true: ArrayLike, y_pred: ArrayLike) -> PRF1:
    """Return point precision, event recall and their harmonic mean.

    Inputs follow the binary-vector contract of :func:`point_adjusted_prf`.
    Recall counts a labelled event as detected when at least one point in its
    contiguous positive run is predicted positive.
    """
    labels, predictions = _fixed_pair(y_true, y_pred)
    return _triplet(_event_prf(labels, predictions))


def range_prf(y_true: ArrayLike, y_pred: ArrayLike, *, alpha: float = 0.2) -> PRF1:
    """Return Tatbul-style range PRF with flat positional bias.

    Inputs follow the binary-vector contract of :func:`point_adjusted_prf`.
    ``alpha`` must lie in ``[0, 1]`` and controls the existence reward in range
    recall; range precision always uses overlap quality only.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    labels, predictions = _fixed_pair(y_true, y_pred)
    return _triplet(_range_prf(labels, predictions, alpha))


def _affiliation_prf_prepared(
    labels: NDArray[np.uint8], predictions: NDArray[np.uint8]
) -> PRF1:
    return _triplet(_affiliation_prf(labels, predictions))


def affiliation_prf(y_true: ArrayLike, y_pred: ArrayLike) -> PRF1:
    """Return affiliation precision, recall and F1 for a fixed prediction.

    Inputs follow the binary-vector contract of :func:`point_adjusted_prf`.
    Events are contiguous positive runs in sequence order; timestamps are not
    accepted. Metrics are NaN when no labelled event defines an affiliation
    zone.
    """
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
    """Return best point-adjusted F1 on a deterministic threshold grid.

    ``y_true`` must be binary and ``y_score`` must be an equally sized finite
    numeric vector with larger values denoting stronger anomalies.
    ``threshold_count`` must be at least two. The result is oracle performance,
    not a deployable threshold estimate.
    """
    labels, scores, thresholds = _oracle_inputs(y_true, y_score, threshold_count)
    return float(_best_point_adjusted_f1(labels, scores, thresholds))


def best_point_adjusted_prf(
    y_true: ArrayLike, y_score: ArrayLike, *, threshold_count: int = 100
) -> PRF1:
    """Return point-adjusted PRF at the threshold maximizing adjusted F1.

    Inputs and oracle-threshold semantics follow
    :func:`best_point_adjusted_f1`.
    """
    labels, scores, thresholds = _oracle_inputs(y_true, y_score, threshold_count)
    return _triplet(_best_point_adjusted_prf(labels, scores, thresholds))


def best_event_f1(
    y_true: ArrayLike, y_score: ArrayLike, *, threshold_count: int = 100
) -> float:
    """Return best composite event F1 on the deterministic threshold grid.

    Labels, scores and ``threshold_count`` follow
    :func:`best_point_adjusted_f1`; event recall uses contiguous label runs.
    """
    labels, scores, thresholds = _oracle_inputs(y_true, y_score, threshold_count)
    return float(_best_event_f1(labels, scores, thresholds))


def best_event_prf(
    y_true: ArrayLike, y_score: ArrayLike, *, threshold_count: int = 100
) -> PRF1:
    """Return composite-event PRF at the threshold maximizing event F1.

    Labels, scores and ``threshold_count`` follow
    :func:`best_point_adjusted_f1`.
    """
    labels, scores, thresholds = _oracle_inputs(y_true, y_score, threshold_count)
    return _triplet(_best_event_prf(labels, scores, thresholds))


def best_range_f1(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    threshold_count: int = 100,
    alpha: float = 0.2,
) -> float:
    """Return best range F1 on the deterministic threshold grid.

    Labels, scores and ``threshold_count`` follow
    :func:`best_point_adjusted_f1`; ``alpha`` follows :func:`range_prf`.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    labels, scores, thresholds = _oracle_inputs(y_true, y_score, threshold_count)
    return float(_best_range_f1(labels, scores, thresholds, alpha))


def best_range_prf(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    threshold_count: int = 100,
    alpha: float = 0.2,
) -> PRF1:
    """Return Tatbul range PRF at the threshold maximizing range F1.

    Labels, scores and configuration follow :func:`best_range_f1`.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    labels, scores, thresholds = _oracle_inputs(y_true, y_score, threshold_count)
    return _triplet(_best_range_prf(labels, scores, thresholds, alpha))


def best_affiliation_f1(
    y_true: ArrayLike, y_score: ArrayLike, *, threshold_count: int = 100
) -> float:
    """Return best affiliation F1 on the deterministic threshold grid.

    Labels, scores and ``threshold_count`` follow
    :func:`best_point_adjusted_f1`. The result is NaN without labelled events.
    """
    labels, scores, thresholds = _oracle_inputs(y_true, y_score, threshold_count)
    return _best_affiliation_f1_prepared(labels, scores, thresholds)


def best_affiliation_prf(
    y_true: ArrayLike, y_score: ArrayLike, *, threshold_count: int = 100
) -> PRF1:
    """Return affiliation PRF at the threshold maximizing affiliation F1.

    Labels, scores and configuration follow :func:`best_affiliation_f1`.
    """
    labels, scores, thresholds = _oracle_inputs(y_true, y_score, threshold_count)
    return _triplet(_best_affiliation_prf(labels, scores, thresholds))


def _best_affiliation_f1_prepared(
    labels: NDArray[np.uint8],
    scores: NDArray[np.float64],
    thresholds: NDArray[np.float64],
) -> float:
    return float(_best_affiliation_f1(labels, scores, thresholds))


def interval_prf(y_true: ArrayLike, y_pred: ArrayLike) -> PRF1:
    """Return overlap-count precision, recall and F1 over intervals.

    ``y_true`` and ``y_pred`` must be equally sized, non-empty,
    one-dimensional binary vectors. Each contiguous positive run is one
    interval. A predicted interval contributes once for every labelled
    interval it overlaps.
    """
    labels, predictions = _fixed_pair(y_true, y_pred)
    return _triplet(_interval_prf(labels, predictions))


def best_interval_f1(
    y_true: ArrayLike, y_score: ArrayLike, *, threshold_count: int = 100
) -> float:
    """Return oracle-best interval F1 on a deterministic threshold grid.

    ``y_true`` must be binary and ``y_score`` must be an equally sized finite
    numeric vector with larger values denoting stronger anomalies.
    ``threshold_count`` must be at least two.
    """
    labels, scores = validate_pair(y_true, y_score)
    thresholds = threshold_grid(scores, threshold_count)
    return float(_best_interval_prf(labels, scores, thresholds)[2])


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
    """Return early/delayed weighted PATE precision, recall and F1.

    ``y_true`` and ``y_pred`` must be equal-length binary vectors. Buffers are
    non-negative point counts around each contiguous labelled event; adjacent
    event zones are clipped so they do not overlap. Results are NaN when no
    labelled event exists.
    """
    labels, predictions = _fixed_pair(y_true, y_pred)
    early_buffer = non_negative_integer(early_buffer, name="early_buffer")
    delayed_buffer = non_negative_integer(delayed_buffer, name="delayed_buffer")
    precision, recall = _pate_pr(
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
    """Return area under the weighted PATE precision-recall curve.

    ``y_true`` must be binary and ``y_score`` must be an equally sized finite
    numeric vector where larger values mean more anomalous. Buffers are
    non-negative point counts, ``buffer_splits`` is positive, and
    ``threshold_count`` is at least two. The returned area is averaged over the
    Cartesian product of sampled early and delayed buffers.
    """
    labels, scores = validate_pair(y_true, y_score)
    thresholds = _rank_thresholds(scores, threshold_count)
    early = _buffer_points(early_buffer, buffer_splits, include_zero)
    delayed = _buffer_points(delayed_buffer, buffer_splits, include_zero)
    return float(_pate_auc(labels, scores, thresholds, early, delayed)[0])


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
    """Return PATE F1 averaged across configured buffer pairs.

    Labels, scores and buffer parameters follow :func:`pate`. If supplied,
    ``y_pred`` must be an equal-length binary vector and is evaluated as one
    fixed decision. Otherwise the best score threshold is selected separately
    for each buffer pair, so the result is oracle performance.
    """
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
        return float(_pate_f1(labels, predictions, early, delayed))
    thresholds = _rank_thresholds(scores, threshold_count)
    return float(_pate_auc(labels, scores, thresholds, early, delayed)[1])
