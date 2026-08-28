"""Compiled probability-only kernels for Affiliation-F.

The formulas are a scalar specialization of the interval implementation in
this package. Detailed distance outputs remain in the public Python API.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .._kernels import binary_runs_kernel

__all__ = ["affiliation_prf_kernel", "best_affiliation_f1_kernel"]


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
def affiliation_prf_kernel(labels, predictions):
    gt_starts, gt_stops = binary_runs_kernel(labels)
    pred_starts, pred_stops = binary_runs_kernel(predictions)
    precision, recall = _affiliation_pr_from_runs(
        pred_starts, pred_stops, gt_starts, gt_stops, labels.size
    )
    if np.isnan(precision) or np.isnan(recall):
        return precision, recall, np.nan
    f1 = 2.0 * precision * recall / (precision + recall + 1e-15)
    return precision, recall, f1


@njit(cache=True, nogil=True)
def best_affiliation_f1_kernel(labels, scores, thresholds):
    gt_starts, gt_stops = binary_runs_kernel(labels)
    if gt_starts.size == 0:
        return np.nan
    prediction = np.empty(labels.size, dtype=np.uint8)
    best = 0.0
    for threshold in thresholds:
        for index in range(scores.size):
            prediction[index] = 1 if scores[index] > threshold else 0
        pred_starts, pred_stops = binary_runs_kernel(prediction)
        precision, recall = _affiliation_pr_from_runs(
            pred_starts, pred_stops, gt_starts, gt_stops, labels.size
        )
        value = 2.0 * precision * recall / (precision + recall + 1e-15)
        if not np.isnan(value) and value > best:
            best = value
    return best
