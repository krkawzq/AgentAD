"""Affiliation precision/recall over event lists.

Ported from ``TSB_AD/evaluation/affiliation/metrics.py``
(TheDatumOrg/TSB-AD, Apache License 2.0), upstream
https://github.com/Horea201/Affiliation-F1Score.  The file-system driver
``produce_all_results`` is dropped; everything else is numerically identical.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

from typing import Any

from ._events import (
    affiliation_precision_distance,
    affiliation_precision_proba,
    affiliation_recall_distance,
    affiliation_recall_proba,
)
from ._generics import (
    _len_wo_nan,
    _sum_wo_nan,
    has_point_anomalies,
    infer_Trange,
)
from ._zones import (
    affiliation_partition,
    get_all_E_gt_func,
)

__license__ = "Apache-2.0"
__source__ = "TSB_AD/evaluation/affiliation/metrics.py @ TheDatumOrg/TSB-AD"

__all__ = ["test_events", "pr_from_events"]


def test_events(events) -> None:
    """Validate an event list; raise on malformed or unordered events."""
    if type(events) is not list:
        raise TypeError("Input `events` should be a list of couples")
    if not all([type(x) is tuple for x in events]):
        raise TypeError("Input `events` should be a list of tuples")
    if not all([len(x) == 2 for x in events]):
        raise ValueError("Input `events` should be a list of couples (start, stop)")
    if not all([x[0] <= x[1] for x in events]):
        raise ValueError(
            "Input `events` should be a list of couples (start, stop) with start <= stop"
        )
    if not all([events[i][1] < events[i + 1][0] for i in range(len(events) - 1)]):
        raise ValueError("Couples of input `events` should be disjoint and ordered")


def pr_from_events(
    events_pred: list[tuple[int, int]],
    events_gt: list[tuple[int, int]],
    Trange: tuple[int, int],
) -> dict[str, Any]:
    """Affiliation precision/recall (with individual terms) over event lists.

    :param events_pred: predicted events, each a ``(start, stop)`` couple
    :param events_gt: ground-truth events, each a ``(start, stop)`` couple
    :param Trange: range of the series containing all events, ``(start, stop)``
    :return: dict with ``Affiliation_Precision``, ``Affiliation_Recall`` and the
        individual distances/probabilities
    """
    test_events(events_pred)
    test_events(events_gt)

    minimal_Trange = infer_Trange(events_pred, events_gt)
    if not Trange[0] <= minimal_Trange[0]:
        raise ValueError("`Trange` should include all the events")
    if not minimal_Trange[1] <= Trange[1]:
        raise ValueError("`Trange` should include all the events")

    if len(events_gt) == 0:
        raise ValueError("Input `events_gt` should have at least one event")

    if has_point_anomalies(events_pred) or has_point_anomalies(events_gt):
        raise ValueError("Cannot manage point anomalies currently")

    E_gt = get_all_E_gt_func(events_gt, Trange)
    aff_partition = affiliation_partition(events_pred, E_gt)

    (
        p_precision_average,
        p_recall_average,
        p_precision,
        p_recall,
    ) = _probability_terms(aff_partition, events_gt, E_gt)

    # Computing precision distance
    d_precision = [
        affiliation_precision_distance(Is, J) for Is, J in zip(aff_partition, events_gt)
    ]
    # Computing recall distance
    d_recall = [
        affiliation_recall_distance(Is, J) for Is, J in zip(aff_partition, events_gt)
    ]
    dict_out = dict(
        {
            "Affiliation_Precision": p_precision_average,
            "Affiliation_Recall": p_recall_average,
            "individual_precision_probabilities": p_precision,
            "individual_recall_probabilities": p_recall,
            "individual_precision_distances": d_precision,
            "individual_recall_distances": d_recall,
        }
    )
    return dict_out


def _probability_terms(
    aff_partition: list[list[tuple[float, float] | None]],
    events_gt: list[tuple[int, int]],
    E_gt: list[tuple[float, float]],
) -> tuple[float, float, list[float], list[float]]:
    p_precision = [
        affiliation_precision_proba(Is, J, E)
        for Is, J, E in zip(aff_partition, events_gt, E_gt)
    ]
    p_recall = [
        affiliation_recall_proba(Is, J, E)
        for Is, J, E in zip(aff_partition, events_gt, E_gt)
    ]
    count = _len_wo_nan(p_precision)
    p_precision_average = _sum_wo_nan(p_precision) / count if count else p_precision[0]
    return p_precision_average, sum(p_recall) / len(p_recall), p_precision, p_recall
