# ruff: noqa: E741
"""Per-event affiliation distance and probability primitives.

Ported from ``TSB_AD/evaluation/affiliation/_single_ground_truth_event.py``
(TheDatumOrg/TSB-AD, Apache License 2.0), upstream
https://github.com/Horea201/Affiliation-F1Score.  Numeric operations are kept
identical to the source.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import math

from ._integral import (
    integral_interval_distance,
    integral_interval_probaCDF_precision,
    integral_interval_probaCDF_recall,
    interval_length,
    sum_interval_lengths,
)
from ._zones import (
    affiliation_partition,
    get_all_E_gt_func,
)

__license__ = "Apache-2.0"
__source__ = (
    "TSB_AD/evaluation/affiliation/_single_ground_truth_event.py @ TheDatumOrg/TSB-AD"
)

__all__ = [
    "affiliation_precision_distance",
    "affiliation_precision_proba",
    "affiliation_recall_distance",
    "affiliation_recall_proba",
]


def affiliation_precision_distance(
    Is: list[tuple[float, float] | None],
    J: tuple[float, float],
) -> float:
    """Individual average distance from ``Is`` to a single ground truth ``J``."""
    if all([I is None for I in Is]):  # no prediction in the current area
        return math.nan  # undefined
    return sum([integral_interval_distance(I, J) for I in Is]) / sum_interval_lengths(
        Is
    )


def affiliation_precision_proba(
    Is: list[tuple[float, float] | None],
    J: tuple[float, float],
    E: tuple[float, float],
) -> float:
    """Individual precision probability from ``Is`` to a single ground truth ``J``."""
    if all([I is None for I in Is]):  # no prediction in the current area
        return math.nan  # undefined
    return sum(
        [integral_interval_probaCDF_precision(I, J, E) for I in Is]
    ) / sum_interval_lengths(Is)


def affiliation_recall_distance(
    Is: list[tuple[float, float] | None],
    J: tuple[float, float],
) -> float:
    """Individual average distance from a single ``J`` to the predictions ``Is``."""
    Is = [I for I in Is if I is not None]  # filter possible None in Is
    if len(Is) == 0:  # there is no prediction in the current area
        return math.inf
    E_gt_recall = get_all_E_gt_func(
        Is, (-math.inf, math.inf)
    )  # from the predictions' viewpoint
    Js = affiliation_partition([J], E_gt_recall)
    return sum(
        [
            integral_interval_distance(J_parts[0] if J_parts else None, I)
            for I, J_parts in zip(Is, Js)
        ]
    ) / interval_length(J)


def affiliation_recall_proba(
    Is: list[tuple[float, float] | None],
    J: tuple[float, float],
    E: tuple[float, float],
) -> float:
    """Individual recall probability from a single ground truth ``J`` to ``Is``."""
    Is = [I for I in Is if I is not None]  # filter possible None in Is
    if len(Is) == 0:  # there is no prediction in the current area
        return 0
    E_gt_recall = get_all_E_gt_func(Is, E)
    Js = affiliation_partition([J], E_gt_recall)
    return sum(
        [
            integral_interval_probaCDF_recall(I, J_parts[0] if J_parts else None, E)
            for I, J_parts in zip(Is, Js)
        ]
    ) / interval_length(J)
