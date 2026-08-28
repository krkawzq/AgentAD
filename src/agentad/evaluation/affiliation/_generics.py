"""Event helpers shared across the affiliation metrics.

Ported from ``TSB_AD/evaluation/affiliation/generics.py``
(TheDatumOrg/TSB-AD, Apache License 2.0), which itself packages the
affiliation metrics of Hwang, Lyu, Paoletti, ... published at
https://github.com/TheDatumOrg/TSB-AD with upstream at
https://github.com/Horea201/Affiliation-F1Score.  File-system helpers that
read ``data/*.gz`` corpora are dropped; everything reachable from
``pr_from_events`` is kept numerically identical.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import groupby
from operator import itemgetter

__license__ = "Apache-2.0"
__source__ = "TSB_AD/evaluation/affiliation/generics.py @ TheDatumOrg/TSB-AD"

__all__ = [
    "convert_vector_to_events",
    "infer_Trange",
    "has_point_anomalies",
    "f1_func",
    "_sum_wo_nan",
    "_len_wo_nan",
]


def convert_vector_to_events(vector: Sequence[float]) -> list[tuple[int, int]]:
    """Convert a binary vector to a list of ``(start, stop)`` event couples.

    A positive index ``i`` denotes the anomalous interval ``[i, i+1)``, so a
    run ``[s, e]`` of ones becomes the event ``(s, e+1)``.
    """
    positive_indexes = [idx for idx, val in enumerate(vector) if val > 0]
    events: list[tuple[int, int]] = []
    for _, group in groupby(enumerate(positive_indexes), lambda ix: ix[0] - ix[1]):
        run = list(map(itemgetter(1), group))
        events.append((run[0], run[-1]))
    # A positive index i is the interval [i, i+1), so the last index moves by 1.
    events = [(x, y + 1) for (x, y) in events]
    return events


def infer_Trange(
    events_pred: list[tuple[int, int]], events_gt: list[tuple[int, int]]
) -> tuple[float, float]:
    """Smallest ``(start, stop)`` range covering both event lists."""
    if len(events_gt) == 0:
        raise ValueError("The gt events should contain at least one event")
    if len(events_pred) == 0:
        # Empty prediction: base Trange only on events_gt (which is non-empty).
        return infer_Trange(events_gt, events_gt)

    min_pred = min(x[0] for x in events_pred)
    min_gt = min(x[0] for x in events_gt)
    max_pred = max(x[1] for x in events_pred)
    max_gt = max(x[1] for x in events_gt)
    return (min(min_pred, min_gt), max(max_pred, max_gt))


def has_point_anomalies(events: list[tuple[int, int]]) -> bool:
    """Whether any event starts and stops at the same time."""
    if len(events) == 0:
        return False
    return min(x[1] - x[0] for x in events) == 0


def f1_func(p: float, r: float) -> float:
    """F1 of precision ``p`` and recall ``r``."""
    return 2 * p * r / (p + r)


def _sum_wo_nan(vec: Sequence[float]) -> float:
    """Sum of the elements, ignoring ``math.isnan`` ones."""
    return sum([e for e in vec if not math.isnan(e)])


def _len_wo_nan(vec: Sequence[float]) -> int:
    """Count of the elements, ignoring ``math.isnan`` ones."""
    return len([e for e in vec if not math.isnan(e)])
