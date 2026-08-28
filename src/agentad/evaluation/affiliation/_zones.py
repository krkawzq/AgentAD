# ruff: noqa: E741
"""Affiliation zone construction from the ground-truth point of view.

Ported from ``TSB_AD/evaluation/affiliation/_affiliation_zone.py``
(TheDatumOrg/TSB-AD, Apache License 2.0), upstream
https://github.com/Horea201/Affiliation-F1Score.  Numeric operations are kept
identical to the source.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

from ._integral import interval_intersection

__license__ = "Apache-2.0"
__source__ = "TSB_AD/evaluation/affiliation/_affiliation_zone.py @ TheDatumOrg/TSB-AD"

__all__ = [
    "t_start",
    "t_stop",
    "E_gt_func",
    "get_all_E_gt_func",
    "affiliation_partition",
]


def t_start(
    j: int,
    Js: list[tuple[float, float]],
    Trange: tuple[float, float],
) -> float:
    """Generalized start such that the midpoint with ``t_stop`` gives the zone."""
    b = max(Trange)
    n = len(Js)
    if j == n:
        return 2 * b - t_stop(n - 1, Js, Trange)
    return Js[j][0]


def t_stop(
    j: int,
    Js: list[tuple[float, float]],
    Trange: tuple[float, float],
) -> float:
    """Generalized stop such that the midpoint with ``t_start`` gives the zone."""
    if j == -1:
        a = min(Trange)
        return 2 * a - t_start(0, Js, Trange)
    return Js[j][1]


def E_gt_func(
    j: int, Js: list[tuple[float, float]], Trange: tuple[float, float]
) -> tuple[float, float]:
    """Affiliation zone of ground-truth event ``j``."""
    range_left = (t_stop(j - 1, Js, Trange) + t_start(j, Js, Trange)) / 2
    range_right = (t_stop(j, Js, Trange) + t_start(j + 1, Js, Trange)) / 2
    return (range_left, range_right)


def get_all_E_gt_func(
    Js: list[tuple[float, float]], Trange: tuple[float, float]
) -> list[tuple[float, float]]:
    """Affiliation partition limits from the ground-truth point of view."""
    E_gt = [E_gt_func(j, Js, Trange) for j in range(len(Js))]
    return E_gt


def affiliation_partition(
    Is: list[tuple[float, float]],
    E_gt: list[tuple[float, float]],
) -> list[list[tuple[float, float] | None]]:
    """Cut predicted events into the affiliation zones.

    The outer list is indexed by each affiliation zone of ``E_gt``; the inner
    list holds each predicted event cut to that zone (``None`` for empty
    intersections).
    """
    out: list[list[tuple[float, float] | None]] = []
    first_candidate = 0
    for zone in E_gt:
        while first_candidate < len(Is) and Is[first_candidate][1] < zone[0]:
            first_candidate += 1

        partition: list[tuple[float, float] | None] = []
        candidate = first_candidate
        while candidate < len(Is) and Is[candidate][0] <= zone[1]:
            partition.append(interval_intersection(Is[candidate], zone))
            candidate += 1
        out.append(partition)
    return out
