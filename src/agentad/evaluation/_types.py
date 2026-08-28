"""Small immutable result types used by individual metric families."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PRF1", "PointMetrics"]


@dataclass(frozen=True, slots=True)
class PRF1:
    """Precision, recall and their harmonic mean."""

    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class PointMetrics:
    """Metrics for one fixed binary point prediction."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    mcc: float
