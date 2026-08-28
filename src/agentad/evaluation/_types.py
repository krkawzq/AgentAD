"""Small immutable result types used by individual metric families."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["AUCResult", "FirstHit", "PRF1", "PointMetrics"]


@dataclass(frozen=True, slots=True)
class AUCResult:
    """ROC and precision-recall areas produced by one curve family."""

    roc: float
    pr: float


@dataclass(frozen=True, slots=True)
class FirstHit:
    """Rank statistics for the highest-ranked score inside an anomaly."""

    rank: int
    fraction: float
    hit_at_3_percent: bool
    hit_at_10_percent: bool


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
    f05: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
