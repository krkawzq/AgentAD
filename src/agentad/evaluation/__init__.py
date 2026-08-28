"""Evaluation tools for ordinary time-series anomaly detection."""

from __future__ import annotations

from .collection import (
    EvaluationResult,
    ReconstructedSeries,
    ScoreFn,
    evaluate_collection,
    reconstruct_series,
)
from .metrics import (
    DEFAULT_METRICS,
    THRESHOLD_INDEPENDENT_METRICS,
    MetricName,
    evaluate,
    generate_curve,
    get_metrics,
)
from .period import find_length_rank, find_period
from .point import (
    PointMetrics,
    average_precision,
    best_point_f1,
    point_adjust,
    point_metrics,
    roc_auc,
)
from .range import (
    PRF1,
    affiliation_prf,
    best_affiliation_f1,
    best_event_f1,
    best_point_adjusted_f1,
    best_range_f1,
    event_prf,
    point_adjusted_prf,
    range_prf,
)
from .vus import VUSResult, volume_under_surface

__all__ = [
    "DEFAULT_METRICS",
    "THRESHOLD_INDEPENDENT_METRICS",
    "MetricName",
    "evaluate",
    "generate_curve",
    "get_metrics",
    "EvaluationResult",
    "ReconstructedSeries",
    "ScoreFn",
    "evaluate_collection",
    "reconstruct_series",
    "find_length_rank",
    "find_period",
    "PointMetrics",
    "average_precision",
    "best_point_f1",
    "point_adjust",
    "point_metrics",
    "roc_auc",
    "PRF1",
    "affiliation_prf",
    "best_affiliation_f1",
    "best_event_f1",
    "best_point_adjusted_f1",
    "best_range_f1",
    "event_prf",
    "point_adjusted_prf",
    "range_prf",
    "VUSResult",
    "volume_under_surface",
]
