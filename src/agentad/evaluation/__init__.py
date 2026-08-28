"""Evaluation tools for ordinary time-series anomaly detection.

Public symbols are loaded on first access so importing the package does not
eagerly initialize the numerical and collection backends.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "DEFAULT_METRICS": ".metrics",
    "THRESHOLD_INDEPENDENT_METRICS": ".metrics",
    "MetricName": ".metrics",
    "evaluate": ".metrics",
    "generate_curve": ".metrics",
    "get_metrics": ".metrics",
    "warmup": ".metrics",
    "EvaluationResult": ".collection",
    "ReconstructedSeries": ".collection",
    "ScoreFn": ".collection",
    "evaluate_collection": ".collection",
    "reconstruct_series": ".collection",
    "find_length_rank": ".period",
    "find_period": ".period",
    "PointMetrics": ".point",
    "average_precision": ".point",
    "best_point_f1": ".point",
    "point_adjust": ".point",
    "point_metrics": ".point",
    "roc_auc": ".point",
    "PRF1": ".range",
    "affiliation_prf": ".range",
    "best_affiliation_f1": ".range",
    "best_event_f1": ".range",
    "best_point_adjusted_f1": ".range",
    "best_range_f1": ".range",
    "event_prf": ".range",
    "point_adjusted_prf": ".range",
    "range_prf": ".range",
    "VUSResult": ".vus",
    "volume_under_surface": ".vus",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
