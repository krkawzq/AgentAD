"""Single-series TSB-AD-compatible metric-suite orchestration.

Adapted from TheDatumOrg/TSB-AD (Apache License 2.0).

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike

from ._kernels import (
    best_event_f1_kernel,
    best_oracle_f1s_kernel,
    best_point_adjusted_f1_kernel,
    best_range_f1_kernel,
    event_prf_kernel,
    point_adjust_kernel,
    range_prf_kernel,
)
from ._validation import binary_array, threshold_grid, validate_pair
from .point import _point_metrics_prepared, _score_metrics_prepared
from .range import (
    _affiliation_prf_prepared,
    _best_affiliation_f1_prepared,
)
from .vus import VUSResult, _volume_under_surface_prepared

MetricName: TypeAlias = Literal[
    "AUC-PR",
    "AUC-ROC",
    "VUS-PR",
    "VUS-ROC",
    "Standard-F1",
    "PA-F1",
    "Event-based-F1",
    "R-based-F1",
    "Affiliation-F",
]

DEFAULT_METRICS: tuple[MetricName, ...] = (
    "AUC-PR",
    "AUC-ROC",
    "VUS-PR",
    "VUS-ROC",
    "Standard-F1",
    "PA-F1",
    "Event-based-F1",
    "R-based-F1",
    "Affiliation-F",
)

THRESHOLD_INDEPENDENT_METRICS: tuple[MetricName, ...] = (
    "AUC-PR",
    "AUC-ROC",
    "VUS-PR",
    "VUS-ROC",
)

_ORACLE_GRID_METRICS: tuple[MetricName, ...] = (
    "PA-F1",
    "Event-based-F1",
    "R-based-F1",
    "Affiliation-F",
)

__all__ = [
    "DEFAULT_METRICS",
    "MetricName",
    "THRESHOLD_INDEPENDENT_METRICS",
    "evaluate",
    "generate_curve",
    "get_metrics",
    "warmup",
]


def _metric_selection(metrics: Iterable[MetricName]) -> tuple[MetricName, ...]:
    if isinstance(metrics, str):
        raise TypeError("metrics must be an iterable of metric names, not one string")
    requested = tuple(metrics)
    unknown = [name for name in requested if name not in DEFAULT_METRICS]
    if unknown:
        raise ValueError(
            f"unknown metrics: {unknown}; available: {list(DEFAULT_METRICS)}"
        )
    if len(set(requested)) != len(requested):
        raise ValueError("metrics must not contain duplicates")
    return cast(tuple[MetricName, ...], requested)


def evaluate(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    y_pred: ArrayLike | None = None,
    metrics: Iterable[MetricName] = DEFAULT_METRICS,
    sliding_window: int = 100,
    threshold_count: int = 100,
    vus_threshold_count: int = 250,
    range_alpha: float = 0.2,
) -> dict[MetricName, float]:
    """Evaluate selected metrics for one score/label pair.

    ``y_pred`` makes all threshold-dependent metrics evaluate the same fixed
    decision. Without it, each such metric uses its own oracle-best threshold,
    matching TSB-AD. Oracle results are useful for score-ranking comparison but
    must not be reported as deployable threshold performance.
    """
    selected = _metric_selection(metrics)
    labels, scores = validate_pair(y_true, y_score)
    if not 0.0 <= range_alpha <= 1.0:
        raise ValueError("range_alpha must be in [0, 1]")
    if isinstance(sliding_window, bool) or not isinstance(
        sliding_window, (int, np.integer)
    ):
        raise TypeError("sliding_window must be an integer")
    if sliding_window < 0:
        raise ValueError("sliding_window must be non-negative")

    predictions = None
    if y_pred is not None:
        predictions = binary_array(y_pred, name="y_pred")
        if predictions.size != labels.size:
            raise ValueError(
                "y_true and y_pred must have the same length: "
                f"{labels.size} != {predictions.size}"
            )

    return _evaluate_prepared(
        labels,
        scores,
        predictions=predictions,
        selected=selected,
        sliding_window=int(sliding_window),
        threshold_count=threshold_count,
        vus_threshold_count=vus_threshold_count,
        range_alpha=range_alpha,
    )


def _evaluate_prepared(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    predictions: np.ndarray | None,
    selected: tuple[MetricName, ...],
    sliding_window: int,
    threshold_count: int,
    vus_threshold_count: int,
    range_alpha: float,
) -> dict[MetricName, float]:
    """Evaluate contiguous, finite uint8/float64 inputs without revalidation."""
    values: dict[MetricName, float] = {}
    needs_vus = "VUS-PR" in selected or "VUS-ROC" in selected
    vus_result: VUSResult | None = None
    if needs_vus:
        if isinstance(vus_threshold_count, bool) or not isinstance(
            vus_threshold_count, (int, np.integer)
        ):
            raise TypeError("vus_threshold_count must be an integer")
        if vus_threshold_count < 2:
            raise ValueError("vus_threshold_count must be at least 2")
        vus_result = _volume_under_surface_prepared(
            labels,
            scores,
            window_size=int(sliding_window),
            threshold_count=int(vus_threshold_count),
            return_surface=False,
        )

    thresholds = None
    if predictions is None and any(name in _ORACLE_GRID_METRICS for name in selected):
        thresholds = threshold_grid(scores, threshold_count)

    ranking = None
    if (
        "AUC-PR" in selected
        or "AUC-ROC" in selected
        or (predictions is None and "Standard-F1" in selected)
    ):
        ranking = _score_metrics_prepared(labels, scores, smoothing=1e-5)

    oracle_values = None
    fused_names = ("PA-F1", "Event-based-F1", "R-based-F1")
    if predictions is None and sum(name in selected for name in fused_names) >= 2:
        assert thresholds is not None
        oracle_values = best_oracle_f1s_kernel(labels, scores, thresholds, range_alpha)

    fixed_point = (
        _point_metrics_prepared(labels, predictions)
        if predictions is not None and "Standard-F1" in selected
        else None
    )
    fixed_adjusted = (
        point_adjust_kernel(labels, predictions)
        if predictions is not None and "PA-F1" in selected
        else None
    )

    for name in selected:
        if name == "AUC-PR":
            assert ranking is not None
            values[name] = ranking[0]
        elif name == "AUC-ROC":
            assert ranking is not None
            values[name] = ranking[1]
        elif name == "VUS-PR":
            assert vus_result is not None
            values[name] = vus_result.pr
        elif name == "VUS-ROC":
            assert vus_result is not None
            values[name] = vus_result.roc
        elif name == "Standard-F1":
            if predictions is not None:
                assert fixed_point is not None
                values[name] = fixed_point.f1
            else:
                assert ranking is not None
                values[name] = ranking[2]
        elif name == "PA-F1":
            if predictions is not None:
                assert fixed_adjusted is not None
                values[name] = _point_metrics_prepared(labels, fixed_adjusted).f1
            else:
                values[name] = (
                    float(oracle_values[0])
                    if oracle_values is not None
                    else float(
                        best_point_adjusted_f1_kernel(labels, scores, thresholds)
                    )
                )
        elif name == "Event-based-F1":
            values[name] = (
                float(event_prf_kernel(labels, predictions)[2])
                if predictions is not None
                else (
                    float(oracle_values[1])
                    if oracle_values is not None
                    else float(best_event_f1_kernel(labels, scores, thresholds))
                )
            )
        elif name == "R-based-F1":
            values[name] = (
                float(range_prf_kernel(labels, predictions, range_alpha)[2])
                if predictions is not None
                else (
                    float(oracle_values[2])
                    if oracle_values is not None
                    else float(
                        best_range_f1_kernel(labels, scores, thresholds, range_alpha)
                    )
                )
            )
        elif name == "Affiliation-F":
            values[name] = (
                _affiliation_prf_prepared(labels, predictions).f1
                if predictions is not None
                else _best_affiliation_f1_prepared(labels, scores, thresholds)
            )
    return values


def warmup() -> None:
    """Compile and load all Numba metric kernels before timed batch evaluation."""
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 0], dtype=np.uint8)
    scores = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9, 0.4, 0.0], dtype=np.float64)
    thresholds = threshold_grid(scores, 3)
    best_oracle_f1s_kernel(labels, scores, thresholds, 0.2)
    best_point_adjusted_f1_kernel(labels, scores, thresholds)
    best_event_f1_kernel(labels, scores, thresholds)
    best_range_f1_kernel(labels, scores, thresholds, 0.2)
    _best_affiliation_f1_prepared(labels, scores, thresholds)
    prediction = np.asarray(scores > 0.5, dtype=np.uint8)
    point_adjust_kernel(labels, prediction)
    event_prf_kernel(labels, prediction)
    range_prf_kernel(labels, prediction, 0.2)
    _volume_under_surface_prepared(
        labels,
        scores,
        window_size=2,
        threshold_count=3,
        return_surface=False,
    )


def get_metrics(
    score: ArrayLike,
    labels: ArrayLike,
    slidingWindow: int = 100,
    pred: ArrayLike | None = None,
    version: str = "opt",
    thre: int = 250,
) -> dict[MetricName, float]:
    """Compatibility wrapper for TSB-AD's score-first API."""
    if version != "opt":
        raise NotImplementedError("only version='opt' is supported")
    return evaluate(
        labels,
        score,
        y_pred=pred,
        sliding_window=slidingWindow,
        vus_threshold_count=thre,
    )


def generate_curve(
    label: ArrayLike,
    score: ArrayLike,
    slidingWindow: int,
    version: str = "opt",
    thre: int = 250,
):
    """Compatibility representation of the VUS surfaces."""
    if version != "opt":
        raise NotImplementedError("only version='opt' is supported")
    labels, scores = validate_pair(label, score)
    result = _volume_under_surface_prepared(
        labels,
        scores,
        window_size=int(slidingWindow),
        threshold_count=int(thre),
    )
    tpr = result.tpr.ravel()
    fpr = result.fpr.ravel()
    precision = result.precision.ravel()
    window = np.repeat(result.windows, result.tpr.shape[1])
    tpr_ap = result.tpr[:, :-1].ravel()
    window_ap = np.repeat(result.windows, result.tpr.shape[1] - 1)
    return fpr, window, tpr, tpr_ap, precision, window_ap, result.roc, result.pr
