"""Single-series TSB-AD-compatible metric-suite orchestration.

Parts are adapted from TheDatumOrg/TSB-AD under Apache-2.0; see
``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike

from ._kernels import (
    best_affiliation_prf_kernel,
    best_oracle_prfs_kernel,
    best_point_adjusted_counts_kernel,
    best_point_adjusted_prf_kernel,
    event_prf_kernel,
    point_adjust_kernel,
    range_prf_kernel,
)
from ._protocol_kernels import (
    best_interval_prf_kernel,
    dilate_labels_kernel,
    interval_prf_kernel,
    pate_auc_kernel,
    pate_f1_kernel,
)
from ._validation import (
    binary_array,
    non_negative_integer,
    threshold_count as validate_threshold_count,
    threshold_grid,
    validate_pair,
)
from .point import (
    _best_point_metrics_prepared,
    _point_adjusted_average_precision_prepared,
    _point_metrics_from_counts,
    _point_metrics_prepared,
    _precision_at_k_prepared,
    _score_metrics_prepared,
)
from .protocols import (
    EventScale,
    _buffer_points,
    _event_adjusted_prepared,
    _k_delay_prepared,
    _rank_thresholds,
)
from .range import (
    _affiliation_prf_prepared,
)
from .vus import (
    VUSResult,
    _fixed_vus_prf_kernel,
    _volume_under_surface_prepared,
)

MetricName: TypeAlias = Literal[
    "AUC-PR",
    "AUC-ROC",
    "R-AUC-PR",
    "R-AUC-ROC",
    "VUS-PR",
    "VUS-ROC",
    "Standard-Accuracy",
    "Standard-Precision",
    "Standard-Recall",
    "Standard-F0.5",
    "Standard-F1",
    "Standard-MCC",
    "Standard-TP",
    "Standard-FP",
    "Standard-FN",
    "Standard-TN",
    "Precision@K",
    "PA-Accuracy",
    "PA-Precision",
    "PA-Recall",
    "PA-F1",
    "PA-AUC-PR",
    "K-Delay-PA-F1",
    "Event-PA-F1",
    "Event-based-Precision",
    "Event-based-Recall",
    "Event-based-F1",
    "R-based-Precision",
    "R-based-Recall",
    "R-based-F1",
    "Affiliation-Precision",
    "Affiliation-Recall",
    "Affiliation-F",
    "VUS-Precision",
    "VUS-Recall",
    "VUS-F",
    "Interval-Precision",
    "Interval-Recall",
    "Interval-F1",
    "Tolerance-PA-Precision",
    "Tolerance-PA-Recall",
    "Tolerance-PA-F1",
    "PATE",
    "PATE-F1",
    "First-Hit-Rank",
    "First-Hit-Fraction",
    "Hit@3%",
    "Hit@10%",
]

DEFAULT_METRICS: tuple[MetricName, ...] = (
    "AUC-PR",
    "AUC-ROC",
    "R-AUC-PR",
    "R-AUC-ROC",
    "VUS-PR",
    "VUS-ROC",
    "Standard-F1",
    "PA-F1",
    "Event-based-F1",
    "R-based-F1",
    "Affiliation-F",
)

TSB_AD_METRICS: tuple[MetricName, ...] = (
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

AVAILABLE_METRICS: tuple[MetricName, ...] = (
    "AUC-PR",
    "AUC-ROC",
    "R-AUC-PR",
    "R-AUC-ROC",
    "VUS-PR",
    "VUS-ROC",
    "Standard-Accuracy",
    "Standard-Precision",
    "Standard-Recall",
    "Standard-F0.5",
    "Standard-F1",
    "Standard-MCC",
    "Standard-TP",
    "Standard-FP",
    "Standard-FN",
    "Standard-TN",
    "Precision@K",
    "PA-Accuracy",
    "PA-Precision",
    "PA-Recall",
    "PA-F1",
    "PA-AUC-PR",
    "K-Delay-PA-F1",
    "Event-PA-F1",
    "Event-based-Precision",
    "Event-based-Recall",
    "Event-based-F1",
    "R-based-Precision",
    "R-based-Recall",
    "R-based-F1",
    "Affiliation-Precision",
    "Affiliation-Recall",
    "Affiliation-F",
    "VUS-Precision",
    "VUS-Recall",
    "VUS-F",
    "Interval-Precision",
    "Interval-Recall",
    "Interval-F1",
    "Tolerance-PA-Precision",
    "Tolerance-PA-Recall",
    "Tolerance-PA-F1",
    "PATE",
    "PATE-F1",
    "First-Hit-Rank",
    "First-Hit-Fraction",
    "Hit@3%",
    "Hit@10%",
)

THRESHOLD_INDEPENDENT_METRICS: tuple[MetricName, ...] = (
    "AUC-PR",
    "AUC-ROC",
    "R-AUC-PR",
    "R-AUC-ROC",
    "VUS-PR",
    "VUS-ROC",
    "Precision@K",
    "PA-AUC-PR",
    "PATE",
    "First-Hit-Rank",
    "First-Hit-Fraction",
    "Hit@3%",
    "Hit@10%",
)

_ORACLE_GRID_METRICS: tuple[MetricName, ...] = (
    "PA-Accuracy",
    "PA-Precision",
    "PA-Recall",
    "PA-F1",
    "Event-based-Precision",
    "Event-based-Recall",
    "Event-based-F1",
    "R-based-Precision",
    "R-based-Recall",
    "R-based-F1",
    "Affiliation-Precision",
    "Affiliation-Recall",
    "Affiliation-F",
    "Interval-Precision",
    "Interval-Recall",
    "Interval-F1",
    "Tolerance-PA-Precision",
    "Tolerance-PA-Recall",
    "Tolerance-PA-F1",
)

_SURFACE_METRICS = (
    "R-AUC-PR",
    "R-AUC-ROC",
    "VUS-PR",
    "VUS-ROC",
)
_FIXED_VUS_METRICS = ("VUS-Precision", "VUS-Recall", "VUS-F")
_STANDARD_METRICS = (
    "Standard-Accuracy",
    "Standard-Precision",
    "Standard-Recall",
    "Standard-F0.5",
    "Standard-F1",
    "Standard-MCC",
    "Standard-TP",
    "Standard-FP",
    "Standard-FN",
    "Standard-TN",
)
_STANDARD_COUNT_METRICS = (
    "Standard-TP",
    "Standard-FP",
    "Standard-FN",
    "Standard-TN",
)
_PA_METRICS = ("PA-Accuracy", "PA-Precision", "PA-Recall", "PA-F1")
_EVENT_METRICS = (
    "Event-based-Precision",
    "Event-based-Recall",
    "Event-based-F1",
)
_RANGE_METRICS = ("R-based-Precision", "R-based-Recall", "R-based-F1")
_AFFILIATION_METRICS = (
    "Affiliation-Precision",
    "Affiliation-Recall",
    "Affiliation-F",
)
_INTERVAL_METRICS = ("Interval-Precision", "Interval-Recall", "Interval-F1")
_TOLERANCE_METRICS = (
    "Tolerance-PA-Precision",
    "Tolerance-PA-Recall",
    "Tolerance-PA-F1",
)

__all__ = [
    "AVAILABLE_METRICS",
    "DEFAULT_METRICS",
    "MetricName",
    "THRESHOLD_INDEPENDENT_METRICS",
    "TSB_AD_METRICS",
    "evaluate",
    "generate_curve",
    "get_metrics",
]


def _metric_selection(metrics: Iterable[MetricName]) -> tuple[MetricName, ...]:
    if isinstance(metrics, str):
        raise TypeError("metrics must be an iterable of metric names, not one string")
    requested = tuple(metrics)
    unknown = [name for name in requested if name not in AVAILABLE_METRICS]
    if unknown:
        raise ValueError(
            f"unknown metrics: {unknown}; available: {list(AVAILABLE_METRICS)}"
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
    precision_k: int | None = None,
    k_delay: int = 5,
    event_scale: EventScale = "squeeze",
    event_base: int = 3,
    tolerance: int = 5,
    pate_early_buffer: int = 100,
    pate_delayed_buffer: int = 100,
    pate_buffer_splits: int = 1,
    pate_include_zero: bool = False,
    pate_threshold_count: int = 250,
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
    sliding_window = non_negative_integer(sliding_window, name="sliding_window")

    predictions = None
    if y_pred is not None:
        predictions = binary_array(y_pred, name="y_pred")
        if predictions.size != labels.size:
            raise ValueError(
                "y_true and y_pred must have the same length: "
                f"{labels.size} != {predictions.size}"
            )

    if predictions is None and any(name in _ORACLE_GRID_METRICS for name in selected):
        threshold_count = validate_threshold_count(threshold_count)
    if any(name in _FIXED_VUS_METRICS for name in selected) and predictions is None:
        raise ValueError("VUS-Precision, VUS-Recall and VUS-F require y_pred")
    if any(name in _SURFACE_METRICS for name in selected):
        vus_threshold_count = validate_threshold_count(
            vus_threshold_count, name="vus_threshold_count"
        )
    if "PATE" in selected or ("PATE-F1" in selected and predictions is None):
        pate_threshold_count = validate_threshold_count(
            pate_threshold_count, name="pate_threshold_count"
        )

    return _evaluate_prepared(
        labels,
        scores,
        predictions=predictions,
        selected=selected,
        sliding_window=sliding_window,
        threshold_count=threshold_count,
        vus_threshold_count=vus_threshold_count,
        range_alpha=range_alpha,
        precision_k=precision_k,
        k_delay=k_delay,
        event_scale=event_scale,
        event_base=event_base,
        tolerance=tolerance,
        pate_early_buffer=pate_early_buffer,
        pate_delayed_buffer=pate_delayed_buffer,
        pate_buffer_splits=pate_buffer_splits,
        pate_include_zero=pate_include_zero,
        pate_threshold_count=pate_threshold_count,
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
    precision_k: int | None,
    k_delay: int,
    event_scale: EventScale,
    event_base: int,
    tolerance: int,
    pate_early_buffer: int,
    pate_delayed_buffer: int,
    pate_buffer_splits: int,
    pate_include_zero: bool,
    pate_threshold_count: int,
) -> dict[MetricName, float]:
    """Evaluate contiguous, finite uint8/float64 inputs without revalidation."""
    computed: dict[str, float] = {}
    needs_vus = any(name in _SURFACE_METRICS for name in selected)
    vus_result: VUSResult | None = None
    if needs_vus:
        vus_threshold_count = validate_threshold_count(
            vus_threshold_count, name="vus_threshold_count"
        )
        vus_result = _volume_under_surface_prepared(
            labels,
            scores,
            window_size=sliding_window,
            threshold_count=vus_threshold_count,
            return_surface=False,
        )
        computed["VUS-PR"] = vus_result.pr
        computed["VUS-ROC"] = vus_result.roc
        computed["R-AUC-PR"] = float(vus_result.pr_by_window[-1])
        computed["R-AUC-ROC"] = float(vus_result.roc_by_window[-1])

    thresholds = None
    if predictions is None and any(name in _ORACLE_GRID_METRICS for name in selected):
        thresholds = threshold_grid(scores, threshold_count)

    ranking = None
    if "AUC-PR" in selected or "AUC-ROC" in selected:
        ranking = _score_metrics_prepared(labels, scores, smoothing=1e-5)
        computed["AUC-PR"] = ranking[0]
        computed["AUC-ROC"] = ranking[1]

    point_result = None
    if any(name in _STANDARD_METRICS for name in selected):
        point_result = (
            _point_metrics_prepared(labels, predictions)
            if predictions is not None
            else _best_point_metrics_prepared(labels, scores, smoothing=1e-5)
        )
        computed.update(
            {
                "Standard-Accuracy": point_result.accuracy,
                "Standard-Precision": point_result.precision,
                "Standard-Recall": point_result.recall,
                "Standard-F0.5": point_result.f05,
                "Standard-F1": point_result.f1,
                "Standard-MCC": point_result.mcc,
                "Standard-TP": float(point_result.tp),
                "Standard-FP": float(point_result.fp),
                "Standard-FN": float(point_result.fn),
                "Standard-TN": float(point_result.tn),
            }
        )

    oracle_prfs = None
    needs_oracle_prfs = any(
        name in _PA_METRICS + _EVENT_METRICS + _RANGE_METRICS for name in selected
    )
    if predictions is None and needs_oracle_prfs:
        assert thresholds is not None
        oracle_prfs = best_oracle_prfs_kernel(
            labels, scores, thresholds, range_alpha
        )

    if any(name in _PA_METRICS for name in selected):
        if predictions is not None:
            adjusted = point_adjust_kernel(labels, predictions)
            pa_metrics = _point_metrics_prepared(labels, adjusted)
            pa_precision, pa_recall, pa_f1 = (
                pa_metrics.precision,
                pa_metrics.recall,
                pa_metrics.f1,
            )
        else:
            assert oracle_prfs is not None
            pa_precision, pa_recall, pa_f1 = oracle_prfs[:3]
            if "PA-Accuracy" in selected:
                assert thresholds is not None
                pa_counts = best_point_adjusted_counts_kernel(
                    labels, scores, thresholds
                )
                pa_metrics = _point_metrics_from_counts(*pa_counts)
        computed["PA-Accuracy"] = pa_metrics.accuracy if "pa_metrics" in locals() else math.nan
        computed["PA-Precision"] = float(pa_precision)
        computed["PA-Recall"] = float(pa_recall)
        computed["PA-F1"] = float(pa_f1)

    if "PA-AUC-PR" in selected:
        computed["PA-AUC-PR"] = _point_adjusted_average_precision_prepared(
            labels, scores
        )

    if any(name in _EVENT_METRICS for name in selected):
        event_values = (
            event_prf_kernel(labels, predictions)
            if predictions is not None
            else oracle_prfs[3:6]
        )
        computed["Event-based-Precision"] = float(event_values[0])
        computed["Event-based-Recall"] = float(event_values[1])
        computed["Event-based-F1"] = float(event_values[2])

    if any(name in _RANGE_METRICS for name in selected):
        range_values = (
            range_prf_kernel(labels, predictions, range_alpha)
            if predictions is not None
            else oracle_prfs[6:9]
        )
        computed["R-based-Precision"] = float(range_values[0])
        computed["R-based-Recall"] = float(range_values[1])
        computed["R-based-F1"] = float(range_values[2])

    if any(name in _AFFILIATION_METRICS for name in selected):
        affiliation = (
            _affiliation_prf_prepared(labels, predictions)
            if predictions is not None
            else None
        )
        if affiliation is None:
            assert thresholds is not None
            affiliation_values = best_affiliation_prf_kernel(
                labels, scores, thresholds
            )
        else:
            affiliation_values = (
                affiliation.precision,
                affiliation.recall,
                affiliation.f1,
            )
        computed["Affiliation-Precision"] = float(affiliation_values[0])
        computed["Affiliation-Recall"] = float(affiliation_values[1])
        computed["Affiliation-F"] = float(affiliation_values[2])

    if "Precision@K" in selected:
        k = int(labels.sum()) if precision_k is None else non_negative_integer(
            precision_k, name="precision_k"
        )
        computed["Precision@K"] = _precision_at_k_prepared(labels, scores, k)

    if "K-Delay-PA-F1" in selected:
        computed["K-Delay-PA-F1"] = _k_delay_prepared(
            labels, scores, predictions, delay=k_delay
        ).f1

    if "Event-PA-F1" in selected:
        computed["Event-PA-F1"] = _event_adjusted_prepared(
            labels,
            scores,
            predictions,
            scale=event_scale,
            base=event_base,
        ).f1

    if any(name in _INTERVAL_METRICS for name in selected):
        interval_values = (
            interval_prf_kernel(labels, predictions)
            if predictions is not None
            else best_interval_prf_kernel(labels, scores, thresholds)
        )
        computed["Interval-Precision"] = float(interval_values[0])
        computed["Interval-Recall"] = float(interval_values[1])
        computed["Interval-F1"] = float(interval_values[2])

    if any(name in _TOLERANCE_METRICS for name in selected):
        tolerance = non_negative_integer(tolerance, name="tolerance")
        tolerant_labels = dilate_labels_kernel(labels, tolerance)
        if predictions is not None:
            tolerant_adjusted = point_adjust_kernel(tolerant_labels, predictions)
            tolerant_metrics = _point_metrics_prepared(
                tolerant_labels, tolerant_adjusted
            )
            tolerance_values = (
                tolerant_metrics.precision,
                tolerant_metrics.recall,
                tolerant_metrics.f1,
            )
        else:
            tolerance_values = best_point_adjusted_prf_kernel(
                tolerant_labels, scores, thresholds
            )
        computed["Tolerance-PA-Precision"] = float(tolerance_values[0])
        computed["Tolerance-PA-Recall"] = float(tolerance_values[1])
        computed["Tolerance-PA-F1"] = float(tolerance_values[2])

    if any(name in _FIXED_VUS_METRICS for name in selected):
        if predictions is None:
            raise ValueError("VUS-Precision, VUS-Recall and VUS-F require y_pred")
        fixed_vus = _fixed_vus_prf_kernel(labels, predictions, sliding_window)
        computed["VUS-Precision"] = float(fixed_vus[0])
        computed["VUS-Recall"] = float(fixed_vus[1])
        computed["VUS-F"] = float(fixed_vus[2])

    if "PATE" in selected or "PATE-F1" in selected:
        early_buffers = _buffer_points(
            pate_early_buffer, pate_buffer_splits, pate_include_zero
        )
        delayed_buffers = _buffer_points(
            pate_delayed_buffer, pate_buffer_splits, pate_include_zero
        )
        if "PATE" in selected or predictions is None:
            pate_thresholds = _rank_thresholds(scores, pate_threshold_count)
            pate_values = pate_auc_kernel(
                labels, scores, pate_thresholds, early_buffers, delayed_buffers
            )
            computed["PATE"] = float(pate_values[0])
            if predictions is None:
                computed["PATE-F1"] = float(pate_values[1])
        if "PATE-F1" in selected and predictions is not None:
            computed["PATE-F1"] = float(
                pate_f1_kernel(labels, predictions, early_buffers, delayed_buffers)
            )

    if any(
        name
        in ("First-Hit-Rank", "First-Hit-Fraction", "Hit@3%", "Hit@10%")
        for name in selected
    ):
        if labels.any():
            order = np.argsort(-scores, kind="stable")
            rank = int(np.flatnonzero(labels[order])[0]) + 1
            fraction = rank / labels.size
            computed["First-Hit-Rank"] = float(rank)
            computed["First-Hit-Fraction"] = float(fraction)
            computed["Hit@3%"] = float(fraction < 0.03)
            computed["Hit@10%"] = float(fraction < 0.10)
        else:
            computed["First-Hit-Rank"] = math.nan
            computed["First-Hit-Fraction"] = math.nan
            computed["Hit@3%"] = math.nan
            computed["Hit@10%"] = math.nan

    return cast(dict[MetricName, float], {name: computed[name] for name in selected})


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
        metrics=TSB_AD_METRICS,
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
    window_size = non_negative_integer(slidingWindow, name="slidingWindow")
    threshold_count = validate_threshold_count(thre, name="thre")
    result = _volume_under_surface_prepared(
        labels,
        scores,
        window_size=window_size,
        threshold_count=threshold_count,
    )
    tpr = result.tpr.ravel()
    fpr = result.fpr.ravel()
    precision = result.precision.ravel()
    window = np.repeat(result.windows, result.tpr.shape[1])
    tpr_ap = result.tpr[:, :-1].ravel()
    window_ap = np.repeat(result.windows, result.tpr.shape[1] - 1)
    return fpr, window, tpr, tpr_ap, precision, window_ap, result.roc, result.pr
