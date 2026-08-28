from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from agentad import SeriesData, SeriesItem
from agentad.evaluation import (
    AVAILABLE_METRICS,
    DEFAULT_METRICS,
    TSB_AD_METRICS,
    average_precision,
    evaluate,
    evaluate_collection,
    event_adjusted_prf,
    first_hit,
    fixed_vus_prf,
    get_metrics,
    interval_prf,
    k_delay_point_adjusted_prf,
    pate,
    pate_prf,
    point_adjusted_average_precision,
    point_metrics,
    precision_at_k,
    range_auc,
    tolerance_point_adjusted_prf,
    volume_under_surface,
)


def test_metric_catalog_is_complete_and_unique():
    assert len(AVAILABLE_METRICS) == 48
    assert len(set(AVAILABLE_METRICS)) == len(AVAILABLE_METRICS)
    assert set(DEFAULT_METRICS) <= set(AVAILABLE_METRICS)
    assert len(TSB_AD_METRICS) == 9


def test_extended_point_metrics_and_rank_metrics():
    labels = np.array([0, 1, 1, 0], dtype=np.uint8)
    predictions = np.array([0, 1, 0, 1], dtype=np.uint8)
    scores = np.array([0.1, 0.9, 0.8, 0.2])

    result = point_metrics(labels, predictions)
    assert (result.tp, result.fp, result.fn, result.tn) == (1, 1, 1, 1)
    assert result.accuracy == 0.5
    assert result.precision == result.recall == result.f1 == result.f05 == 0.5
    assert result.mcc == 0.0
    assert precision_at_k(labels, scores) == 1.0

    hit = first_hit(labels, [0.8, 0.85, 0.6, 0.9])
    assert hit.rank == 2
    assert hit.fraction == 0.5
    assert not hit.hit_at_3_percent
    assert not hit.hit_at_10_percent


def test_point_adjusted_average_precision_uses_event_maxima():
    labels = np.array([0, 1, 1, 0, 1, 1], dtype=np.uint8)
    scores = np.array([0.2, 0.1, 0.9, 0.8, 0.3, 0.7])
    adjusted = np.array([0.2, 0.9, 0.9, 0.8, 0.7, 0.7])
    assert point_adjusted_average_precision(labels, scores) == pytest.approx(
        average_precision(labels, adjusted)
    )


def test_range_auc_is_the_requested_vus_window():
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.4, 0.7, 0.0])
    surface = volume_under_surface(labels, scores, window_size=3, threshold_count=7)
    result = range_auc(labels, scores, window_size=3, threshold_count=7)
    assert result.roc == pytest.approx(surface.roc_by_window[-1])
    assert result.pr == pytest.approx(surface.pr_by_window[-1])
    values = evaluate(
        labels,
        scores,
        metrics=("R-AUC-ROC", "R-AUC-PR", "VUS-ROC", "VUS-PR"),
        sliding_window=3,
        vus_threshold_count=7,
    )
    assert values["R-AUC-ROC"] == pytest.approx(result.roc)
    assert values["R-AUC-PR"] == pytest.approx(result.pr)


def test_fixed_vus_requires_and_uses_one_binary_decision():
    labels = np.array([0, 1, 1, 0, 0], dtype=np.uint8)
    scores = np.array([0.1, 0.9, 0.8, 0.2, 0.3])
    predictions = np.array([0, 1, 0, 0, 0], dtype=np.uint8)
    expected = fixed_vus_prf(labels, predictions, window_size=2)
    values = evaluate(
        labels,
        scores,
        y_pred=predictions,
        metrics=("VUS-Precision", "VUS-Recall", "VUS-F"),
        sliding_window=2,
    )
    assert values == pytest.approx(
        {
            "VUS-Precision": expected.precision,
            "VUS-Recall": expected.recall,
            "VUS-F": expected.f1,
        }
    )
    with pytest.raises(ValueError, match="require y_pred"):
        evaluate(labels, scores, metrics=("VUS-F",))


def test_delay_event_interval_and_tolerance_protocols():
    labels = np.array([0, 1, 1, 1, 0], dtype=np.uint8)
    late = np.array([0, 0, 0, 1, 0], dtype=np.uint8)
    assert k_delay_point_adjusted_prf(labels, late, delay=2).f1 == 0.5
    assert k_delay_point_adjusted_prf(labels, late, delay=3).f1 == 1.0

    dense_event = np.array([0, 1, 1, 1, 1], dtype=np.uint8)
    event_labels = np.array([0, 1, 1, 1, 0], dtype=np.uint8)
    assert event_adjusted_prf(event_labels, dense_event, scale="squeeze").f1 == pytest.approx(
        2 / 3
    )

    interval_labels = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
    spanning_prediction = np.array([0, 1, 1, 1, 0], dtype=np.uint8)
    assert interval_prf(interval_labels, spanning_prediction).f1 == 1.0

    point_label = np.array([0, 0, 1, 0, 0], dtype=np.uint8)
    early_prediction = np.array([1, 0, 0, 0, 0], dtype=np.uint8)
    assert tolerance_point_adjusted_prf(point_label, early_prediction, tolerance=1).f1 == 0.0
    assert tolerance_point_adjusted_prf(point_label, early_prediction, tolerance=2).f1 == 1.0


def test_pate_perfect_detection_and_undefined_labels():
    labels = np.array([0, 0, 1, 1, 0, 0], dtype=np.uint8)
    predictions = labels.copy()
    scores = np.array([0.0, 0.1, 0.9, 0.8, 0.2, 0.3])
    assert pate_prf(labels, predictions, early_buffer=1, delayed_buffer=1).f1 == 1.0
    assert pate(
        labels,
        scores,
        early_buffer=0,
        delayed_buffer=0,
        threshold_count=6,
    ) == pytest.approx(1.0)
    assert math.isnan(pate(np.zeros(4), np.arange(4), threshold_count=4))


def _collection_item(series_id: str, labels: list[int]) -> SeriesItem:
    length = len(labels)
    return SeriesItem(
        series_id,
        np.arange(length, dtype=np.float64).reshape(-1, 1),
        np.arange(length, dtype=np.int64),
        pd.DataFrame({"label": labels}),
    )


def test_collection_fixed_predictions_and_micro_summary():
    test = SeriesData.from_items(
        [
            _collection_item("a", [0, 1, 1]),
            _collection_item("b", [0, 0, 1]),
        ],
        features=pd.DataFrame({"name": ["x"]}),
    )
    result = evaluate_collection(
        test,
        scores={"a": [0.0, 0.9, 0.8], "b": [0.0, 0.1, 0.7]},
        predictions={"a": [0, 1, 0], "b": [0, 1, 1]},
        metrics=("Standard-Accuracy", "Standard-Precision", "Standard-Recall", "Standard-F1"),
        sliding_window=0,
        on_error="raise",
    )
    micro = result.micro_summary()
    assert micro["Standard-Accuracy"] == pytest.approx(4 / 6)
    assert micro["Standard-Precision"] == pytest.approx(2 / 3)
    assert micro["Standard-Recall"] == pytest.approx(2 / 3)
    assert micro["Standard-F1"] == pytest.approx(2 / 3)


def test_full_catalog_selection_returns_requested_order():
    labels = np.array([0, 1, 1, 0, 1, 0], dtype=np.uint8)
    scores = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3])
    predictions = scores > 0.5
    values = evaluate(
        labels,
        scores,
        y_pred=predictions,
        metrics=AVAILABLE_METRICS,
        sliding_window=2,
        threshold_count=7,
        vus_threshold_count=7,
        pate_threshold_count=7,
        pate_early_buffer=1,
        pate_delayed_buffer=1,
    )
    assert tuple(values) == AVAILABLE_METRICS
    assert all(isinstance(value, float) for value in values.values())


def test_tsb_compatibility_wrapper_keeps_nine_columns():
    result = get_metrics([0.0, 0.9, 0.8, 0.1], [0, 1, 1, 0], slidingWindow=1, thre=5)
    assert tuple(result) == TSB_AD_METRICS
