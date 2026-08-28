from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from agentad import SeriesData, SeriesItem, read
from agentad.evaluation import (
    DEFAULT_METRICS,
    evaluate,
    evaluate_collection,
    event_prf,
    find_period,
    point_adjust,
    point_adjusted_prf,
    point_metrics,
    range_prf,
    reconstruct_series,
    volume_under_surface,
    warmup,
)


@pytest.fixture
def binary_case():
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 0], dtype=np.uint8)
    scores = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.4, 0.9, 0.6, 0.5, 0.0])
    predictions = scores > 0.55
    return labels, scores, predictions


def test_fixed_prediction_metric_families(binary_case):
    labels, _, predictions = binary_case

    assert point_metrics(labels, predictions).f1 == pytest.approx(8 / 9)
    assert point_adjusted_prf(labels, predictions).f1 == 1.0
    assert event_prf(labels, predictions).f1 == 1.0
    assert range_prf(labels, predictions).f1 == pytest.approx(13 / 14)


def test_point_adjust_does_not_mutate_prediction(binary_case):
    labels, _, predictions = binary_case
    original = predictions.copy()

    adjusted = point_adjust(labels, predictions)

    np.testing.assert_array_equal(predictions, original)
    assert adjusted.dtype == np.bool_
    assert adjusted[6:9].all()


def test_metric_selection_and_undefined_classes(binary_case):
    labels, scores, _ = binary_case
    result = evaluate(labels, scores, metrics=("AUC-ROC", "Standard-F1"))
    assert tuple(result) == ("AUC-ROC", "Standard-F1")
    assert result["AUC-ROC"] == 1.0

    all_normal = evaluate(np.zeros(5), np.arange(5), metrics=("AUC-ROC", "AUC-PR"))
    assert math.isnan(all_normal["AUC-ROC"])
    assert math.isnan(all_normal["AUC-PR"])

    with pytest.raises(ValueError, match="unknown metrics"):
        evaluate(labels, scores, metrics=("not-a-metric",))


def test_vus_surface_shape_and_scalars(binary_case):
    labels, scores, _ = binary_case
    result = volume_under_surface(labels, scores, window_size=4, threshold_count=11)

    assert result.tpr.shape == (5, 13)
    assert result.fpr.shape == (5, 13)
    assert result.precision.shape == (5, 12)
    assert result.roc == 1.0
    assert result.pr == 1.0


def _collection_item(
    series_id: str, data: list[float], timestamps: list[int], labels: list[int]
) -> SeriesItem:
    return SeriesItem(
        series_id,
        np.asarray(data).reshape(-1, 1),
        np.asarray(timestamps, dtype=np.int64),
        pd.DataFrame({"label": labels}),
    )


def _collection(items: list[SeriesItem]) -> SeriesData:
    return SeriesData.from_items(items, features=pd.DataFrame({"name": ["x"]}))


def test_collection_reconstruction_and_evaluation(tmp_path):
    train = _collection([_collection_item("s", [1.0, 2.0], [0, 1], [0, 0])])
    test = _collection([_collection_item("s", [3.0, 4.0, 5.0], [0, 1, 2], [0, 1, 1])])

    reconstructed = next(reconstruct_series(test, train))
    np.testing.assert_array_equal(reconstructed.data[:, 0], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(reconstructed.timestamps, [0, 1, 0, 1, 2])
    assert reconstructed.train_bounds == (0, 2)

    score_path = tmp_path / "scores.zarr.zip"
    result = evaluate_collection(
        test,
        train,
        scores={"s": np.array([0.0, 0.1, 0.2, 0.8, 0.9])},
        metrics=("AUC-ROC", "Standard-F1"),
        sliding_window=0,
        on_error="raise",
        save_scores_to=score_path,
    )
    assert result.metric_columns == ("AUC-ROC", "Standard-F1")
    assert result.per_series.loc[0, "n_points"] == 5
    assert bool(result.per_series.loc[0, "full_eval"])
    assert result.per_series.loc[0, "AUC-ROC"] == 1.0
    assert result.failures.empty

    saved = read(score_path)["s"]
    np.testing.assert_array_equal(saved.data[:, 0], [0.0, 0.1, 0.2, 0.8, 0.9])
    np.testing.assert_array_equal(saved.timestamps, [0, 1, 0, 1, 2])
    np.testing.assert_array_equal(saved.labels["label"], [0, 0, 0, 1, 1])


def test_collection_error_policy_raises():
    test = _collection([_collection_item("s", [1.0, 2.0], [0, 1], [0, 1])])
    with pytest.raises(KeyError, match="no score"):
        evaluate_collection(
            test,
            scores={},
            metrics=("AUC-ROC",),
            sliding_window=0,
            on_error="raise",
        )


def test_precomputed_scores_skip_unused_data_period_and_score_items(monkeypatch):
    import agentad.evaluation.collection as collection_module
    import agentad.evaluation.period as period_module

    train_item = SeriesItem(
        "s",
        np.array([[1.0], [2.0]]),
        np.array([0, 1], dtype=np.int64),
        pd.DataFrame({"label": [0, 0]}),
    )
    test_item = SeriesItem(
        "s",
        np.array([[3.0, 30.0], [4.0, 40.0]]),
        np.array([0, 1], dtype=np.int64),
        pd.DataFrame({"label": [0, 1]}),
    )
    train = SeriesData.from_items(
        [train_item], features=pd.DataFrame({"name": ["train_feature"]})
    )
    test = SeriesData.from_items(
        [test_item], features=pd.DataFrame({"name": ["test_a", "test_b"]})
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("unused evaluation path was called")

    monkeypatch.setattr(collection_module, "SeriesItem", unexpected)
    monkeypatch.setattr(period_module, "find_period", unexpected)
    result = evaluate_collection(
        test,
        train,
        scores={"s": np.array([0.0, 0.1, 0.2, 0.9])},
        metrics=("AUC-ROC",),
        sliding_window="auto",
        on_error="raise",
    )
    assert result.per_series.loc[0, "AUC-ROC"] == 1.0


def test_period_fallback_for_constant_series():
    assert find_period(np.ones(20)) == 125
    values = np.ones(20_001)
    values[-1] = np.nan
    assert find_period(values) == 125


def test_explicit_warmup():
    assert warmup() is None


def test_default_metric_names_are_stable_and_unique():
    assert len(DEFAULT_METRICS) == 9
    assert len(set(DEFAULT_METRICS)) == len(DEFAULT_METRICS)
