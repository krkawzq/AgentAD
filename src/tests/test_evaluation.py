from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from agentad import SeriesData, SeriesItem, read
from agentad.evaluation import (
    affiliation_prf,
    evaluate,
    evaluate_collection,
    event_prf,
    find_period,
    generate_curve,
    point_adjust,
    point_adjusted_prf,
    point_metrics,
    range_prf,
    reconstruct_series,
    volume_under_surface,
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


def test_point_adjust_fills_whole_event_including_series_start():
    labels = np.array([1, 1, 1, 1, 0, 0], dtype=np.uint8)
    predictions = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)

    adjusted = point_adjust(labels, predictions)

    np.testing.assert_array_equal(adjusted, [1, 1, 1, 1, 0, 0])


def test_event_and_affiliation_boundary_semantics():
    assert event_prf([0, 1, 1], [0, 0, 1]).f1 == 1.0
    assert affiliation_prf([0, 1, 1], [0, 0, 0]).f1 == 0.0


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


def test_reconstruction_and_scoring_are_isolated_from_source():
    test = _collection([_collection_item("s", [1.0, 2.0], [10, 11], [0, 1])])
    expected_data = test["s"].data.copy()
    expected_timestamps = test["s"].timestamps.copy()
    expected_labels = test["s"].labels.copy(deep=True)

    reconstructed = next(reconstruct_series(test))
    reconstructed.data[0, 0] = 99.0
    reconstructed.timestamps[0] = 999
    reconstructed.labels[0] = 1

    np.testing.assert_array_equal(test["s"].data, expected_data)
    np.testing.assert_array_equal(test["s"].timestamps, expected_timestamps)
    pd.testing.assert_frame_equal(test["s"].labels, expected_labels)

    def mutating_score(train_data, data):
        assert train_data is None
        data[:, 0] = -1.0
        return np.array([0.0, 1.0])

    evaluate_collection(
        test,
        score_fn=mutating_score,
        metrics=("AUC-ROC",),
        on_error="raise",
    )
    np.testing.assert_array_equal(test["s"].data, expected_data)


def test_data_consumers_reject_mismatched_feature_schemas():
    train = _collection([_collection_item("s", [1.0, 2.0], [0, 1], [0, 0])])
    test = _collection([_collection_item("s", [3.0, 4.0], [0, 1], [0, 1])])
    train.features.loc[0, "name"] = "temperature"
    test.features.loc[0, "name"] = "pressure"

    with pytest.raises(ValueError, match="feature schemas"):
        next(reconstruct_series(test, train))
    with pytest.raises(ValueError, match="feature schemas"):
        evaluate_collection(
            test,
            train,
            score_fn=lambda train_data, data: np.arange(len(data)),
            metrics=("AUC-ROC",),
        )


def test_data_consumers_reject_mismatched_data_dtypes():
    labels = pd.DataFrame({"label": [0, 1]})
    train = SeriesData.from_items(
        [
            SeriesItem(
                "s",
                np.array([[1.0], [2.0]], dtype=np.float32),
                np.array([0, 1], dtype=np.int64),
                labels,
            )
        ],
        features=pd.DataFrame({"name": ["x"]}),
    )
    test = SeriesData.from_items(
        [
            SeriesItem(
                "s",
                np.array([[3.0], [4.0]], dtype=np.float64),
                np.array([0, 1], dtype=np.int64),
                labels,
            )
        ],
        features=pd.DataFrame({"name": ["x"]}),
    )

    with pytest.raises(ValueError, match="data dtypes"):
        next(reconstruct_series(test, train))


def test_data_consumers_reject_mismatched_feature_attrs():
    train = _collection([_collection_item("s", [1.0, 2.0], [0, 1], [0, 0])])
    test = _collection([_collection_item("s", [3.0, 4.0], [0, 1], [0, 1])])
    train.features.attrs["unit"] = "celsius"
    test.features.attrs["unit"] = "fahrenheit"

    with pytest.raises(ValueError, match="feature attrs"):
        next(reconstruct_series(test, train))


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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"score_fn": object()}, "score_fn must be callable"),
        ({"scores": []}, "scores must be a mapping"),
    ],
)
def test_collection_rejects_invalid_score_provider_configuration(kwargs, message):
    test = _collection([_collection_item("s", [1.0, 2.0], [0, 1], [0, 1])])
    with pytest.raises(TypeError, match=message):
        evaluate_collection(test, metrics=("AUC-ROC",), **kwargs)


def test_collection_error_policy_contains_label_validation_failures():
    test = _collection(
        [
            _collection_item("bad", [1.0, 2.0], [0, 1], [0, 2]),
            _collection_item("good", [1.0, 2.0], [0, 1], [0, 1]),
        ]
    )
    result = evaluate_collection(
        test,
        scores={"bad": [0.0, 1.0], "good": [0.0, 1.0]},
        metrics=("AUC-ROC",),
        on_error="nan",
    )

    assert result.per_series["id"].tolist() == ["bad", "good"]
    assert result.failures["id"].tolist() == ["bad"]
    assert math.isnan(result.per_series.loc[0, "AUC-ROC"])
    assert result.per_series.loc[1, "AUC-ROC"] == 1.0


@pytest.mark.parametrize(
    ("metrics", "kwargs", "message"),
    [
        (("PA-F1",), {"threshold_count": 1}, "threshold_count"),
        (
            ("VUS-ROC",),
            {"sliding_window": 0, "vus_threshold_count": 1},
            "vus_threshold_count",
        ),
    ],
)
def test_collection_validates_global_thresholds_before_scoring(
    metrics, kwargs, message
):
    test = _collection([_collection_item("s", [1.0, 2.0], [0, 1], [0, 1])])
    calls = 0

    def score_fn(train_data, data):
        nonlocal calls
        calls += 1
        return np.arange(len(data))

    with pytest.raises(ValueError, match=message):
        evaluate_collection(
            test,
            score_fn=score_fn,
            metrics=metrics,
            on_error="nan",
            **kwargs,
        )
    assert calls == 0


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


@pytest.mark.parametrize(
    ("sliding_window", "thresholds", "error", "message"),
    [
        (-1, 3, ValueError, "slidingWindow must be non-negative"),
        (1.5, 3, TypeError, "slidingWindow must be an integer"),
        (True, 3, TypeError, "slidingWindow must be an integer"),
        (1, 1, ValueError, "thre must be at least 2"),
        (1, 3.5, TypeError, "thre must be an integer"),
        (1, True, TypeError, "thre must be an integer"),
    ],
)
def test_generate_curve_validates_integer_parameters(
    sliding_window, thresholds, error, message
):
    with pytest.raises(error, match=message):
        generate_curve([0, 1, 0], [0.0, 1.0, 0.5], sliding_window, thre=thresholds)
