"""Collection reconstruction, detector execution and metric aggregation."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .metrics import (
    DEFAULT_METRICS,
    MetricName,
    _FIXED_VUS_METRICS,
    _ORACLE_GRID_METRICS,
    _STANDARD_COUNT_METRICS,
    _STANDARD_METRICS,
    _SURFACE_METRICS,
    _evaluate_prepared,
    _metric_selection,
)
from .point import EventScale, _event_scale_code, _point_metrics_from_counts
from .range import _buffer_points
from ._validation import (
    binary_array,
    non_negative_integer,
    score_array,
    threshold_count as validate_threshold_count,
)
from ..series import SeriesData, SeriesItem, write as write_series
from ..series._data import _attrs_signature

ScoreFn: TypeAlias = Callable[
    [NDArray[np.generic] | None, NDArray[np.generic]], ArrayLike
]

__all__ = [
    "EvaluationResult",
    "ReconstructedSeries",
    "ScoreFn",
    "evaluate_collection",
    "reconstruct_series",
]


@dataclass(frozen=True, slots=True)
class ReconstructedSeries:
    """One full evaluation sequence reconstructed from stored splits.

    Arrays have a shared first-axis length and own their storage. ``labels`` is
    contiguous ``uint8`` containing only zero and one; ``timestamps`` preserves
    the source's opaque ``int64`` values without sorting or unit conversion.
    """

    id: str
    data: np.ndarray
    timestamps: NDArray[np.int64]
    labels: NDArray[np.uint8]
    full_eval: bool
    train_bounds: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class _EvaluationSeries:
    id: str
    labels: NDArray[np.uint8]
    full_eval: bool
    data: np.ndarray | None
    timestamps: NDArray[np.int64] | None
    period_values: np.ndarray | None
    train_bounds: tuple[int, int] | None


@dataclass(slots=True)
class EvaluationResult:
    """Per-series results with explicit metric-column ownership.

    ``per_series`` contains one row per requested test id plus audit columns;
    ``metric_columns`` records the selected metric order. Failed rows remain in
    the table according to the collection evaluation error policy.
    """

    per_series: pd.DataFrame
    metric_columns: tuple[MetricName, ...]

    def summary(self, stat: Literal["mean", "median"] = "mean") -> pd.Series:
        """Macro-average or median across series, skipping NaN metric values.

        ``stat`` must be ``"mean"`` or ``"median"``. Every series contributes
        at most one value per metric, independent of its length.
        """
        if stat not in ("mean", "median"):
            raise ValueError(f"stat must be 'mean' or 'median', got {stat!r}")
        if self.per_series.empty:
            return pd.Series({name: math.nan for name in self.metric_columns})
        return self.per_series[list(self.metric_columns)].agg(stat, axis=0)

    def micro_summary(self) -> pd.Series:
        """Aggregate standard point metrics from summed confusion counts.

        The result is defined only when evaluation selected a standard point
        metric, because TP/FP/FN/TN must be available for every retained row.
        """
        if self.per_series.empty:
            raise ValueError("cannot micro-aggregate an empty evaluation")
        missing = [
            name for name in _STANDARD_COUNT_METRICS if name not in self.per_series
        ]
        if missing:
            raise ValueError("standard point metrics were not evaluated")
        counts = [
            int(self.per_series[name].sum(skipna=True))
            for name in _STANDARD_COUNT_METRICS
        ]
        result = _point_metrics_from_counts(*counts)
        values = {
            "Standard-Accuracy": result.accuracy,
            "Standard-Precision": result.precision,
            "Standard-Recall": result.recall,
            "Standard-F0.5": result.f05,
            "Standard-F1": result.f1,
            "Standard-MCC": result.mcc,
            "Standard-TP": float(result.tp),
            "Standard-FP": float(result.fp),
            "Standard-FN": float(result.fn),
            "Standard-TN": float(result.tn),
        }
        selected = [name for name in self.metric_columns if name in values]
        return pd.Series({name: values[name] for name in selected}, dtype=np.float64)

    def anomalous_summary(self, stat: Literal["mean", "median"] = "mean") -> pd.Series:
        """Macro-aggregate only series containing labelled anomaly points.

        ``stat`` follows :meth:`summary`. This requires the ``n_anomalies``
        audit column produced by :func:`evaluate_collection`.
        """
        if stat not in ("mean", "median"):
            raise ValueError(f"stat must be 'mean' or 'median', got {stat!r}")
        if "n_anomalies" not in self.per_series:
            raise ValueError("per-series anomaly counts are unavailable")
        selected = self.per_series[self.per_series["n_anomalies"] > 0]
        if selected.empty:
            return pd.Series({name: math.nan for name in self.metric_columns})
        return selected[list(self.metric_columns)].agg(stat, axis=0)

    @property
    def failures(self) -> pd.DataFrame:
        """Return rows with a non-null per-series scoring or validation error."""
        if "error" not in self.per_series:
            return self.per_series.iloc[0:0]
        return self.per_series[self.per_series["error"].notna()]


def _labels_of(item: SeriesItem, label_column: str) -> NDArray[np.uint8]:
    labels = item.labels
    if label_column not in labels.columns:
        raise ValueError(
            f"series {item.id!r}: label column {label_column!r} missing; "
            f"available: {list(labels.columns)}"
        )
    try:
        return binary_array(labels[label_column].to_numpy(), name="labels").copy()
    except (TypeError, ValueError) as error:
        raise type(error)(f"series {item.id!r}: {error}") from error


def _validate_collections(test: SeriesData, train: SeriesData | None) -> None:
    if not isinstance(test, SeriesData):
        raise TypeError("test must be a SeriesData collection")
    if train is not None and not isinstance(train, SeriesData):
        raise TypeError("train must be a SeriesData collection or None")
    if train is not None:
        missing = [series_id for series_id in test.ids if series_id not in train]
        if missing:
            raise ValueError(
                f"{len(missing)} test series lack a train split, e.g. {missing[:3]}"
            )


def _validate_data_compatibility(test: SeriesData, train: SeriesData | None) -> None:
    if train is None:
        return
    try:
        pd.testing.assert_frame_equal(
            train.features,
            test.features,
            check_exact=True,
            check_like=False,
        )
    except AssertionError as error:
        raise ValueError(
            "train and test feature schemas must match positionally"
        ) from error
    if _attrs_signature(train.features.attrs) != _attrs_signature(test.features.attrs):
        raise ValueError("train and test feature attrs must match")
    if train.data.dtype != test.data.dtype:
        raise ValueError(
            "train and test data dtypes must match: "
            f"{train.data.dtype} != {test.data.dtype}"
        )


def _prepare_evaluation_series(
    test: SeriesData,
    train: SeriesData | None,
    series_id: str,
    *,
    label_column: str,
    need_data: bool,
    need_timestamps: bool,
    need_period: bool,
) -> _EvaluationSeries:
    test_item = test[series_id]
    test_labels = _labels_of(test_item, label_column)
    train_item = train[series_id] if train is not None else None
    train_labels = (
        _labels_of(train_item, label_column) if train_item is not None else None
    )
    labels = (
        np.concatenate((train_labels, test_labels))
        if train_labels is not None
        else test_labels
    )
    data = None
    if need_data:
        data = (
            np.concatenate((train_item.data, test_item.data), axis=0)
            if train_item is not None
            else np.array(test_item.data, copy=True, order="C")
        )
    timestamps = None
    if need_timestamps:
        timestamps = (
            np.concatenate((train_item.timestamps, test_item.timestamps))
            if train_item is not None
            else np.array(test_item.timestamps, dtype=np.int64, copy=True)
        )
    period_values = None
    if need_period:
        if data is not None:
            period_values = data[:, 0]
        else:
            period_values = (
                np.concatenate((train_item.data[:, 0], test_item.data[:, 0]))
                if train_item is not None
                else np.asarray(test_item.data[:, 0])
            )
    return _EvaluationSeries(
        series_id,
        labels,
        train_item is not None,
        data,
        timestamps,
        period_values,
        (0, len(train_item)) if train_item is not None else None,
    )


def _evaluation_series(
    test: SeriesData,
    train: SeriesData | None,
    *,
    label_column: str,
    need_data: bool,
    need_timestamps: bool,
    need_period: bool,
) -> Iterator[_EvaluationSeries]:
    _validate_collections(test, train)
    if need_data or need_period:
        _validate_data_compatibility(test, train)
    for series_id in test.ids:
        yield _prepare_evaluation_series(
            test,
            train,
            series_id,
            label_column=label_column,
            need_data=need_data,
            need_timestamps=need_timestamps,
            need_period=need_period,
        )


def reconstruct_series(
    test: SeriesData,
    train: SeriesData | None = None,
    *,
    label_column: str = "label",
) -> Iterator[ReconstructedSeries]:
    """Yield test series, optionally prefixed by their matching train split.

    ``test`` and optional ``train`` must be :class:`SeriesData` collections.
    Every test id must exist in ``train`` when it is supplied; feature schemas,
    feature attrs and data dtypes must match exactly. Each item must expose the
    requested binary ``label_column`` with one value per data row.

    Split identity defines order: training data is always the prefix even when
    individual split timestamps restart from zero. Every test id must exist in
    ``train`` when a train collection is supplied. Returned arrays own their
    storage and can be modified without changing either input collection.
    """
    for series in _evaluation_series(
        test,
        train,
        label_column=label_column,
        need_data=True,
        need_timestamps=True,
        need_period=False,
    ):
        assert series.data is not None
        assert series.timestamps is not None
        yield ReconstructedSeries(
            series.id,
            series.data,
            series.timestamps,
            series.labels,
            series.full_eval,
            series.train_bounds,
        )


def _resolve_score(
    score_fn: ScoreFn | None,
    scores: Mapping[str, ArrayLike] | None,
    series: _EvaluationSeries,
) -> NDArray[np.float64]:
    if score_fn is not None:
        assert series.data is not None
        train_data = (
            series.data[slice(*series.train_bounds)]
            if series.train_bounds is not None
            else None
        )
        raw_score = score_fn(train_data, series.data)
    else:
        if scores is None or series.id not in scores:
            raise KeyError(f"no score for series {series.id!r}")
        raw_score = scores[series.id]
    score = score_array(raw_score, name="detector score")
    if score.size != series.labels.size:
        raise ValueError(
            f"score length {score.size} != series length {series.labels.size}"
        )
    return score


def _resolve_prediction(
    predictions: Mapping[str, ArrayLike] | None,
    series: _EvaluationSeries,
) -> NDArray[np.uint8] | None:
    if predictions is None:
        return None
    if series.id not in predictions:
        raise KeyError(f"no prediction for series {series.id!r}")
    prediction = binary_array(predictions[series.id], name="detector prediction")
    if prediction.size != series.labels.size:
        raise ValueError(
            f"prediction length {prediction.size} != series length {series.labels.size}"
        )
    return prediction


def _save_scores(items: list[SeriesItem], path: str | Path) -> None:
    if not items:
        raise ValueError("no successfully evaluated scores to save")
    features = pd.DataFrame(
        {"dtype": ["float64"]},
        index=pd.Index(["anomaly_score"], name="feature"),
    )
    collection = SeriesData.from_items(items, features=features)
    write_series(collection, path)


def evaluate_collection(
    test: SeriesData,
    train: SeriesData | None = None,
    *,
    score_fn: ScoreFn | None = None,
    scores: Mapping[str, ArrayLike] | None = None,
    predictions: Mapping[str, ArrayLike] | None = None,
    metrics: Iterable[MetricName] = DEFAULT_METRICS,
    label_column: str = "label",
    sliding_window: int | Literal["auto"] = "auto",
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
    on_error: Literal["nan", "zero", "raise"] = "nan",
    save_scores_to: str | Path | None = None,
    progress: bool = False,
) -> EvaluationResult:
    """Score and evaluate every series in a collection.

    ``test`` and optional ``train`` follow :func:`reconstruct_series`. Exactly
    one of ``score_fn`` and ``scores`` is required. A score callback receives
    ``(train_data_or_none, full_data)`` and must return one finite numeric score
    per full-sequence row, with larger values meaning more anomalous. A score
    mapping and optional prediction mapping must contain every evaluated series
    id; mapped predictions are equal-length binary vectors.

    ``sliding_window="auto"`` estimates a point-count window from the first
    feature, so train/test feature metadata and dtypes must match. Explicit
    windows, buffers, delays and tolerances are interpreted in concatenated
    array order, not timestamp units. ``on_error`` is a per-series policy;
    ``raise`` preserves the original exception, while ``nan`` and ``zero``
    retain an error column for auditability. ``save_scores_to`` writes only
    successfully evaluated, full-length scores as a new series collection.
    ``score_seconds`` times callback execution only; precomputed mappings leave
    it undefined and report their validation overhead as ``score_load_seconds``.
    """
    if (score_fn is None) == (scores is None):
        raise ValueError("pass exactly one of score_fn and scores")
    if score_fn is not None and not callable(score_fn):
        raise TypeError("score_fn must be callable")
    if scores is not None and not isinstance(scores, Mapping):
        raise TypeError("scores must be a mapping")
    if predictions is not None and not isinstance(predictions, Mapping):
        raise TypeError("predictions must be a mapping")
    if on_error not in ("nan", "zero", "raise"):
        raise ValueError(f"invalid on_error policy: {on_error!r}")
    if sliding_window != "auto":
        sliding_window = non_negative_integer(sliding_window, name="sliding_window")

    selected = _metric_selection(metrics)
    needs_vus = any(name in _SURFACE_METRICS + _FIXED_VUS_METRICS for name in selected)
    needs_oracle_grid = predictions is None and any(
        name in _ORACLE_GRID_METRICS for name in selected
    )
    if needs_oracle_grid:
        threshold_count = validate_threshold_count(threshold_count)
    if any(name in _SURFACE_METRICS for name in selected):
        vus_threshold_count = validate_threshold_count(
            vus_threshold_count, name="vus_threshold_count"
        )
    if any(name in _FIXED_VUS_METRICS for name in selected) and predictions is None:
        raise ValueError("VUS-Precision, VUS-Recall and VUS-F require predictions")
    if not 0.0 <= range_alpha <= 1.0:
        raise ValueError("range_alpha must be in [0, 1]")
    if "Precision@K" in selected and precision_k is not None:
        precision_k = non_negative_integer(precision_k, name="precision_k")
    if "K-Delay-PA-F1" in selected:
        k_delay = non_negative_integer(k_delay, name="k_delay")
        if k_delay < 1:
            raise ValueError("k_delay must be at least 1")
    if any(name.startswith("Tolerance-PA-") for name in selected):
        tolerance = non_negative_integer(tolerance, name="tolerance")
    if "Event-PA-F1" in selected:
        _event_scale_code(event_scale, event_base)
    if "PATE" in selected or "PATE-F1" in selected:
        _buffer_points(pate_early_buffer, pate_buffer_splits, pate_include_zero)
        _buffer_points(pate_delayed_buffer, pate_buffer_splits, pate_include_zero)
        if "PATE" in selected or predictions is None:
            pate_threshold_count = validate_threshold_count(
                pate_threshold_count, name="pate_threshold_count"
            )
    internal_selected = selected
    if any(name in _STANDARD_METRICS for name in selected):
        internal_selected = selected + tuple(
            name for name in _STANDARD_COUNT_METRICS if name not in selected
        )
    need_period = needs_vus and sliding_window == "auto"
    _validate_collections(test, train)
    if score_fn is not None or need_period:
        _validate_data_compatibility(test, train)
    rows: list[dict[str, object]] = []
    score_items: list[SeriesItem] | None = [] if save_scores_to is not None else None
    total = len(test)
    for position, series_id in enumerate(test.ids, start=1):
        train_item = train[series_id] if train is not None else None
        row: dict[str, object] = {
            "id": series_id,
            "n_points": len(test[series_id])
            + (len(train_item) if train_item is not None else 0),
            "full_eval": train_item is not None,
        }
        try:
            series = _prepare_evaluation_series(
                test,
                train,
                series_id,
                label_column=label_column,
                need_data=score_fn is not None,
                need_timestamps=score_items is not None,
                need_period=need_period,
            )
            row["n_anomalies"] = int(series.labels.sum())
            row["n_events"] = int(
                np.count_nonzero(
                    (series.labels != 0)
                    & np.concatenate((np.array([True]), series.labels[:-1] == 0))
                )
            )
            started = time.perf_counter()
            score = _resolve_score(score_fn, scores, series)
            prediction = _resolve_prediction(predictions, series)
            score_elapsed = time.perf_counter() - started
            if score_fn is None:
                row["score_seconds"] = math.nan
                row["score_load_seconds"] = score_elapsed
            else:
                row["score_seconds"] = score_elapsed
                row["score_load_seconds"] = math.nan
            if need_period:
                from .period import find_period

                assert series.period_values is not None
                window = find_period(series.period_values)
            elif needs_vus:
                window = int(sliding_window)
            else:
                window = 0
            started = time.perf_counter()
            row.update(
                cast(
                    "dict[str, object]",
                    _evaluate_prepared(
                        series.labels,
                        score,
                        predictions=prediction,
                        selected=internal_selected,
                        sliding_window=window,
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
                    ),
                )
            )
            row["metric_seconds"] = time.perf_counter() - started
            row["sliding_window"] = window if needs_vus else math.nan
            if score_items is not None:
                assert series.timestamps is not None
                score_items.append(
                    SeriesItem(
                        series.id,
                        score.reshape(-1, 1),
                        series.timestamps,
                        pd.DataFrame({label_column: series.labels}),
                        {"full_eval": series.full_eval},
                    )
                )
        except Exception as error:  # noqa: BLE001 - explicit per-series boundary
            if on_error == "raise":
                raise
            row["error"] = f"{type(error).__name__}: {error}"
            fill = 0.0 if on_error == "zero" else math.nan
            row.update({name: fill for name in internal_selected})
            row.setdefault("score_seconds", math.nan)
            row.setdefault("score_load_seconds", math.nan)
            row.setdefault("metric_seconds", math.nan)
            row["sliding_window"] = math.nan
        rows.append(row)
        if progress and (position % 50 == 0 or position == total):
            print(f"evaluated {position}/{total} series", flush=True)

    if save_scores_to is not None:
        assert score_items is not None
        _save_scores(score_items, save_scores_to)
    frame = pd.DataFrame(rows)
    for name in selected:
        if name not in frame:
            frame[name] = pd.Series(dtype=np.float64)
    return EvaluationResult(frame, selected)
