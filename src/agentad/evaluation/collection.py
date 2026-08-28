"""Collection reconstruction, detector execution and metric aggregation."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .metrics import (
    DEFAULT_METRICS,
    MetricName,
    _evaluate_prepared,
    _metric_selection,
)
from ._validation import binary_array, score_array
from ..series import SeriesData, SeriesItem, write as write_series

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
    """One full evaluation sequence reconstructed from stored splits."""

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
    """Per-series results with macro aggregation helpers."""

    per_series: pd.DataFrame
    metric_columns: tuple[MetricName, ...]

    def summary(self, stat: Literal["mean", "median"] = "mean") -> pd.Series:
        """Macro-average or median across series, skipping undefined metrics."""
        if stat not in ("mean", "median"):
            raise ValueError(f"stat must be 'mean' or 'median', got {stat!r}")
        if self.per_series.empty:
            return pd.Series({name: math.nan for name in self.metric_columns})
        return self.per_series[list(self.metric_columns)].agg(stat, axis=0)

    @property
    def failures(self) -> pd.DataFrame:
        """Rows that failed scoring or validation."""
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
        return binary_array(labels[label_column].to_numpy(), name="labels")
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
    for series_id in test.ids:
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
                else np.asarray(test_item.data)
            )
        timestamps = None
        if need_timestamps:
            timestamps = (
                np.concatenate((train_item.timestamps, test_item.timestamps))
                if train_item is not None
                else np.asarray(test_item.timestamps, dtype=np.int64)
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
        yield _EvaluationSeries(
            series_id,
            labels,
            train_item is not None,
            data,
            timestamps,
            period_values,
            (0, len(train_item)) if train_item is not None else None,
        )


def reconstruct_series(
    test: SeriesData,
    train: SeriesData | None = None,
    *,
    label_column: str = "label",
) -> Iterator[ReconstructedSeries]:
    """Yield test series, optionally prefixed by their matching train split.

    Split identity defines order: training data is always the prefix even when
    individual split timestamps restart from zero. Every test id must exist in
    ``train`` when a train collection is supplied.
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
    metrics: Iterable[MetricName] = DEFAULT_METRICS,
    label_column: str = "label",
    sliding_window: int | Literal["auto"] = "auto",
    threshold_count: int = 100,
    vus_threshold_count: int = 250,
    on_error: Literal["nan", "zero", "raise"] = "nan",
    save_scores_to: str | Path | None = None,
    progress: bool = False,
) -> EvaluationResult:
    """Score and evaluate every series in a collection.

    Exactly one of ``score_fn`` and ``scores`` is required. ``on_error`` is a
    per-series policy; ``raise`` preserves the original exception while the
    other modes retain an error column for auditability.
    """
    if (score_fn is None) == (scores is None):
        raise ValueError("pass exactly one of score_fn and scores")
    if on_error not in ("nan", "zero", "raise"):
        raise ValueError(f"invalid on_error policy: {on_error!r}")
    if sliding_window != "auto" and (
        isinstance(sliding_window, bool)
        or not isinstance(sliding_window, (int, np.integer))
        or sliding_window < 0
    ):
        raise ValueError("sliding_window must be 'auto' or a non-negative integer")

    selected = _metric_selection(metrics)
    needs_vus = "VUS-PR" in selected or "VUS-ROC" in selected
    need_period = needs_vus and sliding_window == "auto"
    rows: list[dict[str, object]] = []
    score_items: list[SeriesItem] | None = [] if save_scores_to is not None else None
    total = len(test)
    reconstructed = _evaluation_series(
        test,
        train,
        label_column=label_column,
        need_data=score_fn is not None,
        need_timestamps=score_items is not None,
        need_period=need_period,
    )
    for position, series in enumerate(reconstructed, start=1):
        row: dict[str, object] = {
            "id": series.id,
            "n_points": len(series.labels),
            "full_eval": series.full_eval,
        }
        try:
            started = time.perf_counter()
            score = _resolve_score(score_fn, scores, series)
            row["score_seconds"] = time.perf_counter() - started
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
                _evaluate_prepared(
                    series.labels,
                    score,
                    predictions=None,
                    selected=selected,
                    sliding_window=window,
                    threshold_count=threshold_count,
                    vus_threshold_count=vus_threshold_count,
                    range_alpha=0.2,
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
            row.update({name: fill for name in selected})
            row["score_seconds"] = math.nan
            row["metric_seconds"] = math.nan
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
