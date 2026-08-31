"""Metric-table assembly from stored per-series scores."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import pandas as pd

from ..evaluation import (
    DEFAULT_METRICS,
    MetricName,
    evaluate_collection,
)
from .datasets import DatasetSplit
from .layout import has_score, load_score, metrics_path

__all__ = ["write_metrics"]


def write_metrics(
    split: DatasetSplit,
    unit: str | Path,
    results_dir: str | Path,
    *,
    metrics: Iterable[MetricName] = DEFAULT_METRICS,
    label_column: str = "label",
    sliding_window: int | Literal["auto"] = "auto",
) -> pd.DataFrame:
    """Evaluate every stored score of one dataset unit and write metrics.csv.

    Scores are read from ``unit`` (under the outputs root); the metric table
    is written into ``results_dir`` (under the results root). Series without
    a stored score stay in the table with an ``error`` row so partial runs
    remain auditable. Returns the per-series frame.
    """
    scores = {
        series_id: load_score(unit, series_id)
        for series_id in split.test.ids
        if has_score(unit, series_id)
    }
    if not scores:
        raise ValueError(f"no stored scores under {Path(unit)}")
    result = evaluate_collection(
        split.test,
        split.train,
        scores=scores,
        metrics=metrics,
        label_column=label_column,
        sliding_window=sliding_window,
    )
    frame = result.per_series
    path = metrics_path(unit)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame
