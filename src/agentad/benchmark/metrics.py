"""Metric-table assembly from stored per-series scores."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ..evaluation import (
    DEFAULT_METRICS,
    MetricName,
    evaluate_collection,
)
from .datasets import DatasetSplit
from .layout import (
    COMPLETION_SCHEMA,
    _atomic_write,
    _file_sha256,
    atomic_write_json,
    completion_path,
    has_score,
    load_run_manifest,
    load_score,
    metrics_path,
)

__all__ = ["write_metrics"]


def write_metrics(
    split: DatasetSplit,
    unit: str | Path,
    results_dir: str | Path,
    *,
    metrics: Iterable[MetricName] = DEFAULT_METRICS,
    label_column: str = "label",
    sliding_window: int | Literal["auto"] | Mapping[str, int] = "auto",
    score_scope: Literal["full", "test"] = "full",
) -> pd.DataFrame:
    """Evaluate every stored score of one dataset unit and write metrics.csv.

    Scores are read from the active fingerprint under ``unit``. Every requested
    series must have a valid score and every selected metric must be finite;
    partial runs fail without publishing a completion marker. ``metrics.csv``
    and ``completion.json`` are replaced atomically in that order.
    """
    if score_scope not in ("full", "test"):
        raise ValueError(f"invalid score scope: {score_scope!r}")
    selected = tuple(metrics)
    missing = [
        series_id for series_id in split.test.ids if not has_score(unit, series_id)
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} series have no score for the active run, e.g. {missing[:3]}"
        )
    scores = {series_id: load_score(unit, series_id) for series_id in split.test.ids}
    result = evaluate_collection(
        split.test,
        split.train if score_scope == "full" else None,
        scores=scores,
        metrics=selected,
        label_column=label_column,
        sliding_window=sliding_window,
        on_error="raise",
    )
    frame = result.per_series
    if frame["id"].tolist() != list(split.test.ids):
        raise RuntimeError("metric rows do not match the requested series order")
    values = frame[list(selected)].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("selected benchmark metrics contain non-finite values")
    path = metrics_path(results_dir)
    encoded = frame.to_csv(index=False).encode("utf-8")
    _atomic_write(path, lambda handle: handle.write(encoded))
    manifest = load_run_manifest(unit)
    identifiers = "\0".join(frame["id"].astype(str)).encode("utf-8")
    atomic_write_json(
        completion_path(results_dir),
        {
            "schema": COMPLETION_SCHEMA,
            "run_fingerprint": manifest["fingerprint"],
            "metrics_sha256": _file_sha256(path),
            "rows": len(frame),
            "ids_sha256": hashlib.sha256(identifiers).hexdigest(),
            "metric_columns": list(selected),
        },
    )
    return frame
