"""Histogram-density outlier detector.

Builds a fixed-bin density histogram per windowed feature column and
scores each window by the negative sum of ``log2(density + alpha)`` over
features: dense regions contribute large (near-zero) log densities, while
any feature landing in an empty or thin bin dominates the sum. Window
scores center-align to points with boundary replication.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from ...benchmark import (
    has_score,
    load_split,
    save_score,
    unit_dir,
    write_metrics,
)
from ...evaluation.period import find_period
from .._utils import validate_series
from ._common import infer_period, pad_window_scores, windowed_features


@dataclass(frozen=True, slots=True)
class HistogramConfig:
    window: int | None = 1
    bins: int = 10
    alpha: float = 0.1
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.window is not None and self.window < 1:
            raise ValueError("window must be positive when provided")
        if self.bins < 2:
            raise ValueError("bins must be at least two")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")


def _histograms(features: Tensor, bins: int, alpha: float) -> tuple[Tensor, Tensor]:
    """Per-column log2 density scores and per-sample bin assignments."""
    count, width = features.shape
    minimum = features.min(dim=0).values
    maximum = features.max(dim=0).values
    # A zero-range column widens to (min - 0.5, max + 0.5) so every bin
    # stays non-degenerate.
    low = torch.where(minimum == maximum, minimum - 0.5, minimum)
    high = torch.where(minimum == maximum, maximum + 0.5, maximum)
    edges = (
        torch.linspace(
            0.0, 1.0, bins + 1, device=features.device, dtype=features.dtype
        )[:, None]
        * (high - low)[None, :]
        + low[None, :]
    )  # [bins+1, width]
    widths = (edges[1:] - edges[:-1]).clamp_min(torch.finfo(features.dtype).tiny)

    # Values on the lower edge fall into bin 0 and on the upper edge into
    # the last bin; interior values take the bin below the edge they hit.
    indices = torch.searchsorted(
        edges.T.contiguous(), features.T.contiguous()
    ).T  # [count, width]
    assignment = (indices - 1).clamp(0, bins - 1)
    counts = (
        (
            assignment[None, :, :]
            == torch.arange(bins, device=features.device)[:, None, None]
        )
        .sum(dim=1)
        .to(features.dtype)
    )  # [bins, width]
    density = counts / (count * widths)
    return torch.log2(density + alpha), assignment


def _score_item(
    features: Tensor, config: HistogramConfig, window: int, length: int
) -> Tensor:
    bin_scores, assignment = _histograms(features, config.bins, config.alpha)
    window_scores = -torch.gather(bin_scores, 0, assignment).sum(dim=1)
    if features.shape[0] == length:
        return window_scores
    return pad_window_scores(window_scores[None], window, length)[0]


@torch.inference_mode()
def score(series: Tensor, config: HistogramConfig) -> Tensor:
    window = config.window or infer_period(series[0, :, 0])
    series = validate_series(series, min_length=window)
    features = windowed_features(series, window, config.normalize)
    return torch.stack(
        [_score_item(item, config, window, series.shape[1]) for item in features]
    )


# ---------------------------------------------------------------------------
# TSB-AD baseline evaluation (spec: tmp/method_train_eval_spec.md §4 shape C)
# ---------------------------------------------------------------------------


METHOD_NAME = "Histogram"

# TSB-AD-M plain HBOS: Optimal_Multi_algo_HP_dict["HBOS"] = {"n_bins": 30,
# "tol": 0.5} (forks/TSB-AD/TSB_AD/HP_list.py) maps n_bins -> bins; the
# wrapper default run_HBOS(slidingWindow=1) keeps the per-point histogram.
# pyod's tol (tolerance for samples falling outside the bins) has no
# counterpart in this implementation, which smooths densities with alpha
# instead; alpha/normalize keep the HBOS constructor defaults.
DEFAULT_HP: Mapping[str, Any] = {
    "window": 1,
    "bins": 30,
    "alpha": 0.1,
    "normalize": True,
}

# TSB-AD-U subsequence variant: Optimal_Uni_algo_HP_dict["Sub_HBOS"] =
# {"periodicity": 1, "n_bins": 10}; the window is data-derived (rank-1 period
# of the full series), so it is injected at runtime instead of living here.
_UNI_DEFAULT_HP: Mapping[str, Any] = {"bins": 10, "alpha": 0.1, "normalize": True}


def _check_scores(scores: np.ndarray, length: int) -> None:
    if scores.shape != (length,):
        raise ValueError(f"score length {scores.shape[0]} != series length {length}")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")


def _resolve_config(full: np.ndarray, hp: Mapping[str, Any] | None) -> HistogramConfig:
    values: dict[str, Any] = {
        **(_UNI_DEFAULT_HP if full.shape[1] == 1 else DEFAULT_HP),
        **(hp or {}),
    }
    # Sub_HBOS protocol: periodicity=1 selects the rank-1 period of the
    # concatenated full series; an explicit hp "window" overrides it.
    if "window" not in values:
        values["window"] = find_period(full[:, 0])
    return HistogramConfig(**values)


def evaluate(
    dataset_dir: str | Path,
    output_root: str | Path = "outputs/benchmark",
    *,
    artifact: str | None = None,
    partition: str | None = "Eva",
    seed: int = 2024,
    hp: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Score every series of one artifact on its train+test full length.

    Univariate series follow the Sub_HBOS protocol (rank-1 period window,
    10 bins), multivariate series the plain HBOS protocol (per-point
    window, 30 bins); ``hp`` overrides the selected defaults. Scores are
    stored under ``output_root`` (already-scored series are skipped when
    ``resume``) and the per-series metric table is returned.
    """
    split = load_split(dataset_dir, artifact, partition=partition)
    unit = unit_dir(output_root, METHOD_NAME, split)
    for series_id in split.test.ids:
        if resume and has_score(unit, series_id):
            continue
        L.seed_everything(seed)
        train_item = split.train[series_id] if split.train is not None else None
        test_item = split.test[series_id]
        full = (
            np.concatenate([train_item.data, test_item.data])
            if train_item is not None
            else test_item.data
        )
        config = _resolve_config(full, hp)
        scores = score(torch.from_numpy(full)[None], config)[0].numpy()
        _check_scores(scores, len(full))
        save_score(unit, series_id, scores)
    return write_metrics(split, unit)
