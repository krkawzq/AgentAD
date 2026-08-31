"""MMPAD evaluation (形态 C, pure algorithm, full-length self-join scoring).

Protocol source: the official TSB-AD unsupervised usage — ``run_MMPAD``
receives the concatenated train+test series with the HP of
``Optimal_Multi/Uni_algo_HP_dict`` (``forks/TSB-AD/TSB_AD/HP_list.py``);
no reference library is built.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch

from ...benchmark import (
    has_score,
    load_split,
    save_score,
    unit_dir,
    write_metrics,
)
from .algorithm import score
from .config import MMPADConfig

METHOD_NAME = "MMPAD"

# The channel-count presets already carry the TSB-AD optimal HP; ``hp``
# only overrides MMPADConfig fields (e.g. subsequence_length, neighbors).
DEFAULT_HP: Mapping[str, Any] = {}


def _build_config(features: int, hp: Mapping[str, Any]) -> MMPADConfig:
    """Pick the TSB-AD optimal preset by channel count, then apply ``hp``.

    ``subsequence_length=None`` keeps the built-in autocorrelation window
    inference; the remaining keys map to ``MMPADConfig`` fields through
    ``dataclasses.replace``.
    """
    factory = (
        MMPADConfig.original_multivariate_hp0
        if features > 1
        else MMPADConfig.original_univariate_hp0
    )
    config = factory(subsequence_length=hp.get("subsequence_length"))
    overrides = {
        key: value for key, value in hp.items() if key != "subsequence_length"
    }
    return replace(config, **overrides) if overrides else config


def _check_scores(scores: np.ndarray, length: int) -> None:
    """Enforce the one-score-per-point finite contract before persisting."""
    if scores.shape != (length,):
        raise ValueError(f"expected {length} scores, got shape {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("MMPAD produced non-finite scores")


def _score_series(full: np.ndarray, hp: Mapping[str, Any]) -> np.ndarray:
    """Self-join score one full-length ``[T, C]`` series."""
    config = _build_config(full.shape[1], hp)
    scores = score(torch.from_numpy(full)[None], config)[0].numpy()
    _check_scores(scores, len(full))
    return scores


def evaluate(
    dataset_dir: str | Path,
    output_root: str | Path = "benchmarks/MMPAD",
    *,
    artifact: str | None = None,
    partition: str | None = "Eva",
    seed: int = 2024,
    hp: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Self-join score every series of one artifact pair on its full length.

    Scores cover ``len(train) + len(test)`` points, mirroring the official
    TSB-AD unsupervised ``run_MMPAD`` call on the whole series; stored
    scores are skipped when ``resume``. Returns the per-series metric
    table, also written to ``metrics.csv`` under the MMPAD unit directory.
    """
    split = load_split(dataset_dir, artifact, partition=partition)
    unit = unit_dir(output_root, METHOD_NAME, split)
    merged_hp = {**DEFAULT_HP, **(hp or {})}
    for series_id in split.test.ids:
        if resume and has_score(unit, series_id):
            continue
        L.seed_everything(seed)
        train_item = split.train[series_id] if split.train is not None else None
        test_item = split.test[series_id]
        full = (
            np.concatenate([train_item.data, test_item.data])
            if train_item is not None
            else np.asarray(test_item.data)
        )
        save_score(unit, series_id, _score_series(full, merged_hp))
    return write_metrics(split, unit)
