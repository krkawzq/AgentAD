"""DADA form-B evaluation: released-checkpoint zero-shot scoring per TSB-AD.

Protocol mirrors ``forks/DADA`` zero-shot inference: the concatenated
train+test series is standardized with a ``StandardScaler`` fitted on the
train prefix (``data_provider.TrainSegLoader``), tiled into non-overlapping
windows of ``sequence_length`` (win_size=step=100), and each window is scored
by masked-copy variance with the published Hugging Face weights.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from agentad.benchmark import (
    has_score,
    load_split,
    save_score,
    unit_dir,
    write_metrics,
)

from .config import DADAConfig
from .model import DADA

METHOD_NAME = "DADA"
CHECKPOINT_PATH = Path("pretrained/dada/pytorch_model.bin")

# Original protocol: all scripts but three pass `--norm 0` (also the run.py
# default), keeping the model's instance normalization off, and
# `cal_anomaly_score` without an `anomaly_criterion` returns pure
# masked-copy variance.
DEFAULT_HP: Mapping[str, Any] = {"normalization": False, "variance_weight": 1.0}


def _standardize(full: np.ndarray, train_length: int) -> np.ndarray:
    """Fit ``StandardScaler`` on the train prefix and transform the full series.

    ``TrainSegLoader`` fits the scaler on the train segment only and shares
    the transform with test; without a train segment the full series is the
    only available normal reference. Returns ``[T, C]`` float64.
    """
    scaler = StandardScaler()
    scaler.fit(full[:train_length] if train_length else full)
    return scaler.transform(full)


def _score_window(
    model: DADA,
    window: np.ndarray,
    config: DADAConfig,
    device: torch.device,
) -> np.ndarray:
    """Score one ``[window, C]`` segment as ``[1, window, C]``; returns ``[window]``."""
    batch = torch.from_numpy(window).to(torch.float32)[None].to(device)
    score = model.score(
        batch,
        normalize=config.normalization,
        variance_weight=config.variance_weight,
    )
    return score[0].cpu().numpy()


def _score_series(
    model: DADA,
    series: np.ndarray,
    config: DADAConfig,
    device: torch.device,
) -> np.ndarray:
    """Score the full series with the original non-overlapping window tiling.

    ``TrainSegLoader`` yields ``(length - win_size) // step + 1`` windows with
    win_size=step=100, so the trailing ``length % win_size`` points fall
    outside every window and the original never scores them — its evaluator
    truncates the ground truth the same way (drop, never pad). The TSB-AD
    protocol requires one score per full-length point, so those trailing
    slots reuse the last scored value. Series shorter than one window are
    scored through a single edge-padded window, keeping the real prefix
    scores. Returns ``[T]`` for a ``[T, C]`` input.
    """
    window = config.sequence_length
    total = series.shape[0]
    pieces = [
        _score_window(model, series[start : start + window], config, device)
        for start in range(0, total // window * window, window)
    ]
    if pieces:
        scores: np.ndarray = np.concatenate(pieces)
    else:
        padded = np.pad(series, ((0, window - total), (0, 0)), mode="edge")
        scores = _score_window(model, padded, config, device)[:total]
    if scores.shape[0] < total:
        scores = np.concatenate([scores, np.full(total - scores.shape[0], scores[-1])])
    return scores


def evaluate(
    dataset_dir: str | Path,
    output_root: str | Path = "benchmarks/DADA",
    *,
    artifact: str | None = None,
    partition: str | None = "Eva",
    device: str = "cuda",
    seed: int = 2024,
    hp: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Score one dataset unit with the released DADA weights (no training).

    Skips series with stored scores when ``resume``; returns the per-series
    metric frame written to ``<unit>/metrics.csv``.
    """
    split = load_split(dataset_dir, artifact, partition=partition)
    unit = unit_dir(output_root, METHOD_NAME, split)
    config = replace(DADAConfig.original_pretrained(), **{**DEFAULT_HP, **(hp or {})})
    target = torch.device(device)
    model = DADA.from_official_checkpoint(CHECKPOINT_PATH).to(target)
    for series_id in split.test.ids:
        if resume and has_score(unit, series_id):
            continue
        # Per-series reseed keeps scores reproducible when resume skips
        # earlier series, since masked-copy scoring draws random masks.
        L.seed_everything(seed)
        test_item = split.test[series_id]
        train_item = split.train[series_id] if split.train is not None else None
        test_data = np.asarray(test_item.data, dtype=np.float64)
        if train_item is None:
            full = test_data
            train_length = 0
        else:
            train_data = np.asarray(train_item.data, dtype=np.float64)
            full = np.concatenate([train_data, test_data])
            train_length = train_data.shape[0]
        standardized = _standardize(full, train_length)
        scores = _score_series(model, standardized, config, target)
        if scores.shape != (len(full),) or not np.isfinite(scores).all():
            raise RuntimeError(
                f"DADA produced invalid scores for {series_id}: shape={scores.shape}"
            )
        save_score(unit, series_id, scores)
    return write_metrics(split, unit)


if __name__ == "__main__":
    evaluate(
        dataset_dir="data/processed/tsb-ad/TSB-AD-M/CATSv2",
        output_root="benchmarks/DADA",
        device="cuda",
    )
