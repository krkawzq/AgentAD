"""Per-series full-length scoring for CARLA on one TSB-AD artifact (form A).

Protocol: the series' trained CARLA scores the concatenated train+test full
length z-scored with the train segment's per-channel statistics — the same
statistics the training windows used. ``calibrate_normal_clusters`` fixes the
majority normal cluster on the train windows and ``CARLA.score`` averages
window scores over the points they cover.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch

from ...benchmark import (
    checkpoints_dir,
    has_score,
    load_split,
    save_score,
    unit_dir,
    write_metrics,
)

from .config import CARLAConfig
from .model import CARLA
from .train import (
    METHOD_NAME,
    _min_train_length,
    build_config,
    fit_normalization,
    train_series,
)


def _load_checkpoint(directory: str | Path, device: torch.device) -> CARLA:
    """Rebuild the calibrated model stored by ``train``."""
    payload = torch.load(
        Path(directory) / "classification.pt", map_location="cpu", weights_only=False
    )
    model = CARLA(CARLAConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    model.normal_clusters.copy_(payload["normal_clusters"])
    return model.to(device)


def _check_scores(scores: np.ndarray, length: int) -> None:
    """One finite score per full-length point, as the benchmark requires."""
    if scores.shape != (length,):
        raise ValueError(
            f"score length {scores.shape[0]} does not match series length {length}"
        )
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")


def evaluate(
    dataset_dir: str | Path,
    output_root: str | Path = "benchmarks",
    *,
    artifact: str | None = None,
    partition: str | None = "Eva",
    device: str = "cuda",
    seed: int = 2024,
    hp: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Score every series of one artifact on its full length and write metrics.

    Per series: load the trained checkpoint (or train in memory when absent),
    score the z-scored concatenated train+test length with the train
    statistics, and store one score per point. Series too short for one full
    window are reported and left unscored. Returns the per-series metric
    table written to ``metrics.csv``.
    """
    split = load_split(dataset_dir, artifact, partition=partition)
    if split.train is None:
        raise ValueError(
            f"CARLA is semisupervised; {split.unit_name} has no train split"
        )
    unit = unit_dir(output_root, METHOD_NAME, split)
    target = torch.device(device)
    for series_id in split.test.ids:
        if resume and has_score(unit, series_id):
            continue
        L.seed_everything(seed)
        train_data = np.asarray(split.train[series_id].data)
        test_data = np.asarray(split.test[series_id].data)
        full = np.concatenate([train_data, test_data], axis=0)
        mean, std = fit_normalization(train_data)
        checkpoint = checkpoints_dir(unit) / series_id
        if (checkpoint / "classification.pt").is_file():
            model = _load_checkpoint(checkpoint, target)
            config = model.config
        elif checkpoint.is_dir():
            # An interrupted train leaves the directory without its final
            # stage; fail loudly instead of silently retraining over it.
            raise FileNotFoundError(
                f"incomplete CARLA checkpoint {checkpoint}: "
                "classification.pt is missing"
            )
        else:
            config = build_config(train_data.shape[1], hp)
            model = None
        if (
            train_data.shape[0] < _min_train_length(config)
            or full.shape[0] < config.window_length
        ):
            print(
                f"skipping {series_id!r}: too short for window {config.window_length}"
            )
            continue
        if model is None:
            model = train_series(
                config,
                torch.from_numpy(train_data),
                device=target,
            )
        scores = (
            model.score(
                torch.from_numpy((full - mean) / std).float()[None].to(target),
                stride=config.window_stride,
            )[0]
            .cpu()
            .numpy()
        )
        _check_scores(scores, full.shape[0])
        save_score(unit, series_id, scores)
    return write_metrics(split, unit)
