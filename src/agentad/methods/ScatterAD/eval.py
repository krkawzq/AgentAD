"""Dataset-level evaluation for ScatterAD (form A).

Loads or trains the dataset-level model once before the series loop and
scores the concatenated train+test full length of every test series with
the migrated ``ScatterAD.score`` (no thresholding); TSB-AD metrics are
assembled by ``write_metrics``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import ScatterADConfig
from .model import ScatterADLightningModule
from .train import METHOD_NAME, checkpoint_path, train_dataset
from ...benchmark import (
    has_score,
    load_split,
    save_score,
    unit_dir,
    write_metrics,
)


def _load_model(
    payload: Mapping[str, Any], device: str
) -> tuple[ScatterADLightningModule, np.ndarray, np.ndarray]:
    module = ScatterADLightningModule(ScatterADConfig(**payload["config"]))
    module.load_state_dict(payload["state_dict"])
    return (
        module.to(torch.device(device)),
        payload["mean"].numpy(),
        payload["std"].numpy(),
    )


def _check_scores(scores: np.ndarray, length: int) -> None:
    if scores.shape != (length,):
        raise RuntimeError(
            f"ScatterAD scores have shape {scores.shape}, expected ({length},)"
        )
    if not np.isfinite(scores).all():
        raise RuntimeError("ScatterAD produced non-finite scores")


def evaluate(
    dataset_dir: str | Path,
    output_root: str | Path = "outputs/benchmark",
    *,
    artifact: str | None = None,
    partition: str | None = "Eva",
    device: str = "cuda",
    seed: int = 2024,
    hp: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Score every test series of one artifact and return its metric table.

    The dataset model comes from the unit checkpoint, or from an in-memory
    :func:`train_dataset` call when the checkpoint is absent. The same
    train-fitted scaler statistics normalize training and full-length
    scoring. ``resume=True`` skips series whose scores are already stored.
    """
    split = load_split(dataset_dir, artifact, partition=partition)
    if split.train is None:
        raise ValueError("ScatterAD is semisupervised and requires a train split")
    unit = unit_dir(output_root, METHOD_NAME, split)
    checkpoint = checkpoint_path(unit, split.name)
    if checkpoint.is_file():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    else:
        payload = train_dataset(split, device=device, seed=seed, hp=hp)
    module, mean, std = _load_model(payload, device)
    target = torch.device(device)
    for series_id in split.test.ids:
        if resume and has_score(unit, series_id):
            continue
        train_item = split.train.get(series_id)
        test_item = split.test[series_id]
        full = (
            np.concatenate([train_item.data, test_item.data])
            if train_item is not None
            else test_item.data
        )
        normalized = ((full - mean) / std).astype(np.float32)
        series = torch.from_numpy(normalized)[None].to(target)
        scores = module.model.score(series)[0].cpu().numpy()
        _check_scores(scores, len(full))
        save_score(unit, series_id, scores)
    return write_metrics(split, unit)
