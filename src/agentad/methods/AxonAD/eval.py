"""Form-A per-series scoring for AxonAD on the concatenated train+test full length."""

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
from .config import AxonADConfig
from .model import AxonADLightningModule
from .train import (
    METHOD_NAME,
    _segment_tensor,
    _split_train_validation,
    build_config,
    train_series,
)


def _load_or_train(
    unit: str | Path,
    series_id: str,
    train_data: np.ndarray,
    config: AxonADConfig,
    *,
    device: str,
) -> AxonADLightningModule:
    """Load a stored checkpoint, or train in memory when it is missing."""
    checkpoint = checkpoints_dir(unit) / f"{series_id}.pt"
    if not checkpoint.is_file():
        return train_series(train_data, config, device=device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    stored_config = AxonADConfig(**payload["config"])
    module = AxonADLightningModule(stored_config).to(device)
    module.load_state_dict(payload["state_dict"])
    # The calibrated buffers are not persisted; rebuild them from the
    # training segment with the checkpoint's own protocol.
    train_segment = _split_train_validation(train_data, stored_config)[0]
    module.model.calibrate(
        _segment_tensor(train_segment).to(module.device),
        batch_size=stored_config.batch_size,
    )
    return module


def _check_scores(scores: np.ndarray, length: int, series_id: str) -> None:
    if scores.shape != (length,):
        raise ValueError(
            f"series {series_id!r}: expected {length} scores, got shape {scores.shape}"
        )
    if not np.isfinite(scores).all():
        raise ValueError(f"series {series_id!r}: scores contain non-finite values")


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
    """Score every series of one dataset artifact pair and return the metric table.

    Series with a stored score are skipped when ``resume``; missing
    checkpoints are trained on the fly in memory.
    """
    split = load_split(dataset_dir, artifact, partition=partition)
    if split.train is None:
        raise FileNotFoundError(
            f"no train archive for {split.unit_name}; AxonAD is semi-supervised"
        )
    unit = unit_dir(output_root, METHOD_NAME, split)
    config = build_config(split.test.n_features, hp)
    for series_id in split.test.ids:
        if resume and has_score(unit, series_id):
            continue
        L.seed_everything(seed)
        if series_id not in split.train:
            raise KeyError(f"series {series_id!r} missing from the train archive")
        train_item = split.train[series_id]
        test_item = split.test[series_id]
        full = np.concatenate([train_item.data, test_item.data])
        if len(full) < config.window_length:
            raise ValueError(
                f"series {series_id!r}: full length {len(full)} is shorter than "
                f"the window {config.window_length}"
            )
        module = _load_or_train(unit, series_id, train_item.data, config, device=device)
        scores = (
            module.model.score(
                _segment_tensor(full).to(module.device),
                batch_size=config.batch_size,
            )[0]
            .cpu()
            .numpy()
        )
        _check_scores(scores, len(full), series_id)
        save_score(unit, series_id, scores)
    return write_metrics(split, unit)
