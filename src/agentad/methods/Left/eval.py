"""Full-length scoring for LEFT on one TSB-AD artifact (form A).

Protocol: the artifact-level LEFT scores the concatenated train+test full
length, z-scored with the full series' own statistics, using non-overlapping
windows (stride ``sequence_length``). The fork drops the tail shorter than
one window; the benchmark requires one score per point, so the series is
edge-padded to a whole window count and the padding scores are trimmed.
Window scores are per-point channel means produced by ``Left.score``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ...benchmark import (
    DatasetSplit,
    checkpoints_dir,
    has_score,
    load_split,
    save_score,
    unit_dir,
    write_metrics,
)

from .config import LeftConfig
from .model import LeftLightningModule
from .train import (
    _ZERO_VARIANCE_EPSILON,
    METHOD_NAME,
    build_config,
    save_checkpoint,
    train_split,
)

_SCORE_CHUNK = 256


def _load_checkpoint(path: str | Path, device: str) -> LeftLightningModule:
    """Rebuild the module stored by ``save_checkpoint``."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    module = LeftLightningModule(LeftConfig(**payload["config"]))
    module.load_state_dict(payload["state_dict"])
    return module.eval().to(torch.device(device))


def _load_or_train(
    split: DatasetSplit,
    output_root: str | Path,
    *,
    device: str,
    seed: int,
    hp: Mapping[str, Any] | None,
) -> LeftLightningModule:
    """Return the artifact-level LEFT, training it once when no checkpoint exists."""
    unit = unit_dir(output_root, METHOD_NAME, split)
    checkpoint = checkpoints_dir(unit) / f"{split.name}.pt"
    if checkpoint.is_file():
        return _load_checkpoint(checkpoint, device)
    config = build_config(split.test.n_features, hp)
    module = train_split(split, config, device=device, seed=seed)
    save_checkpoint(module, config, checkpoint)
    return module


def _full_series(split: DatasetSplit, series_id: str) -> np.ndarray:
    """Concatenated ``[T, C]`` train prefix and test segment of one series."""
    test_item = split.test[series_id]
    train_item = split.train[series_id] if split.train is not None else None
    if train_item is None:
        return np.asarray(test_item.data)
    return np.concatenate(
        [np.asarray(train_item.data), np.asarray(test_item.data)], axis=0
    )


def _score_series(
    module: LeftLightningModule, full: np.ndarray, window: int, target: torch.device
) -> np.ndarray:
    """Non-overlapping-window scores over the z-scored full length, [T]."""
    std = full.std(axis=0)
    std = np.where(std == 0, _ZERO_VARIANCE_EPSILON, std)
    z = (full - full.mean(axis=0)) / std
    padding = (-len(z)) % window
    if padding:
        z = np.pad(z, ((0, padding), (0, 0)), mode="edge")
    windows = torch.from_numpy(
        np.ascontiguousarray(z, dtype=np.float32).reshape(-1, window, z.shape[1])
    )
    chunks = [
        module.model.score(windows[start : start + _SCORE_CHUNK].to(target)).cpu()
        for start in range(0, len(windows), _SCORE_CHUNK)
    ]
    scores: np.ndarray = np.concatenate(chunks, axis=0).reshape(-1)
    return scores[: len(full)].astype(np.float64)


def _check_scores(scores: np.ndarray, length: int) -> None:
    if scores.shape != (length,):
        raise ValueError(
            f"score length {scores.shape[0]} does not match series length {length}"
        )
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")


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
    """Score every series of one artifact on its full length and write metrics.

    The LEFT is loaded (or trained) once outside the series loop; each
    series contributes one ``len(train)+len(test)`` score array. Returns
    the per-series metric table written to ``metrics.csv``.
    """
    split = load_split(dataset_dir, artifact, partition=partition)
    unit = unit_dir(output_root, METHOD_NAME, split)
    pending = [
        series_id
        for series_id in split.test.ids
        if not (resume and has_score(unit, series_id))
    ]
    if pending:
        module = _load_or_train(split, output_root, device=device, seed=seed, hp=hp)
        window = module.config.sequence_length
        target = torch.device(device)
        for series_id in pending:
            full = _full_series(split, series_id)
            if len(full) < window:
                raise ValueError(
                    f"series {series_id!r} is shorter than one window ({window})"
                )
            scores = _score_series(module, full, window, target)
            _check_scores(scores, len(full))
            save_score(unit, series_id, scores)
    return write_metrics(split, unit)
