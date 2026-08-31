"""TSB-AD Form B for Time-RCD: zero-shot evaluation with the official
uni/multi checkpoints, scoring the concatenated train+test full length
(TSB-AD Optimal HP row ``Time_RCD`` in the original ``HP_list.py``).
The MAFT fine-tuning variant is not implemented in this script.
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
    DatasetSplit,
    has_score,
    load_split,
    save_score,
    unit_dir,
    write_metrics,
)

from .model import TimeRCD, load_official_checkpoint

METHOD_NAME = "Time-RCD"

# Published checkpoints vendored under the repository root; the uni/multi
# choice follows the input channel count like the original wrapper.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_UNI_CHECKPOINT = (
    _REPO_ROOT / "pretrained" / "time_rcd" / "uni" / "pretrain_checkpoint_best_uni.pth"
)
_MULTI_CHECKPOINT = (
    _REPO_ROOT
    / "pretrained"
    / "time_rcd"
    / "multi"
    / "pretrain_checkpoint_best_multi.pth"
)

# TSB-AD Optimal row "Time_RCD" in the original HP_list.py.
DEFAULT_HP: Mapping[str, Any] = {"win_size": 15_000, "batch_size": 64}


def _checkpoint_path(channels: int) -> Path:
    """Checkpoint matching the series channel count (uni only for C == 1)."""
    return _UNI_CHECKPOINT if channels == 1 else _MULTI_CHECKPOINT


def _series_channels(split: DatasetSplit) -> int:
    """Channel count shared by every series of one artifact pair."""
    widths = {split.test[series_id].n_features for series_id in split.test.ids}
    if split.train is not None:
        widths |= {split.train[series_id].n_features for series_id in split.train.ids}
    if len(widths) != 1:
        raise ValueError(f"artifact mixes channel counts: {sorted(widths)}")
    return widths.pop()


def _load_model(
    channels: int, settings: Mapping[str, Any], device: torch.device
) -> TimeRCD:
    """Build the official model for ``channels`` and apply the scoring HP."""
    model = load_official_checkpoint(
        _checkpoint_path(channels),
        variant="uni" if channels == 1 else "multi",
        input_features=channels,
    )
    # The TSB-AD wrapper feeds non-overlapping ``win_size`` windows; the
    # vendored config carries that length as ``inference_window``. Multi-channel
    # scoring keeps the factory batch size (1) instead of the TSB-AD HP value.
    batch_size = (
        int(settings["batch_size"]) if channels == 1 else model.config.batch_size
    )
    model.config = replace(
        model.config,
        inference_window=int(settings["win_size"]),
        batch_size=batch_size,
    )
    return model.to(device=device)


def _score_full(model: TimeRCD, full: np.ndarray, device: torch.device) -> np.ndarray:
    """Zero-shot scores for one concatenated ``[T, C]`` full length."""
    # score() already applies the original protocol (per-channel z-scoring,
    # non-overlapping windows, softmax anomaly probability); batch_size only
    # caps how many windows share one forward pass.
    series = torch.from_numpy(np.asarray(full, dtype=np.float32))[None].to(device)
    return model.score(series, batch_size=model.config.batch_size)[0].cpu().numpy()


def _check_scores(scores: np.ndarray, length: int) -> None:
    if scores.shape != (length,):
        raise ValueError(f"expected {length} scores, got shape {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")


def evaluate(
    dataset_dir: str | Path,
    output_root: str | Path = "outputs/benchmark",
    *,
    artifact: str | None = None,
    results_root: str | Path = "results/benchmark",
    partition: str | None = "Eva",
    device: str = "cuda",
    seed: int = 2024,
    hp: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Score every series of one artifact with the official Time-RCD checkpoint.

    The uni/multi checkpoint is selected once per artifact from the shared
    channel count; each series is scored zero-shot over its concatenated
    train+test full length. Returns the per-series metric table.
    """
    settings: Mapping[str, Any] = {**DEFAULT_HP, **(hp or {})}
    split = load_split(dataset_dir, artifact, partition=partition)
    unit = unit_dir(output_root, METHOD_NAME, split)
    results = unit_dir(results_root, METHOD_NAME, split)
    target = torch.device(device)
    L.seed_everything(seed)
    model = _load_model(_series_channels(split), settings, target)
    for series_id in split.test.ids:
        if resume and has_score(unit, series_id):
            continue
        train_item = split.train[series_id] if split.train is not None else None
        test_item = split.test[series_id]
        full = (
            np.concatenate([train_item.data, test_item.data])
            if train_item is not None
            else test_item.data
        )
        scores = _score_full(model, full, target)
        _check_scores(scores, len(full))
        save_score(unit, series_id, scores)
    return write_metrics(split, unit, results)
