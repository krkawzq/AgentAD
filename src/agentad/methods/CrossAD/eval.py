"""Form-A evaluation for CrossAD under the authors' TSB-AD protocol.

Each series' concatenated train+test data is scored at full length with
non-overlapping windows and the per-series checkpoint, mirroring
``forks/CrossAD/run_TSBAD.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as F

from agentad.benchmark import (
    DatasetSplit,
    checkpoints_dir,
    has_score,
    load_split,
    save_score,
    unit_dir,
    write_metrics,
)
from agentad.series import SeriesItem

from .config import CrossADConfig
from .model import CrossADLightningModule
from .train import METHOD_NAME, select_config, train_series, zscore

_SCORE_BATCH = 256


def _load_or_train(
    unit: Path,
    series_id: str,
    train_data: np.ndarray,
    *,
    source: str,
    hp: Mapping[str, Any] | None,
    device: str,
    resume: bool,
) -> CrossADLightningModule:
    """Load the stored checkpoint or train on the fly; training here stays in
    memory and never writes checkpoints."""
    path = checkpoints_dir(unit) / f"{series_id}.pt"
    if resume and path.is_file():
        payload = torch.load(path, map_location="cpu")
        module = CrossADLightningModule(CrossADConfig(**payload["config"]))
        module.load_state_dict(payload["state_dict"])
        return module
    config = select_config(series_id, source, hp)
    return train_series(train_data, config, device=device)


def _score_windows(
    module: CrossADLightningModule, windows: Tensor, device: str
) -> np.ndarray:
    outputs = []
    for start in range(0, len(windows), _SCORE_BATCH):
        chunk = windows[start : start + _SCORE_BATCH].to(device)
        outputs.append(module.model.score(chunk).cpu().numpy())
    return np.concatenate(outputs).reshape(-1)


def score_series(
    module: CrossADLightningModule,
    full: np.ndarray,
    config: CrossADConfig,
    *,
    device: str = "cuda",
) -> np.ndarray:
    """Full-length anomaly scores for the concatenated train+test series.

    The full series is z-scored independently and scored on non-overlapping
    windows; the ragged tail is scored through an edge-padded final window
    because the benchmark requires one score per point, while the original
    drops the tail and truncates the labels. Scores are min-maxed to [0, 1]
    as ``run_TSBAD.py`` does before saving.
    """
    module = module.to(device).eval()
    window = config.sequence_length
    values = torch.from_numpy(zscore(np.asarray(full, dtype=np.float64))).float()
    covered = (len(values) // window) * window
    parts: list[np.ndarray] = []
    if covered:
        tiled = values[:covered].reshape(-1, window, values.shape[-1])
        parts.append(_score_windows(module, tiled, device))
    tail = len(values) - covered
    if tail:
        padded = F.pad(
            values[covered:].unsqueeze(0), (0, 0, window - tail, 0), mode="replicate"
        )
        scores = module.model.score(padded.to(device))[0, -tail:]
        parts.append(scores.float().cpu().numpy())
    flat = np.concatenate(parts) if parts else np.empty(0)
    return _min_max(flat)


def _min_max(scores: np.ndarray) -> np.ndarray:
    """Min-max to [0, 1]; constant scores map to zero."""
    low = float(scores.min())
    span = float(scores.max()) - low
    if span <= 0:
        return np.zeros_like(scores)
    return (scores - low) / span


def _check_scores(scores: np.ndarray, length: int) -> None:
    if scores.shape != (length,):
        raise ValueError(f"score length {scores.shape[0]} != series length {length}")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")


def _train_item(split: DatasetSplit, series_id: str) -> SeriesItem:
    if split.train is None or series_id not in split.train:
        raise ValueError(f"CrossAD needs a normal train prefix for {series_id!r}")
    return split.train[series_id]


def evaluate(
    dataset_dir: str | Path,
    output_root: str | Path = "benchmarks/CrossAD",
    *,
    artifact: str | None = None,
    partition: str | None = "Eva",
    device: str = "cuda",
    seed: int = 2024,
    hp: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Score every series of one dataset artifact and return the per-series
    metric table; with ``resume`` set, series that already have scores are
    skipped."""
    split = load_split(dataset_dir, artifact, partition=partition)
    unit = unit_dir(output_root, METHOD_NAME, split)
    for series_id in split.test.ids:
        if resume and has_score(unit, series_id):
            continue
        L.seed_everything(seed)
        train_item = _train_item(split, series_id)
        test_item = split.test[series_id]
        full = np.concatenate([train_item.data, test_item.data])
        module = _load_or_train(
            unit,
            series_id,
            train_item.data,
            source=split.source,
            hp=hp,
            device=device,
            resume=resume,
        )
        scores = score_series(module, full, module.config, device=device)
        _check_scores(scores, len(full))
        save_score(unit, series_id, scores)
    return write_metrics(split, unit)


if __name__ == "__main__":
    evaluate(
        dataset_dir="data/processed/tsb-ad/TSB-AD-M/CATSv2",
        output_root="benchmarks/CrossAD",
        device="cuda",
    )
