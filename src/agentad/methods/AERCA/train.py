"""Dataset-level AERCA training for TSB-AD artifacts (form A, spec v1)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from ...benchmark import checkpoints_dir, load_split, unit_dir

from .config import AERCAConfig
from .model import AERCA, AERCALightningModule

METHOD_NAME = "AERCA"

# Original AERCA protocol defaults (the generic paper protocol, not the
# swat-specific tuning in AERCAConfig.original_swat); smoke runs shrink
# them via ``hp``.
DEFAULT_HP: Mapping[str, Any] = {
    "epochs": 5_000,
    "patience": 20,
    "learning_rate": 1e-4,
    "hidden_dim": 50,
    "hidden_layers": 1,
    "window": 1,
}


class TrainedAERCA(NamedTuple):
    """In-memory result of one dataset-level training unit.

    ``mean``/``std`` are the per-channel normalization fitted on every
    train point; scoring must reuse the exact same statistics.
    """

    model: AERCA
    config: AERCAConfig
    mean: np.ndarray
    std: np.ndarray


def _build_config(input_features: int, hp: Mapping[str, Any] | None) -> AERCAConfig:
    """Merge ``hp`` over DEFAULT_HP; ``input_features`` always comes from data."""
    overrides = {**DEFAULT_HP, **(hp or {})}
    return AERCAConfig(input_features=input_features, **overrides)


def _normalization_stats(
    series: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over every train point, StandardScaler semantics."""
    stacked = np.concatenate(
        [np.asarray(data, dtype=np.float64) for data in series], axis=0
    )
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    # StandardScaler keeps zero-variance channels at scale one.
    return mean, np.where(std > 0, std, 1.0)


def _split_80_20(
    series: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Original AERCA split: by series count, or by time for a single series.

    The original protocol shuffles the normal series with a seeded
    permutation before the 80/20 cut (``forks/AERCA/datasets/swat.py``);
    the TSB-AD preprocessing performs no permutation, so this split follows
    the fixed ``split.train.ids`` file order — a deliberate landing-point
    difference rather than a reproduced shuffle.
    """
    if len(series) == 1:
        data = series[0]
        cut = int(0.8 * len(data))
        return [data[:cut]], [data[cut:]]
    cut = int(0.8 * len(series))
    return series[:cut], series[cut:]


def _length_batches(series: Sequence[np.ndarray]) -> list[torch.Tensor]:
    """Stack same-length series into one [batch, time, feature] tensor each."""
    groups: dict[int, list[np.ndarray]] = {}
    for data in series:
        groups.setdefault(len(data), []).append(data)
    return [
        torch.from_numpy(np.stack(groups[length]).astype(np.float32))
        for length in sorted(groups)
    ]


def train_dataset(
    series: Sequence[np.ndarray],
    *,
    device: str = "cuda",
    seed: int = 2024,
    hp: Mapping[str, Any] | None = None,
) -> TrainedAERCA:
    """Train one AERCA on all normal series of a dataset unit.

    Splits the series 80/20 into train/validation (by series count, or by
    time for a single series), early-stops on validation loss with the best
    weights restored, and returns the model plus normalization stats.
    """
    if not series:
        raise ValueError("train_dataset requires at least one series")
    L.seed_everything(seed)
    mean, std = _normalization_stats(series)
    normalized = [
        (np.asarray(data, dtype=np.float64) - mean) / std for data in series
    ]
    train_part, val_part = _split_80_20(normalized)
    config = _build_config(normalized[0].shape[1], hp)
    module = AERCALightningModule(config)
    trainer = L.Trainer(
        accelerator="gpu" if torch.device(device).type == "cuda" else "cpu",
        devices=1,
        max_epochs=config.epochs,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    # batch_size=None yields each pre-stacked tensor unchanged, so one batch
    # can hold every same-length series while ragged lengths stay supported.
    trainer.fit(
        module,
        train_dataloaders=DataLoader(_length_batches(train_part), batch_size=None),
        val_dataloaders=DataLoader(_length_batches(val_part), batch_size=None),
    )
    return TrainedAERCA(module.model, config, mean, std)


def train(
    dataset_dir: str | Path,
    output_root: str | Path = "benchmarks",
    *,
    artifact: str | None = None,
    partition: str | None = "Eva",
    device: str = "cuda",
    seed: int = 2024,
    hp: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> None:
    """Train the dataset-level AERCA of one artifact and store its checkpoint."""
    split = load_split(dataset_dir, artifact, partition=partition)
    if split.train is None:
        raise FileNotFoundError(
            f"AERCA needs a normal train prefix: {split.unit_name}"
        )
    checkpoint = (
        checkpoints_dir(unit_dir(output_root, METHOD_NAME, split))
        / f"{split.name}.pt"
    )
    if resume and checkpoint.is_file():
        return
    trained = train_dataset(
        [split.train[series_id].data for series_id in split.train.ids],
        device=device,
        seed=seed,
        hp=hp,
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(trained.config),
            "state_dict": trained.model.state_dict(),
            "mean": torch.from_numpy(trained.mean),
            "std": torch.from_numpy(trained.std),
        },
        checkpoint,
    )
