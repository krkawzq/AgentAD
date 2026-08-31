"""Dataset-level training for LEFT on one TSB-AD artifact (form A).

Protocol (``forks/Left/exp/exp_TSBAD.py`` + ``LeftLightningModule`` docstring):
every series segment is z-scored with its own statistics, the normal train
prefix is split 80/20 in time, training windows slide at stride one and
validation windows are non-overlapping (stride ``sequence_length``). The
early-stopping / best-weight-restore contract lives in the migrated
``ValidationEarlyStopping`` mixin.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from ...benchmark import (
    DatasetSplit,
    checkpoints_dir,
    load_split,
    unit_dir,
)

from .config import LeftConfig
from .model import LeftLightningModule

METHOD_NAME = "Left"

# Left has no official TSB-AD hyperparameters: the fork ships only CrossAD
# settings under configs/TSB-AD-U/, and original_msl() is the factory that
# keeps every structural field at the dataclass default. The fork's seven
# DADA dataset configs train with batch_size 128 and TSB-AD-U with 32; this
# implementation takes 128 as the generic default, overridable via ``hp``.
DEFAULT_HP: Mapping[str, Any] = {
    "train_epochs": 20,
    "batch_size": 128,
    "learning_rate": 1e-4,
    "patience": 3,
}

_ZERO_VARIANCE_EPSILON = 1e-8


def build_config(n_features: int, hp: Mapping[str, Any] | None = None) -> LeftConfig:
    """Build the model config for an artifact with ``n_features`` channels.

    ``input_features`` is derived from the data; the remaining structure
    keeps the most generic original factory (``original_msl``, all dataclass
    defaults). ``hp`` overrides config fields by name on top of DEFAULT_HP.
    """
    base = replace(LeftConfig.original_msl(), input_features=n_features)
    return replace(base, **{**DEFAULT_HP, **(hp or {})})


class _SegmentWindows(Dataset):
    """Lazily sliced windows of one z-scored series segment.

    The fork standardizes the whole segment before windowing and guards
    zero-variance channels with epsilon rather than unit scale.
    """

    def __init__(
        self, data: np.ndarray, start: int, end: int, window: int, stride: int
    ) -> None:
        segment = np.asarray(data[start:end])
        self._segment = segment
        self._mean = segment.mean(axis=0)
        std = segment.std(axis=0)
        self._std = np.where(std == 0, _ZERO_VARIANCE_EPSILON, std)
        self._window = window
        self._stride = stride
        self._count = max(0, (end - start - window) // stride + 1)

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> Tensor:
        start = index * self._stride
        window = self._segment[start : start + self._window]
        z = (window - self._mean) / self._std
        return torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32))


def train_split(
    split: DatasetSplit,
    config: LeftConfig,
    *,
    device: str = "cuda",
    seed: int = 2024,
) -> LeftLightningModule:
    """Fit one LEFT jointly on every training series of ``split``, in memory.

    Each series contributes its own 80/20 train/validation windows; without
    any validation window the stopper monitors the training windows instead,
    which keeps the same normal-reconstruction objective.
    """
    window = config.sequence_length
    train_parts: list[Dataset] = []
    val_parts: list[Dataset] = []
    if split.train is None:
        raise ValueError(f"LEFT is semisupervised; {split.unit_name} has no train split")
    for series_id in split.train.ids:
        data = split.train[series_id].data
        train_end = int(len(data) * 0.8)
        train_windows = _SegmentWindows(data, 0, train_end, window, stride=1)
        val_windows = _SegmentWindows(data, train_end, len(data), window, stride=window)
        if len(train_windows):
            train_parts.append(train_windows)
        if len(val_windows):
            val_parts.append(val_windows)
    if not train_parts:
        raise ValueError(
            f"no training windows of length {window} in {split.unit_name}"
        )
    train_set: Dataset = ConcatDataset(train_parts)
    val_set: Dataset = ConcatDataset(val_parts) if val_parts else train_set

    L.seed_everything(seed)
    module = LeftLightningModule(config)
    trainer = L.Trainer(
        accelerator="gpu" if torch.device(device).type == "cuda" else "cpu",
        devices=1,
        max_epochs=config.train_epochs,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(
        module,
        DataLoader(train_set, batch_size=config.batch_size, shuffle=True),
        DataLoader(val_set, batch_size=config.batch_size, shuffle=False),
    )
    # on_train_end has already restored the best-validation weights.
    return module


def save_checkpoint(
    module: LeftLightningModule, config: LeftConfig, path: str | Path
) -> None:
    """Store the config and CPU weights so eval can rebuild the exact model."""
    state = {key: value.cpu() for key, value in module.state_dict().items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(config), "state_dict": state}, path)


def train(
    dataset_dir: str | Path,
    output_root: str | Path = "benchmarks/Left",
    *,
    artifact: str | None = None,
    partition: str | None = "Eva",
    device: str = "cuda",
    seed: int = 2024,
    hp: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> None:
    """Train one LEFT per artifact and store ``checkpoints/<artifact>.pt``.

    With ``resume=True`` an existing checkpoint makes the call a no-op.
    """
    split = load_split(dataset_dir, artifact, partition=partition)
    unit = unit_dir(output_root, METHOD_NAME, split)
    checkpoint = checkpoints_dir(unit) / f"{split.name}.pt"
    if resume and checkpoint.is_file():
        return
    config = build_config(split.test.n_features, hp)
    module = train_split(split, config, device=device, seed=seed)
    save_checkpoint(module, config, checkpoint)
