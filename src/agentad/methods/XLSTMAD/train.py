"""Form A per-series training for xLSTMAD under the TSB-AD benchmark protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from ...benchmark import checkpoints_dir, load_split, unit_dir

from .config import XLSTMADConfig
from .model import XLSTMADLightningModule

METHOD_NAME = "xLSTMAD"

# TSB-AD benchmark HPs. Optimal_Multi/Uni_algo_HP_dict (HP_list.py:139,335)
# are identical (window_size=50, lr=1e-3, embedding_dim=40, no uni/multi
# difference), while the xLSTMAD class default and run_xLSTMAD function
# default use window_size=100. The benchmark contract here follows the class
# default window 100; pass hp={"window_length": 50, "embedding_dim": 40} to
# switch to the Optimal HP dict values. epochs=50, patience=3 and
# min_delta=1e-3 come from the original Trainer (models/xLSTMAD.py:159-171).
DEFAULT_HP: Mapping[str, Any] = {
    "window_length": 100,
    "learning_rate": 1e-3,
    "epochs": 50,
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 1e-3,
    "validation_fraction": 0.2,
}


def _zscore(data: np.ndarray) -> np.ndarray:
    """Per-channel z-score; zero std maps to 1e-8 (ReconstructDataset)."""
    mean = data.mean(axis=0)
    std = np.where(data.std(axis=0) == 0, 1e-8, data.std(axis=0))
    return (data - mean) / std


def build_config(
    input_features: int,
    *,
    device: str,
    hp: Mapping[str, Any] | None = None,
) -> XLSTMADConfig:
    """Build one config from ``DEFAULT_HP`` overridden by ``hp``.

    ``input_features`` comes from the data and ``scalar_backend`` follows the
    device (the original sLSTM blocks use the CUDA backend; CPU needs the
    vanilla backend), so neither is part of ``DEFAULT_HP``.
    """
    merged = {**DEFAULT_HP, **(hp or {})}
    merged.setdefault(
        "scalar_backend",
        "cuda" if torch.device(device).type == "cuda" else "vanilla",
    )
    config = XLSTMADConfig.original_default(
        input_features=input_features,
        window_length=merged.pop("window_length"),
        scalar_backend=merged.pop("scalar_backend"),
    )
    return replace(config, **merged)


def _window_dataset(normalized: np.ndarray, window: int) -> TensorDataset:
    """Stride-1 reconstruction windows as a float32 ``[n, window, features]`` set."""
    tensor = torch.from_numpy(np.ascontiguousarray(normalized, dtype=np.float32))
    return TensorDataset(tensor.unfold(0, window, 1).transpose(1, 2))


def train_series(
    train_data: np.ndarray,
    config: XLSTMADConfig,
    *,
    device: str = "cuda",
) -> XLSTMADLightningModule:
    """Train one model on one train segment and return the fitted module.

    Follows the original ``xLSTMAD.fit``: the segment is z-scored with its
    own statistics, windowed with stride 1, and split on window indices into
    the first ``1 - validation_fraction`` fraction for training and the tail
    for validation; early stopping reloads the best weights.
    """
    windows = _window_dataset(_zscore(train_data), config.window_length)
    split_index = int((1.0 - config.validation_fraction) * len(windows))
    if config.validation_fraction > 0 and not 0 < split_index < len(windows):
        raise ValueError(
            f"train segment of {len(windows)} windows cannot be split with "
            f"validation_fraction={config.validation_fraction}"
        )
    train_loader = DataLoader(
        Subset(windows, range(split_index)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = None
    if config.validation_fraction > 0:
        val_loader = DataLoader(
            Subset(windows, range(split_index, len(windows))),
            batch_size=4 * config.batch_size,
        )
    module = XLSTMADLightningModule(config).to(device)
    trainer = L.Trainer(
        accelerator="gpu" if torch.device(device).type == "cuda" else "cpu",
        devices=1,
        max_epochs=config.epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return module


def save_checkpoint(
    path: str | Path,
    module: XLSTMADLightningModule,
    config: XLSTMADConfig,
) -> None:
    """Persist config and weights so eval can rebuild the model exactly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in module.state_dict().items()}
    torch.save({"config": asdict(config), "state_dict": state}, path)


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
    """Train one xLSTMAD model per series train segment and store checkpoints.

    The training unit is one series; each checkpoint lives at
    ``<unit>/checkpoints/<series_id>.pt`` and existing files are skipped when
    ``resume`` is set.
    """
    split = load_split(dataset_dir, artifact, partition=partition)
    if split.train is None:
        raise ValueError(f"xLSTMAD needs a train archive: {split.unit_name}")
    unit = unit_dir(output_root, METHOD_NAME, split)
    for series_id in split.test.ids:
        ckpt_path = checkpoints_dir(unit) / f"{series_id}.pt"
        if resume and ckpt_path.is_file():
            continue
        L.seed_everything(seed)
        train_data = split.train[series_id].data
        config = build_config(train_data.shape[1], device=device, hp=hp)
        module = train_series(train_data, config, device=device)
        save_checkpoint(ckpt_path, module, config)
