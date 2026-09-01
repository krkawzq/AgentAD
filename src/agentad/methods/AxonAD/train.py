"""Form-A per-series training for AxonAD on the normal train prefix (TSB-AD Optimal_Multi HP)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from ...benchmark import (
    atomic_write_file,
    checkpoints_dir,
    load_split,
    prepare_run,
    unit_dir,
)
from .._utils import sliding_windows
from .config import AxonADConfig
from .model import AxonAD, AxonADLightningModule

METHOD_NAME = "AxonAD"

# TSB-AD Optimal_Multi (forks/AxonAD/TSB_AD/HP_list.py); Optimal_Uni has no
# entry, so both M and U share these values, plus the original fit defaults.
DEFAULT_HP: Mapping[str, Any] = {
    "win_size": 100,
    "d_model": 128,
    "num_heads": 8,
    "lr": 5e-4,
    "kl_tail_k": 10,
    "forecast_steps": 1,
    "epochs": 50,
    "batch_size": 128,
    "validation_size": 0.2,
}

# Original TSB-AD HP keys -> AxonADConfig fields.
_HP_TO_CONFIG: Mapping[str, str] = {
    "win_size": "window_length",
    "d_model": "model_dim",
    "num_heads": "heads",
    "lr": "learning_rate",
    "kl_tail_k": "tail_length",
    "forecast_steps": "forecast_steps",
    "epochs": "epochs",
    "batch_size": "batch_size",
    "validation_size": "validation_fraction",
}


def build_config(
    input_features: int, hp: Mapping[str, Any] | None = None
) -> AxonADConfig:
    """Optimal_Multi config with ``hp`` overrides (original TSB-AD keys or config field names)."""
    config_names = {field.name for field in fields(AxonADConfig)}
    overrides: dict[str, Any] = {}
    for key, value in {**DEFAULT_HP, **(hp or {})}.items():
        name = _HP_TO_CONFIG.get(key, key)
        if name not in config_names:
            raise ValueError(f"unknown AxonAD hyperparameter: {key!r}")
        overrides[name] = value
    base = AxonADConfig.original_optimal_multivariate(input_features=input_features)
    return replace(base, **overrides)


def _segment_tensor(data: np.ndarray) -> Tensor:
    """One series segment as a ``[1, T, C]`` float32 tensor, un-normalized."""
    values = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
    return values[None]


def _window_tensor(data: np.ndarray, window: int) -> Tensor:
    """Z-score one segment per feature and slide windows -> ``[N, window, C]``."""
    windows = sliding_windows(AxonAD.segment_zscore(_segment_tensor(data)), window)
    return windows.flatten(0, 1)


def _split_train_validation(
    data: np.ndarray, config: AxonADConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out the last ``validation_fraction`` tail for validation.

    Segments no longer than one training window train on everything without
    a validation split, matching the original ``split <= win_size`` branch.
    """
    split = int((1.0 - config.validation_fraction) * len(data))
    if split <= config.window_length:
        return data, data[:0]
    return data[:split], data[split:]


def train_series(
    train_data: np.ndarray,
    config: AxonADConfig,
    *,
    device: str | torch.device = "cuda",
) -> AxonADLightningModule:
    """Train one series and calibrate on its training segment.

    The training and validation segments are each z-scored per feature
    before windowing; the returned module is ready for ``score`` on an
    un-normalized full-length series.
    """
    device = torch.device(device)
    train_segment, valid_segment = _split_train_validation(train_data, config)
    train_windows = _window_tensor(train_segment, config.window_length)
    train_loader = DataLoader(
        TensorDataset(train_windows),
        batch_size=config.batch_size,
        shuffle=True,
    )
    valid_loader: DataLoader | None = None
    # A validation tail shorter than one window yields no windows; the
    # original then trains without validation, so skip early stopping here.
    if len(valid_segment) >= config.window_length:
        valid_windows = _window_tensor(valid_segment, config.window_length)
        if len(valid_windows) > 0:
            valid_loader = DataLoader(
                TensorDataset(valid_windows),
                batch_size=config.batch_size,
                shuffle=False,
            )
    module = AxonADLightningModule(config).to(device)
    trainer = L.Trainer(
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        max_epochs=config.epochs,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    # Calibration consumes the un-normalized training segment; score()
    # z-scores whatever series it receives.
    module.model.calibrate(
        _segment_tensor(train_segment).to(module.device),
        batch_size=config.batch_size,
    )
    return module


def train(
    dataset_dir: str | Path,
    output_root: str | Path = "outputs/benchmark",
    *,
    artifact: str | None = None,
    partition: str | None = "Eva",
    device: str = "cuda",
    seed: int = 2024,
    hp: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> None:
    """Train one checkpoint per series of one dataset artifact pair."""
    split = load_split(dataset_dir, artifact, partition=partition)
    if split.train is None:
        raise FileNotFoundError(
            f"no train archive for {split.unit_name}; AxonAD is semi-supervised"
        )
    unit = unit_dir(output_root, METHOD_NAME, split)
    prepare_run(
        unit,
        split,
        method=METHOD_NAME,
        partition=partition,
        seed=seed,
        hp=hp,
        resume=resume,
        implementation_file=__file__,
        device=device,
    )
    checkpoints = checkpoints_dir(unit)
    config = build_config(split.test.n_features, hp)
    for series_id in split.test.ids:
        checkpoint = checkpoints / f"{series_id}.pt"
        if resume and checkpoint.is_file():
            continue
        L.seed_everything(seed)
        if series_id not in split.train:
            raise KeyError(f"series {series_id!r} missing from the train archive")
        train_item = split.train[series_id]
        module = train_series(train_item.data, config, device=device)
        checkpoints.mkdir(parents=True, exist_ok=True)
        payload = {"config": asdict(config), "state_dict": module.state_dict()}
        atomic_write_file(
            checkpoint,
            lambda handle: torch.save(payload, handle),
        )
