"""Form A per-series training: KAN-AD one-step prediction, protocol in ``KANADConfig.original_default``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ...benchmark import checkpoints_dir, load_split, unit_dir

from .config import KANADConfig
from .model import KANAD, KANADLightningModule

METHOD_NAME = "KAN-AD"

# Original config.toml [Model_Params.Default] plus kanad.py's
# EarlyStoppingTorch: patience=3; StepLR(step 5, gamma 0.75) and early
# stopping min_delta=1e-4 are built into KANADConfig.original_default. The
# model is channel-independent (per-feature forecasting), so the feature
# count comes from the input shape and needs no config field.
DEFAULT_HP: Mapping[str, Any] = {
    "window": 96,
    "order": 2,
    "batch_size": 1024,
    "epochs": 100,
    "learning_rate": 0.01,
    "early_stopping_patience": 3,
}

# EasyTSAD GlobalCfg carves the validation segment from the training tail
# with valid_proportion=0.2.
_VALID_PROPORTION = 0.2


def _resolve_config(hp: Mapping[str, Any] | None) -> KANADConfig:
    """Merge ``hp`` over :data:`DEFAULT_HP` on top of the original defaults."""
    return replace(KANADConfig.original_default(), **{**DEFAULT_HP, **(hp or {})})


def difference_standardize(
    train_data: np.ndarray, full_data: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """First-order difference over the full length, z-scored with train stats.

    ``full_data`` is the concatenated train+test series; the z-score statistics
    come from the train segment's differences, which equal the first
    ``len(train_data) - 1`` points of the full-length difference, so no
    difference crosses the train/test boundary into the statistics. Returns
    ``(z_train, z_full)`` on the differenced domain (each one point shorter
    than the corresponding input).
    """
    diff_full = np.diff(np.asarray(full_data, dtype=np.float64), axis=0)
    n_train_diff = len(train_data) - 1
    diff_train = diff_full[:n_train_diff]
    mean = diff_train.mean(axis=0)
    std = diff_train.std(axis=0)
    # EasyTSAD's StandardScaler replaces zero scales with 1.
    std = np.where(std == 0, 1.0, std)
    z_full = (diff_full - mean) / std
    return z_full[:n_train_diff], z_full


def _split_valid(z_train: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Carve the validation tail off the differenced train segment."""
    valid_len = max(1, int(len(z_train) * _VALID_PROPORTION))
    fit, valid = z_train[:-valid_len], z_train[-valid_len:]
    needed = window + 1  # pairs() needs window context points plus 1 target
    if len(fit) < needed or len(valid) < needed:
        raise ValueError(
            f"train segment too short for window={window}: {len(fit)} fit / "
            f"{len(valid)} valid differenced points, need at least {needed} each"
        )
    return fit, valid


def _pair_loader(
    model: KANAD, z: np.ndarray, *, batch_size: int, shuffle: bool
) -> DataLoader:
    series = torch.from_numpy(np.ascontiguousarray(z)).to(torch.float32)[None]
    windows, targets = model.pairs(series)
    return DataLoader(
        TensorDataset(windows, targets), batch_size=batch_size, shuffle=shuffle
    )


def train_series(
    train_data: np.ndarray,
    *,
    config: KANADConfig,
    device: str = "cpu",
    checkpoint: str | Path | None = None,
) -> KANADLightningModule:
    """Train one series on its differenced normal prefix and return the module.

    ``checkpoint=None`` keeps the trained module in memory only (the
    on-the-fly path eval uses when a checkpoint is missing); otherwise the
    resolved config and state dict are saved for eval-time reconstruction.
    """
    z_train, _ = difference_standardize(train_data, train_data)
    fit, valid = _split_valid(z_train, config.window)
    module = KANADLightningModule(config)
    trainer = L.Trainer(
        accelerator="gpu" if str(device).startswith(("cuda", "gpu")) else "cpu",
        devices=1,
        max_epochs=config.epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(
        module,
        _pair_loader(module.model, fit, batch_size=config.batch_size, shuffle=True),
        _pair_loader(module.model, valid, batch_size=config.batch_size, shuffle=False),
    )
    if checkpoint is not None:
        path = Path(checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(config), "state_dict": module.state_dict()}, path)
    return module


def train(
    dataset_dir: str | Path,
    output_root: str | Path = f"benchmarks/{METHOD_NAME}",
    *,
    artifact: str | None = None,
    partition: str | None = "Eva",
    device: str = "cuda",
    seed: int = 2024,
    hp: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> None:
    """Train one checkpoint per test series under ``checkpoints/<series_id>.pt``."""
    split = load_split(dataset_dir, artifact, partition=partition)
    if split.train is None:
        raise ValueError(
            f"KAN-AD trains per series and requires a train split in {dataset_dir}"
        )
    unit = unit_dir(output_root, METHOD_NAME, split)
    config = _resolve_config(hp)
    ckpt_dir = checkpoints_dir(unit)
    for series_id in split.test.ids:
        ckpt_path = ckpt_dir / f"{series_id}.pt"
        if resume and ckpt_path.is_file():
            continue
        if series_id not in split.train:
            raise KeyError(f"no train segment for series {series_id!r}")
        L.seed_everything(seed)
        train_series(
            split.train[series_id].data,
            config=config,
            device=device,
            checkpoint=ckpt_path,
        )
