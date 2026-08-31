"""Form-A per-series training for CrossAD under the authors' TSB-AD protocol.

One CrossAD per series is trained on the normal prefix, mirroring
``forks/CrossAD/run_TSBAD.py``: independent z-score per split, stride-one
training windows, non-overlapping validation windows, type1 lr decay.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ...benchmark import DatasetSplit, checkpoints_dir, load_split, unit_dir

from .config import CrossADConfig
from .model import CrossADLightningModule

METHOD_NAME = "CrossAD"

# Training recipe of the authors' TSB-AD runs
# (forks/CrossAD/configs/TSB-AD-U/train_configs.json); structural fields such
# as sequence_length come from the per-series config factory instead.
DEFAULT_HP: Mapping[str, Any] = {
    "train_epochs": 20,
    "batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "learning_rate_schedule": "type1",
    "patience": 3,
}

# ``None`` (no authors' mapping, e.g. TSB-AD-M) falls back to the default
# factory through ``dict.get``.
_SETTING_FACTORIES: dict[int | None, Callable[[], CrossADConfig]] = {
    1: CrossADConfig.original_tsb_1,
    2: CrossADConfig.original_tsb_2,
    3: CrossADConfig.original_tsb_3,
    4: CrossADConfig.original_tsb_4,
    5: CrossADConfig.original_tsb_5,
    6: CrossADConfig.original_tsb_6,
}
# Generic TSB recipe (model_configs_2.json); TSB-AD-M has no official mapping.
_DEFAULT_FACTORY = CrossADConfig.original_tsb_2

# Authors' per-series TSB-AD-U setting table, keyed by source-file stem; the
# table lives with the fork and a missing one degrades to the default factory.
_SETTINGS_CSV = (
    Path(__file__).resolve().parents[4]
    / "forks/CrossAD/configs/TSB-AD-U/TSB-AD-U_setting_draft.csv"
)


@lru_cache(maxsize=1)
def _tsb_ad_u_settings() -> dict[str, int]:
    """Read the authors' file-name -> setting table; missing file -> empty."""
    try:
        with _SETTINGS_CSV.open(newline="") as handle:
            return {
                row["name"].strip().removesuffix(".csv"): int(row["model_setting"])
                for row in csv.DictReader(handle)
            }
    except OSError:
        return {}


def select_config(
    series_id: str, source: str, hp: Mapping[str, Any] | None = None
) -> CrossADConfig:
    """Build one series' config: the authors' TSB-AD-U factory when their
    setting table maps the series id, the generic TSB factory otherwise, then
    ``hp`` overrides applied last."""
    setting = _tsb_ad_u_settings().get(series_id) if source == "TSB-AD-U" else None
    factory = _SETTING_FACTORIES.get(setting, _DEFAULT_FACTORY)
    return replace(factory(), **{**DEFAULT_HP, **(hp or {})})


def zscore(data: np.ndarray) -> np.ndarray:
    """Per-channel z-score with the original's zero-std guard (epsilon 1e-8)."""
    std = data.std(axis=0)
    return (data - data.mean(axis=0)) / np.where(std == 0, 1e-8, std)


class _WindowDataset(Dataset):
    """Lazily sliced windows; stride-one windows are far too numerous to
    materialize as the original's stacked tensor does."""

    def __init__(self, data: np.ndarray, window: int, stride: int) -> None:
        self.values = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
        self.window = window
        self.stride = stride
        self.count = max(0, (len(data) - window) // stride + 1)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> Tensor:
        start = index * self.stride
        return self.values[start : start + self.window]


def _train_valid_windows(
    train_data: np.ndarray, window: int
) -> tuple[_WindowDataset, _WindowDataset]:
    """80/20 prefix split with the original's windowing.

    Both parts are z-scored independently, training windows slide at stride
    one and validation windows do not overlap. A prefix too short for a
    validation window reuses its last training window so ``val/loss`` stays
    monitorable, and a prefix shorter than one window is edge-padded.
    """
    if len(train_data) < window:
        train_data = np.pad(
            train_data, ((0, window - len(train_data)), (0, 0)), mode="edge"
        )
    cut = int(0.8 * len(train_data))
    train_part = train_data[:cut]
    if len(train_part) < window:
        # An 80/20 cut below one window cannot train; fall back to the prefix.
        train_part = train_data
    train_windows = _WindowDataset(zscore(train_part), window, stride=1)
    valid_windows = _WindowDataset(zscore(train_data[cut:]), window, stride=window)
    if len(valid_windows) == 0:
        valid_windows = _WindowDataset(
            zscore(train_part[-window:]), window, stride=window
        )
    return train_windows, valid_windows


def _accelerator(device: str) -> str:
    if not device.startswith("cuda"):
        return "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"device={device!r} requested but CUDA is unavailable; use device='cpu'"
        )
    return "gpu"


def train_series(
    train_data: np.ndarray,
    config: CrossADConfig,
    *,
    device: str = "cuda",
    seed: int = 2024,
) -> CrossADLightningModule:
    """Train one series' model on its normal prefix and return the module
    restored to its best validation weights."""
    L.seed_everything(seed)
    window = config.sequence_length
    train_windows, valid_windows = _train_valid_windows(
        np.asarray(train_data, dtype=np.float64), window
    )
    train_loader = DataLoader(train_windows, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_windows, batch_size=config.batch_size)
    module = CrossADLightningModule(config)
    trainer = L.Trainer(
        accelerator=_accelerator(device),
        devices=1,
        max_epochs=config.train_epochs,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, train_loader, valid_loader)
    return module


def _train_data(split: DatasetSplit, series_id: str) -> np.ndarray:
    if split.train is None or series_id not in split.train:
        raise ValueError(f"CrossAD needs a normal train prefix for {series_id!r}")
    return split.train[series_id].data


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
    """Train one CrossAD per series of a dataset artifact and store each as
    ``checkpoints/<series_id>.pt``; with ``resume`` set, existing checkpoints
    are kept."""
    split = load_split(dataset_dir, artifact, partition=partition)
    ckpt_dir = checkpoints_dir(unit_dir(output_root, METHOD_NAME, split))
    for series_id in split.test.ids:
        ckpt_path = ckpt_dir / f"{series_id}.pt"
        if resume and ckpt_path.is_file():
            continue
        config = select_config(series_id, split.source, hp)
        module = train_series(
            _train_data(split, series_id), config, device=device, seed=seed
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"config": asdict(config), "state_dict": module.state_dict()}, ckpt_path
        )
