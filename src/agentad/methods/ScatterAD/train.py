"""Dataset-level training for ScatterAD (form A).

Protocol mirrors the vendored fork ``forks/ScatterAD`` (``solver.py``,
``data_factory/data_loader.py``): a standard scaler fitted on the whole
training data, stride-one sliding windows, Adam with the fork's defaults,
exponential target-encoder updates, and no early stopping — the final
epoch's weights are the model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ...series import SeriesData
from .config import ScatterADConfig
from .model import ScatterADLightningModule
from ...benchmark import DatasetSplit, checkpoints_dir, load_split, unit_dir

METHOD_NAME = "ScatterAD"

# Fork defaults (``main.py`` + ``scripts/all.sh``). ``win_size`` is the
# fork's CLI name for the config's ``window_length``; ``stride`` is the
# loader step and stays a windowing knob of this module.
DEFAULT_HP: Mapping[str, Any] = {
    "win_size": 110,
    "stride": 1,
    "epochs": 2,
    "batch_size": 128,
    "learning_rate": 1e-4,
    "graph_radius": 1,
    "target_momentum": 0.5,
}

_HP_CONFIG_ALIASES = {"win_size": "window_length"}
_HP_SCRIPT_FIELDS = frozenset({"stride"})


def _merged_hp(hp: Mapping[str, Any] | None) -> dict[str, Any]:
    return {**DEFAULT_HP, **(hp or {})}


def _windowing(hp: Mapping[str, Any] | None) -> tuple[int, int]:
    window = int(_merged_hp(hp)["win_size"])
    stride = int(_merged_hp(hp)["stride"])
    if window < 2:
        raise ValueError("win_size must cover at least 2 time steps")
    if stride <= 0:
        raise ValueError("stride must be positive")
    return window, stride


def _build_config(
    input_features: int, hp: Mapping[str, Any] | None
) -> ScatterADConfig:
    merged = {
        name: value
        for name, value in _merged_hp(hp).items()
        if name not in _HP_SCRIPT_FIELDS
    }
    renamed = {
        _HP_CONFIG_ALIASES.get(name, name): value for name, value in merged.items()
    }
    known = {field.name for field in fields(ScatterADConfig)}
    unknown = sorted(set(renamed) - known)
    if unknown:
        raise ValueError(f"unknown ScatterAD hyperparameters: {unknown}")
    return replace(ScatterADConfig(input_features=input_features), **renamed)


def _fit_scaler(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over the concatenated train segments.

    The fork fits its StandardScaler on the whole training file before the
    80/20 cut, so the statistics cover the validation series too.
    Zero-variance channels keep scale 1 (sklearn semantics).
    """
    stacked = np.concatenate(arrays, axis=0)
    mean = stacked.mean(axis=0)
    std = np.where(stacked.std(axis=0) > 0, stacked.std(axis=0), 1.0)
    return mean, std


def _normalized_arrays(
    collection: SeriesData, mean: np.ndarray, std: np.ndarray
) -> list[np.ndarray]:
    return [((item.data - mean) / std).astype(np.float32) for item in collection]


class _WindowDataset(Dataset):
    """Lazily materialized sliding windows over pre-normalized series."""

    def __init__(self, arrays: list[np.ndarray], window: int, stride: int) -> None:
        self._arrays = arrays
        self._window = window
        self._stride = stride
        counts = [max(0, (a.shape[0] - window) // stride + 1) for a in arrays]
        self._bounds = np.cumsum([0, *counts])

    def __len__(self) -> int:
        return int(self._bounds[-1])

    def __getitem__(self, index: int) -> torch.Tensor:
        source = int(np.searchsorted(self._bounds, index, side="right")) - 1
        offset = (index - int(self._bounds[source])) * self._stride
        return torch.from_numpy(
            self._arrays[source][offset : offset + self._window]
        )


def _series_split(train: SeriesData) -> tuple[list[str], list[str]]:
    """Internal train/validation split over the train series ids.

    The fork validates on the tail 20% of its single training file; with
    several series the tail 20% of the series count plays that role, and a
    single-series artifact trains without validation. The fork's datasets
    that score on the test file (WADI/Water/Swan) are not reproduced.
    """
    ids = list(train.ids)
    if len(ids) < 2:
        return ids, []
    n_train = min(max(1, int(0.8 * len(ids))), len(ids) - 1)
    return ids[:n_train], ids[n_train:]


def _loaders(
    config: ScatterADConfig,
    train: SeriesData,
    mean: np.ndarray,
    std: np.ndarray,
    window: int,
    stride: int,
) -> tuple[DataLoader, DataLoader | None]:
    train_ids, val_ids = _series_split(train)
    train_set = _WindowDataset(
        _normalized_arrays(train.select(train_ids), mean, std), window, stride
    )
    if len(train_set) < 2:
        raise RuntimeError(
            f"ScatterAD training needs at least two windows of length {window}"
        )
    # BatchNorm1d rejects single-sample training batches, so the trailing
    # batch is dropped only when it would hold exactly one window.
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=(len(train_set) % config.batch_size == 1),
    )
    val_loader: DataLoader | None = None
    if val_ids:
        val_set = _WindowDataset(
            _normalized_arrays(train.select(val_ids), mean, std), window, stride
        )
        if len(val_set) > 0:
            val_loader = DataLoader(val_set, batch_size=config.batch_size)
    return train_loader, val_loader


def train_dataset(
    split: DatasetSplit,
    *,
    device: str = "cuda",
    seed: int = 2024,
    hp: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one ScatterAD model jointly on every series of ``split.train``.

    Returns the checkpoint payload (``config`` dict, CPU ``state_dict``,
    per-channel scaler ``mean``/``std`` tensors) without writing to disk;
    :func:`train` persists it under the unit's checkpoint directory.
    """
    if split.train is None or len(split.train) == 0:
        raise ValueError("ScatterAD training requires a non-empty train split")
    L.seed_everything(seed)
    mean, std = _fit_scaler([item.data for item in split.train])
    window, stride = _windowing(hp)
    config = _build_config(split.train.n_features, hp)
    train_loader, val_loader = _loaders(
        config, split.train, mean, std, window, stride
    )
    module = ScatterADLightningModule(config)
    trainer = L.Trainer(
        accelerator="gpu" if torch.device(device).type == "cuda" else "cpu",
        devices=1,
        max_epochs=config.epochs,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return {
        "config": asdict(config),
        "state_dict": {
            name: value.detach().cpu() for name, value in module.state_dict().items()
        },
        "mean": torch.from_numpy(mean),
        "std": torch.from_numpy(std),
    }


def checkpoint_path(unit: str | Path, artifact: str) -> Path:
    """Checkpoint file of one dataset-level unit."""
    return checkpoints_dir(unit) / f"{artifact}.pt"


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
    """Train ScatterAD on one dataset artifact and persist its checkpoint.

    With ``resume=True`` an existing checkpoint skips training. Scoring is
    handled by :func:`agentad.methods.ScatterAD.eval.evaluate`.
    """
    split = load_split(dataset_dir, artifact, partition=partition)
    unit = unit_dir(output_root, METHOD_NAME, split)
    checkpoint = checkpoint_path(unit, split.name)
    if resume and checkpoint.is_file():
        return
    payload = train_dataset(split, device=device, seed=seed, hp=hp)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint)
