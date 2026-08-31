"""Form-A per-series fine-tuning script for TSPulse (TSB-AD ``TSPulse_FT``).

Trains one checkpoint per series from the normal prefix, following the
reference ``TSPulsePipeline.fit`` recipe: stride-1 context windows expanded
into enumerated patch-masking self-supervision samples, AdamW + OneCycleLR
inside :class:`TSPulseLightningModule`, and the original short-series rules.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, fields, replace
import math
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from ...benchmark import checkpoints_dir, load_split, unit_dir

from .config import TSPulseConfig
from .model import TSPulseFineTune, TSPulseLightningModule

METHOD_NAME = "TSPulse-FT"
PRETRAINED_PATH = "pretrained/tspulse"

# TSB-AD Optimal HP (HP_list.py 'TSPulse_FT') plus the official fine-tuning
# defaults (models/TSPulse.py fit): context 512, win_size 96, epochs 20,
# lr 1e-4, batch 256, validation_fraction 0.2. ``one_cycle_pct_start`` 0.475
# is the torch OneCycleLR default the reference scheduler relies on (it never
# passes pct_start); the config default 0.3 deviates from that. Smoke runs
# override ``epochs``/``batch_size``; ``context`` must match the pretrained core.
DEFAULT_HP: Mapping[str, Any] = {
    "context": 512,
    "win_size": 96,
    "epochs": 20,
    "lr": 1e-4,
    "batch_size": 256,
    "validation_fraction": 0.2,
    "one_cycle_pct_start": 0.475,
}

_HP_FIELDS: Mapping[str, str] = {
    "context": "context_length",
    "win_size": "aggregation_length",
    "epochs": "finetune_epochs",
    "lr": "finetune_learning_rate",
    "batch_size": "finetune_batch_size",
    "validation_fraction": "finetune_validation",
}

# Short-series thresholds of the reference fit: no validation split and a
# five-epoch cap below 3000 points, subsampling above 100k enumerated
# samples, batch 8 below 500 samples, and skip below 100 samples.
_SMALL_SEGMENT = 3000
_MAX_FINETUNE_SAMPLES = 100_000
_SMALL_FINETUNE_SAMPLES = 500
_MIN_FINETUNE_SAMPLES = 100

_CONFIG_FIELDS = frozenset(field.name for field in fields(TSPulseConfig))


def _resolve_config(input_features: int, hp: Mapping[str, Any] | None) -> TSPulseConfig:
    """Build the fine-tuning config from the ``original_tsb_finetune`` factory."""
    merged = {**DEFAULT_HP, **(hp or {})}
    # The scoring prediction mode is fixed to the TSB-AD optimum ("time")
    # and has no config field, so the key is accepted and ignored here.
    merged.pop("prediction_mode", None)
    overrides: dict[str, Any] = {}
    for key, value in merged.items():
        name = _HP_FIELDS.get(key, key)
        if name not in _CONFIG_FIELDS:
            raise ValueError(f"unknown TSPulse fine-tuning hp: {key!r}")
        overrides[name] = value
    config = TSPulseConfig.original_tsb_finetune(input_features=input_features)
    return replace(config, **overrides)


class _PatchMaskWindows(Dataset):
    """Enumerated patch-masking self-supervision samples of one segment.

    Replicates the reference data recipe: ``TSPulseFinetuneDataset``
    (per-channel z-score with ``std == 0 -> 1e-8``, front-padding with the
    first row, stride-1 windows) composed with
    ``PatchMaskingDatasetWrapper(window_position="last")``, which hides each
    patch of the aggregation window exactly once.
    """

    def __init__(self, segment: np.ndarray, config: TSPulseConfig) -> None:
        context = config.context_length
        mean = segment.mean(axis=0)
        deviation = segment.std(axis=0)
        deviation = np.where(deviation == 0, 1e-8, deviation)
        data = (segment - mean) / deviation
        if len(data) < context:
            padding = np.repeat(data[:1], context - len(data), axis=0)
            data = np.concatenate((padding, data), axis=0)
        self._data = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
        self._context = context
        self._patch = config.patch_length
        self._window_start = max(0, context - config.aggregation_length)
        self.patches = math.ceil(config.aggregation_length / self._patch)
        self.windows = max(len(data) - context + 1, 0)

    def __len__(self) -> int:
        return self.windows * self.patches

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        window, patch = divmod(index, self.patches)
        values = self._data[window : window + self._context]
        observed = torch.ones_like(values, dtype=torch.bool)
        start = self._window_start + patch * self._patch
        observed[start : min(start + self._patch, self._context)] = False
        return {"past_values": values, "past_observed_mask": observed}


def _fine_tune(
    segment: np.ndarray, config: TSPulseConfig, *, device: str, seed: int
) -> TSPulseFineTune | None:
    """Fine-tune on one normal prefix; ``None`` means the reference fit rules skip it."""
    # The reference fit applies its short-series checks to the series passed
    # to ``fit``, which is the normal prefix (train segment) in this protocol.
    if len(segment) < config.context_length:
        return None
    create_valid = len(segment) >= _SMALL_SEGMENT
    if create_valid:
        split_at = int((1 - config.finetune_validation) * len(segment))
        train_set = _PatchMaskWindows(segment[:split_at], config)
        valid_set = _PatchMaskWindows(segment[split_at:], config)
    else:
        train_set = _PatchMaskWindows(segment, config)
        valid_set = train_set
    if len(train_set) < _MIN_FINETUNE_SAMPLES:
        return None
    if len(train_set) > _MAX_FINETUNE_SAMPLES:
        fraction = _MAX_FINETUNE_SAMPLES / len(train_set)
        train_set, _ = random_split(train_set, [fraction, 1 - fraction])
        valid_set, _ = random_split(valid_set, [fraction, 1 - fraction])
    batch_size = config.finetune_batch_size
    if len(train_set) < _SMALL_FINETUNE_SAMPLES:
        batch_size = 8
    epochs = config.finetune_epochs
    if not create_valid:
        epochs = min(5, epochs)

    L.seed_everything(seed)
    model = TSPulseFineTune.from_pretrained(config, model_name_or_path=PRETRAINED_PATH)
    module = TSPulseLightningModule(config, model=model)
    trainer = L.Trainer(
        accelerator="gpu" if torch.device(device).type == "cuda" else "cpu",
        devices=1,
        max_epochs=epochs,
        # HF TrainingArguments clips at max_grad_norm=1.0 by default.
        gradient_clip_val=1.0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(
        module,
        train_dataloaders=DataLoader(train_set, batch_size=batch_size, shuffle=True),
        val_dataloaders=DataLoader(valid_set, batch_size=batch_size),
    )
    return model


def train_series(
    train_data: np.ndarray,
    *,
    device: str = "cuda",
    seed: int = 2024,
    hp: Mapping[str, Any] | None = None,
) -> TSPulseFineTune | None:
    """Fine-tune one series' normal prefix in memory.

    Returns the fine-tuned model, or ``None`` when the short-series rules
    skip fine-tuning (callers fall back to zero-shot scoring).
    """
    segment = np.asarray(train_data, dtype=np.float64)
    config = _resolve_config(segment.shape[1], hp)
    return _fine_tune(segment, config, device=device, seed=seed)


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
    """Fine-tune every series of one artifact pair and store per-series checkpoints."""
    split = load_split(dataset_dir, artifact, partition=partition)
    unit = unit_dir(output_root, METHOD_NAME, split)
    input_features = split.test.n_features
    config = _resolve_config(input_features, hp)
    for series_id in split.test.ids:
        path = checkpoints_dir(unit) / f"{series_id}.pt"
        if resume and path.is_file():
            continue
        train_item = split.train[series_id] if split.train is not None else None
        if train_item is None:
            continue
        model = _fine_tune(train_item.data, config, device=device, seed=seed)
        if model is None:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(config), "state_dict": model.state_dict()}, path)
