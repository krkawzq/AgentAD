"""Per-series PaAno training for TSB-AD artifacts (form A: train.py + eval.py).

The training unit is one series' normal train segment: stride-1 patches
train the triplet+pretext encoder for a fixed iteration count and the
normal memory bank is fit on the same patches, following
``forks/PaAno/train.py``; scoring the train+test concatenation lives in
``eval.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ...benchmark import DatasetSplit, checkpoints_dir, load_split, unit_dir

from .._utils import sliding_windows
from .config import PaAnoConfig
from .model import PaAnoLightningModule

METHOD_NAME = "PaAno"

# TSB-AD PaAno_PAI optimum (forks/TSB-AD/TSB_AD/models/HP_list.py); embed_dim
# names the last convolution width, and use_revin keeps normalization inside
# the model, so the scripts apply no external z-score.
DEFAULT_HP: Mapping[str, Any] = {
    "patch_size": 64,
    "num_iters": 200,
    "batch_size": 512,
    "lr": 1e-4,
    "top_k": 3,
    "triplet_margin": 0.5,
    "embed_dim": 64,
    "projection_dim": 256,
    "use_revin": True,
}


def build_config(
    input_features: int, hp: Mapping[str, Any] | None = None
) -> PaAnoConfig:
    """Merge ``hp`` overrides into ``DEFAULT_HP`` as one per-series config.

    ``original_default`` already matches the TSB-AD optimum apart from RevIN
    and the triplet margin, so it is the base; ``embed_dim`` replaces the
    last convolution width as in the original
    ``PatchEncoder(layers=[128, 256, 128, embed_dim])``.
    """
    merged = {**DEFAULT_HP, **(hp or {})}
    unknown = sorted(set(merged) - set(DEFAULT_HP))
    if unknown:
        supported = ", ".join(sorted(DEFAULT_HP))
        raise ValueError(f"unknown PaAno hp keys {unknown}; supported: {supported}")
    widths = PaAnoConfig.original_default().convolution_widths
    return replace(
        PaAnoConfig.original_default(input_features=input_features),
        patch_length=int(merged["patch_size"]),
        training_iterations=int(merged["num_iters"]),
        batch_size=int(merged["batch_size"]),
        learning_rate=float(merged["lr"]),
        top_k=int(merged["top_k"]),
        triplet_margin=float(merged["triplet_margin"]),
        projection_dim=int(merged["projection_dim"]),
        use_revin=bool(merged["use_revin"]),
        convolution_widths=(*widths[:-1], int(merged["embed_dim"])),
    )


class _AnchorBatches(Dataset):
    """One pre-drawn anchor batch per item; the patch tensor is shared."""

    def __init__(self, patches: Tensor, anchor_batches: list[Tensor]) -> None:
        super().__init__()
        self.patches = patches
        self.anchor_batches = anchor_batches

    def __len__(self) -> int:
        return len(self.anchor_batches)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.patches, self.anchor_batches[index]


def _sample_anchor_batches(
    n_patches: int, batch_size: int, iterations: int, seed: int
) -> list[Tensor]:
    """Pre-draw one shuffled anchor-index batch per training iteration."""
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randperm(n_patches, generator=generator)[:batch_size]
        for _ in range(iterations)
    ]


def _train_patches(
    train_data: ArrayLike, config: PaAnoConfig, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Return the train patches ``[P, C, L]`` and series ``[1, T, C]``."""
    values = np.ascontiguousarray(train_data, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(
            f"train_data must be one- or two-dimensional, got shape {values.shape}"
        )
    series = torch.from_numpy(values).to(device)[None]
    if series.shape[1] < config.patch_length + 1:
        raise ValueError(
            f"train length {series.shape[1]} yields fewer than two "
            f"{config.patch_length}-long patches; extend the train segment "
            "or lower patch_size"
        )
    windows = sliding_windows(series, config.patch_length)
    return windows.flatten(0, 1).transpose(1, 2).contiguous(), series


def train_series(
    train_data: ArrayLike,
    *,
    config: PaAnoConfig,
    device: str | torch.device = "cuda",
    seed: int = 2024,
) -> PaAnoLightningModule:
    """Train one series' encoder in memory and fit its normal memory bank.

    Runs ``config.training_iterations`` optimizer steps over stride-1 train
    patches; the Lightning module restores its best-iteration weights at
    train end, which then fit the memory bank. The returned module is
    scoring-ready via ``module.model.score`` on the concatenated full
    series, and nothing is persisted.
    """
    if not config.use_revin:
        # The non-RevIN protocol (forks/PaAno main.py) z-scores with train
        # statistics; that branch is not implemented here, so fail fast
        # instead of silently training on unnormalized data.
        raise ValueError("use_revin=False is not supported by this script")
    L.seed_everything(seed)
    target = torch.device(device)
    patches, series = _train_patches(train_data, config, target)
    batches = _AnchorBatches(
        patches,
        _sample_anchor_batches(
            patches.shape[0], config.batch_size, config.training_iterations, seed
        ),
    )
    module = PaAnoLightningModule(config)
    trainer = L.Trainer(
        accelerator="gpu" if target.type == "cuda" else "cpu",
        devices=1,
        max_epochs=1,
        max_steps=config.training_iterations,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    # batch_size=None feeds each collate call the single (patches, anchors)
    # tuple itself; an identity collate keeps it intact for _batch.
    trainer.fit(
        module,
        train_dataloaders=DataLoader(
            batches, batch_size=None, collate_fn=lambda item: item
        ),
    )
    module.model.fit_memory(series)
    return module


def _save_checkpoint(
    path: Path, module: PaAnoLightningModule, config: PaAnoConfig
) -> None:
    """Persist the config, encoder weights, and non-persistent memory bank."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(config),
            "state_dict": {
                key: value.detach().cpu()
                for key, value in module.model.state_dict().items()
            },
            "normal_memory": module.model.normal_memory.detach().cpu(),
        },
        path,
    )


def _train_segment(split: DatasetSplit, series_id: str) -> np.ndarray:
    """Return the series' normal train segment ``[T, C]``."""
    if split.train is None or series_id not in split.train:
        raise ValueError(f"PaAno requires a train segment for series {series_id!r}")
    return split.train[series_id].data


def train(
    dataset_dir: str | Path,
    output_root: str | Path = "benchmarks/PaAno",
    *,
    artifact: str | None = None,
    partition: str | None = "Eva",
    device: str = "cuda",
    seed: int = 2024,
    hp: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> None:
    """Train one checkpoint per series' train segment of one artifact pair.

    ``resume=True`` skips series whose ``checkpoints/<series_id>.pt`` already
    exists; ``resume=False`` retrains every series of the pair.
    """
    split = load_split(dataset_dir, artifact, partition=partition)
    unit = unit_dir(output_root, METHOD_NAME, split)
    for series_id in split.test.ids:
        ckpt_path = checkpoints_dir(unit) / f"{series_id}.pt"
        if resume and ckpt_path.is_file():
            continue
        train_data = _train_segment(split, series_id)
        config = build_config(train_data.shape[1], hp)
        module = train_series(train_data, config=config, device=device, seed=seed)
        _save_checkpoint(ckpt_path, module, config)
