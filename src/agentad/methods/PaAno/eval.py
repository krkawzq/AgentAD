"""Per-series PaAno scoring for TSB-AD artifacts (form A: train.py + eval.py).

Scores the train+test concatenation with the trained encoder and its normal
memory bank, as the original concatenates before scoring
(``forks/PaAno/main.py``). A missing checkpoint is trained on the fly via
``train_series`` and stays in memory.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike

from agentad.benchmark import (
    DatasetSplit,
    checkpoints_dir,
    has_score,
    load_split,
    save_score,
    unit_dir,
    write_metrics,
)

from .config import PaAnoConfig
from .model import PaAnoLightningModule
from .train import METHOD_NAME, build_config, train_series


def _full_series(split: DatasetSplit, series_id: str) -> np.ndarray:
    """Concatenate the train prefix and test suffix into one ``[T, C]`` array."""
    if split.train is None or series_id not in split.train:
        raise ValueError(f"PaAno requires a train segment for series {series_id!r}")
    train_item = split.train[series_id]
    test_item = split.test[series_id]
    return np.concatenate([train_item.data, test_item.data], axis=0)


def _module_from_checkpoint(
    path: Path, device: str | torch.device
) -> PaAnoLightningModule:
    """Rebuild a scoring-ready module from one stored checkpoint.

    The stored config wins over the caller config, so scores stay
    reproducible when hp changes after training; the memory bank is a
    non-persistent buffer and is restored from its dedicated tensor.
    """
    payload = torch.load(path, map_location="cpu")
    module = PaAnoLightningModule(PaAnoConfig(**payload["config"]))
    module.model.load_state_dict(payload["state_dict"])
    module.model.normal_memory = payload["normal_memory"].to(device)
    return module.to(torch.device(device))


def _load_or_train(
    split: DatasetSplit,
    unit: Path,
    series_id: str,
    *,
    config: PaAnoConfig,
    device: str,
    seed: int,
    resume: bool,
) -> PaAnoLightningModule:
    """Load the stored checkpoint when present, otherwise train in memory."""
    ckpt_path = checkpoints_dir(unit) / f"{series_id}.pt"
    if resume and ckpt_path.is_file():
        return _module_from_checkpoint(ckpt_path, device)
    train_data = split.train[series_id].data
    return train_series(train_data, config=config, device=device, seed=seed)


def _score_full(
    module: PaAnoLightningModule, full: ArrayLike, device: str | torch.device
) -> np.ndarray:
    """Score the concatenated full series and return a one-dimensional array."""
    values = np.ascontiguousarray(full, dtype=np.float32)
    x = torch.from_numpy(values).to(device)[None]
    return module.model.score(x)[0].cpu().numpy()


def _check_scores(scores: np.ndarray, length: int) -> None:
    """Fail loudly unless every full-length point carries a finite score."""
    if scores.shape != (length,):
        raise ValueError(
            f"expected one score per point ({length}), got shape {scores.shape}"
        )
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")


def evaluate(
    dataset_dir: str | Path,
    output_root: str | Path = "benchmarks/PaAno",
    *,
    artifact: str | None = None,
    partition: str | None = "Eva",
    device: str = "cuda",
    seed: int = 2024,
    hp: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Score each series on its train+test concatenation, one row per series.

    Series with a stored score are skipped when ``resume`` is set; a missing
    checkpoint is trained in memory without persisting it. Writes
    ``<unit>/metrics.csv`` and returns the per-series metric frame.
    """
    split = load_split(dataset_dir, artifact, partition=partition)
    unit = unit_dir(output_root, METHOD_NAME, split)
    for series_id in split.test.ids:
        if resume and has_score(unit, series_id):
            continue
        L.seed_everything(seed)
        full = _full_series(split, series_id)
        config = build_config(full.shape[1], hp)
        module = _load_or_train(
            split,
            unit,
            series_id,
            config=config,
            device=device,
            seed=seed,
            resume=resume,
        )
        scores = _score_full(module, full, device)
        _check_scores(scores, full.shape[0])
        save_score(unit, series_id, scores)
    return write_metrics(split, unit)


if __name__ == "__main__":
    evaluate(
        dataset_dir="data/processed/tsb-ad/TSB-AD-M/CATSv2",
        output_root="benchmarks/PaAno",
        device="cuda",
    )
