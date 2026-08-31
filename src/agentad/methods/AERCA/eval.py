"""Dataset-level AERCA scoring for TSB-AD artifacts (form A, spec v1)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from agentad.benchmark import (
    checkpoints_dir,
    has_score,
    load_split,
    save_score,
    unit_dir,
    write_metrics,
)

from .config import AERCAConfig
from .model import AERCA
from .train import METHOD_NAME, TrainedAERCA, train_dataset


def _check_scores(scores: np.ndarray, length: int) -> None:
    """Raise unless the scores cover the full series with finite values."""
    if scores.shape != (length,):
        raise ValueError(f"expected {length} scores, got shape {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")


def _load_trained(checkpoint: Path) -> TrainedAERCA:
    payload = torch.load(checkpoint, map_location="cpu")
    config = AERCAConfig(**payload["config"])
    model = AERCA(config)
    model.load_state_dict(payload["state_dict"])
    return TrainedAERCA(
        model=model,
        config=config,
        mean=payload["mean"].numpy(),
        std=payload["std"].numpy(),
    )


def _load_or_train(
    checkpoint: Path,
    series: Sequence[np.ndarray],
    *,
    device: str,
    seed: int,
    hp: Mapping[str, Any] | None,
    resume: bool,
) -> TrainedAERCA:
    """Prefer the stored checkpoint; otherwise train in memory without saving."""
    if resume and checkpoint.is_file():
        return _load_trained(checkpoint)
    return train_dataset(series, device=device, seed=seed, hp=hp)


def evaluate(
    dataset_dir: str | Path,
    output_root: str | Path = "benchmarks/AERCA",
    *,
    artifact: str | None = None,
    partition: str | None = "Eva",
    device: str = "cuda",
    seed: int = 2024,
    hp: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Score every test series of one artifact over its full train+test length.

    Loads or trains the dataset-level model once, then writes one score
    array per pending series and returns the per-series metrics table.
    """
    split = load_split(dataset_dir, artifact, partition=partition)
    if split.train is None:
        raise FileNotFoundError(
            f"AERCA needs a normal train prefix: {split.unit_name}"
        )
    unit = unit_dir(output_root, METHOD_NAME, split)
    pending = [
        series_id
        for series_id in split.test.ids
        if not (resume and has_score(unit, series_id))
    ]
    if pending:
        trained = _load_or_train(
            checkpoints_dir(unit) / f"{split.name}.pt",
            [split.train[series_id].data for series_id in split.train.ids],
            device=device,
            seed=seed,
            hp=hp,
            resume=resume,
        )
        target = torch.device(device)
        model = trained.model.to(target)
        for series_id in pending:
            full = np.concatenate(
                [split.train[series_id].data, split.test[series_id].data]
            ).astype(np.float64)
            normalized = (full - trained.mean) / trained.std
            x = torch.from_numpy(normalized).to(torch.float32)
            scores = model.score(x[None].to(target))[0].cpu().numpy()
            _check_scores(scores, len(full))
            save_score(unit, series_id, scores)
    return write_metrics(split, unit)


if __name__ == "__main__":
    evaluate(
        dataset_dir="data/processed/tsb-ad/TSB-AD-M/CATSv2",
        output_root="benchmarks/AERCA",
        device="cuda",
    )
