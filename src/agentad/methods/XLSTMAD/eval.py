"""Form A evaluation for xLSTMAD under the TSB-AD benchmark protocol."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch

from ...benchmark import (
    checkpoints_dir,
    has_score,
    load_split,
    prepare_run,
    save_score,
    unit_dir,
    write_metrics,
)

from .config import XLSTMADConfig
from .model import XLSTMADLightningModule
from .train import METHOD_NAME, _zscore, build_config, train_series


def load_series_checkpoint(
    path: str | Path,
    *,
    device: str = "cuda",
) -> XLSTMADLightningModule:
    """Rebuild one trained module from a ``train`` checkpoint."""
    payload = torch.load(path, map_location="cpu")
    module = XLSTMADLightningModule(XLSTMADConfig(**payload["config"]))
    module.load_state_dict(payload["state_dict"])
    return module.to(device)


def _score_full(
    module: XLSTMADLightningModule,
    full: np.ndarray,
    device: str,
) -> np.ndarray:
    """Score the concatenated train+test series on the full length.

    The original ``decision_function`` builds a fresh
    ``ReconstructDataset(normalize=True)``, so the full series is z-scored
    with its own statistics, independent of the training normalization.
    Scores keep the migrated zero padding on the first ``window - 1`` points.
    """
    series = torch.from_numpy(np.ascontiguousarray(_zscore(full), dtype=np.float32))
    scores = module.model.score(series.unsqueeze(0).to(device))[0]
    return scores.detach().cpu().numpy().astype(np.float64)


def _check_scores(scores: np.ndarray, length: int, series_id: str) -> None:
    if scores.shape != (length,):
        raise ValueError(
            f"series {series_id!r}: score length {scores.shape} != {length}"
        )
    if not np.isfinite(scores).all():
        raise ValueError(f"series {series_id!r}: non-finite anomaly scores")


def evaluate(
    dataset_dir: str | Path,
    output_root: str | Path = "outputs/benchmark",
    *,
    artifact: str | None = None,
    results_root: str | Path = "results/benchmark",
    partition: str | None = "Eva",
    device: str = "cuda",
    seed: int = 2024,
    hp: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Score every series of one dataset unit on its full length and write
    ``metrics.csv``.

    Existing checkpoints are loaded; a missing checkpoint triggers a live
    single-series training without saving. Existing score files are skipped
    when ``resume`` is set. Returns the per-series metric table.
    """
    split = load_split(dataset_dir, artifact, partition=partition)
    if split.train is None:
        raise ValueError(f"xLSTMAD needs a train archive: {split.unit_name}")
    unit = unit_dir(output_root, METHOD_NAME, split)
    results = unit_dir(results_root, METHOD_NAME, split)
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
    for series_id in split.test.ids:
        if resume and has_score(unit, series_id):
            continue
        L.seed_everything(seed)
        train_item = split.train[series_id]
        test_item = split.test[series_id]
        full = np.concatenate([train_item.data, test_item.data])
        config = build_config(full.shape[1], device=device, hp=hp)
        ckpt_path = checkpoints_dir(unit) / f"{series_id}.pt"
        if resume and ckpt_path.is_file():
            module = load_series_checkpoint(ckpt_path, device=device)
        else:
            module = train_series(train_item.data, config, device=device)
        scores = _score_full(module, full, device)
        _check_scores(scores, len(full), series_id)
        save_score(unit, series_id, scores)
    return write_metrics(split, unit, results)
