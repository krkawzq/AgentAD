"""Result layout under ``<output_root>/<Method>/<source>/<dataset>``.

Every method owns one directory per evaluated dataset unit::

    <output_root>/<Method>/<source>/<dataset>/scores/<series_id>.npy
    <output_root>/<Method>/<source>/<dataset>/checkpoints/...
    <output_root>/<Method>/<source>/<dataset>/metrics.csv

Scores are one flat ``float64`` array per series, as long as the
concatenated train+test full length. ``has_score`` is the resume check:
methods skip series whose score file already exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .datasets import DatasetSplit

__all__ = [
    "checkpoints_dir",
    "has_score",
    "load_score",
    "method_root",
    "metrics_path",
    "save_score",
    "score_path",
    "scores_dir",
    "unit_dir",
]


def method_root(output_root: str | Path, method: str) -> Path:
    """Directory of one method, e.g. ``outputs/benchmark/xLSTMAD``."""
    return Path(output_root) / method


def unit_dir(output_root: str | Path, method: str, split: DatasetSplit) -> Path:
    """Directory of one method on one dataset unit, e.g. ``outputs/benchmark/xLSTMAD/TSB-AD-M/CATSv2``."""
    return method_root(output_root, method) / split.source / split.name


def scores_dir(unit: str | Path) -> Path:
    return Path(unit) / "scores"


def score_path(unit: str | Path, series_id: str) -> Path:
    return scores_dir(unit) / f"{series_id}.npy"


def has_score(unit: str | Path, series_id: str) -> bool:
    return score_path(unit, series_id).is_file()


def save_score(unit: str | Path, series_id: str, scores: np.ndarray) -> None:
    """Persist one per-series score array, flattened to one dimension."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    path = score_path(unit, series_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values)


def load_score(unit: str | Path, series_id: str) -> np.ndarray:
    return np.load(score_path(unit, series_id))


def checkpoints_dir(unit: str | Path) -> Path:
    """Directory for method-owned checkpoints of one dataset unit."""
    return Path(unit) / "checkpoints"


def metrics_path(unit: str | Path) -> Path:
    return Path(unit) / "metrics.csv"
