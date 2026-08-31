"""Zero-shot (form B) and fine-tuned scoring entries for TSPulse.

``evaluate_zero_shot`` scores the concatenated full length with the official
pretrained checkpoint (TSB-AD ``TSPulse_ZS``); ``evaluate_finetune`` reuses
the per-series checkpoints written by :mod:`.train` (TSB-AD ``TSPulse_FT``),
fine-tuning in memory when a checkpoint is missing and falling back to
zero-shot scoring for series the short-series rules skip. Series shorter
than ``context_length + 1`` are skipped without a score, as in the original
TSB-AD protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import lightning as L
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

from .config import TSPulseConfig
from .model import TSPulseFineTune, TSPulseZeroShot
from .train import (
    METHOD_NAME,
    PRETRAINED_PATH,
    _resolve_config as _resolve_finetune_config,
    train_series,
)

ZERO_SHOT_NAME = "TSPulse-ZS"

# TSB-AD Optimal HP (HP_list.py 'TSPulse_ZS') plus the reference
# ``run_TSPulse_ZS`` batch size; ``batch_size`` only bounds score chunking.
DEFAULT_HP: Mapping[str, Any] = {
    "win_size": 96,
    "prediction_mode": "time",
    "batch_size": 256,
}

_PREDICTION_MODES: Mapping[str, tuple[str, ...]] = {
    "time": ("time",),
    "frequency": ("frequency",),
    "forecast": ("forecast",),
}

_CONFIG_FIELDS = frozenset(field.name for field in fields(TSPulseConfig))


def _resolve_zero_shot_config(
    input_features: int, hp: Mapping[str, Any] | None
) -> tuple[TSPulseConfig, int]:
    """Build the zero-shot config and the scoring batch size from ``hp``."""
    merged = {**DEFAULT_HP, **(hp or {})}
    mode = merged.pop("prediction_mode")
    batch_size = int(merged.pop("batch_size"))
    if isinstance(mode, str):
        try:
            mode = _PREDICTION_MODES[mode]
        except KeyError:
            raise ValueError(f"unknown TSPulse prediction mode: {mode!r}") from None
    overrides: dict[str, Any] = {
        "aggregation_length": int(merged.pop("win_size")),
        "score_modes": tuple(mode),
    }
    for key, value in merged.items():
        if key not in _CONFIG_FIELDS:
            raise ValueError(f"unknown TSPulse zero-shot hp: {key!r}")
        overrides[key] = value
    config = TSPulseConfig.original_tsb_zero_shot(input_features=input_features)
    return replace(config, **overrides), batch_size


def _score_full(
    model: TSPulseFineTune | TSPulseZeroShot,
    full: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Score the concatenated train+test length; whole-series z-score is built in."""
    tensor = torch.from_numpy(np.ascontiguousarray(full, dtype=np.float32))[None]
    scores = model.score(tensor.to(device), batch_size=batch_size)[0].cpu().numpy()
    if scores.shape != (tensor.shape[1],) or not np.isfinite(scores).all():
        raise ValueError("TSPulse scores must be finite and match the full length")
    return scores


def _load_finetuned(
    unit: str | Path, series_id: str, device: str
) -> TSPulseFineTune | None:
    """Rebuild one fine-tuned model from its checkpoint, or ``None`` if absent."""
    path = checkpoints_dir(unit) / f"{series_id}.pt"
    if not path.is_file():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = TSPulseFineTune.from_pretrained(
        TSPulseConfig(**checkpoint["config"]),
        model_name_or_path=PRETRAINED_PATH,
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)


def evaluate_zero_shot(
    dataset_dir: str | Path,
    output_root: str | Path = "benchmarks/TSPulse-ZS",
    *,
    artifact: str | None = None,
    partition: str | None = "Eva",
    device: str = "cuda",
    seed: int = 2024,
    hp: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Score every series of one artifact pair with the official pretrained model."""
    split = load_split(dataset_dir, artifact, partition=partition)
    unit = unit_dir(output_root, ZERO_SHOT_NAME, split)
    config, batch_size = _resolve_zero_shot_config(split.test.n_features, hp)
    L.seed_everything(seed)
    model = TSPulseZeroShot.from_pretrained(
        config, model_name_or_path=PRETRAINED_PATH
    ).to(device)
    for series_id in split.test.ids:
        if resume and has_score(unit, series_id):
            continue
        train_item = split.train[series_id] if split.train else None
        test_item = split.test[series_id]
        full = (
            np.concatenate([train_item.data, test_item.data])
            if train_item is not None
            else test_item.data
        )
        if full.shape[0] < config.context_length + 1:
            print(
                f"skipping {series_id!r}: {full.shape[0]} points < "
                f"context_length + 1 ({config.context_length + 1})"
            )
            continue
        save_score(
            unit, series_id, _score_full(model, full, device=device, batch_size=batch_size)
        )
    return write_metrics(split, unit)


def evaluate_finetune(
    dataset_dir: str | Path,
    output_root: str | Path = "benchmarks/TSPulse-FT",
    *,
    artifact: str | None = None,
    partition: str | None = "Eva",
    device: str = "cuda",
    seed: int = 2024,
    hp: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Score every series with its fine-tuned checkpoint, fine-tuning in memory when absent."""
    split = load_split(dataset_dir, artifact, partition=partition)
    unit = unit_dir(output_root, METHOD_NAME, split)
    config = _resolve_finetune_config(split.test.n_features, hp)
    zero_shot: TSPulseZeroShot | None = None
    for series_id in split.test.ids:
        if resume and has_score(unit, series_id):
            continue
        L.seed_everything(seed)
        train_item = split.train[series_id] if split.train else None
        test_item = split.test[series_id]
        full = (
            np.concatenate([train_item.data, test_item.data])
            if train_item is not None
            else test_item.data
        )
        if full.shape[0] < config.context_length + 1:
            print(
                f"skipping {series_id!r}: {full.shape[0]} points < "
                f"context_length + 1 ({config.context_length + 1})"
            )
            continue
        model = _load_finetuned(unit, series_id, device)
        if model is None and train_item is not None:
            model = train_series(train_item.data, device=device, seed=seed, hp=hp)
        if model is None:
            # Zero-shot fallback applies only to series of at least
            # context_length + 1 points; shorter series are skipped above
            # and produce no score at all.
            if zero_shot is None:
                zero_shot = TSPulseZeroShot.from_pretrained(
                    config, model_name_or_path=PRETRAINED_PATH
                ).to(device)
            model = zero_shot
        save_score(
            unit,
            series_id,
            _score_full(
                model, full, device=device, batch_size=config.finetune_batch_size
            ),
        )
    return write_metrics(split, unit)


if __name__ == "__main__":
    evaluate_zero_shot(
        dataset_dir="data/processed/tsb-ad/TSB-AD-M/CATSv2",
        output_root="benchmarks/TSPulse-ZS",
        device="cuda",
    )
