"""形态 A 评估：逐序列加载或现场训练 KAN-AD，对拼接全长打分（TSB-AD 口径）."""

from __future__ import annotations

from collections.abc import Mapping
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

from .config import KANADConfig
from .model import KANADLightningModule
from .train import METHOD_NAME, _resolve_config, difference_standardize, train_series


def _load_module(checkpoint: Path) -> KANADLightningModule:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    module = KANADLightningModule(KANADConfig(**payload["config"]))
    module.load_state_dict(payload["state_dict"])
    return module


def _load_or_train(
    train_data: np.ndarray,
    *,
    config: KANADConfig,
    checkpoint: Path,
    device: str,
) -> KANADLightningModule:
    if checkpoint.is_file():
        return _load_module(checkpoint)
    # 缺 ckpt 时现场训练单序列，模型只在内存传递、不落盘。
    return train_series(train_data, config=config, device=device)


def _score_full_length(
    module: KANADLightningModule, z_full: np.ndarray, *, device: str
) -> np.ndarray:
    series = (
        torch.from_numpy(np.ascontiguousarray(z_full))
        .to(torch.float32)[None]
        .to(device)
    )
    scores = module.model.score(series)[0].cpu().numpy()
    # np.diff 丢弃原始首点（差分点 k 对应原始点 k+1），迁移版 score 又把
    # 无上下文的前缀 0 pad 在差分域头部；差分损失的这 1 个点因此在头部
    # 补 0（prepend），保证最终 scores[i] 对齐原始点 i。
    return np.concatenate(([0.0], scores))


def _check_scores(scores: np.ndarray, length: int, series_id: str) -> None:
    if scores.shape != (length,):
        raise ValueError(
            f"series {series_id!r}: score length {len(scores)} != full length {length}"
        )
    if not np.isfinite(scores).all():
        raise ValueError(f"series {series_id!r}: non-finite anomaly scores")


def evaluate(
    dataset_dir: str | Path,
    output_root: str | Path = f"benchmarks/{METHOD_NAME}",
    *,
    artifact: str | None = None,
    partition: str | None = "Eva",
    device: str = "cuda",
    seed: int = 2024,
    hp: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Score each test series on the concatenated full length; return metrics."""
    split = load_split(dataset_dir, artifact, partition=partition)
    if split.train is None:
        raise ValueError(
            f"KAN-AD scores the concatenated full length and requires a train "
            f"split in {dataset_dir}"
        )
    unit = unit_dir(output_root, METHOD_NAME, split)
    config = _resolve_config(hp)
    ckpt_dir = checkpoints_dir(unit)
    for series_id in split.test.ids:
        if resume and has_score(unit, series_id):
            continue
        if series_id not in split.train:
            raise KeyError(f"no train segment for series {series_id!r}")
        L.seed_everything(seed)
        train_data = split.train[series_id].data
        full = np.concatenate([train_data, split.test[series_id].data], axis=0)
        _, z_full = difference_standardize(train_data, full)
        module = _load_or_train(
            train_data,
            config=config,
            checkpoint=ckpt_dir / f"{series_id}.pt",
            device=device,
        )
        module.to(torch.device(device))
        scores = _score_full_length(module, z_full, device=device)
        _check_scores(scores, len(full), series_id)
        save_score(unit, series_id, scores)
    return write_metrics(split, unit)


if __name__ == "__main__":
    evaluate(
        dataset_dir="data/processed/tsb-ad/TSB-AD-M/CATSv2",
        output_root=f"benchmarks/{METHOD_NAME}",
        device="cuda",
    )
