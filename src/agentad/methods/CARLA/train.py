"""Per-series two-stage CARLA training on one TSB-AD artifact (form A).

Protocol (``forks/CARLA`` pretext + classification entry points, MSL/SMD
settings): windows of 200 points at stride 1 z-scored per channel with the
train-segment statistics; stage one contrastively trains the projection
against injected sub-sequence anomalies, the pretext projection mines each
window's nearest/furthest neighbors, and stage two self-labels the windows
into ten clusters from those neighbors. Each series trains its own model and
stores both stages under ``checkpoints/<series_id>/``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, NamedTuple

import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ...benchmark import (
    atomic_write_file,
    checkpoints_dir,
    load_split,
    prepare_run,
    unit_dir,
)

from .._utils import sliding_windows
from .config import CARLAConfig
from .model import (
    CARLA,
    CARLAClassificationLightningModule,
    CARLAPretextLightningModule,
    inject_anomalies,
)

METHOD_NAME = "CARLA"

# The original pretext stage mines top-10 neighbors per window and the
# classification dataset then consumes ``num_neighbors`` entries per role
# (the five closest of the nearest, the five most distant of the furthest).
MINE_TOP_K = 10

# Validation windows for the classification stage, as a fraction of the train
# windows taken from the tail of the train prefix.
_VAL_TAIL_FRACTION = 0.1

DEFAULT_HP: Mapping[str, Any] = {
    "window_length": 200,
    "window_stride": 1,
    "mid_channels": 4,
    "projection_dim": 4,
    "clusters": 10,
    "cluster_heads": 1,
    "temperature": 0.4,
    "pretext_margin": 1.0,
    "margin_adjustment": 0.1,
    "noise_sigma": 0.01,
    "anomaly_min_fraction": 0.1,
    "anomaly_max_fraction": 0.9,
    "pretext_epochs": 30,
    "pretext_batch_size": 50,
    "pretext_learning_rate": 1e-3,
    "pretext_weight_decay": 0.01,
    "pretext_cosine_decay_rate": 0.01,
    "neighbors": 5,
    "classification_epochs": 50,
    "classification_batch_size": 50,
    "classification_learning_rate": 0.01,
    "classification_weight_decay": 0.001,
    "entropy_weight": 2.0,
    "inconsistency_weight": 0.0,
}


def build_config(
    input_features: int, hp: Mapping[str, Any] | None = None
) -> CARLAConfig:
    """Build the model config for a series with ``input_features`` channels.

    ``original_msl`` is the generic factory (window 200 at stride 1, every
    structural field at the original MSL/SMD value); ``hp`` overrides config
    fields by name on top of DEFAULT_HP.
    """
    base = replace(CARLAConfig.original_msl(), input_features=input_features)
    return replace(base, **{**DEFAULT_HP, **(hp or {})})


def fit_normalization(train_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std of the train segment; constant channels get std 1.

    Train and test windows share these statistics, as in the original MSL/SMD
    datasets.
    """
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    return mean, np.where(std == 0.0, 1.0, std)


def _min_train_length(config: CARLAConfig) -> int:
    """Shortest train segment yielding a full pretext batch and a minable bank."""
    per_stage = max(
        config.pretext_batch_size,
        (config.classification_batch_size + 1) // 2,
        MINE_TOP_K,
    )
    return config.window_length + per_stage - 1


def _trainer(max_epochs: int, device: torch.device) -> L.Trainer:
    return L.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )


def _cpu_state(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.detach().cpu() for key, value in state_dict.items()}


def _inject_anomalies_chunked(
    windows: Tensor, config: CARLAConfig, *, chunk_size: int = 2048
) -> Tensor:
    """Inject the synthetic sub-sequence anomalies once for the whole set.

    The original builds ``ts_ss_augment`` a single time per window so every
    epoch reuses the same negatives; chunking bounds temporary memory.
    """
    return torch.cat(
        [
            inject_anomalies(
                segment,
                min_fraction=config.anomaly_min_fraction,
                max_fraction=config.anomaly_max_fraction,
            )
            for segment in windows.split(chunk_size)
        ]
    )


def _pretext_positives(windows: Tensor, *, noise_sigma: float) -> Tensor:
    """Fixed pretext positives: one of the ten preceding windows, or noise.

    Mirrors the original AugmentedDataset, which draws each positive once at
    construction; the first ten windows fall back to additive noise.
    """
    count = windows.shape[0]
    positions = torch.arange(count, device=windows.device)
    offsets = torch.randint(1, 11, (count,), device=windows.device)
    positives = windows[(positions - offsets).clamp_min(0)]
    early = positions <= 10
    positives[early] = windows[early] + noise_sigma * torch.randn_like(windows[early])
    return positives


def _make_pretext_collate(
    bank: Tensor, positives: Tensor
) -> Callable[[list[Tensor]], dict[str, Tensor]]:
    """Batch pretext items as anchor/positive/injected windows of the bank."""
    anchor_count = positives.shape[0]

    def collate(items: list[Tensor]) -> dict[str, Tensor]:
        rows = torch.stack(list(items)).to(bank.device)
        return {
            "anchors": bank[rows],
            "positives": positives[rows],
            "injected_anomalies": bank[anchor_count + rows],
        }

    return collate


def _make_neighbor_collate(
    bank: Tensor, nearest: Tensor, furthest: Tensor
) -> Callable[[list[Tensor]], dict[str, Tensor]]:
    """Batch classification items as anchor/nearest/furthest windows."""

    def collate(items: list[Tensor]) -> dict[str, Tensor]:
        rows = torch.stack(list(items)).to(bank.device)
        return {
            "anchors": bank[rows],
            "nearest": bank[nearest[rows]],
            "furthest": bank[furthest[rows]],
        }

    return collate


def _pick_neighbors(
    nearest: Tensor, furthest: Tensor, *, num_neighbors: int
) -> tuple[Tensor, Tensor]:
    """Draw one neighbor per anchor from the closest and most distant pools.

    The original NeighborsDataset slices the mined top-k lists to
    ``num_neighbors`` per role and fixes one random draw for the whole run.
    """
    near_pool = nearest[:, :num_neighbors]
    far_pool = furthest[:, -num_neighbors:]
    columns = torch.randint(
        0, num_neighbors, (nearest.shape[0], 1), device=nearest.device
    )
    return near_pool.gather(1, columns)[:, 0], far_pool.gather(1, columns)[:, 0]


class CarlaArtifacts(NamedTuple):
    """Checkpoint artifacts of one series: pretext weights and mined neighbors."""

    pretext_state_dict: dict[str, Tensor]
    nearest: Tensor
    furthest: Tensor


class _IndexSeries(Dataset[Tensor]):
    """Row view over one 1-D index tensor for collate-based sampling."""

    def __init__(self, indices: Tensor) -> None:
        self._indices = indices

    def __len__(self) -> int:
        return self._indices.shape[0]

    def __getitem__(self, index: int) -> Tensor:
        return self._indices[index]


def _train_series(
    config: CARLAConfig, train_data: Tensor, *, device: torch.device
) -> tuple[CARLA, CarlaArtifacts]:
    """Fit both stages on one series; return the model and ckpt artifacts.

    ``train_data`` is the raw train segment: the original injects anomalies
    and draws pretext noise on raw windows and normalizes only afterwards
    with the train-segment statistics (``forks/CARLA/data/augment.py`` acts
    on ``ts_org``; the z-score lives in ``custom_dataset.py``).
    """
    if config.neighbors > MINE_TOP_K:
        raise ValueError("neighbors must not exceed MINE_TOP_K")
    model = CARLA(config).to(device)
    mean_array, std_array = fit_normalization(train_data.cpu().numpy())
    mean = torch.from_numpy(mean_array).to(device)
    std = torch.from_numpy(std_array).to(device)
    raw_windows = sliding_windows(
        train_data.to(device)[None], config.window_length, config.window_stride
    )[0]
    # Stage one. The original pretext entry point builds Adam without the
    # configured weight decay and keeps the final-epoch weights (its best
    # model is tracked by reference), so no pretext validation runs here.
    # Injection and the noise fallback act on the raw windows; anchors,
    # positives and the bank are normalized with the train statistics after.
    injected = _inject_anomalies_chunked(raw_windows, config)
    raw_bank = torch.cat((raw_windows, injected))
    del injected
    raw_positives = _pretext_positives(raw_windows, noise_sigma=config.noise_sigma)
    windows = ((raw_windows - mean) / std).float()
    bank = ((raw_bank - mean) / std).float()
    positives = ((raw_positives - mean) / std).float()
    pretext = CARLAPretextLightningModule(config, model)
    _trainer(config.pretext_epochs, device).fit(
        pretext,
        train_dataloaders=DataLoader(
            _IndexSeries(torch.arange(windows.shape[0])),
            batch_size=config.pretext_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=_make_pretext_collate(bank, positives),
        ),
    )
    pretext_state = _cpu_state(model.state_dict())
    del positives
    # Neighbor mining in the pretext projection space. Deviation from the
    # original: its faiss mining ranks the whole index against a single
    # random query vector, so every window shares one degenerate global
    # neighbor ordering (forks/CARLA/utils/repository.py); this migration
    # mines the true per-window top-10 the paper describes instead.
    nearest, furthest = model.mine_neighbors(bank, k=MINE_TOP_K)
    nearest_idx, furthest_idx = _pick_neighbors(
        nearest, furthest, num_neighbors=config.neighbors
    )
    # The original selects the classification checkpoint by best F1 on the
    # labeled test series — test leakage the benchmark must not reproduce.
    # Validation therefore uses unlabeled windows from the tail of the train
    # prefix; without window labels the module keeps the final weights (its
    # documented no-label rule) and the normal cluster is calibrated below.
    tail = max(1, round(windows.shape[0] * _VAL_TAIL_FRACTION))
    neighbor_collate = _make_neighbor_collate(bank, nearest_idx, furthest_idx)
    classification = CARLAClassificationLightningModule(config, model)
    _trainer(config.classification_epochs, device).fit(
        classification,
        train_dataloaders=DataLoader(
            _IndexSeries(torch.arange(bank.shape[0])),
            batch_size=config.classification_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=neighbor_collate,
        ),
        val_dataloaders=DataLoader(
            _IndexSeries(torch.arange(windows.shape[0] - tail, windows.shape[0])),
            batch_size=config.classification_batch_size,
            collate_fn=neighbor_collate,
        ),
    )
    # Majority (normal) cluster over the normal train windows; the original
    # counted the org+injected bank, whose injected pseudo anomalies would
    # bias the vote.
    model.calibrate_normal_clusters(windows)
    artifacts = CarlaArtifacts(
        pretext_state_dict=pretext_state,
        nearest=nearest.cpu(),
        furthest=furthest.cpu(),
    )
    return model, artifacts


def train_series(
    config: CARLAConfig,
    train_data: Tensor,
    *,
    device: str | torch.device = "cpu",
) -> CARLA:
    """Fit both CARLA stages on one series' raw train segment ``[T, C]``.

    In-memory single-series entry point for ``evaluate`` when no checkpoint
    exists; returns the model with the final classification weights and the
    normal cluster calibrated on the normalized train windows.
    """
    model, _ = _train_series(config, train_data, device=torch.device(device))
    return model


def _save_checkpoint(
    directory: Path,
    config: CARLAConfig,
    model: CARLA,
    artifacts: CarlaArtifacts,
) -> None:
    """Store the pretext weights, mined neighbors and calibrated model."""
    directory.mkdir(parents=True, exist_ok=True)
    config_payload = asdict(config)
    pretext_payload = {
        "config": config_payload,
        "state_dict": artifacts.pretext_state_dict,
    }
    atomic_write_file(
        directory / "pretext.pt",
        lambda handle: torch.save(pretext_payload, handle),
    )
    neighbor_payload = {"nearest": artifacts.nearest, "furthest": artifacts.furthest}
    atomic_write_file(
        directory / "neighbors.pt",
        lambda handle: torch.save(neighbor_payload, handle),
    )
    classification_payload = {
        "config": config_payload,
        "state_dict": _cpu_state(model.state_dict()),
        "normal_clusters": model.normal_clusters.detach().cpu(),
    }
    atomic_write_file(
        directory / "classification.pt",
        lambda handle: torch.save(classification_payload, handle),
    )


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
    """Train one two-stage CARLA per series under ``checkpoints/<series_id>/``.

    With ``resume=True`` series whose classification checkpoint already
    exists are skipped; series too short for one full pretext batch are
    reported and left without a checkpoint.
    """
    split = load_split(dataset_dir, artifact, partition=partition)
    if split.train is None:
        raise ValueError(
            f"CARLA is semisupervised; {split.unit_name} has no train split"
        )
    unit = unit_dir(output_root, METHOD_NAME, split)
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
        checkpoint = checkpoints_dir(unit) / series_id
        if resume and (checkpoint / "classification.pt").is_file():
            continue
        L.seed_everything(seed)
        train_data = np.asarray(split.train[series_id].data)
        config = build_config(train_data.shape[1], hp)
        minimum = _min_train_length(config)
        if train_data.shape[0] < minimum:
            print(f"skipping {series_id!r}: fewer than {minimum} train points")
            continue
        model, artifacts = _train_series(
            config,
            torch.from_numpy(train_data),
            device=torch.device(device),
        )
        _save_checkpoint(checkpoint, config, model, artifacts)
