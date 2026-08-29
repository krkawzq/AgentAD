"""Lightning training module for ScatterAD."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import lightning as L
import torch
from torch import Tensor

from .config import ScatterADConfig
from .model import ScatterAD


class ScatterADLightningModule(L.LightningModule):
    def __init__(self, config: ScatterADConfig) -> None:
        super().__init__()
        self.config = config
        self.model = ScatterAD(config)

    def forward(self, series: Tensor):
        return self.model(series)

    def _series(self, batch: object) -> Tensor:
        if isinstance(batch, Tensor):
            series = batch
        elif isinstance(batch, Mapping):
            series = batch.get("series")
        elif isinstance(batch, (tuple, list)) and batch:
            series = batch[0]
        else:
            series = None
        if not isinstance(series, Tensor):
            raise TypeError("ScatterAD batches must provide a series tensor")
        if series.shape[1] != self.config.window_length:
            raise ValueError(
                f"ScatterAD expects windows of length {self.config.window_length}"
            )
        return series

    def _step(self, batch: object, stage: str) -> Tensor:
        series = self._series(batch)
        losses = self.model.compute_loss(series)
        self.log(
            f"{stage}/loss",
            losses.total,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=series.shape[0],
        )
        for name in ("scattering", "temporal", "consistency"):
            self.log(
                f"{stage}/{name}",
                getattr(losses, name),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=series.shape[0],
            )
        return losses.total

    def training_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure: Any | None = None,
    ) -> None:
        optimizer.step(closure=optimizer_closure)
        self.model.update_target()

    def predict_step(
        self, batch: object, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        return self.model.score(self._series(batch))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
