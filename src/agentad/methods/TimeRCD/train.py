"""Lightning training module for Time-RCD pretraining objectives."""

from __future__ import annotations

from collections.abc import Mapping

import lightning as L
import torch
from torch import Tensor

from .config import TimeRCDConfig
from .model import TimeRCD


class TimeRCDLightningModule(L.LightningModule):
    def __init__(self, config: TimeRCDConfig) -> None:
        super().__init__()
        self.config = config
        self.model = TimeRCD(config)
        if not config.train_encoder:
            for parameter in self.model.ts_encoder.parameters():
                parameter.requires_grad_(False)

    def forward(self, series: Tensor, valid_mask: Tensor | None = None):
        return self.model(series, valid_mask)

    @staticmethod
    def _batch(batch: object) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if isinstance(batch, Tensor):
            return batch, None, None
        if isinstance(batch, Mapping):
            series = batch.get("series", batch.get("time_series"))
            labels = batch.get("labels")
            masked = batch.get("masked_points", batch.get("mask"))
        elif isinstance(batch, (tuple, list)) and batch:
            series = batch[0]
            labels = batch[1] if len(batch) > 1 else None
            masked = batch[2] if len(batch) > 2 else None
        else:
            series = labels = masked = None
        if not isinstance(series, Tensor):
            raise TypeError("Time-RCD batches must provide a series tensor")
        if labels is not None and not isinstance(labels, Tensor):
            raise TypeError("Time-RCD labels must be a tensor")
        if masked is not None and not isinstance(masked, Tensor):
            raise TypeError("Time-RCD mask must be a tensor")
        return series, labels, masked

    def _step(self, batch: object, stage: str) -> Tensor:
        series, labels, masked = self._batch(batch)
        losses = self.model.compute_loss(
            series,
            labels=labels,
            masked_points=masked,
        )
        for name in ("total", "reconstruction", "classification"):
            metric = "loss" if name == "total" else name
            self.log(
                f"{stage}/{metric}",
                getattr(losses, name),
                on_step=stage == "train" and name == "total",
                on_epoch=True,
                prog_bar=name == "total",
                sync_dist=True,
                batch_size=series.shape[0],
            )
        return losses.total

    def training_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def predict_step(
        self, batch: object, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        series, _, _ = self._batch(batch)
        return self.model.score(series, batch_size=self.config.batch_size)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        return torch.optim.AdamW(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
