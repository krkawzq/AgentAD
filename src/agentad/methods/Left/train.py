"""Lightning training module for Left."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Callable

import lightning as L
import torch
from torch import Tensor

from .._utils import ValidationEarlyStopping
from .config import LeftConfig
from .model import Left


class LeftLightningModule(ValidationEarlyStopping, L.LightningModule):
    def __init__(self, config: LeftConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Left(config)
        self._init_validation_early_stopping(
            monitor="val/loss",
            patience=config.patience,
        )

    def forward(self, series: Tensor):
        return self.model(series)

    @staticmethod
    def _series(batch: object) -> Tensor:
        if isinstance(batch, Tensor):
            series = batch
        elif isinstance(batch, Mapping):
            series = batch.get("series")
        elif isinstance(batch, (tuple, list)) and batch:
            series = batch[0]
        else:
            series = None
        if not isinstance(series, Tensor):
            raise TypeError("Left batches must provide a series tensor")
        return series

    def _step(self, batch: object, stage: str) -> Tensor:
        series = self._series(batch)
        output = self.model(series)
        if stage == "train":
            self.model.update_prototypes(series, output)
        losses = self.model.compute_loss_from_output(series, output)
        for name in ("total", "multiscale", "cycle", "consistency"):
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
        return self.model.score(self._series(batch))

    def _type2_schedule(self) -> Callable[[int], float]:
        milestones = (
            (2, 5e-5), (4, 1e-5), (6, 5e-6), (8, 1e-6),
            (10, 5e-7), (15, 1e-7), (20, 5e-8),
        )

        def scale(epoch: int) -> float:
            current = self.config.learning_rate
            for milestone, value in milestones:
                if epoch + 1 < milestone:
                    break
                current = value
            return current / self.config.learning_rate

        return scale

    def configure_optimizers(self) -> dict[str, object]:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.learning_rate
        )
        if self.config.learning_rate_schedule == "type1":
            scheduler: object = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=0.5
            )
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, self._type2_schedule()
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
