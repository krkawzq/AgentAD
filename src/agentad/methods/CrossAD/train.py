"""Lightning training module for CrossAD."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Callable

import lightning as L
import torch
from torch import Tensor

from .._utils import ValidationEarlyStopping
from .config import CrossADConfig
from .model import CrossAD


class CrossADLightningModule(ValidationEarlyStopping, L.LightningModule):
    def __init__(self, config: CrossADConfig | None = None) -> None:
        super().__init__()
        self.config = config or CrossADConfig.original_msl()
        self.model = CrossAD(self.config)
        # The original stopper treats val_loss <= best as an improvement.
        self._init_validation_early_stopping(
            monitor="val/loss",
            patience=self.config.patience,
            ties_improve=True,
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
            raise TypeError("CrossAD batches must provide a series tensor")
        return series

    def _step(self, batch: object, stage: str) -> Tensor:
        series = self._series(batch)
        losses = self.model.compute_loss(series)
        for name in ("total", "reconstruction", "context"):
            metric = "loss" if name == "total" else name
            self.log(
                f"{stage}/{metric}",
                getattr(losses, name),
                on_step=stage == "train" and name == "total",
                on_epoch=True,
                prog_bar=name == "total",
                sync_dist=True,
                # batch_size=1 makes Lightning's epoch reduction an unweighted
                # mean over batches, matching the original np.average(val_loss).
                batch_size=series.shape[0] if stage == "train" else 1,
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
            (2, 5e-5),
            (4, 1e-5),
            (6, 5e-6),
            (8, 1e-6),
            (10, 5e-7),
            (15, 1e-7),
            (20, 5e-8),
        )

        def scale(epoch: int) -> float:
            # The original applies milestone m at the end of epoch m-1, so
            # epoch m is the first one running with the milestone value.
            current = self.config.learning_rate
            for milestone, value in milestones:
                if epoch < milestone:
                    break
                current = value
            return current / self.config.learning_rate

        return scale

    @staticmethod
    def _type1_schedule() -> Callable[[int], float]:
        # The original halves the learning rate only from the third epoch on:
        # it calls _adjust_learning_rate(epoch + 1) after each epoch, and the
        # first call keeps the base rate for the second epoch.
        def scale(epoch: int) -> float:
            return 0.5 ** max(0, epoch - 1)

        return scale

    def configure_optimizers(self) -> dict[str, object]:
        optimizer_type = (
            torch.optim.AdamW if self.config.optimizer == "adamw" else torch.optim.Adam
        )
        optimizer = optimizer_type(
            self.parameters(), lr=self.config.learning_rate
        )
        if self.config.learning_rate_schedule == "type1":
            scheduler: object = torch.optim.lr_scheduler.LambdaLR(
                optimizer, self._type1_schedule()
            )
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, self._type2_schedule()
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
