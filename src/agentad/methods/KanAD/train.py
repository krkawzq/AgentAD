"""Lightning training module for the original KAN-AD objective."""

from __future__ import annotations

from typing import Any, Mapping

import lightning as L
import torch
from torch import Tensor

from .._utils import ValidationEarlyStopping
from .config import KANADConfig
from .model import KANAD


class KANADLightningModule(ValidationEarlyStopping, L.LightningModule):
    def __init__(self, config: KANADConfig | None = None) -> None:
        super().__init__()
        self.config = config or KANADConfig.original_default()
        self.model = KANAD(self.config)
        # The original constructs EarlyStoppingTorch without a save path,
        # treats val_loss == best - delta as an improvement, and scores with
        # the final weights.
        self._init_validation_early_stopping(
            monitor="val/loss",
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
            ties_improve=True,
            restore_best=False,
        )

    def forward(self, windows: Tensor):
        return self.model(windows)

    @staticmethod
    def _batch(batch: object) -> tuple[Tensor, Tensor]:
        if isinstance(batch, Mapping):
            windows, targets = batch.get("windows"), batch.get("targets")
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            windows, targets = batch
        else:
            raise TypeError("KAN-AD batches must provide windows and targets")
        if not isinstance(windows, Tensor) or not isinstance(targets, Tensor):
            raise TypeError("KAN-AD windows and targets must be tensors")
        return windows, targets

    def _step(self, batch: object, stage: str) -> Tensor:
        windows, targets = self._batch(batch)
        loss = self.model.compute_loss(windows, targets)
        self.log(
            f"{stage}/loss",
            loss.total,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            # batch_size=1 makes Lightning's epoch reduction an unweighted
            # mean over batches, matching the original's
            # valid_loss / len(valid_loader) validation average.
            batch_size=windows.shape[0] if stage == "train" else 1,
        )
        return loss.total

    def training_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def predict_step(
        self, batch: object, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        windows, targets = self._batch(batch)
        return self.model.window_score(windows, targets)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.scheduler_step_size,
            gamma=self.config.scheduler_gamma,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
