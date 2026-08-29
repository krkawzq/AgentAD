"""Lightning training module for AERCA."""

from __future__ import annotations

from collections.abc import Mapping

import lightning as L
import torch
from torch import Tensor

from .._utils import ValidationEarlyStopping
from .config import AERCAConfig
from .model import AERCA, AERCALoss


class AERCALightningModule(ValidationEarlyStopping, L.LightningModule):
    def __init__(self, config: AERCAConfig) -> None:
        super().__init__()
        self.config = config
        self.model = AERCA(config)
        self._init_validation_early_stopping(
            monitor="val/loss",
            patience=config.patience,
        )

    def forward(self, series: Tensor, *, include_current_residual: bool = True):
        return self.model(
            series, include_current_residual=include_current_residual
        )

    @staticmethod
    def _series(batch: object) -> Tensor:
        if isinstance(batch, Tensor):
            return batch
        if isinstance(batch, Mapping):
            value = batch.get("series")
        elif isinstance(batch, (tuple, list)) and batch:
            value = batch[0]
        else:
            value = None
        if not isinstance(value, Tensor):
            raise TypeError("AERCA batches must provide a series tensor")
        return value

    def _step(self, batch: object, stage: str) -> Tensor:
        series = self._series(batch)
        losses: AERCALoss = self.model.compute_loss(series)
        self.log(
            f"{stage}/loss",
            losses.total,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=series.shape[0],
        )
        for name in (
            "reconstruction",
            "encoder_sparsity",
            "decoder_sparsity",
            "encoder_smoothness",
            "decoder_smoothness",
            "kl_divergence",
        ):
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

    def predict_step(
        self, batch: object, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        return self.model.score(self._series(batch))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
