"""Lightning training module for the official xLSTMAD reconstruction objective."""

from __future__ import annotations

from collections.abc import Mapping

import lightning as L
import torch
from torch import Tensor

from .config import XLSTMADConfig
from .model import XLSTMAD


class XLSTMADLightningModule(L.LightningModule):
    def __init__(self, config: XLSTMADConfig) -> None:
        super().__init__()
        self.config = config
        self.model = XLSTMAD(config)

    def forward(self, windows: Tensor):
        return self.model(windows)

    @staticmethod
    def _batch(batch: object) -> tuple[Tensor, Tensor | None]:
        if isinstance(batch, Tensor):
            return batch, None
        if isinstance(batch, Mapping):
            windows = batch.get("windows", batch.get("series"))
            labels = batch.get("labels")
        elif isinstance(batch, (tuple, list)) and batch:
            windows = batch[0]
            labels = batch[1] if len(batch) > 1 else None
        else:
            windows = labels = None
        if not isinstance(windows, Tensor):
            raise TypeError("xLSTMAD batches must provide windows")
        if labels is not None and not isinstance(labels, Tensor):
            raise TypeError("xLSTMAD labels must be tensors when provided")
        return windows, labels

    def _step(self, batch: object, stage: str) -> Tensor:
        windows, _ = self._batch(batch)
        loss = self.model.compute_loss(windows).total
        self.log(
            f"{stage}/loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=windows.shape[0],
        )
        return loss

    def training_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def predict_step(
        self, batch: object, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        windows, _ = self._batch(batch)
        return self.model.window_score(windows)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
