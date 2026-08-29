"""Lightning training module for AxonAD."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import lightning as L
import torch
from torch import Tensor
from torch.nn import functional as F

from .._utils import ValidationEarlyStopping
from .config import AxonADConfig
from .model import AxonAD, AxonADLoss


class AxonADLightningModule(ValidationEarlyStopping, L.LightningModule):
    def __init__(self, config: AxonADConfig) -> None:
        super().__init__()
        self.config = config
        self.model = AxonAD(config)
        # The original early-stops on the validation reconstruction MSE.
        self._init_validation_early_stopping(
            monitor="val/reconstruction",
            patience=config.patience,
            min_delta=config.early_stopping_min_delta,
        )

    def forward(self, windows: Tensor):
        return self.model(windows)

    @staticmethod
    def _windows(batch: object) -> Tensor:
        if isinstance(batch, Tensor):
            windows = batch
        elif isinstance(batch, Mapping):
            windows = batch.get("windows")
        elif isinstance(batch, (tuple, list)) and batch:
            windows = batch[0]
        else:
            windows = None
        if not isinstance(windows, Tensor):
            raise TypeError("AxonAD batches must provide a windows tensor")
        return windows

    def _train_loss(self, windows: Tensor) -> AxonADLoss:
        return self.model.compute_loss(windows)

    def _validation_loss(self, windows: Tensor) -> AxonADLoss:
        output = self.model(windows)
        reconstruction = F.mse_loss(output.reconstruction, windows)
        representation = self.model._tail_representation_distance(output).mean()
        total = (
            torch.exp(-self.model.reconstruction_balance) * reconstruction
            + self.model.reconstruction_balance
            + torch.exp(-self.model.representation_balance) * representation
            + self.model.representation_balance
        )
        return AxonADLoss(total, reconstruction, representation)

    def _log(self, stage: str, windows: Tensor, losses: AxonADLoss) -> Tensor:
        for name in ("total", "reconstruction", "representation"):
            metric = "loss" if name == "total" else name
            self.log(
                f"{stage}/{metric}",
                getattr(losses, name),
                on_step=stage == "train" and name == "total",
                on_epoch=True,
                prog_bar=name == "total",
                sync_dist=True,
                batch_size=windows.shape[0],
            )
        return losses.total

    def training_step(self, batch: object, batch_idx: int) -> Tensor:
        windows = self._windows(batch)
        return self._log("train", windows, self._train_loss(windows))

    def validation_step(self, batch: object, batch_idx: int) -> Tensor:
        windows = self._windows(batch)
        return self._log("val", windows, self._validation_loss(windows))

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure: Any | None = None,
    ) -> None:
        optimizer.step(closure=optimizer_closure)
        self.model.update_target()

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | int | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        self.clip_gradients(
            optimizer,
            gradient_clip_val=self.config.gradient_clip_val,
            gradient_clip_algorithm="norm",
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
