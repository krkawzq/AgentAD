"""Lightning implementation of PaAno's iteration-based training procedure."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import lightning as L
import torch
from torch import Tensor

from .config import PaAnoConfig
from .model import PaAno


class PaAnoLightningModule(L.LightningModule):
    def __init__(self, config: PaAnoConfig | None = None) -> None:
        super().__init__()
        self.config = config or PaAnoConfig.original_default()
        self.model = PaAno(self.config)
        self._last_iteration_loss: float | None = None
        self._best_iteration_loss: float | None = None
        self._best_iteration_state: dict[str, Tensor] | None = None

    def forward(self, patches: Tensor) -> Tensor:
        return self.model(patches)

    @staticmethod
    def _batch(batch: object) -> tuple[Tensor, Tensor]:
        if isinstance(batch, Mapping):
            patches = batch.get("all_patches")
            indices = batch.get("anchor_indices")
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            patches, indices = batch
        else:
            raise TypeError("PaAno batches must provide all_patches and anchor_indices")
        if not isinstance(patches, Tensor) or not isinstance(indices, Tensor):
            raise TypeError("PaAno patches and anchor indices must be tensors")
        return patches, indices.long()

    def _step(self, batch: object, stage: str) -> Tensor:
        patches, indices = self._batch(batch)
        iteration = int(self.global_step) + 1 if stage == "train" else 1
        losses = self.model.compute_loss(
            patches,
            indices,
            iteration=iteration,
            total_iterations=self.config.training_iterations,
        )
        self.log(
            f"{stage}/loss",
            losses.total,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=indices.numel(),
        )
        self.log(
            f"{stage}/triplet",
            losses.triplet,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=indices.numel(),
        )
        self.log(
            f"{stage}/pretext",
            losses.pretext,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=indices.numel(),
        )
        if stage == "train":
            self._last_iteration_loss = float(losses.total)
        return losses.total

    def training_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def on_train_batch_end(self, outputs: object, batch: object, batch_idx: int) -> None:
        # The original compares the iteration loss after optimizer.step() and
        # deep-copies the post-update weights, so the snapshot lags the loss
        # by one update.
        value = self._last_iteration_loss
        if value is not None and (
            self._best_iteration_loss is None or value < self._best_iteration_loss
        ):
            self._best_iteration_loss = value
            self._best_iteration_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in self.state_dict().items()
            }
        self._last_iteration_loss = None

    def on_train_end(self) -> None:
        if self._best_iteration_state is not None:
            self.load_state_dict(self._best_iteration_state)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        final = self.config.final_learning_rate_factor
        total = self.config.training_iterations

        def schedule(step: int) -> float:
            progress = min(step + 1, total) / total
            return final + (1 - final) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
