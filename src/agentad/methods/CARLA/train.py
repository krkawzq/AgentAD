"""Independent Lightning modules for CARLA's two published training stages."""

from __future__ import annotations

from collections.abc import Mapping

import lightning as L
import torch
from torch import Tensor

from .config import CARLAConfig
from .model import CARLA, inject_anomalies


class CARLAPretextLightningModule(L.LightningModule):
    def __init__(self, config: CARLAConfig, model: CARLA | None = None) -> None:
        super().__init__()
        self.config = config
        self.model = model or CARLA(config)
        self._previous_loss: Tensor | None = None

    def forward(self, windows: Tensor) -> Tensor:
        return self.model.project(windows)

    def _batch(self, batch: object) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(batch, Mapping):
            raise TypeError(
                "CARLA pretext batches must map anchors/positives/injected_anomalies"
            )
        anchors = batch.get("anchors", batch.get("ts_org"))
        positives = batch.get("positives", batch.get("ts_w_augment"))
        anomalies = batch.get("injected_anomalies", batch.get("ts_ss_augment"))
        if not isinstance(anchors, Tensor):
            raise TypeError("CARLA pretext anchors must be a tensor")
        if positives is None:
            positives = anchors + self.config.noise_sigma * torch.randn_like(anchors)
        if anomalies is None:
            anomalies = inject_anomalies(
                anchors,
                min_fraction=self.config.anomaly_min_fraction,
                max_fraction=self.config.anomaly_max_fraction,
            )
        if not isinstance(positives, Tensor) or not isinstance(anomalies, Tensor):
            raise TypeError("CARLA positives and injected anomalies must be tensors")
        return anchors, positives, anomalies

    def _step(self, batch: object, stage: str) -> Tensor:
        anchors, positives, anomalies = self._batch(batch)
        previous = self._previous_loss if stage == "train" else None
        loss = self.model.compute_pretext_loss(
            anchors,
            positives,
            anomalies,
            previous_loss=previous,
        )
        if stage == "train":
            self._previous_loss = loss.detach()
        self.log(
            f"{stage}/loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=anchors.shape[0],
        )
        return loss

    def training_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self) -> dict[str, object]:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.pretext_learning_rate
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.pretext_epochs,
            eta_min=self.config.pretext_learning_rate
            * self.config.pretext_cosine_decay_rate**3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class CARLAClassificationLightningModule(L.LightningModule):
    def __init__(self, config: CARLAConfig, model: CARLA | None = None) -> None:
        super().__init__()
        self.config = config
        self.model = model or CARLA(config)

    def forward(self, windows: Tensor):
        return self.model(windows)

    @staticmethod
    def _batch(batch: object) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(batch, Mapping):
            raise TypeError(
                "CARLA classification batches must map anchors/nearest/furthest"
            )
        anchors = batch.get("anchors", batch.get("anchor"))
        nearest = batch.get("nearest", batch.get("NNeighbor"))
        furthest = batch.get("furthest", batch.get("FNeighbor"))
        if not all(isinstance(value, Tensor) for value in (anchors, nearest, furthest)):
            raise TypeError("CARLA classification batch values must be tensors")
        return anchors, nearest, furthest

    def _step(self, batch: object, stage: str) -> Tensor:
        anchors, nearest, furthest = self._batch(batch)
        losses = self.model.compute_classification_loss(anchors, nearest, furthest)
        for name in ("total", "consistency", "inconsistency", "entropy"):
            metric = "loss" if name == "total" else name
            self.log(
                f"{stage}/{metric}",
                getattr(losses, name),
                on_step=stage == "train" and name == "total",
                on_epoch=True,
                prog_bar=name == "total",
                sync_dist=True,
                batch_size=anchors.shape[0],
            )
        return losses.total

    def training_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config.classification_learning_rate,
            weight_decay=self.config.classification_weight_decay,
        )
