"""Lightning fine-tuning module for the complete TSPulse core."""

from __future__ import annotations

from collections.abc import Mapping

import lightning as L
import torch
from torch import Tensor

from .config import TSPulseConfig
from .model import TSPulseFineTune


class TSPulseLightningModule(L.LightningModule):
    def __init__(
        self,
        config: TSPulseConfig,
        model: TSPulseFineTune | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model or TSPulseFineTune(config)
        if config.finetune_freeze_backbone:
            for parameter in self.model.core.backbone.parameters():
                parameter.requires_grad_(False)

    def forward(
        self,
        series: Tensor,
        observed_mask: Tensor | None = None,
        future_values: Tensor | None = None,
    ):
        return self.model(
            series,
            observed_mask,
            future_values=future_values,
            return_loss=False,
        )

    @staticmethod
    def _batch(
        batch: object,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if isinstance(batch, Tensor):
            return batch, None, None
        if isinstance(batch, Mapping):
            series = batch.get("series", batch.get("past_values"))
            future = batch.get("future_values")
            observed = batch.get("observed_mask", batch.get("past_observed_mask"))
        elif isinstance(batch, (tuple, list)) and batch:
            series = batch[0]
            future = batch[1] if len(batch) > 1 else None
            observed = batch[2] if len(batch) > 2 else None
        else:
            series = future = observed = None
        if not isinstance(series, Tensor):
            raise TypeError("TSPulse batches must provide past_values/series")
        if future is not None and not isinstance(future, Tensor):
            raise TypeError("TSPulse future_values must be a tensor")
        if observed is not None and not isinstance(observed, Tensor):
            raise TypeError("TSPulse observed mask must be a tensor")
        return series, future, observed

    def _step(self, batch: object, stage: str) -> Tensor:
        series, future, observed = self._batch(batch)
        losses = self.model.compute_loss(
            series,
            future_values=future,
            observed_mask=observed,
        )
        names = (
            "total",
            "time_reconstruction",
            "masked_time_reconstruction",
            "frequency_reconstruction",
            "frequency_time_reconstruction",
            "frequency_probability",
            "forecast",
        )
        for name in names:
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
        return self.model.score(series, batch_size=self.config.finetune_batch_size)

    def configure_optimizers(self) -> dict[str, object]:
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.finetune_learning_rate,
            weight_decay=self.config.finetune_weight_decay,
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.finetune_learning_rate,
            total_steps=total_steps,
            pct_start=self.config.one_cycle_pct_start,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
