"""Lightning inference module for the released zero-shot DADA checkpoint."""

from __future__ import annotations

from collections.abc import Mapping

import lightning as L
from torch import Tensor

from .config import DADAConfig
from .model import DADA


class DADALightningModule(L.LightningModule):
    """DADA has no public training objective; this module never invents one."""

    def __init__(self, config: DADAConfig | None = None) -> None:
        super().__init__()
        self.config = config or DADAConfig.original_pretrained()
        self.model = DADA(self.config)
        if self.config.reference_checkpoint is not None:
            self.model.load_reference_checkpoint(self.config.reference_checkpoint)
        self.automatic_optimization = False

    def forward(self, series: Tensor, **kwargs: object):
        return self.model(series, **kwargs)

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
            raise TypeError("DADA batches must provide a series tensor")
        return series

    def training_step(self, batch: object, batch_idx: int) -> None:
        raise RuntimeError(
            "The released DADA project provides pretrained zero-shot inference but "
            "does not publish a training objective. Load its reference checkpoint."
        )

    def validation_step(self, batch: object, batch_idx: int) -> Tensor:
        score = self.model.score(
            self._series(batch),
            normalize=self.config.normalization,
            variance_weight=self.config.variance_weight,
        )
        value = score.mean()
        self.log("val/mean_score", value, sync_dist=True, batch_size=score.shape[0])
        return value

    def predict_step(
        self, batch: object, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        return self.model.score(
            self._series(batch),
            normalize=self.config.normalization,
            variance_weight=self.config.variance_weight,
        )

    def configure_optimizers(self) -> None:
        return None
