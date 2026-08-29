"""Lightning prediction module for non-parametric MMPAD."""

from __future__ import annotations

from collections.abc import Mapping

import lightning as L
from torch import Tensor

from .config import MMPADConfig
from .model import MMPAD


class MMPADLightningModule(L.LightningModule):
    def __init__(self, config: MMPADConfig) -> None:
        super().__init__()
        self.config = config
        self.model = MMPAD(config)
        self.automatic_optimization = False

    def forward(self, series: Tensor) -> Tensor:
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
            raise TypeError("MMPAD batches must provide a series tensor")
        return series

    def fit_reference(self, normal_series: Tensor) -> None:
        self.model.fit(normal_series)

    def training_step(self, batch: object, batch_idx: int) -> None:
        if self.config.use_train_reference:
            self.fit_reference(self._series(batch))
            return None
        raise RuntimeError(
            "MMPAD is non-parametric; use predict directly or set "
            "use_train_reference=True to fit a reference profile."
        )

    def predict_step(
        self, batch: object, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        return self.model.score(self._series(batch))

    def configure_optimizers(self) -> None:
        return None
