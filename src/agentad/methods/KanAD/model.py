"""Periodic Kolmogorov-Arnold network for one-step anomaly prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from typing import Mapping
import lightning as L
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

from .._utils import evaluation_mode, sliding_windows, validate_series
from .config import KANADConfig
from .._utils import ValidationEarlyStopping


@dataclass(frozen=True, slots=True)
class KANADOutput:
    prediction: Tensor


@dataclass(frozen=True, slots=True)
class KANADLoss:
    total: Tensor
    forecast: Tensor


class KANAD(nn.Module):
    """KAN-AD with channel-independent multivariate batching.

    The published model is univariate. Multivariate inputs are evaluated by the
    same shared predictor independently per feature, avoiding one model copy per
    channel while retaining the original hypothesis class.
    """

    periodic_basis: Tensor
    harmonic_orders: Tensor

    def __init__(self, config: KANADConfig = KANADConfig()) -> None:
        super().__init__()
        self.config = config
        channels = 2 * config.order + 1
        time = torch.arange(config.window, dtype=torch.float32)
        frequencies = torch.arange(1, config.order + 1, dtype=torch.float32)[:, None]
        self.register_buffer(
            "periodic_basis",
            torch.cos(2 * torch.pi * frequencies * time[None, :] / config.window),
        )
        self.register_buffer("harmonic_orders", frequencies[:, 0])
        self.initial = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.inner = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.reduce = nn.Conv1d(channels, 1, 1, bias=False)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.norm3 = nn.BatchNorm1d(1)
        self.activation = nn.GELU()
        self.readout = nn.Conv1d(1, 1, config.window)

    def _predict(self, windows: Tensor) -> Tensor:
        windows = validate_series(windows, length=self.config.window, name="windows")
        batch, _, features = windows.shape
        values = windows.transpose(1, 2).reshape(batch * features, self.config.window)
        basis = self.periodic_basis.to(values.dtype).expand(values.shape[0], -1, -1)
        nonlinear = torch.cos(
            self.harmonic_orders.to(values.dtype)[None, :, None] * values[:, None, :]
        )
        expanded = torch.cat((basis, nonlinear, values[:, None, :]), dim=1)
        hidden = self.activation(self.norm1(self.initial(expanded)))
        hidden = self.activation(self.norm2(self.inner(hidden) + expanded))
        hidden = self.activation(self.norm3(self.reduce(hidden) + values[:, None, :]))
        return self.readout(hidden).reshape(batch, features)

    def forward(self, windows: Tensor) -> KANADOutput:
        """Predict the next value for windows shaped ``[B, window, C]``."""
        return KANADOutput(self._predict(windows))

    @staticmethod
    def _targets(targets: Tensor, batch: int, features: int) -> Tensor:
        if targets.ndim == 3 and targets.shape[1] == 1:
            targets = targets[:, 0]
        if targets.ndim == 1 and features == 1:
            targets = targets[:, None]
        if targets.shape != (batch, features):
            raise ValueError(
                f"targets must have shape [{batch}, {features}] or [{batch}, 1, {features}]"
            )
        if not targets.is_floating_point():
            raise TypeError("targets must use a floating dtype")
        return targets

    def compute_loss(self, windows: Tensor, targets: Tensor) -> KANADLoss:
        prediction = self(windows).prediction
        targets = self._targets(targets, prediction.shape[0], prediction.shape[1])
        forecast = F.mse_loss(prediction, targets)
        return KANADLoss(forecast, forecast)

    @torch.inference_mode()
    def window_score(self, windows: Tensor, targets: Tensor) -> Tensor:
        """Return the published absolute one-step forecast error as ``[B]``."""
        with evaluation_mode(self):
            prediction = self(windows).prediction
            targets = self._targets(targets, prediction.shape[0], prediction.shape[1])
            error = (prediction - targets).abs().mean(dim=1)
            # NaN forecasts are pinned to the original sentinel score.
            return torch.where(
                torch.isnan(error), torch.full_like(error, 1000.0), error
            )

    @torch.inference_mode()
    def score(self, series: Tensor) -> Tensor:
        """Return aligned point scores, with the context-only prefix set to zero."""
        with evaluation_mode(self):
            series = validate_series(
                series, min_length=self.config.window + 1, name="series"
            )
            windows = sliding_windows(series[:, :-1], self.config.window)
            targets = series[:, self.config.window :]
            prediction = self._predict(windows.flatten(0, 1)).reshape_as(targets)
            error = (prediction - targets).abs().mean(2)
            # Non-finite forecasts are pinned to the original sentinel score.
            error = torch.where(
                torch.isnan(error), torch.full_like(error, 1000.0), error
            )
            return F.pad(error, (self.config.window, 0))

    def pairs(self, series: Tensor, *, stride: int = 1) -> tuple[Tensor, Tensor]:
        """Create forecast windows and their immediately following targets."""
        series = validate_series(
            series, min_length=self.config.window + 1, name="series"
        )
        windows = sliding_windows(series[:, :-1], self.config.window, stride)
        starts = torch.arange(windows.shape[1], device=series.device) * stride
        targets = series.index_select(1, starts + self.config.window)
        return windows.flatten(0, 1), targets.flatten(0, 1)


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

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
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
