"""Periodic Kolmogorov-Arnold network for one-step anomaly prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .._utils import evaluation_mode, sliding_windows, validate_series
from .config import KANADConfig


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
            return (prediction - targets).abs().mean(dim=1)

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
