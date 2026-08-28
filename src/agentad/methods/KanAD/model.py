"""Periodic Kolmogorov-Arnold network for one-step anomaly prediction."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .._utils import sliding_windows, validate_series
from .config import KANADConfig


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
        self.initial = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.inner = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.reduce = nn.Conv1d(channels, 1, 1, bias=False)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.norm3 = nn.BatchNorm1d(1)
        self.activation = nn.GELU()
        self.readout = nn.Conv1d(1, 1, config.window)

    def forward(self, windows: Tensor) -> Tensor:
        """Predict the next value for windows shaped ``[B, window, C]``."""
        windows = validate_series(windows, length=self.config.window, name="windows")
        batch, _, features = windows.shape
        values = windows.transpose(1, 2).reshape(batch * features, self.config.window)
        basis = self.periodic_basis.to(values.dtype).expand(values.shape[0], -1, -1)
        nonlinear = torch.cos(
            torch.arange(1, self.config.order + 1, device=values.device, dtype=values.dtype)[
                None, :, None
            ]
            * values[:, None, :]
        )
        expanded = torch.cat((basis, nonlinear, values[:, None, :]), dim=1)
        hidden = self.activation(self.norm1(self.initial(expanded)))
        hidden = self.activation(self.norm2(self.inner(hidden) + expanded))
        hidden = self.activation(self.norm3(self.reduce(hidden) + values[:, None, :]))
        prediction = self.readout(hidden).reshape(batch, features)
        return prediction

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

    def loss(self, windows: Tensor, targets: Tensor) -> Tensor:
        prediction = self(windows)
        targets = self._targets(targets, prediction.shape[0], prediction.shape[1])
        return F.mse_loss(prediction, targets)

    def score(self, windows: Tensor, targets: Tensor) -> Tensor:
        """Return the published absolute one-step forecast error as ``[B]``."""
        prediction = self(windows)
        targets = self._targets(targets, prediction.shape[0], prediction.shape[1])
        return (prediction - targets).abs().mean(dim=1)

    def pairs(self, series: Tensor, *, stride: int = 1) -> tuple[Tensor, Tensor]:
        """Create forecast windows and their immediately following targets."""
        series = validate_series(series, min_length=self.config.window + 1, name="series")
        windows = sliding_windows(series[:, :-1], self.config.window, stride)
        starts = torch.arange(windows.shape[1], device=series.device) * stride
        targets = series.index_select(1, starts + self.config.window)
        return windows.flatten(0, 1), targets.flatten(0, 1)
