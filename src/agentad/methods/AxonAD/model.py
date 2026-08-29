"""Predictable query dynamics with EMA representation targets."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .._utils import evaluation_mode, sliding_windows, validate_series
from .config import AxonADConfig


class _CausalConv1d(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dilation: int) -> None:
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(
            input_dim, output_dim, 3, padding=self.padding, dilation=dilation
        )

    def forward(self, x: Tensor) -> Tensor:
        output = self.conv(x)
        return output[..., : -self.padding] if self.padding else output


class _QueryPredictor(nn.Module):
    def __init__(self, config: AxonADConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for dilation in config.query_dilations:
            layers.extend(
                (
                    _CausalConv1d(config.model_dim, config.model_dim, dilation),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                )
            )
        layers.append(nn.Conv1d(config.model_dim, config.model_dim, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x.transpose(1, 2)).transpose(1, 2)


class _PredictiveAttention(nn.Module):
    def __init__(self, config: AxonADConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.model_dim // config.heads
        self.forecast_steps = config.forecast_steps
        self.query = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.key = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.value = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.output = nn.Linear(config.model_dim, config.model_dim)
        self.predictor = _QueryPredictor(config)

    def _heads(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], x.shape[1], self.heads, self.head_dim).transpose(
            1, 2
        )

    def forward(
        self, teacher: Tensor, history: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        real_query = self._heads(self.query(teacher))
        key = self._heads(self.key(teacher))
        value = self._heads(self.value(teacher))
        shift = self.forecast_steps
        shifted = (
            history
            if shift == 0
            else torch.cat(
                (
                    history.new_zeros(history.shape[0], shift, history.shape[2]),
                    history[:, :-shift],
                ),
                dim=1,
            )
        )
        predicted_query = self._heads(self.predictor(shifted))
        context = F.scaled_dot_product_attention(real_query, key, value)
        context = context.transpose(1, 2).flatten(2)
        return self.output(context), real_query, predicted_query, key


@dataclass(frozen=True, slots=True)
class AxonADOutput:
    reconstruction: Tensor
    real_query: Tensor
    predicted_query: Tensor
    key: Tensor
    target_query: Tensor
    training_mask: Tensor


@dataclass(frozen=True, slots=True)
class AxonADLoss:
    total: Tensor
    reconstruction: Tensor
    representation: Tensor


class AxonAD(nn.Module):
    """AxonAD with explicit post-optimizer EMA target updates."""

    def __init__(self, config: AxonADConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(config.input_features, config.model_dim)
        self.position = nn.Parameter(
            torch.randn(1, config.window_length, config.model_dim) * 0.02
        )
        self.input_norm = nn.LayerNorm(config.model_dim)
        self.output_norm = nn.LayerNorm(config.model_dim)
        self.attention = _PredictiveAttention(config)
        self.feedforward = nn.Sequential(
            nn.Linear(config.model_dim, 2 * config.model_dim),
            nn.ReLU(),
            nn.Linear(2 * config.model_dim, config.model_dim),
        )
        self.reconstruction_head = nn.Linear(config.model_dim, config.input_features)
        self.target_embedding = nn.Linear(config.input_features, config.model_dim)
        self.target_position = nn.Parameter(self.position.detach().clone())
        self.target_norm = nn.LayerNorm(config.model_dim)
        self.target_query = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self._initialize_target()
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.reconstruction_balance = nn.Parameter(torch.zeros(()))
        self.representation_balance = nn.Parameter(torch.zeros(()))
        self.register_buffer("mse_median", torch.tensor(0.0))
        self.register_buffer("mse_iqr", torch.tensor(1.0))
        self.register_buffer("representation_median", torch.tensor(0.0))
        self.register_buffer("representation_iqr", torch.tensor(1.0))
        self.register_buffer("calibrated", torch.tensor(False), persistent=False)

    def _initialize_target(self) -> None:
        with torch.no_grad():
            self.target_embedding.load_state_dict(self.embedding.state_dict())
            self.target_position.copy_(self.position)
            self.target_norm.load_state_dict(self.input_norm.state_dict())
            self.target_query.weight.copy_(self.attention.query.weight)
        for module in (self.target_embedding, self.target_norm, self.target_query):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.target_position.requires_grad_(False)

    def _training_mask(self, batch: int, device: torch.device) -> Tensor:
        if not self.training:
            return torch.zeros(
                batch, self.config.window_length, device=device, dtype=torch.bool
            )
        start = self.config.forecast_steps
        available = self.config.window_length - start
        tail_length = max(1, available // 2)
        tail_start = max(start, self.config.window_length - tail_length)
        tail_length = self.config.window_length - tail_start
        block = max(1, round(tail_length * self.config.mask_block_fraction))
        block = min(block, tail_length)
        target_masked = max(1, round(tail_length * self.config.mask_ratio))
        blocks = max(1, math.ceil(target_masked / block))
        starts = torch.randint(
            tail_start,
            max(tail_start + 1, self.config.window_length - block + 1),
            (batch, blocks),
            device=device,
        )
        positions = torch.arange(self.config.window_length, device=device)
        return (
            (positions[None, None] >= starts[:, :, None])
            & (positions[None, None] < starts[:, :, None] + block)
        ).any(1)

    def forward(self, windows: Tensor) -> AxonADOutput:
        windows = validate_series(
            windows, length=self.config.window_length, name="windows"
        )
        if windows.shape[2] != self.config.input_features:
            raise ValueError("window feature count does not match the AxonAD config")
        hidden = self.embedding(windows) + self.position
        attended, real_query, predicted_query, key = self.attention(
            self.input_norm(hidden), self.input_norm(hidden)
        )
        hidden = hidden + attended
        hidden = hidden + self.feedforward(self.output_norm(hidden))
        reconstruction = self.reconstruction_head(hidden)
        with torch.no_grad():
            target_hidden = self.target_norm(
                self.target_embedding(windows) + self.target_position
            )
            target_query = self.attention._heads(self.target_query(target_hidden))
        return AxonADOutput(
            reconstruction,
            real_query,
            predicted_query,
            key,
            target_query,
            self._training_mask(windows.shape[0], windows.device),
        )

    def _tail_representation_distance(self, output: AxonADOutput) -> Tensor:
        start = max(
            self.config.forecast_steps,
            self.config.window_length - self.config.tail_length,
        )
        predicted = F.normalize(output.predicted_query[:, :, start:], dim=-1)
        target = F.normalize(output.target_query[:, :, start:], dim=-1)
        return (1 - (predicted * target).sum(-1)).mean((1, 2))

    def _masked_representation_loss(self, output: AxonADOutput) -> Tensor:
        distance = 1 - (
            F.normalize(output.predicted_query, dim=-1)
            * F.normalize(output.target_query, dim=-1)
        ).sum(-1)
        mask = output.training_mask[:, None].expand_as(distance)
        return (
            distance.masked_select(mask).mean()
            if mask.any()
            else distance.new_zeros(())
        )

    def compute_loss(self, windows: Tensor) -> AxonADLoss:
        output = self(windows)
        reconstruction = F.mse_loss(output.reconstruction, windows)
        representation = self._masked_representation_loss(output)
        total = (
            torch.exp(-self.reconstruction_balance) * reconstruction
            + self.reconstruction_balance
            + torch.exp(-self.representation_balance) * representation
            + self.representation_balance
        )
        return AxonADLoss(total, reconstruction, representation)

    @torch.no_grad()
    def update_target(self) -> None:
        momentum = self.config.target_momentum
        pairs = (
            (self.target_embedding, self.embedding),
            (self.target_norm, self.input_norm),
            (self.target_query, self.attention.query),
        )
        for target, online in pairs:
            for target_parameter, online_parameter in zip(
                target.parameters(), online.parameters()
            ):
                target_parameter.mul_(momentum).add_(
                    online_parameter, alpha=1 - momentum
                )
        self.target_position.mul_(momentum).add_(self.position, alpha=1 - momentum)

    def _window_components(self, windows: Tensor) -> tuple[Tensor, Tensor]:
        output = self(windows)
        reconstruction = (output.reconstruction - windows).square().sum(-1).mean(1)
        return reconstruction, self._tail_representation_distance(output)

    @torch.inference_mode()
    def calibrate(self, normal_series: Tensor, *, batch_size: int = 256) -> None:
        normal_series = validate_series(
            normal_series, min_length=self.config.window_length, name="normal_series"
        )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        with evaluation_mode(self):
            windows = sliding_windows(normal_series, self.config.window_length).flatten(
                0, 1
            )
            components = [
                self._window_components(chunk) for chunk in windows.split(batch_size)
            ]
            mse = torch.cat([item[0] for item in components])
            representation = torch.cat([item[1] for item in components])
            self.mse_median.copy_(mse.median())
            self.mse_iqr.copy_(
                (mse.quantile(0.75) - mse.quantile(0.25)).clamp_min(self.config.epsilon)
            )
            self.representation_median.copy_(representation.median())
            self.representation_iqr.copy_(
                (
                    representation.quantile(0.75) - representation.quantile(0.25)
                ).clamp_min(self.config.epsilon)
            )
            self.calibrated.fill_(True)

    @torch.inference_mode()
    def score(self, series: Tensor, *, batch_size: int = 256) -> Tensor:
        if not self.calibrated:
            raise RuntimeError("call calibrate() before score()")
        series = validate_series(series, min_length=self.config.window_length)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        with evaluation_mode(self):
            windows = sliding_windows(series, self.config.window_length)
            flattened = windows.flatten(0, 1)
            components = [
                self._window_components(chunk) for chunk in flattened.split(batch_size)
            ]
            mse = torch.cat([item[0] for item in components])
            representation = torch.cat([item[1] for item in components])
            window_scores = (
                (mse - self.mse_median) / self.mse_iqr
                + (representation - self.representation_median)
                / self.representation_iqr
            ).reshape(series.shape[0], windows.shape[1])
            left = math.ceil((self.config.window_length - 1) / 2)
            right = (self.config.window_length - 1) // 2
            return F.pad(window_scores, (left, right), mode="replicate")
