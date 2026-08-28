"""Encoder-decoder xLSTM reconstruction for anomaly detection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .._utils import evaluation_mode, sliding_windows, validate_series
from .config import XLSTMADConfig


class _CausalDepthwiseConv(nn.Module):
    def __init__(self, width: int, kernel_size: int) -> None:
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(
            width, width, kernel_size, padding=self.padding, groups=width
        )

    def forward(self, x: Tensor) -> Tensor:
        output = self.conv(x.transpose(1, 2))
        if self.padding:
            output = output[..., :-self.padding]
        return output.transpose(1, 2)


class _ScalarLSTM(nn.Module):
    def __init__(self, config: XLSTMADConfig) -> None:
        super().__init__()
        width = config.embedding_dim
        self.heads = config.heads
        self.head_dim = width // config.heads
        self.conv = _CausalDepthwiseConv(width, config.scalar_kernel)
        self.input_projection = nn.Linear(width, 4 * width)
        self.recurrent_projection = nn.Linear(width, 4 * width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        convolved = self.conv(x)
        batch = x.shape[0]
        hidden = x.new_zeros(batch, self.heads, self.head_dim)
        cell = torch.zeros_like(hidden)
        normalizer = torch.zeros_like(hidden)
        stabilizer = x.new_full(hidden.shape, -torch.inf)
        outputs: list[Tensor] = []
        for value in convolved.unbind(1):
            gates = self.input_projection(value) + self.recurrent_projection(
                hidden.flatten(1)
            )
            candidate, input_gate, forget_gate, output_gate = gates.chunk(4, dim=1)
            candidate = candidate.reshape_as(hidden).tanh()
            input_gate = input_gate.reshape_as(hidden)
            forget_gate = F.logsigmoid(forget_gate.reshape_as(hidden))
            next_stabilizer = torch.maximum(forget_gate + stabilizer, input_gate)
            stabilized_input = torch.exp(input_gate - next_stabilizer)
            stabilized_forget = torch.exp(forget_gate + stabilizer - next_stabilizer)
            cell = stabilized_forget * cell + stabilized_input * candidate
            normalizer = stabilized_forget * normalizer + stabilized_input
            hidden = torch.sigmoid(output_gate.reshape_as(hidden)) * cell / normalizer.clamp_min(1)
            stabilizer = next_stabilizer
            outputs.append(hidden.flatten(1))
        return torch.stack(outputs, dim=1)


class _MatrixLSTM(nn.Module):
    def __init__(self, config: XLSTMADConfig) -> None:
        super().__init__()
        width = config.embedding_dim
        self.heads = config.heads
        self.head_dim = width // config.heads
        self.conv = _CausalDepthwiseConv(width, config.matrix_kernel)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output_gate = nn.Linear(width, width)
        self.input_gate = nn.Linear(width, config.heads)
        self.forget_gate = nn.Linear(width, config.heads)

    def forward(self, x: Tensor) -> Tensor:
        convolved = self.conv(x)
        batch = x.shape[0]
        matrix = x.new_zeros(batch, self.heads, self.head_dim, self.head_dim)
        normalizer = x.new_zeros(batch, self.heads, self.head_dim)
        stabilizer = x.new_full((batch, self.heads), -torch.inf)
        outputs: list[Tensor] = []
        for value in convolved.unbind(1):
            query = self.query(value).reshape(batch, self.heads, self.head_dim)
            key = self.key(value).reshape_as(query) / math.sqrt(self.head_dim)
            candidate = self.value(value).reshape_as(query)
            input_gate = self.input_gate(value)
            forget_gate = F.logsigmoid(self.forget_gate(value))
            next_stabilizer = torch.maximum(forget_gate + stabilizer, input_gate)
            stabilized_input = torch.exp(input_gate - next_stabilizer)
            stabilized_forget = torch.exp(forget_gate + stabilizer - next_stabilizer)
            matrix = (
                stabilized_forget[..., None, None] * matrix
                + stabilized_input[..., None, None]
                * candidate[..., :, None]
                * key[..., None, :]
            )
            normalizer = (
                stabilized_forget[..., None] * normalizer
                + stabilized_input[..., None] * key
            )
            numerator = (matrix @ query[..., None]).squeeze(-1)
            denominator = (normalizer * query).sum(-1).abs().clamp_min(1)[..., None]
            hidden = torch.sigmoid(self.output_gate(value)).reshape_as(query) * numerator / denominator
            stabilizer = next_stabilizer
            outputs.append(hidden.flatten(1))
        return torch.stack(outputs, dim=1)


class _XLSTMBlock(nn.Module):
    def __init__(self, config: XLSTMADConfig, *, scalar_memory: bool) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.cell = _ScalarLSTM(config) if scalar_memory else _MatrixLSTM(config)
        hidden = max(1, round(config.embedding_dim * config.feedforward_factor))
        self.feedforward_norm = nn.LayerNorm(config.embedding_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(config.embedding_dim, 2 * hidden),
            nn.GLU(),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.embedding_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.cell(self.norm(x))
        return x + self.feedforward(self.feedforward_norm(x))


class _XLSTMStack(nn.Module):
    def __init__(self, config: XLSTMADConfig) -> None:
        super().__init__()
        scalar = set(config.scalar_memory_blocks)
        self.blocks = nn.ModuleList(
            _XLSTMBlock(config, scalar_memory=index in scalar)
            for index in range(config.blocks)
        )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


@dataclass(frozen=True, slots=True)
class XLSTMADOutput:
    reconstruction: Tensor


@dataclass(frozen=True, slots=True)
class XLSTMADLoss:
    total: Tensor
    reconstruction: Tensor


class XLSTMAD(nn.Module):
    """Full xLSTM encoder-decoder with MSE anomaly scoring."""

    def __init__(self, config: XLSTMADConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_features, config.embedding_dim)
        self.encoder = _XLSTMStack(config)
        self.decoder = _XLSTMStack(config)
        self.output_projection = nn.Linear(config.embedding_dim, config.input_features)

    def forward(self, windows: Tensor) -> XLSTMADOutput:
        windows = validate_series(
            windows, length=self.config.window_length, name="windows"
        )
        if windows.shape[2] != self.config.input_features:
            raise ValueError("window feature count does not match the xLSTMAD config")
        hidden = self.encoder(self.input_projection(windows))
        reconstruction = self.output_projection(F.gelu(self.decoder(hidden)))
        return XLSTMADOutput(reconstruction)

    def compute_loss(self, windows: Tensor) -> XLSTMADLoss:
        output = self(windows)
        reconstruction = F.mse_loss(output.reconstruction, windows)
        return XLSTMADLoss(reconstruction, reconstruction)

    @torch.inference_mode()
    def window_score(self, windows: Tensor) -> Tensor:
        with evaluation_mode(self):
            reconstruction = self(windows).reconstruction
            return (reconstruction - windows).square().mean((1, 2))

    @torch.inference_mode()
    def score(self, series: Tensor, *, batch_size: int = 256) -> Tensor:
        series = validate_series(
            series, min_length=self.config.window_length, name="series"
        )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        with evaluation_mode(self):
            windows = sliding_windows(series, self.config.window_length)
            flattened = windows.flatten(0, 1)
            scores = torch.cat(
                [
                    (self(chunk).reconstruction - chunk).square().mean((1, 2))
                    for chunk in flattened.split(batch_size)
                ]
            ).reshape(series.shape[0], -1)
            return F.pad(scores, (self.config.window_length - 1, 0))
