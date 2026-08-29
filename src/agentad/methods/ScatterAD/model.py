"""Temporal-topological scattering with a momentum target encoder."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from collections.abc import Mapping
from typing import Any
import lightning as L
from torch import Tensor

from .._utils import evaluation_mode, validate_series
from .config import ScatterADConfig


class _TemporalBlock(nn.Module):
    """Temporal convolution over the time axis.

    The vendored fork flattens each PyG batch into a node matrix before this
    block, so its kernel-3 convolution runs on a length-one axis and reduces
    to a point-wise linear map. This implementation runs the convolution the
    fork's own docstring describes, mixing adjacent time steps.
    """
    def __init__(self, input_dim: int, output_dim: int, kernel: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(input_dim, output_dim, kernel, padding=kernel // 2)
        self.norm = nn.BatchNorm1d(output_dim)
        self.activation = nn.PReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.norm(self.conv(x)))


class _LocalGraphAttention(nn.Module):
    """Multi-head GAT over a fixed temporal neighborhood without PyG materialization."""

    def __init__(self, config: ScatterADConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.hidden_dim // config.heads
        self.radius = config.graph_radius
        self.linear = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.attention_source = nn.Parameter(torch.empty(config.heads, self.head_dim))
        self.attention_target = nn.Parameter(torch.empty(config.heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(config.hidden_dim))
        self.dropout = nn.Dropout(config.attention_dropout)
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_target)

    def _split(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], x.shape[1], self.heads, self.head_dim)

    def forward(self, x: Tensor) -> Tensor:
        batch, time, _ = x.shape
        # GATConv adds self-loops to the explicit temporal edges by default.
        offsets = torch.arange(-self.radius, self.radius + 1, device=x.device)
        positions = torch.arange(time, device=x.device)[:, None] + offsets[None, :]
        valid = (positions >= 0) & (positions < time)
        positions = positions.clamp(0, time - 1)

        transformed = self._split(self.linear(x))
        source_logits = (transformed * self.attention_source).sum(-1)
        neighbors = transformed[:, positions]
        target_logits = (neighbors * self.attention_target).sum(-1)
        logits = F.leaky_relu(
            source_logits[:, :, None] + target_logits,
            negative_slope=0.2,
        )
        logits = logits.masked_fill(~valid[None, :, :, None], -torch.inf)
        weights = self.dropout(logits.softmax(dim=2))
        return (weights[..., None] * neighbors).sum(dim=2).flatten(2) + self.bias


class _NodeEncoder(nn.Module):
    def __init__(self, config: ScatterADConfig) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            _TemporalBlock(
                config.input_features, config.hidden_dim, config.temporal_kernel
            ),
            nn.Dropout(config.temporal_dropout),
            _TemporalBlock(
                config.hidden_dim, config.hidden_dim, config.temporal_kernel
            ),
        )
        self.graph_layers = nn.ModuleList(
            _LocalGraphAttention(config) for _ in range(config.graph_layers)
        )
        self.skip = nn.Linear(config.input_features, config.hidden_dim)
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim, 2 * config.hidden_dim),
            nn.BatchNorm1d(2 * config.hidden_dim),
            nn.PReLU(),
            nn.Dropout(config.projection_dropout),
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        temporal = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        graph = temporal
        for layer in self.graph_layers:
            graph = F.elu(layer(graph))
        hidden = graph + self.skip(x)
        projected = self.projection(graph.flatten(0, 1)).reshape_as(graph)
        return hidden, projected


@dataclass(frozen=True, slots=True)
class ScatterADOutput:
    online_hidden: Tensor
    prediction: Tensor
    target_hidden: Tensor


@dataclass(frozen=True, slots=True)
class ScatterADLoss:
    total: Tensor
    scattering: Tensor
    temporal: Tensor
    consistency: Tensor


class ScatterAD(nn.Module):
    """ScatterAD using implicit local graphs and window-aware loss boundaries."""

    def __init__(self, config: ScatterADConfig) -> None:
        super().__init__()
        self.config = config
        self.online_encoder = _NodeEncoder(config)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, 2 * config.hidden_dim),
            nn.BatchNorm1d(2 * config.hidden_dim),
            nn.PReLU(),
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )
        # The original normalizes a 1-D center with norm(dim=1), which errors
        # in PyTorch; the whole-vector norm implements its evident intent,
        # with a zero-norm guard the original lacks.
        center = torch.randn(config.hidden_dim)
        center = (
            center
            / center.norm().clamp_min(config.score_epsilon)
            * torch.rand(()).sqrt()
        )
        self.center = nn.Parameter(center)

    def forward(self, x: Tensor) -> ScatterADOutput:
        x = validate_series(x, min_length=2)
        if x.shape[2] != self.config.input_features:
            raise ValueError("x feature count does not match the ScatterAD config")
        online_hidden, online_projection = self.online_encoder(x)
        prediction = self.predictor(online_projection.flatten(0, 1)).reshape_as(
            online_projection
        )
        with torch.no_grad():
            target_hidden, _ = self.target_encoder(x)
            target_hidden = F.normalize(target_hidden, dim=2)
        return ScatterADOutput(online_hidden, prediction, target_hidden)

    def compute_loss(self, x: Tensor) -> ScatterADLoss:
        output = self(x)
        center = F.normalize(self.center, dim=0)
        scattering = -F.cosine_similarity(
            output.target_hidden,
            center[None, None, :],
            dim=2,
        ).mean()
        # Each window contributes only in-window adjacent pairs; the original
        # flattens shuffled batches into node lists, so its version also mixes
        # unrelated window boundaries.
        temporal = F.mse_loss(output.prediction[:, 1:], output.prediction[:, :-1])
        # The original derives the consistency pairs from the loader's directed
        # edges: offsets 1..radius in both directions (the Water configuration
        # ships radius 2, the remaining datasets radius 1).
        pairs: list[Tensor] = []
        for offset in range(1, self.config.graph_radius + 1):
            pairs.append(
                F.cosine_similarity(
                    output.prediction[:, :-offset],
                    output.target_hidden[:, offset:],
                    dim=2,
                )
            )
            pairs.append(
                F.cosine_similarity(
                    output.prediction[:, offset:],
                    output.target_hidden[:, :-offset],
                    dim=2,
                )
            )
        consistency = (
            -torch.cat(pairs, dim=1).sigmoid().log().mean()
        )
        return ScatterADLoss(
            scattering + temporal + consistency,
            scattering,
            temporal,
            consistency,
        )

    @torch.no_grad()
    def update_target(self) -> None:
        """Apply the published exponential moving-average target update."""
        momentum = self.config.target_momentum
        for online, target in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target.mul_(momentum).add_(online, alpha=1 - momentum)

    @torch.inference_mode()
    def score(self, x: Tensor) -> Tensor:
        """Return center scattering plus adjacent-time inconsistency as ``[B, T]``."""
        x = validate_series(x, min_length=2)
        if x.shape[2] != self.config.input_features:
            raise ValueError("x feature count does not match the ScatterAD config")
        with evaluation_mode(self):
            hidden, _ = self.online_encoder(x)
            # The trailing time step of each window gets a zero difference;
            # the original's flattened batches give it a cross-window pair.
            temporal = (hidden[:, 1:] - hidden[:, :-1]).square().mean(dim=2)
            temporal = F.pad(temporal, (0, 1))
            distance = torch.linalg.vector_norm(
                hidden - self.center[None, None, :], dim=2
            )
            scattering = distance.clamp_min(self.config.score_epsilon).reciprocal()
            return scattering + temporal


class ScatterADLightningModule(L.LightningModule):
    def __init__(self, config: ScatterADConfig) -> None:
        super().__init__()
        self.config = config
        self.model = ScatterAD(config)

    def forward(self, series: Tensor):
        return self.model(series)

    def _series(self, batch: object) -> Tensor:
        if isinstance(batch, Tensor):
            series = batch
        elif isinstance(batch, Mapping):
            series = batch.get("series")
        elif isinstance(batch, (tuple, list)) and batch:
            series = batch[0]
        else:
            series = None
        if not isinstance(series, Tensor):
            raise TypeError("ScatterAD batches must provide a series tensor")
        if series.shape[1] != self.config.window_length:
            raise ValueError(
                f"ScatterAD expects windows of length {self.config.window_length}"
            )
        return series

    def _step(self, batch: object, stage: str) -> Tensor:
        series = self._series(batch)
        losses = self.model.compute_loss(series)
        self.log(
            f"{stage}/loss",
            losses.total,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=series.shape[0],
        )
        for name in ("scattering", "temporal", "consistency"):
            self.log(
                f"{stage}/{name}",
                getattr(losses, name),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=series.shape[0],
            )
        return losses.total

    def training_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure: Any | None = None,
    ) -> None:
        optimizer.step(closure=optimizer_closure)
        self.model.update_target()

    def predict_step(
        self, batch: object, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        return self.model.score(self._series(batch))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
