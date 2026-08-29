"""Cross-scale reconstruction and cross-window context modeling."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .._utils import evaluation_mode, sinusoidal_encoding, validate_series
from .config import CrossADConfig


@dataclass(frozen=True, slots=True)
class CrossADOutput:
    target: Tensor
    reconstruction: Tensor
    context_loss: Tensor


@dataclass(frozen=True, slots=True)
class CrossADLoss:
    total: Tensor
    reconstruction: Tensor
    context: Tensor


class _SequenceNorm(nn.Module):
    def __init__(self, width: int, kind: str) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(width) if kind == "batch" else nn.LayerNorm(width)
        self.batch = kind == "batch"

    def forward(self, x: Tensor) -> Tensor:
        if self.batch:
            channels_first = x.transpose(1, 2)
            if self.training and x.shape[0] * x.shape[1] == 1:
                return F.batch_norm(
                    channels_first,
                    self.norm.running_mean,
                    self.norm.running_var,
                    self.norm.weight,
                    self.norm.bias,
                    training=False,
                    momentum=0.0,
                    eps=self.norm.eps,
                ).transpose(1, 2)
            return self.norm(channels_first).transpose(1, 2)
        return self.norm(x)


class _Attention(nn.Module):
    def __init__(self, config: CrossADConfig) -> None:
        super().__init__()
        self.layer = nn.MultiheadAttention(
            config.model_dim,
            config.heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.output_dropout = nn.Dropout(config.projection_dropout)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        output, _ = self.layer(query, key, value, attn_mask=mask, need_weights=False)
        return self.output_dropout(output)


class _FeedForward(nn.Module):
    def __init__(self, config: CrossADConfig) -> None:
        super().__init__()
        hidden = config.feedforward_dim or 4 * config.model_dim
        activation: nn.Module = nn.ReLU() if config.activation == "relu" else nn.GELU()
        self.net = nn.Sequential(
            nn.Linear(config.model_dim, hidden),
            activation,
            nn.Dropout(config.feedforward_dropout),
            nn.Linear(hidden, config.model_dim),
            nn.Dropout(config.feedforward_dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _EncoderLayer(nn.Module):
    def __init__(self, config: CrossADConfig) -> None:
        super().__init__()
        self.attention = _Attention(config)
        self.feedforward = _FeedForward(config)
        # The original applies the ff_dropout rate on the attention residual
        # branch in addition to the dropout inside the attention projection.
        self.residual_dropout = nn.Dropout(config.feedforward_dropout)
        self.norm1 = _SequenceNorm(config.model_dim, config.normalization)
        self.norm2 = _SequenceNorm(config.model_dim, config.normalization)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.norm1(x + self.residual_dropout(self.attention(x, x, x, mask)))
        return self.norm2(x + self.feedforward(x))


class _DecoderLayer(nn.Module):
    def __init__(self, config: CrossADConfig) -> None:
        super().__init__()
        self.self_attention = _Attention(config)
        self.cross_attention = _Attention(config)
        self.feedforward = _FeedForward(config)
        self.residual_dropout = nn.Dropout(config.feedforward_dropout)
        self.norm1 = _SequenceNorm(config.model_dim, config.normalization)
        self.norm2 = _SequenceNorm(config.model_dim, config.normalization)
        self.norm3 = _SequenceNorm(config.model_dim, config.normalization)

    def forward(self, x: Tensor, context: Tensor, mask: Tensor) -> Tensor:
        x = self.norm1(x + self.residual_dropout(self.self_attention(x, x, x, mask)))
        x = self.norm2(
            x + self.residual_dropout(self.cross_attention(x, context, context))
        )
        return self.norm3(x + self.feedforward(x))


class _ExtractorLayer(nn.Module):
    def __init__(self, config: CrossADConfig) -> None:
        super().__init__()
        self.cross_attention = _Attention(config)
        self.feedforward = _FeedForward(config)
        self.residual_dropout = nn.Dropout(config.feedforward_dropout)
        self.norm1 = _SequenceNorm(config.model_dim, config.normalization)
        self.norm2 = _SequenceNorm(config.model_dim, config.normalization)

    def forward(self, query: Tensor, local: Tensor) -> Tensor:
        query = self.norm1(
            query + self.residual_dropout(self.cross_attention(query, local, local))
        )
        return self.norm2(query + self.feedforward(query))


class _PatchEmbedding(nn.Module):
    def __init__(self, patch_length: int, model_dim: int) -> None:
        super().__init__()
        self.patch_length = patch_length
        self.projection = nn.Linear(patch_length, model_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # Channels are independent samples for the CrossAD encoder.
        x = F.pad(x.transpose(1, 2), (0, self.patch_length - 1), mode="replicate")
        patches = x.unfold(-1, self.patch_length, self.patch_length)
        return self.projection(
            patches.reshape(-1, patches.shape[-2], self.patch_length)
        )


class _MultiScale(nn.Module):
    def __init__(self, kernels: tuple[int, ...], method: str) -> None:
        super().__init__()
        self.kernels = kernels
        self.method = method

    def down(self, x: Tensor) -> list[Tensor]:
        channels_first = x.transpose(1, 2)
        outputs: list[Tensor] = []
        for kernel in self.kernels:
            windows = F.pad(channels_first, (0, kernel - 1), mode="replicate").unfold(
                -1, kernel, kernel
            )
            sampled = (
                windows.mean(dim=-1) if self.method == "average" else windows[..., 0]
            )
            outputs.append(sampled.transpose(1, 2))
        return outputs


class _ContextMemory(nn.Module):
    def __init__(self, config: CrossADConfig) -> None:
        super().__init__()
        self.top_frequencies = config.top_frequencies
        self.decay = config.context_decay
        self.epsilon = config.context_epsilon
        self.query_templates = nn.Parameter(
            torch.randn(config.query_count, config.query_length, config.model_dim)
        )
        self.router = nn.Linear(config.sequence_length, config.query_count)
        self.extractor = nn.ModuleList(
            _ExtractorLayer(config) for _ in range(config.extractor_layers)
        )
        self.register_buffer(
            "context",
            torch.randn(config.context_size, config.query_length, config.model_dim),
        )
        self.register_buffer("ema_count", torch.ones(config.context_size))
        self.register_buffer("ema_sum", torch.zeros_like(self.context))

    def _route(self, x: Tensor) -> Tensor:
        spectrum = torch.fft.rfft(x.squeeze(-1), dim=1)
        count = min(self.top_frequencies, spectrum.shape[1])
        indices = spectrum.abs().topk(count, dim=1).indices
        filtered = torch.zeros_like(spectrum)
        filtered.scatter_(1, indices, spectrum.gather(1, indices))
        periodic = torch.fft.irfft(filtered, n=x.shape[1], dim=1)
        return F.gumbel_softmax(self.router(periodic), tau=1.0, hard=True, dim=1)

    def _quantize(self, query: Tensor) -> tuple[Tensor, Tensor]:
        flat_query = query.flatten(1)
        flat_context = self.context.flatten(1)
        distances = (
            flat_query.square().sum(1, keepdim=True)
            + flat_context.square().sum(1)
            - 2 * flat_query @ flat_context.T
        )
        assignments = F.one_hot(distances.argmin(1), self.context.shape[0]).to(
            query.dtype
        )
        selected = assignments @ flat_context
        loss = F.mse_loss(flat_query, selected.detach(), reduction="none").mean(1)
        assigned_sum = torch.einsum("bn,bqd->nqd", assignments, query)

        with torch.no_grad():
            self.ema_count.mul_(self.decay).add_(
                assignments.sum(0), alpha=1 - self.decay
            )
            self.ema_sum.mul_(self.decay).add_(assigned_sum, alpha=1 - self.decay)
            total = self.ema_count.sum()
            dimension = flat_context.shape[1]
            # The original writes the Laplace-smoothed counts back into the
            # EMA state, so later updates accumulate on the smoothed counts.
            self.ema_count.copy_(
                (self.ema_count + self.epsilon)
                / (total + dimension * self.epsilon)
                * total
            )
            self.context.copy_(self.ema_sum / self.ema_count[:, None, None])

        # The forward value is the full EMA bank; gradients reach selected queries.
        straight_through = assigned_sum + self.context.detach() - assigned_sum.detach()
        return straight_through.flatten(0, 1), loss

    def forward(self, x: Tensor, local: Tensor) -> tuple[Tensor, Tensor]:
        if not self.training:
            return self.context.flatten(0, 1), x.new_zeros(x.shape[0])
        query = torch.einsum("bn,nqd->bqd", self._route(x), self.query_templates)
        for layer in self.extractor:
            query = layer(query, local)
        return self._quantize(query)


class CrossAD(nn.Module):
    """CrossAD with device-safe masks and a bounded EMA context library."""

    def __init__(self, config: CrossADConfig = CrossADConfig()) -> None:
        super().__init__()
        self.config = config
        self.multi_scale = _MultiScale(config.scale_kernels, config.scale_method)
        self.patch_embedding = _PatchEmbedding(config.patch_length, config.model_dim)
        self.encoder = nn.ModuleList(
            _EncoderLayer(config) for _ in range(config.encoder_layers)
        )
        self.decoder = nn.ModuleList(
            _DecoderLayer(config) for _ in range(config.decoder_layers)
        )
        self.encoder_norm = _SequenceNorm(config.model_dim, config.normalization)
        self.decoder_norm = _SequenceNorm(config.model_dim, config.normalization)
        self.patch_projection = nn.Linear(config.model_dim, config.patch_length)
        self.context_memory = _ContextMemory(config)

        self.temporal_lengths = tuple(
            math.ceil(config.sequence_length / kernel)
            for kernel in config.scale_kernels
        ) + (config.sequence_length,)
        self.patch_counts = tuple(
            math.ceil(length / config.patch_length) for length in self.temporal_lengths
        )

        base_positions = sinusoidal_encoding(self.patch_counts[-1], config.model_dim)
        position_scales = self.multi_scale.down(base_positions)
        self.register_buffer(
            "position_encoding", torch.cat(position_scales, dim=1), persistent=False
        )
        self.register_buffer(
            "encoder_mask",
            self._group_mask(self.patch_counts[:-1], causal=False),
            persistent=False,
        )
        self.register_buffer(
            "decoder_mask",
            self._group_mask(self.patch_counts[1:], causal=True),
            persistent=False,
        )

    @staticmethod
    def _group_mask(lengths: tuple[int, ...], *, causal: bool) -> Tensor:
        groups = torch.cat(
            [torch.full((length,), index) for index, length in enumerate(lengths)]
        )
        if causal:
            return groups[:, None] < groups[None, :]
        return groups[:, None] != groups[None, :]

    def _encode(self, x: Tensor) -> tuple[Tensor, list[Tensor], Tensor]:
        batch, _, features = x.shape
        independent = x.transpose(1, 2).reshape(
            batch * features, self.config.sequence_length, 1
        )
        scaled_values = self.multi_scale.down(independent)
        encoded_groups = [self.patch_embedding(scale) for scale in scaled_values]
        encoded = torch.cat(encoded_groups, dim=1) + self.position_encoding.to(x.dtype)
        for layer in self.encoder:
            encoded = layer(encoded, self.encoder_mask)
        # The original encoder normalizes its output before the context
        # extractor and the decoder consume it.
        encoded = self.encoder_norm(encoded)
        target = torch.cat(scaled_values[1:] + [independent], dim=1)
        return independent, list(encoded.split(self.patch_counts[:-1], dim=1)), target

    def forward(self, x: Tensor) -> CrossADOutput:
        x = validate_series(x, length=self.config.sequence_length)
        batch, _, features = x.shape
        independent, encoded_groups, target = self._encode(x)
        encoded = torch.cat(encoded_groups, dim=1)
        context, context_loss = self.context_memory(independent, encoded)
        context = context.unsqueeze(0).expand(batch * features, -1, -1)

        decoder_groups = [
            F.interpolate(
                group.transpose(1, 2), size=next_count, mode="nearest"
            ).transpose(1, 2)
            for group, next_count in zip(encoded_groups, self.patch_counts[1:])
        ]
        decoded = torch.cat(decoder_groups, dim=1)
        for layer in self.decoder:
            decoded = layer(decoded, context, self.decoder_mask)
        decoded = self.patch_projection(self.decoder_norm(decoded))

        reconstructed_groups: list[Tensor] = []
        for group, target_length in zip(
            decoded.split(self.patch_counts[1:], dim=1), self.temporal_lengths[1:]
        ):
            reconstructed_groups.append(group.flatten(1)[:, :target_length, None])
        reconstruction = torch.cat(reconstructed_groups, dim=1)

        target = target.reshape(batch, features, -1).transpose(1, 2)
        reconstruction = reconstruction.reshape(batch, features, -1).transpose(1, 2)
        context_loss = context_loss.reshape(batch, features).mean(1)
        return CrossADOutput(target, reconstruction, context_loss)

    def compute_loss(self, x: Tensor) -> CrossADLoss:
        output = self(x)
        reconstruction = F.mse_loss(output.reconstruction, output.target)
        context = output.context_loss.mean()
        total = reconstruction + self.config.context_loss_weight * context
        return CrossADLoss(total, reconstruction, context)

    @torch.inference_mode()
    def score(self, x: Tensor) -> Tensor:
        """Return the published multi-scale reconstruction score as ``[B, T]``."""
        with evaluation_mode(self):
            output = self(x)
            errors = (output.reconstruction - output.target).square()
            groups = errors.split(self.temporal_lengths[1:], dim=1)
            finest = groups[-1]
            for coarse in groups[:-1]:
                finest = finest + F.interpolate(
                    coarse.transpose(1, 2),
                    size=self.config.sequence_length,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            return finest.mean(dim=2)
