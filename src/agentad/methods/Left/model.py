"""Frequency-aware multi-view reconstruction for Left."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from collections.abc import Mapping
from typing import Callable
import lightning as L
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

from .._utils import evaluation_mode, sinusoidal_encoding, validate_series
from .config import LeftConfig
from .._utils import ValidationEarlyStopping


class _SpectralBands(nn.Module):
    cutoffs: Tensor
    order: Tensor
    inverse_order: Tensor

    def __init__(self, config: LeftConfig) -> None:
        super().__init__()
        self.factors = config.scale_factors
        self.boundary = config.spectral_boundary
        self.temperature = config.spectral_temperature
        self.epsilon = config.epsilon
        self.minimum_width = config.spectral_min_bandwidth
        cutoffs = torch.tensor([0.5 / factor for factor in self.factors])
        order = cutoffs.argsort()
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel())
        self.register_buffer("cutoffs", cutoffs)
        self.register_buffer("order", order)
        self.register_buffer("inverse_order", inverse)
        self.edge_logits = nn.Parameter(torch.zeros(len(self.factors)))

    def _edges(self) -> Tensor:
        sorted_cutoffs = self.cutoffs[self.order]
        edges: list[Tensor] = []
        previous = sorted_cutoffs.new_zeros(())
        for cutoff, logit in zip(sorted_cutoffs, self.edge_logits):
            edge = previous + (cutoff - previous).clamp_min(0) * logit.sigmoid()
            edges.append(torch.minimum(edge, cutoff))
            previous = edges[-1]
        return torch.stack(edges)

    def _masks(self, frequencies: Tensor) -> tuple[Tensor, Tensor]:
        edges = self._edges()
        lower = torch.cat((edges.new_zeros(1), edges[:-1]))
        # The original soft step divides by tau + eps.
        width = self.temperature + self.epsilon
        masks = torch.stack(
            [
                (
                    torch.sigmoid((frequencies - low) / width)
                    - torch.sigmoid((frequencies - high) / width)
                ).clamp_min(0)
                for low, high in zip(lower, edges)
            ]
        )
        residual = torch.sigmoid((frequencies - edges[-1]) / width)
        denominator = masks.sum(0) + residual + self.epsilon
        return masks[self.inverse_order] / denominator, residual / denominator

    def forward(self, x: Tensor) -> tuple[list[Tensor], Tensor]:
        batch, time, features = x.shape
        if self.boundary == "reflect":
            extended = torch.cat((x, x.flip(1)), dim=1)
        else:
            extended = x
        spectrum = torch.fft.rfft(extended, dim=1).transpose(1, 2)
        frequencies = torch.linspace(
            0,
            0.5,
            spectrum.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )
        masks, residual_mask = self._masks(frequencies)
        outputs: list[Tensor] = []
        for factor, mask in zip(self.factors, masks):
            filtered = torch.fft.irfft(
                (spectrum * mask[None, None]).transpose(1, 2),
                n=extended.shape[1],
                dim=1,
            )[:, :time]
            output_length = math.ceil(time / factor)
            required = output_length * factor
            if required > time:
                filtered = F.pad(
                    filtered.transpose(1, 2),
                    (0, required - time),
                    mode="replicate",
                ).transpose(1, 2)
            outputs.append(filtered[:, ::factor])
        residual = torch.fft.irfft(
            (spectrum * residual_mask[None, None]).transpose(1, 2),
            n=extended.shape[1],
            dim=1,
        )[:, :time]
        return outputs, residual

    def regularizer(self) -> Tensor:
        if self.minimum_width == 0:
            return self.edge_logits.new_zeros(())
        edges = self._edges()
        lower = torch.cat((edges.new_zeros(1), edges[:-1]))
        return F.relu(self.minimum_width - (edges - lower)).square().mean()


class _TimeFrequencyTransform(nn.Module):
    def __init__(self, config: LeftConfig) -> None:
        super().__init__()
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.window_length = config.window_length
        self.window_epsilon = 1e-6
        self.window_parameter = nn.Parameter(torch.hann_window(config.window_length))

    def _window(self, reference: Tensor) -> Tensor:
        window = F.softplus(self.window_parameter).to(reference) + self.window_epsilon
        return window / window.norm().clamp_min(1e-12) * math.sqrt(self.window_length)

    def forward(self, x: Tensor) -> Tensor:
        batch, time, features = x.shape
        flattened = x.transpose(1, 2).reshape(batch * features, time)
        spectrum = torch.stft(
            flattened,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self._window(flattened),
            center=True,
            return_complex=True,
        )
        real_imag = torch.stack((spectrum.real, spectrum.imag), dim=1)
        return real_imag.reshape(
            batch, features, 2, spectrum.shape[-2], spectrum.shape[-1]
        )

    def inverse(self, real_imag: Tensor, *, length: int) -> Tensor:
        batch, features, _, frequencies, frames = real_imag.shape
        flattened = real_imag.reshape(batch * features, 2, frequencies, frames)
        spectrum = torch.complex(flattened[:, 0], flattened[:, 1])
        result = torch.istft(
            spectrum,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self._window(spectrum.real),
            center=True,
            length=length,
        )
        return result.reshape(batch, features, length).transpose(1, 2)


class _PatchEmbedding(nn.Module):
    def __init__(
        self, patch_length: int, stride: int, model_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.patch_length = patch_length
        self.stride = stride
        self.projection = nn.Linear(patch_length, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x.transpose(1, 2), (0, self.patch_length - 1), mode="replicate")
        patches = x.unfold(-1, self.patch_length, self.stride)
        return self.dropout(
            self.projection(patches.reshape(-1, patches.shape[-2], self.patch_length))
        )


class _ScaleEncoderLayer(nn.Module):
    def __init__(self, config: LeftConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.model_dim,
            config.heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.projection_dropout = nn.Dropout(config.projection_dropout)
        self.norm1 = nn.LayerNorm(config.model_dim)
        self.norm2 = nn.LayerNorm(config.model_dim)
        activation: nn.Module = nn.ReLU() if config.activation == "relu" else nn.GELU()
        self.feedforward = nn.Sequential(
            nn.Linear(config.model_dim, config.feedforward_dim),
            activation,
            nn.Dropout(config.feedforward_dropout),
            nn.Linear(config.feedforward_dim, config.model_dim),
            nn.Dropout(config.feedforward_dropout),
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        attended, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + self.projection_dropout(attended))
        return self.norm2(x + self.feedforward(x))


class _CrossAttentionBlock(nn.Module):
    def __init__(self, config: LeftConfig) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(config.model_dim)
        self.context_norm = nn.LayerNorm(config.model_dim)
        self.attention = nn.MultiheadAttention(
            config.model_dim,
            config.heads,
            dropout=config.fusion_dropout,
            batch_first=True,
        )
        self.output_dropout = nn.Dropout(config.fusion_dropout)
        self.feedforward_norm = nn.LayerNorm(config.model_dim)
        activation: nn.Module = nn.ReLU() if config.activation == "relu" else nn.GELU()
        self.feedforward = nn.Sequential(
            nn.Linear(config.model_dim, config.fusion_feedforward_dim),
            activation,
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_feedforward_dim, config.model_dim),
            nn.Dropout(config.fusion_dropout),
        )

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        normalized_query = self.query_norm(query)
        normalized_context = self.context_norm(context)
        attended, _ = self.attention(
            normalized_query,
            normalized_context,
            normalized_context,
            need_weights=False,
        )
        query = query + self.output_dropout(attended)
        return query + self.feedforward(self.feedforward_norm(query))


class _MultiViewFusion(nn.Module):
    def __init__(self, config: LeftConfig) -> None:
        super().__init__()
        self.time_from_frequency = nn.ModuleList(
            _CrossAttentionBlock(config) for _ in range(config.fusion_layers)
        )
        self.frequency_from_time = nn.ModuleList(
            _CrossAttentionBlock(config) for _ in range(config.fusion_layers)
        )
        self.multiscale_from_time = nn.ModuleList(
            _CrossAttentionBlock(config) for _ in range(config.fusion_layers)
        )

    def forward(
        self, time: Tensor, frequency: Tensor, multiscale: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        for time_layer, frequency_layer, multiscale_layer in zip(
            self.time_from_frequency,
            self.frequency_from_time,
            self.multiscale_from_time,
        ):
            time = time_layer(time, frequency)
            frequency = frequency_layer(frequency, time)
            multiscale = multiscale_layer(multiscale, time)
        return time, frequency, multiscale


class _PrototypeBank(nn.Module):
    def __init__(self, config: LeftConfig) -> None:
        super().__init__()
        self.temperature = config.prototype_temperature
        self.epsilon = config.epsilon
        self.prototypes = nn.Parameter(
            torch.randn(config.prototypes, config.model_dim) * 0.02
        )

    def normalize(self, x: Tensor) -> Tensor:
        # The original divides by norm + eps rather than clamping the norm.
        return x / (x.norm(p=2, dim=-1, keepdim=True) + self.epsilon)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        similarity = torch.einsum(
            "bld,md->blm", self.normalize(tokens), self.normalize(self.prototypes)
        )
        membership = (self.temperature * similarity).softmax(-1)
        confidence = membership.max(-1).values
        entropy = -(membership * membership.clamp_min(self.epsilon).log()).sum(-1)
        count = membership.shape[-1]
        entropy = entropy / (math.log(count + 1e-6) if count > 1 else 1.0)
        return membership, confidence, entropy

    @torch.no_grad()
    def update(self, latents: Tensor, assignments: Tensor, momentum: float) -> None:
        latents = self.normalize(latents)
        counts = torch.bincount(assignments, minlength=self.prototypes.shape[0]).to(
            latents.dtype
        )
        sums = torch.zeros_like(self.prototypes).index_add(0, assignments, latents)
        active = counts > 0
        means = sums[active] / counts[active, None]
        # Assign through .data: the update runs between the forward pass and
        # the backward pass of the same training step, and a tracked in-place
        # write would invalidate the saved autograd graph.
        self.prototypes.data[active] = (
            momentum * self.prototypes.data[active] + (1 - momentum) * means
        )


class _TimeEncoder(nn.Module):
    def __init__(self, config: LeftConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(config.input_features, config.model_dim, 5, padding=2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.model_dim, config.model_dim, 3, padding=1),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.model_dim, config.model_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class _FrequencyEncoder(nn.Module):
    def __init__(self, config: LeftConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2 * config.input_features, config.model_dim, 3, padding=1),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Conv2d(config.model_dim, config.model_dim, 3, padding=1),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.projection = nn.Conv1d(config.model_dim, config.model_dim, 1)

    def forward(self, spectrum: Tensor) -> Tensor:
        batch, features, _, frequencies, frames = spectrum.shape
        encoded = self.net(spectrum.reshape(batch, 2 * features, frequencies, frames))
        return self.projection(encoded.mean(2)).transpose(1, 2)


@dataclass(frozen=True, slots=True)
class LeftOutput:
    multiscale_target: Tensor
    multiscale_reconstruction: Tensor
    temporal_reconstruction: Tensor
    spectral_reconstruction: Tensor
    multiscale_full_reconstruction: Tensor
    time_membership: Tensor
    frequency_membership: Tensor
    time_latent: Tensor
    frequency_latent: Tensor
    time_prototype_weights: Tensor
    frequency_prototype_weights: Tensor
    time_confidence: Tensor
    frequency_confidence: Tensor
    disagreement: Tensor
    spectrum: Tensor


@dataclass(frozen=True, slots=True)
class LeftLoss:
    total: Tensor
    multiscale: Tensor
    cycle: Tensor
    consistency: Tensor


class Left(nn.Module):
    """Left detector with learnable spectral scales and deep multi-view fusion."""

    positions: Tensor
    scale_mask: Tensor
    training_steps: Tensor

    def __init__(self, config: LeftConfig) -> None:
        super().__init__()
        self.config = config
        self.stride = config.patch_stride or config.patch_length
        self.spectral_bands = _SpectralBands(config)
        self.transform = _TimeFrequencyTransform(config)
        self.patch_embedding = _PatchEmbedding(
            config.patch_length,
            self.stride,
            config.model_dim,
            config.patch_dropout,
        )
        self.scale_encoder = nn.ModuleList(
            _ScaleEncoderLayer(config) for _ in range(config.encoder_layers)
        )
        self.scale_norm = nn.LayerNorm(config.model_dim)
        self.time_encoder = _TimeEncoder(config)
        self.frequency_encoder = _FrequencyEncoder(config)
        self.fusion = _MultiViewFusion(config)
        activation: nn.Module = nn.ReLU() if config.activation == "relu" else nn.GELU()
        self.multiscale_decoder = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, config.decoder_feedforward_dim),
            activation,
            nn.Dropout(config.decoder_dropout),
            nn.Linear(config.decoder_feedforward_dim, config.model_dim),
            nn.Dropout(config.decoder_dropout),
        )
        self.patch_projection = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, config.patch_length),
        )
        self.time_prototypes = _PrototypeBank(config)
        self.frequency_prototypes = _PrototypeBank(config)
        self.latent_fusion = nn.Sequential(
            nn.LayerNorm(2 * config.model_dim),
            nn.Linear(2 * config.model_dim, config.latent_fusion_hidden),
            nn.GELU(),
            nn.Linear(config.latent_fusion_hidden, config.model_dim),
        )
        self.time_decoder = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(
                config.model_dim,
                config.sequence_length * config.input_features,
            ),
        )
        frequencies = config.n_fft // 2 + 1
        frames = config.sequence_length // config.hop_length + 1
        self.frequency_shape = (config.input_features, 2, frequencies, frames)
        self.frequency_decoder = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, math.prod(self.frequency_shape)),
        )
        self.temporal_lengths = tuple(
            math.ceil(config.sequence_length / factor)
            for factor in config.scale_factors
        ) + (config.sequence_length,)
        self.patch_counts = tuple(
            math.ceil(length / self.stride) for length in self.temporal_lengths
        )
        positions = torch.cat(
            [
                sinusoidal_encoding(count, config.model_dim)
                for count in self.patch_counts[:-1]
            ],
            dim=1,
        )
        self.register_buffer("positions", positions, persistent=False)
        group_ids = torch.cat(
            [
                torch.full((count,), index)
                for index, count in enumerate(self.patch_counts[:-1])
            ]
        )
        self.register_buffer(
            "scale_mask", group_ids[:, None] != group_ids[None, :], persistent=False
        )
        self.register_buffer("training_steps", torch.zeros((), dtype=torch.long))

    def _memory_weight(self) -> float | Tensor:
        if not self.training:
            return self.config.memory_weight_max
        progress = (
            (self.training_steps - self.config.memory_warmup_steps)
            / max(self.config.memory_ramp_steps, 1)
        ).clamp(0, 1)
        return self.config.memory_weight_max * progress

    @staticmethod
    def _js_divergence(left: Tensor, right: Tensor, epsilon: float) -> Tensor:
        left = left.clamp_min(epsilon)
        right = right.clamp_min(epsilon)
        middle = 0.5 * (left + right)
        return 0.5 * (
            (left * (left.log() - middle.log())).sum(-1)
            + (right * (right.log() - middle.log())).sum(-1)
        )

    def _smooth(self, score: Tensor) -> Tensor:
        if self.config.score_smoothing == 1:
            return score
        padding = self.config.score_smoothing // 2
        channels_first = F.pad(
            score.transpose(1, 2), (padding, padding), mode="replicate"
        )
        kernel = score.new_full(
            (score.shape[2], 1, self.config.score_smoothing),
            1 / self.config.score_smoothing,
        )
        return F.conv1d(channels_first, kernel, groups=score.shape[2]).transpose(1, 2)

    def _decode_multiscale(
        self, encoded: Tensor, batch: int, features: int
    ) -> tuple[Tensor, list[Tensor]]:
        groups = encoded.split(self.patch_counts[:-1], dim=1)
        upsampled = [
            F.interpolate(group.transpose(1, 2), size=count, mode="nearest").transpose(
                1, 2
            )
            for group, count in zip(groups, self.patch_counts[1:])
        ]
        tokens = torch.cat(upsampled, dim=1)
        decoded = self.patch_projection(tokens + self.multiscale_decoder(tokens))
        decoded = decoded.flatten(1).reshape(batch, features, -1).transpose(1, 2)
        padded_lengths = tuple(
            count * self.config.patch_length for count in self.patch_counts[1:]
        )
        decoded_groups = [
            group[:, :length]
            for group, length in zip(
                decoded.split(padded_lengths, dim=1), self.temporal_lengths[1:]
            )
        ]
        return torch.cat(decoded_groups, dim=1), decoded_groups

    def forward(self, x: Tensor) -> LeftOutput:
        x = validate_series(x, length=self.config.sequence_length)
        if x.shape[2] != self.config.input_features:
            raise ValueError("x feature count does not match the Left config")
        if self.training:
            self.training_steps.add_(1)
        batch, _, features = x.shape
        scales, _ = self.spectral_bands(x)
        multiscale_target = torch.cat(scales[1:] + [x], dim=1)
        scale_tokens = torch.cat(
            [self.patch_embedding(scale) for scale in scales], dim=1
        )
        scale_tokens = scale_tokens + self.positions.to(scale_tokens.dtype)
        for layer in self.scale_encoder:
            scale_tokens = layer(scale_tokens, self.scale_mask)
        scale_tokens = self.scale_norm(scale_tokens)

        spectrum = self.transform(x)
        time_tokens = self.time_encoder(x)
        frequency_tokens = self.frequency_encoder(spectrum)
        expanded_scale = scale_tokens.reshape(
            batch, features * scale_tokens.shape[1], self.config.model_dim
        )
        time_tokens, frequency_tokens, fused_scale = self.fusion(
            time_tokens, frequency_tokens, expanded_scale
        )
        fused_scale = fused_scale.reshape(
            batch * features, scale_tokens.shape[1], self.config.model_dim
        )
        scale_tokens = (
            self.config.residual_weight * scale_tokens
            + self.config.fusion_weight * fused_scale
        )
        multiscale_reconstruction, decoded_groups = self._decode_multiscale(
            scale_tokens, batch, features
        )
        multiscale_full = decoded_groups[-1]

        time_membership, time_confidence, time_entropy = self.time_prototypes(
            time_tokens
        )
        frequency_membership, frequency_confidence, _ = self.frequency_prototypes(
            frequency_tokens
        )
        aligned_frequency_membership = F.interpolate(
            frequency_membership.transpose(1, 2),
            size=self.config.sequence_length,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        disagreement = self._js_divergence(
            time_membership, aligned_frequency_membership, self.config.epsilon
        )

        time_latent = time_tokens.mean(1)
        frequency_latent = frequency_tokens.mean(1)
        time_weights = time_membership.mean(1)
        frequency_weights = frequency_membership.mean(1)
        memory_weight = self._memory_weight()
        time_latent = (1 - memory_weight) * time_latent + memory_weight * (
            time_weights
            @ self.time_prototypes.normalize(self.time_prototypes.prototypes)
        )
        frequency_latent = (1 - memory_weight) * frequency_latent + memory_weight * (
            frequency_weights
            @ self.frequency_prototypes.normalize(self.frequency_prototypes.prototypes)
        )
        joint = self.latent_fusion(torch.cat((time_latent, frequency_latent), dim=1))
        temporal_reconstruction = self.time_decoder(joint).reshape_as(x)
        reconstructed_spectrum = self.frequency_decoder(joint).reshape(
            batch, *self.frequency_shape
        )
        spectral_reconstruction = self.transform.inverse(
            reconstructed_spectrum, length=self.config.sequence_length
        )

        return LeftOutput(
            multiscale_target,
            multiscale_reconstruction,
            temporal_reconstruction,
            spectral_reconstruction,
            multiscale_full,
            time_membership,
            aligned_frequency_membership,
            time_tokens.mean(1),
            frequency_tokens.mean(1),
            time_weights,
            frequency_weights,
            time_confidence,
            frequency_confidence,
            disagreement,
            spectrum,
        )

    def compute_loss_from_output(self, x: Tensor, output: LeftOutput) -> LeftLoss:
        multiscale_errors = F.smooth_l1_loss(
            output.multiscale_reconstruction,
            output.multiscale_target,
            reduction="none",
        ).split(self.temporal_lengths[1:], dim=1)
        multiscale = torch.stack([error.mean() for error in multiscale_errors]).mean()
        multiscale = (
            multiscale
            + self.config.band_regularization_weight * self.spectral_bands.regularizer()
        )
        cycle = F.smooth_l1_loss(
            self.transform(output.temporal_reconstruction), output.spectrum
        ) + F.smooth_l1_loss(output.spectral_reconstruction, x)
        consistency = F.smooth_l1_loss(
            output.multiscale_full_reconstruction,
            output.spectral_reconstruction.detach(),
        )
        total = (
            self.config.multiscale_loss_weight * multiscale
            + self.config.cycle_loss_weight * cycle
            + self.config.consistency_loss_weight * consistency
        )
        return LeftLoss(total, multiscale, cycle, consistency)

    def compute_loss(self, x: Tensor) -> LeftLoss:
        return self.compute_loss_from_output(x, self(x))

    @torch.no_grad()
    def update_prototypes(self, x: Tensor, output: LeftOutput) -> None:
        """Apply the confidence-filtered EMA update inside the training iteration.

        The original runs this update inside its training forward pass, before
        the backward pass and optimizer step of the same iteration.
        """
        error = (output.spectral_reconstruction - x).abs().mean((1, 2))
        disagreement = output.disagreement.mean(1)
        good = (
            (error <= torch.quantile(error, self.config.update_error_quantile))
            & (
                disagreement
                <= torch.quantile(
                    disagreement,
                    self.config.update_disagreement_quantile,
                )
            )
            & (output.time_confidence.mean(1) >= self.config.confidence_threshold)
            & (output.frequency_confidence.mean(1) >= self.config.confidence_threshold)
        )
        if good.any():
            self.time_prototypes.update(
                output.time_latent[good],
                output.time_prototype_weights[good].argmax(1),
                self.config.prototype_momentum,
            )
            self.frequency_prototypes.update(
                output.frequency_latent[good],
                output.frequency_prototype_weights[good].argmax(1),
                self.config.prototype_momentum,
            )

    def _score(self, x: Tensor) -> Tensor:
        output = self(x)
        multiscale_errors = F.smooth_l1_loss(
            output.multiscale_reconstruction,
            output.multiscale_target,
            reduction="none",
        ).split(self.temporal_lengths[1:], dim=1)
        multiscale_score = multiscale_errors[-1]
        for coarse in multiscale_errors[:-1]:
            multiscale_score = multiscale_score + F.interpolate(
                coarse.transpose(1, 2),
                size=self.config.sequence_length,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        time_probability = output.time_membership
        frequency_probability = output.frequency_membership
        # The original normalizes entropy by log(M + 1e-6), or 1.0 when M == 1.
        time_entropy = -(
            time_probability * time_probability.clamp_min(self.config.epsilon).log()
        ).sum(-1) / (
            math.log(time_probability.shape[-1] + 1e-6)
            if time_probability.shape[-1] > 1
            else 1.0
        )
        frequency_entropy = -(
            frequency_probability
            * frequency_probability.clamp_min(self.config.epsilon).log()
        ).sum(-1) / (
            math.log(frequency_probability.shape[-1] + 1e-6)
            if frequency_probability.shape[-1] > 1
            else 1.0
        )
        gate = 0.25 * (
            2
            - time_probability.max(-1).values
            - frequency_probability.max(-1).values
            + time_entropy
            + frequency_entropy
        )
        cycle_score = (
            self.config.frequency_score_weight
            * (output.spectral_reconstruction - x).abs()
            + self.config.time_score_weight * (output.temporal_reconstruction - x).abs()
            + self.config.gate_score_weight * gate.unsqueeze(-1)
            + self.config.consistency_score_weight
            * (
                output.multiscale_full_reconstruction - output.spectral_reconstruction
            ).abs()
        )
        cycle_score = self._smooth(cycle_score)
        score = (
            self.config.cycle_score_weight * cycle_score
            + self.config.multiscale_score_weight * multiscale_score
        )
        return score.mean(2)

    @torch.inference_mode()
    def score(self, x: Tensor) -> Tensor:
        with evaluation_mode(self):
            return self._score(x)


class LeftLightningModule(ValidationEarlyStopping, L.LightningModule):
    """Data contract: the original slides training windows at stride one,
    validates and scores on non-overlapping windows (stride
    ``sequence_length``), and z-scores every split independently; the
    caller owns windowing and standardization."""

    def __init__(self, config: LeftConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Left(config)
        # The original stopper treats val_loss <= best as an improvement.
        self._init_validation_early_stopping(
            monitor="val/loss",
            patience=config.patience,
            ties_improve=True,
        )

    def forward(self, series: Tensor):
        return self.model(series)

    @staticmethod
    def _series(batch: object) -> Tensor:
        series: Tensor | None
        if isinstance(batch, Tensor):
            series = batch
        elif isinstance(batch, Mapping):
            series = batch.get("series")
        elif isinstance(batch, (tuple, list)) and batch:
            series = batch[0]
        else:
            series = None
        if not isinstance(series, Tensor):
            raise TypeError("Left batches must provide a series tensor")
        return series

    def _step(self, batch: object, stage: str) -> Tensor:
        series = self._series(batch)
        output = self.model(series)
        if stage == "train":
            self.model.update_prototypes(series, output)
        losses = self.model.compute_loss_from_output(series, output)
        for name in ("total", "multiscale", "cycle", "consistency"):
            metric = "loss" if name == "total" else name
            self.log(
                f"{stage}/{metric}",
                getattr(losses, name),
                on_step=stage == "train" and name == "total",
                on_epoch=True,
                prog_bar=name == "total",
                sync_dist=True,
                # batch_size=1 makes Lightning's epoch reduction an unweighted
                # mean over batches, matching the original np.average(val_loss).
                batch_size=series.shape[0] if stage == "train" else 1,
            )
        return losses.total

    def training_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def predict_step(
        self, batch: object, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        return self.model.score(self._series(batch))

    def _type2_schedule(self) -> Callable[[int], float]:
        milestones = (
            (2, 5e-5),
            (4, 1e-5),
            (6, 5e-6),
            (8, 1e-6),
            (10, 5e-7),
            (15, 1e-7),
            (20, 5e-8),
        )

        def scale(epoch: int) -> float:
            # The original applies milestone m at the end of epoch m-1, so
            # epoch m is the first one running with the milestone value.
            current = self.config.learning_rate
            for milestone, value in milestones:
                if epoch < milestone:
                    break
                current = value
            return current / self.config.learning_rate

        return scale

    @staticmethod
    def _type1_schedule() -> Callable[[int], float]:
        # The original halves the learning rate only from the third epoch on:
        # it calls _adjust_learning_rate(epoch + 1) after each epoch, and the
        # first call keeps the base rate for the second epoch.
        def scale(epoch: int) -> float:
            return 0.5 ** max(0, epoch - 1)

        return scale

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        if self.config.learning_rate_schedule == "type1":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, self._type1_schedule()
            )
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, self._type2_schedule()
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
