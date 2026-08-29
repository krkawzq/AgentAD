"""Relative-context discrepancy encoder for zero-shot anomaly detection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from collections.abc import Mapping
import lightning as L
from torch import Tensor

from .._utils import evaluation_mode, validate_series
from .config import TimeRCDConfig


class _RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:
        normalized = x * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + self.epsilon
        )
        return (normalized * self.scale).to(x.dtype)


class _RotaryEmbedding(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.register_buffer(
            "inv_freq",
            1 / (10_000 ** (torch.arange(0, width, 2).float() / width)),
        )

    def forward(self, length: int) -> Tensor:
        positions = torch.arange(length, device=self.inv_freq.device).to(self.inv_freq)
        return torch.outer(positions, self.inv_freq)


class _BinaryFeatureBias(nn.Module):
    def __init__(self, heads: int) -> None:
        super().__init__()
        self.emd = nn.Embedding(2, heads)

    def forward(self, feature_ids: Tensor) -> Tensor:
        equal = feature_ids[:, :, None] == feature_ids[:, None, :]
        different = self.emd.weight[0][None, :, None, None]
        same = self.emd.weight[1][None, :, None, None]
        return torch.where(equal[:, None], same, different)


class _RelativeAttention(nn.Module):
    def __init__(self, config: TimeRCDConfig) -> None:
        super().__init__()
        self.num_heads = config.heads
        self.head_dim = config.model_dim // config.heads
        self.q_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.k_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.v_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.out_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.binary_attention_bias = (
            _BinaryFeatureBias(config.heads) if config.input_features > 1 else None
        )

    @staticmethod
    def _apply_rope(x: Tensor, frequencies: Tensor) -> Tensor:
        paired = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        cosine = frequencies.cos()[None, :, :, None]
        sine = frequencies.sin()[None, :, :, None]
        return torch.stack(
            (
                paired[..., 0] * cosine[..., 0] - paired[..., 1] * sine[..., 0],
                paired[..., 0] * sine[..., 0] + paired[..., 1] * cosine[..., 0],
            ),
            dim=-1,
        ).flatten(-2)

    def forward(
        self,
        x: Tensor,
        frequencies: Tensor,
        feature_ids: Tensor,
        valid_tokens: Tensor,
    ) -> Tensor:
        batch, tokens, width = x.shape
        query = self._apply_rope(self.q_proj(x), frequencies)
        key = self._apply_rope(self.k_proj(x), frequencies)
        value = self.v_proj(x)
        query = query.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        key = key.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        if self.binary_attention_bias is None:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=valid_tokens[:, None, None],
            )
        else:
            logits = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
            logits = logits + self.binary_attention_bias(feature_ids)
            logits.masked_fill_(~valid_tokens[:, None, None], -torch.inf)
            attended = logits.softmax(-1) @ value
        attended = attended.transpose(1, 2).reshape(batch, tokens, width)
        return self.out_proj(attended)


class _GatedMLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(width, 4 * width)
        self.up_proj = nn.Linear(width, 4 * width)
        self.down_proj = nn.Linear(4 * width, width)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))


class _EncoderLayer(nn.Module):
    def __init__(self, config: TimeRCDConfig) -> None:
        super().__init__()
        self.self_attn = _RelativeAttention(config)
        self.input_norm = _RMSNorm(config.model_dim, config.epsilon)
        self.output_norm = _RMSNorm(config.model_dim, config.epsilon)
        self.mlp = _GatedMLP(config.model_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        frequencies: Tensor,
        feature_ids: Tensor,
        valid_tokens: Tensor,
    ) -> Tensor:
        x = x + self.self_attn(
            self.input_norm(x), frequencies, feature_ids, valid_tokens
        )
        return x + self.dropout(self.mlp(self.output_norm(x)))


class _TimeSeriesEncoder(nn.Module):
    def __init__(self, config: TimeRCDConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding_layer = nn.Linear(config.patch_length, config.model_dim)
        self.rope_embedder = _RotaryEmbedding(config.model_dim)
        self.transformer_encoder = nn.Module()
        self.transformer_encoder.layers = nn.ModuleList(
            _EncoderLayer(config) for _ in range(config.layers)
        )
        self.projection_layer = nn.Linear(
            config.model_dim, config.patch_length * config.projection_dim
        )
        # The original encoder zeroes every parameter whose full name contains
        # "bias" after construction (its weight branch never matches), which
        # also covers the binary-feature Embedding weight; from-scratch
        # pretraining starts from those zeroed parameters.
        for name, parameter in self.named_parameters():
            if "bias" in name:
                nn.init.zeros_(parameter)

    def forward(self, series: Tensor, valid_mask: Tensor) -> Tensor:
        batch, time, features = series.shape
        padded_length = (
            math.ceil(time / self.config.patch_length) * self.config.patch_length
        )
        if padded_length != time:
            series = F.pad(series.transpose(1, 2), (0, padded_length - time)).transpose(
                1, 2
            )
            valid_mask = F.pad(valid_mask, (0, padded_length - time), value=False)
        patch_count = padded_length // self.config.patch_length
        patches = (
            series.reshape(batch, patch_count, self.config.patch_length, features)
            .permute(0, 3, 1, 2)
            .reshape(batch, features * patch_count, self.config.patch_length)
        )
        feature_ids = torch.arange(features, device=series.device).repeat_interleave(
            patch_count
        )
        feature_ids = feature_ids[None].expand(batch, -1)
        patch_mask = valid_mask.reshape(
            batch, patch_count, self.config.patch_length
        ).any(-1)
        token_mask = patch_mask[:, None].expand(-1, features, -1).reshape(batch, -1)
        hidden = self.embedding_layer(patches)
        frequencies = self.rope_embedder(hidden.shape[1]).to(hidden.device)
        for layer in self.transformer_encoder.layers:
            hidden = layer(hidden, frequencies, feature_ids, token_mask)
        projected = self.projection_layer(hidden).reshape(
            batch,
            features,
            patch_count,
            self.config.patch_length,
            self.config.projection_dim,
        )
        return projected.permute(0, 2, 3, 1, 4).reshape(
            batch, padded_length, features, self.config.projection_dim
        )[:, :time]


@dataclass(frozen=True, slots=True)
class TimeRCDOutput:
    embeddings: Tensor
    logits: Tensor
    reconstruction: Tensor


@dataclass(frozen=True, slots=True)
class TimeRCDLoss:
    total: Tensor
    reconstruction: Tensor
    classification: Tensor


class TimeRCD(nn.Module):
    """Time-RCD encoder with masked reconstruction and anomaly heads."""

    def __init__(self, config: TimeRCDConfig) -> None:
        super().__init__()
        self.config = config
        self.ts_encoder = _TimeSeriesEncoder(config)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(config.projection_dim, 4 * config.projection_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.projection_dim, 4 * config.projection_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.projection_dim, 1),
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(config.projection_dim, config.projection_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.projection_dim // 2, 2),
        )

    def forward(
        self, series: Tensor, valid_mask: Tensor | None = None
    ) -> TimeRCDOutput:
        series = validate_series(series, min_length=1)
        if series.shape[2] != self.config.input_features:
            raise ValueError("series feature count does not match the Time-RCD config")
        if valid_mask is None:
            valid_mask = torch.ones(
                series.shape[:2], device=series.device, dtype=torch.bool
            )
        if valid_mask.shape != series.shape[:2]:
            raise ValueError("valid_mask must have shape [batch, time]")
        valid_mask = valid_mask.bool()
        if not valid_mask.any(1).all():
            raise ValueError("every sample must contain at least one valid time point")
        embeddings = self.ts_encoder(series, valid_mask)
        logits = self.anomaly_head(embeddings).mean(2)
        reconstruction = self.reconstruction_head(embeddings).squeeze(-1)
        return TimeRCDOutput(embeddings, logits, reconstruction)

    def _random_mask(self, series: Tensor, generator: torch.Generator | None) -> Tensor:
        patches = math.ceil(series.shape[1] / self.config.patch_length)
        count = int(patches * self.config.mask_ratio)
        selected = torch.zeros(
            series.shape[0], patches, device=series.device, dtype=torch.bool
        )
        for batch_index in range(series.shape[0]):
            indices = torch.randperm(
                patches, device=series.device, generator=generator
            )[:count]
            selected[batch_index, indices] = True
        return selected.repeat_interleave(self.config.patch_length, dim=1)[
            :, : series.shape[1]
        ]

    def compute_loss(
        self,
        series: Tensor,
        *,
        labels: Tensor | None = None,
        masked_points: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> TimeRCDLoss:
        series = validate_series(series, min_length=1)
        if masked_points is None:
            masked_points = self._random_mask(series, generator)
        if masked_points.shape != series.shape[:2]:
            raise ValueError("masked_points must have shape [batch, time]")
        masked_points = masked_points.bool()
        if not masked_points.any():
            raise ValueError("masked_points must select at least one time point")
        expanded_mask = masked_points[..., None].expand_as(series)
        noise = 0.1 * torch.randn(
            series.shape,
            device=series.device,
            dtype=series.dtype,
            generator=generator,
        )
        masked = torch.where(expanded_mask, noise, series)
        output = self(masked)
        reconstruction = F.mse_loss(
            output.reconstruction[expanded_mask], series[expanded_mask]
        )
        if labels is None:
            classification = reconstruction.new_zeros(())
        else:
            if labels.shape != series.shape[:2]:
                raise ValueError("labels must have shape [batch, time]")
            # The original binarizes labels before its padding check, so -1
            # padding is silently treated as class 0; excluding the padding
            # first follows the original's documented intent.
            valid = labels.flatten() >= 0
            classification = (
                F.cross_entropy(
                    output.logits.flatten(0, 1)[valid],
                    (labels.flatten()[valid] > 0).long(),
                )
                if valid.any()
                else reconstruction.new_zeros(())
            )
        # The vendored inference-only package defines no training loop, so the
        # equal-weight combination is the natural reading of its two losses.
        total = (
            self.config.reconstruction_weight * reconstruction
            + self.config.classification_weight * classification
        )
        return TimeRCDLoss(total, reconstruction, classification)

    @torch.inference_mode()
    def score(self, series: Tensor, *, batch_size: int = 16) -> Tensor:
        series = validate_series(series, min_length=1)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        with evaluation_mode(self):
            mean = series.mean(1, keepdim=True)
            # The original replaces only exactly-constant channels instead of
            # clamping small scales.
            scale = series.std(1, keepdim=True, unbiased=False)
            scale = torch.where(
                scale == 0, scale.new_full((), 1e-8), scale
            )
            normalized = (series - mean) / scale
            batch, time, features = normalized.shape
            window = min(time, self.config.inference_window)
            padded_length = math.ceil(time / window) * window
            if padded_length != time:
                normalized = F.pad(
                    normalized.transpose(1, 2),
                    (0, padded_length - time),
                    mode="replicate",
                ).transpose(1, 2)
            chunks = normalized.reshape(batch, -1, window, features)
            valid = torch.arange(window, device=series.device)[None, :] < (
                time
                - torch.arange(chunks.shape[1], device=series.device)[:, None] * window
            ).clamp(min=0, max=window)
            valid = valid[None].expand(batch, -1, -1).reshape(-1, window)
            flattened = chunks.flatten(0, 1)
            probabilities = torch.cat(
                [
                    self(data, mask).logits.softmax(-1)[..., 1]
                    for data, mask in zip(
                        flattened.split(batch_size), valid.split(batch_size)
                    )
                ]
            )
            return probabilities.reshape(batch, -1)[:, :time]

    def load_checkpoint(self, path: str | Path, *, strict: bool = True) -> None:
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        if not isinstance(state, dict):
            raise TypeError("checkpoint does not contain a state dictionary")
        state = {key.removeprefix("module."): value for key, value in state.items()}
        self.load_state_dict(state, strict=strict)


class TimeRCDLightningModule(L.LightningModule):
    def __init__(self, config: TimeRCDConfig) -> None:
        super().__init__()
        self.config = config
        self.model = TimeRCD(config)
        if not config.train_encoder:
            for parameter in self.model.ts_encoder.parameters():
                parameter.requires_grad_(False)

    def forward(self, series: Tensor, valid_mask: Tensor | None = None):
        return self.model(series, valid_mask)

    @staticmethod
    def _batch(batch: object) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if isinstance(batch, Tensor):
            return batch, None, None
        if isinstance(batch, Mapping):
            series = batch.get("series", batch.get("time_series"))
            labels = batch.get("labels")
            masked = batch.get("masked_points", batch.get("mask"))
        elif isinstance(batch, (tuple, list)) and batch:
            series = batch[0]
            labels = batch[1] if len(batch) > 1 else None
            masked = batch[2] if len(batch) > 2 else None
        else:
            series = labels = masked = None
        if not isinstance(series, Tensor):
            raise TypeError("Time-RCD batches must provide a series tensor")
        if labels is not None and not isinstance(labels, Tensor):
            raise TypeError("Time-RCD labels must be a tensor")
        if masked is not None and not isinstance(masked, Tensor):
            raise TypeError("Time-RCD mask must be a tensor")
        return series, labels, masked

    def _step(self, batch: object, stage: str) -> Tensor:
        series, labels, masked = self._batch(batch)
        losses = self.model.compute_loss(
            series,
            labels=labels,
            masked_points=masked,
        )
        for name in ("total", "reconstruction", "classification"):
            metric = "loss" if name == "total" else name
            self.log(
                f"{stage}/{metric}",
                getattr(losses, name),
                on_step=stage == "train" and name == "total",
                on_epoch=True,
                prog_bar=name == "total",
                sync_dist=True,
                batch_size=series.shape[0],
            )
        return losses.total

    def training_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def predict_step(
        self, batch: object, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        series, _, _ = self._batch(batch)
        return self.model.score(series, batch_size=self.config.batch_size)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        return torch.optim.AdamW(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
