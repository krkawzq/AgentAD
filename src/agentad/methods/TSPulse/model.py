"""Time/FFT patch mixer used by TSPulse zero-shot and fine-tuned modes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .._utils import evaluation_mode, sliding_windows, validate_series
from .config import TSPulseConfig


class _MixerLayer(nn.Module):
    def __init__(self, config: TSPulseConfig, token_count: int) -> None:
        super().__init__()
        hidden = config.expansion_factor * config.model_dim
        self.features = config.input_features
        self.mix_channels = config.channel_mode == "mix" and config.input_features > 1
        self.norm1 = nn.LayerNorm(config.model_dim)
        self.token_mixer = nn.Conv1d(
            config.model_dim,
            config.model_dim,
            3,
            padding=1,
            groups=config.model_dim,
        )
        self.gate = nn.Linear(config.model_dim, 1)
        self.norm2 = nn.LayerNorm(config.model_dim)
        self.feature_mixer = nn.Sequential(
            nn.Linear(config.model_dim, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.model_dim),
            nn.Dropout(config.dropout),
        )
        self.channel_mixer = (
            nn.Linear(config.input_features, config.input_features, bias=False)
            if self.mix_channels
            else None
        )
        self.token_count = token_count

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, tokens, width = x.shape
        normalized = self.norm1(x).reshape(batch * channels, tokens, width)
        mixed = self.token_mixer(normalized.transpose(1, 2)).transpose(1, 2)
        gate = self.gate(normalized).softmax(1)
        x = x + (mixed * (1 + gate)).reshape_as(x)
        x = x + self.feature_mixer(self.norm2(x))
        if self.channel_mixer is not None:
            x = x + self.channel_mixer(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


@dataclass(frozen=True, slots=True)
class TSPulseOutput:
    time_reconstruction: Tensor
    frequency_reconstruction: Tensor
    forecast: Tensor
    masked_points: Tensor


@dataclass(frozen=True, slots=True)
class TSPulseLoss:
    total: Tensor
    time_reconstruction: Tensor
    frequency_reconstruction: Tensor
    forecast: Tensor


class TSPulse(nn.Module):
    """Shared TSPulse architecture for zero-shot scoring and fine-tuning."""

    def __init__(self, config: TSPulseConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_count = config.context_length // config.patch_length
        token_count = 2 * self.patch_count + config.register_tokens
        self.time_embedding = nn.Linear(config.patch_length, config.model_dim)
        self.frequency_embedding = nn.Linear(2, config.model_dim)
        self.register = nn.Parameter(
            torch.randn(config.register_tokens, config.model_dim) * 0.02
        )
        self.mask_token = nn.Parameter(torch.zeros(config.patch_length))
        self.encoder = nn.ModuleList(
            _MixerLayer(config, token_count) for _ in range(config.layers)
        )
        self.decoder = nn.ModuleList(
            _MixerLayer(config, token_count) for _ in range(config.decoder_layers)
        )
        self.time_head = nn.Linear(config.model_dim, config.patch_length)
        self.frequency_head = nn.Linear(config.model_dim, 2)
        self.forecast_head = nn.Linear(
            config.model_dim, config.forecast_length * config.input_features
        )
        if config.revin_affine:
            self.revin_weight = nn.Parameter(torch.ones(1, 1, config.input_features))
            self.revin_bias = nn.Parameter(torch.zeros(1, 1, config.input_features))
        else:
            self.register_parameter("revin_weight", None)
            self.register_parameter("revin_bias", None)

    def _normalize(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        location = x.mean(1, keepdim=True)
        scale = x.var(1, keepdim=True, unbiased=False).add(self.config.epsilon).sqrt()
        normalized = (x - location) / scale
        if self.revin_weight is not None:
            normalized = normalized * self.revin_weight + self.revin_bias
        return normalized, location, scale

    def _denormalize(self, x: Tensor, location: Tensor, scale: Tensor) -> Tensor:
        if self.revin_weight is not None:
            x = (x - self.revin_bias) / self.revin_weight.clamp_min(self.config.epsilon)
        return x * scale + location

    def _random_observed_mask(
        self, series: Tensor, generator: torch.Generator | None
    ) -> Tensor:
        selected = (
            torch.rand(
                series.shape[0],
                self.patch_count,
                device=series.device,
                generator=generator,
            )
            < self.config.mask_ratio
        )
        selected[:, 0] |= ~selected.any(1)
        masked = selected.repeat_interleave(self.config.patch_length, 1)
        return ~masked[..., None].expand_as(series)

    def forward(
        self,
        series: Tensor,
        observed_mask: Tensor | None = None,
    ) -> TSPulseOutput:
        series = validate_series(series, length=self.config.context_length)
        if series.shape[2] != self.config.input_features:
            raise ValueError("series feature count does not match the TSPulse config")
        if observed_mask is None:
            observed_mask = torch.ones_like(series, dtype=torch.bool)
        if observed_mask.shape != series.shape:
            raise ValueError("observed_mask must match series shape")
        normalized, location, scale = self._normalize(series)
        relative = torch.arange(self.config.context_length, device=series.device)
        replacement = self.mask_token[relative % self.config.patch_length]
        replacement = replacement[None, :, None].expand_as(normalized)
        masked = torch.where(observed_mask, normalized, replacement)

        time_patches = masked.transpose(1, 2).reshape(
            series.shape[0],
            self.config.input_features,
            self.patch_count,
            self.config.patch_length,
        )
        time_tokens = self.time_embedding(time_patches)
        spectrum = torch.fft.rfft(masked, dim=1)
        frequency_values = torch.stack((spectrum.real, spectrum.imag), dim=-1)
        frequency_values = (
            F.interpolate(
                frequency_values.permute(0, 2, 3, 1).reshape(
                    series.shape[0] * self.config.input_features, 2, -1
                ),
                size=self.patch_count,
                mode="linear",
                align_corners=False,
            )
            .reshape(series.shape[0], self.config.input_features, 2, self.patch_count)
            .permute(0, 1, 3, 2)
        )
        frequency_tokens = self.frequency_embedding(frequency_values)
        register = self.register[None, None].expand(
            series.shape[0], self.config.input_features, -1, -1
        )
        hidden = torch.cat((time_tokens, frequency_tokens, register), dim=2)
        for layer in self.encoder:
            hidden = layer(hidden)
        for layer in self.decoder:
            hidden = layer(hidden)

        time_values = (
            self.time_head(hidden[:, :, : self.patch_count])
            .reshape(
                series.shape[0], self.config.input_features, self.config.context_length
            )
            .transpose(1, 2)
        )
        frequency_values = self.frequency_head(
            hidden[:, :, self.patch_count : 2 * self.patch_count]
        )
        frequency_values = F.interpolate(
            frequency_values.permute(0, 1, 3, 2).reshape(
                series.shape[0] * self.config.input_features, 2, self.patch_count
            ),
            size=self.config.context_length // 2 + 1,
            mode="linear",
            align_corners=False,
        ).reshape(series.shape[0], self.config.input_features, 2, -1)
        reconstructed_spectrum = torch.complex(
            frequency_values[:, :, 0], frequency_values[:, :, 1]
        ).transpose(1, 2)
        frequency_reconstruction = torch.fft.irfft(
            reconstructed_spectrum,
            n=self.config.context_length,
            dim=1,
        )
        pooled = hidden.mean((1, 2))
        forecast = self.forecast_head(pooled).reshape(
            series.shape[0], self.config.forecast_length, self.config.input_features
        )
        return TSPulseOutput(
            self._denormalize(time_values, location, scale),
            self._denormalize(frequency_reconstruction, location, scale),
            self._denormalize(forecast, location, scale),
            ~observed_mask,
        )

    def compute_loss(
        self,
        series: Tensor,
        *,
        future_values: Tensor | None = None,
        observed_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> TSPulseLoss:
        if observed_mask is None:
            observed_mask = self._random_observed_mask(series, generator)
        output = self(series, observed_mask)
        selected = output.masked_points
        time_loss = F.mse_loss(output.time_reconstruction[selected], series[selected])
        frequency_loss = F.mse_loss(
            output.frequency_reconstruction[selected], series[selected]
        )
        forecast_loss = (
            F.mse_loss(output.forecast, future_values)
            if future_values is not None
            else time_loss.new_zeros(())
        )
        total = (
            self.config.time_loss_weight * time_loss
            + self.config.frequency_loss_weight * frequency_loss
            + self.config.forecast_loss_weight * forecast_loss
        )
        return TSPulseLoss(total, time_loss, frequency_loss, forecast_loss)

    def _stitched_reconstruction(
        self,
        windows: Tensor,
        *,
        frequency: bool,
    ) -> Tensor:
        start = self.config.context_length - self.config.aggregation_length
        reconstruction = windows.clone()
        for patch_start in range(
            start, self.config.context_length, self.config.patch_length
        ):
            observed = torch.ones_like(windows, dtype=torch.bool)
            observed[:, patch_start : patch_start + self.config.patch_length] = False
            output = self(windows, observed)
            source = (
                output.frequency_reconstruction
                if frequency
                else output.time_reconstruction
            )
            reconstruction[:, patch_start : patch_start + self.config.patch_length] = (
                source[:, patch_start : patch_start + self.config.patch_length]
            )
        return reconstruction

    @staticmethod
    def _normalize_score(score: Tensor, epsilon: float) -> Tensor:
        minimum = score.amin(1, keepdim=True)
        maximum = score.amax(1, keepdim=True)
        return (score - minimum) / (maximum - minimum).clamp_min(epsilon)

    def _align_context_scores(self, score: Tensor, output_length: int) -> Tensor:
        left = self.config.context_length - self.config.aggregation_length // 2
        right = output_length - left - score.shape[1]
        return torch.cat(
            (
                score[:, :1].expand(-1, left),
                score,
                score[:, -1:].expand(-1, max(right, 0)),
            ),
            dim=1,
        )[:, :output_length]

    @torch.inference_mode()
    def score(self, series: Tensor, *, batch_size: int = 128) -> Tensor:
        series = validate_series(series, min_length=self.config.context_length + 1)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        with evaluation_mode(self):
            windows = sliding_windows(series, self.config.context_length)
            flattened = windows.flatten(0, 1)
            mode_scores: list[Tensor] = []
            for mode in self.config.score_modes:
                if mode == "forecast":
                    forecast_windows = sliding_windows(
                        series[:, :-1], self.config.context_length
                    )
                    forecast_flat = forecast_windows.flatten(0, 1)
                    targets = series[:, self.config.context_length :].flatten(0, 1)
                    chunks = []
                    offset = 0
                    for chunk in forecast_flat.split(batch_size):
                        output = self(chunk).forecast[:, 0]
                        chunks.append(
                            (output - targets[offset : offset + len(chunk)])
                            .square()
                            .mean(1)
                        )
                        offset += len(chunk)
                    score = torch.cat(chunks).reshape(series.shape[0], -1)
                    score = torch.cat(
                        (
                            score[:, :1].expand(-1, self.config.context_length),
                            score,
                        ),
                        dim=1,
                    )[:, : series.shape[1]]
                else:
                    chunks = []
                    for chunk in flattened.split(batch_size):
                        reconstruction = self._stitched_reconstruction(
                            chunk, frequency=mode == "frequency"
                        )
                        start = (
                            self.config.context_length - self.config.aggregation_length
                        )
                        chunks.append(
                            (reconstruction[:, start:] - chunk[:, start:])
                            .square()
                            .mean((1, 2))
                        )
                    score = torch.cat(chunks).reshape(series.shape[0], -1)
                    score = self._align_context_scores(score, series.shape[1])
                mode_scores.append(self._normalize_score(score, self.config.epsilon))
            combined = torch.stack(mode_scores).mean(0)
            if self.config.smoothing_window > 1:
                kernel = combined.new_full(
                    (1, 1, self.config.smoothing_window),
                    1 / self.config.smoothing_window,
                )
                padding = self.config.smoothing_window // 2
                combined = F.conv1d(
                    F.pad(combined[:, None], (padding, padding), mode="replicate"),
                    kernel,
                )[:, 0, : series.shape[1]]
            return combined


class TSPulseZeroShot(TSPulse):
    """TSPulse used directly with pretrained weights."""


class TSPulseFineTune(TSPulse):
    """TSPulse exposing masked reconstruction fine-tuning through compute_loss()."""
