"""Configuration for Left."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class LeftConfig:
    input_features: int
    sequence_length: int = 96
    n_fft: int = 64
    hop_length: int = 16
    window_length: int = 64
    scale_factors: tuple[int, ...] = (4, 2, 1)
    spectral_boundary: Literal["reflect", "periodic"] = "reflect"
    spectral_temperature: float = 0.03
    spectral_min_bandwidth: float = 0.0
    patch_length: int = 6
    patch_stride: int | None = None
    model_dim: int = 128
    encoder_layers: int = 1
    heads: int = 8
    feedforward_dim: int = 512
    fusion_layers: int = 1
    dropout: float = 0.1
    activation: Literal["relu", "gelu"] = "gelu"
    prototypes: int = 16
    prototype_temperature: float = 5.0
    prototype_momentum: float = 0.99
    confidence_threshold: float = 0.85
    update_error_quantile: float = 0.3
    update_disagreement_quantile: float = 0.3
    memory_warmup_steps: int = 500
    memory_ramp_steps: int = 1_000
    memory_weight_max: float = 0.3
    residual_weight: float = 0.6
    fusion_weight: float = 0.4
    multiscale_loss_weight: float = 0.5
    cycle_loss_weight: float = 0.5
    consistency_loss_weight: float = 0.1
    band_regularization_weight: float = 0.0
    multiscale_score_weight: float = 0.3
    cycle_score_weight: float = 0.7
    consistency_score_weight: float = 0.1
    frequency_score_weight: float = 0.3
    time_score_weight: float = 0.7
    gate_score_weight: float = 1.0
    score_smoothing: int = 15
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        for name in (
            "input_features",
            "sequence_length",
            "n_fft",
            "hop_length",
            "window_length",
            "patch_length",
            "model_dim",
            "encoder_layers",
            "heads",
            "feedforward_dim",
            "fusion_layers",
            "prototypes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.patch_stride is not None and self.patch_stride <= 0:
            raise ValueError("patch_stride must be positive when provided")
        if self.window_length > self.n_fft:
            raise ValueError("window_length cannot exceed n_fft")
        if self.sequence_length <= self.n_fft // 2:
            raise ValueError(
                "sequence_length must exceed n_fft // 2 for reflected STFT padding"
            )
        if not self.scale_factors or any(factor <= 0 for factor in self.scale_factors):
            raise ValueError("scale_factors must contain positive integers")
        if any(
            left < right
            for left, right in zip(self.scale_factors, self.scale_factors[1:])
        ):
            raise ValueError("scale_factors must be ordered from coarse to fine")
        if self.model_dim % self.heads:
            raise ValueError("model_dim must be divisible by heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.spectral_temperature <= 0 or self.epsilon <= 0:
            raise ValueError("spectral_temperature and epsilon must be positive")
        if self.spectral_min_bandwidth < 0:
            raise ValueError("spectral_min_bandwidth must be non-negative")
        if not 0 <= self.prototype_momentum < 1:
            raise ValueError("prototype_momentum must be in [0, 1)")
        for name in (
            "confidence_threshold",
            "update_error_quantile",
            "update_disagreement_quantile",
            "memory_weight_max",
            "residual_weight",
            "fusion_weight",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "multiscale_loss_weight",
            "cycle_loss_weight",
            "consistency_loss_weight",
            "band_regularization_weight",
            "multiscale_score_weight",
            "cycle_score_weight",
            "consistency_score_weight",
            "frequency_score_weight",
            "time_score_weight",
            "gate_score_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.memory_warmup_steps < 0 or self.memory_ramp_steps < 0:
            raise ValueError("memory curriculum steps must be non-negative")
        if self.score_smoothing <= 0 or self.score_smoothing % 2 == 0:
            raise ValueError("score_smoothing must be a positive odd integer")
