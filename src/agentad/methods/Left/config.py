"""Configuration for Left."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self


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
    patch_dropout: float = 0.1
    model_dim: int = 128
    encoder_layers: int = 1
    heads: int = 8
    feedforward_dim: int = 512
    attention_dropout: float = 0.1
    projection_dropout: float = 0.1
    feedforward_dropout: float = 0.1
    fusion_layers: int = 1
    fusion_dropout: float = 0.1
    fusion_feedforward_dim: int = 128
    decoder_dropout: float = 0.1
    decoder_feedforward_dim: int = 128
    latent_fusion_hidden: int = 512
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
    train_epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-4
    learning_rate_schedule: Literal["type1", "type2"] = "type1"
    patience: int = 3

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
            "fusion_feedforward_dim",
            "decoder_feedforward_dim",
            "latent_fusion_hidden",
            "train_epochs",
            "batch_size",
            "patience",
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
        for name in (
            "dropout",
            "patch_dropout",
            "attention_dropout",
            "projection_dropout",
            "feedforward_dropout",
            "fusion_dropout",
            "decoder_dropout",
        ):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
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
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    @classmethod
    def _original(cls, **values: object) -> Self:
        return cls(**values)

    @classmethod
    def original_gecco(cls) -> Self:
        return cls._original(
            input_features=9, sequence_length=128, n_fft=16, hop_length=8,
            window_length=16, scale_factors=(34, 2), spectral_temperature=0.01,
            patch_length=24, patch_dropout=0.5, model_dim=512,
            feedforward_dim=1024, attention_dropout=0.7,
            projection_dropout=0.7, feedforward_dropout=0.7, dropout=0.7,
            fusion_dropout=0.2, fusion_feedforward_dim=128,
            decoder_dropout=0.5, decoder_feedforward_dim=256,
            prototypes=64, prototype_temperature=3.0,
            update_error_quantile=0.2, update_disagreement_quantile=0.2,
            memory_weight_max=0.1, latent_fusion_hidden=128,
            cycle_loss_weight=0.7, multiscale_loss_weight=0.25,
            consistency_loss_weight=0.05, cycle_score_weight=0.05,
            multiscale_score_weight=0.95, consistency_score_weight=0.0,
            residual_weight=0.4, fusion_weight=0.6,
            frequency_score_weight=0.5, time_score_weight=0.5,
            gate_score_weight=0.1, score_smoothing=15,
        )

    @classmethod
    def original_msl(cls) -> Self:
        return cls(input_features=1)

    @classmethod
    def original_psm(cls) -> Self:
        return cls._original(
            input_features=25, sequence_length=192, window_length=32,
            scale_factors=(4,), prototypes=128,
            memory_warmup_steps=1_000, memory_ramp_steps=2_000,
            latent_fusion_hidden=512, residual_weight=0.1, fusion_weight=0.9,
            frequency_score_weight=0.2, time_score_weight=0.8,
            gate_score_weight=0.2, score_smoothing=3,
            learning_rate_schedule="type2",
        )

    @classmethod
    def original_smap(cls) -> Self:
        return cls._original(
            input_features=1, sequence_length=192, scale_factors=(16, 8, 4),
            spectral_temperature=0.01, prototypes=16,
            prototype_temperature=9.0, memory_warmup_steps=1_000,
            memory_ramp_steps=2_000, latent_fusion_hidden=512,
            residual_weight=0.5, fusion_weight=0.5,
            frequency_score_weight=0.5, time_score_weight=0.5,
            gate_score_weight=5.0, score_smoothing=3,
        )

    @classmethod
    def original_smd(cls) -> Self:
        return cls._original(
            input_features=38, sequence_length=192, n_fft=128, hop_length=16,
            window_length=128, scale_factors=(31, 2),
            spectral_temperature=0.001, patch_length=16, patch_dropout=0.2,
            feedforward_dim=256, attention_dropout=0.2,
            projection_dropout=0.2, feedforward_dropout=0.2, dropout=0.2,
            prototypes=32, update_error_quantile=0.1,
            update_disagreement_quantile=0.1, memory_warmup_steps=1_000,
            memory_ramp_steps=2_000, memory_weight_max=0.2,
            latent_fusion_hidden=128, cycle_loss_weight=0.1,
            multiscale_loss_weight=0.85, consistency_loss_weight=0.1,
            cycle_score_weight=0.2, multiscale_score_weight=0.8,
            consistency_score_weight=0.1, residual_weight=0.3,
            fusion_weight=0.7, frequency_score_weight=0.9,
            time_score_weight=0.1, gate_score_weight=5.0,
            score_smoothing=51,
        )

    @classmethod
    def original_swan(cls) -> Self:
        return cls._original(
            input_features=38, sequence_length=192, n_fft=16, hop_length=4,
            window_length=8, scale_factors=(16, 8), spectral_temperature=0.6,
            patch_length=24, decoder_dropout=0.2, prototypes=24,
            prototype_temperature=3.0, memory_warmup_steps=1_000,
            memory_ramp_steps=2_000, latent_fusion_hidden=128,
            cycle_loss_weight=0.2, multiscale_loss_weight=0.8,
            consistency_loss_weight=0.1, cycle_score_weight=0.5,
            multiscale_score_weight=0.5, consistency_score_weight=0.0,
            residual_weight=0.9, fusion_weight=0.1,
            frequency_score_weight=0.5, time_score_weight=0.5,
            gate_score_weight=0.1, score_smoothing=5,
        )

    @classmethod
    def original_swat(cls) -> Self:
        return cls._original(
            input_features=31, sequence_length=192, n_fft=64, hop_length=32,
            window_length=64, scale_factors=(16, 8), spectral_temperature=0.02,
            prototypes=32, prototype_temperature=3.0,
            latent_fusion_hidden=512, cycle_loss_weight=0.1,
            multiscale_loss_weight=0.9, consistency_loss_weight=0.0,
            cycle_score_weight=0.0, multiscale_score_weight=1.0,
            consistency_score_weight=0.0, residual_weight=0.4,
            fusion_weight=0.6, frequency_score_weight=0.5,
            time_score_weight=0.5, gate_score_weight=0.5,
            score_smoothing=3,
        )

    @classmethod
    def original_configs(cls) -> dict[str, Self]:
        return {
            "GECCO": cls.original_gecco(), "MSL": cls.original_msl(),
            "PSM": cls.original_psm(), "SMAP": cls.original_smap(),
            "SMD": cls.original_smd(), "SWAN": cls.original_swan(),
            "SWAT": cls.original_swat(),
        }
