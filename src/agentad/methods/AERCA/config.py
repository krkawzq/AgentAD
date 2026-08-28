"""Configuration for AERCA."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AERCAConfig:
    input_features: int
    window: int = 1
    hidden_dim: int = 50
    hidden_layers: int = 1
    encoder_alpha: float = 0.5
    decoder_alpha: float = 0.5
    encoder_sparsity_weight: float = 0.5
    decoder_sparsity_weight: float = 0.5
    encoder_smoothness_weight: float = 0.5
    decoder_smoothness_weight: float = 0.5
    kl_weight: float = 0.5
    covariance_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        for name in ("input_features", "window", "hidden_dim", "hidden_layers"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("encoder_alpha", "decoder_alpha"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "encoder_sparsity_weight",
            "decoder_sparsity_weight",
            "encoder_smoothness_weight",
            "decoder_smoothness_weight",
            "kl_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.covariance_epsilon <= 0:
            raise ValueError("covariance_epsilon must be positive")
