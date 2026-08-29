"""Configuration for AERCA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self


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
    stride: int = 1
    causal_quantile: float = 0.8
    reconstruction_quantile: float = 0.95
    root_cause_quantile_encoder: float = 0.99
    root_cause_quantile_decoder: float = 0.99
    initial_z_score: float = 3.0
    risk: float = 1e-2
    initial_level: float = 0.98
    num_candidates: int = 100
    learning_rate: float = 1e-4
    epochs: int = 5_000
    patience: int = 20

    def __post_init__(self) -> None:
        for name in (
            "input_features",
            "window",
            "hidden_dim",
            "hidden_layers",
            "stride",
            "num_candidates",
            "epochs",
            "patience",
        ):
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
        for name in (
            "causal_quantile",
            "reconstruction_quantile",
            "root_cause_quantile_encoder",
            "root_cause_quantile_decoder",
            "initial_level",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.initial_z_score <= 0 or self.risk <= 0 or self.learning_rate <= 0:
            raise ValueError("z-score, risk, and learning_rate must be positive")

    @classmethod
    def _original(
        cls,
        *,
        input_features: int,
        window: int,
        hidden_dim: int,
        hidden_layers: int,
        learning_rate: float,
        causal_quantile: float,
        risk: float = 1e-2,
        initial_level: float = 0.98,
    ) -> Self:
        return cls(
            input_features=input_features,
            window=window,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            causal_quantile=causal_quantile,
            risk=risk,
            initial_level=initial_level,
        )

    @classmethod
    def original_linear(cls) -> Self:
        return cls._original(
            input_features=4,
            window=1,
            hidden_dim=50,
            hidden_layers=1,
            learning_rate=1e-4,
            causal_quantile=0.50,
        )

    @classmethod
    def original_lorenz96(cls) -> Self:
        return cls._original(
            input_features=20,
            window=1,
            hidden_dim=50,
            hidden_layers=2,
            learning_rate=1e-4,
            causal_quantile=0.70,
        )

    @classmethod
    def original_lotka_volterra(cls) -> Self:
        return cls._original(
            input_features=40,
            window=1,
            hidden_dim=50,
            hidden_layers=2,
            learning_rate=1e-4,
            causal_quantile=0.90,
        )

    @classmethod
    def original_msds(cls) -> Self:
        return cls._original(
            input_features=10,
            window=1,
            hidden_dim=1_000,
            hidden_layers=4,
            learning_rate=1e-6,
            causal_quantile=0.70,
            initial_level=0.0,
        )

    @classmethod
    def original_nonlinear(cls) -> Self:
        return cls._original(
            input_features=6,
            window=5,
            hidden_dim=50,
            hidden_layers=8,
            learning_rate=1e-4,
            causal_quantile=0.80,
        )

    @classmethod
    def original_swat(cls) -> Self:
        return cls._original(
            input_features=51,
            window=1,
            hidden_dim=1_000,
            hidden_layers=8,
            learning_rate=1e-6,
            causal_quantile=0.70,
            risk=1e-5,
            initial_level=0.9,
        )

    @classmethod
    def original_configs(cls) -> dict[str, Self]:
        return {
            "linear": cls.original_linear(),
            "lorenz96": cls.original_lorenz96(),
            "lotka_volterra": cls.original_lotka_volterra(),
            "msds": cls.original_msds(),
            "nonlinear": cls.original_nonlinear(),
            "swat": cls.original_swat(),
        }
