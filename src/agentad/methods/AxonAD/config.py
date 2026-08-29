"""Configuration for AxonAD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True, slots=True)
class AxonADConfig:
    input_features: int
    window_length: int = 100
    model_dim: int = 64
    heads: int = 4
    forecast_steps: int = 1
    tail_length: int = 5
    query_dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.1
    target_momentum: float = 0.9
    mask_ratio: float = 0.5
    mask_block_fraction: float = 0.5
    epsilon: float = 1e-9
    batch_size: int = 128
    epochs: int = 50
    patience: int = 3
    early_stopping_min_delta: float = 1e-4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    validation_fraction: float = 0.2
    gradient_clip_val: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "input_features",
            "window_length",
            "model_dim",
            "heads",
            "tail_length",
            "batch_size",
            "epochs",
            "patience",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.model_dim % self.heads:
            raise ValueError("model_dim must be divisible by heads")
        if not 0 <= self.forecast_steps < self.window_length:
            raise ValueError("forecast_steps must be in [0, window_length)")
        if not self.query_dilations or any(
            value <= 0 for value in self.query_dilations
        ):
            raise ValueError("query_dilations must contain positive integers")
        if not 0 <= self.dropout < 1 or not 0 <= self.target_momentum < 1:
            raise ValueError("dropout and target_momentum must be in [0,1)")
        if not 0 < self.mask_ratio <= 1 or not 0 < self.mask_block_fraction <= 1:
            raise ValueError("mask fractions must be in (0,1]")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.gradient_clip_val <= 0:
            raise ValueError("gradient_clip_val must be positive")

    @classmethod
    def original_default(cls, *, input_features: int) -> Self:
        return cls(input_features=input_features)

    @classmethod
    def original_optimal_multivariate(cls, *, input_features: int) -> Self:
        return cls(
            input_features=input_features,
            model_dim=128,
            heads=8,
            tail_length=10,
            learning_rate=5e-4,
        )

    @classmethod
    def original_configs(cls, *, input_features: int) -> dict[str, Self]:
        return {
            "default": cls.original_default(input_features=input_features),
            "optimal_multivariate": cls.original_optimal_multivariate(
                input_features=input_features
            ),
        }
