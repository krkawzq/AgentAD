"""Configuration for AxonAD."""

from __future__ import annotations

from dataclasses import dataclass


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

    def __post_init__(self) -> None:
        for name in (
            "input_features",
            "window_length",
            "model_dim",
            "heads",
            "tail_length",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.model_dim % self.heads:
            raise ValueError("model_dim must be divisible by heads")
        if not 0 <= self.forecast_steps < self.window_length:
            raise ValueError("forecast_steps must be in [0, window_length)")
        if not self.query_dilations or any(value <= 0 for value in self.query_dilations):
            raise ValueError("query_dilations must contain positive integers")
        if not 0 <= self.dropout < 1 or not 0 <= self.target_momentum < 1:
            raise ValueError("dropout and target_momentum must be in [0,1)")
        if not 0 < self.mask_ratio <= 1 or not 0 < self.mask_block_fraction <= 1:
            raise ValueError("mask fractions must be in (0,1]")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
