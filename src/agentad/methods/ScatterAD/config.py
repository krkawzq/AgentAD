"""Configuration for ScatterAD."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScatterADConfig:
    input_features: int
    hidden_dim: int = 512
    graph_layers: int = 2
    heads: int = 4
    temporal_kernel: int = 3
    graph_radius: int = 1
    temporal_dropout: float = 0.3
    attention_dropout: float = 0.2
    projection_dropout: float = 0.3
    target_momentum: float = 0.5
    score_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        positive = {
            "input_features": self.input_features,
            "hidden_dim": self.hidden_dim,
            "graph_layers": self.graph_layers,
            "heads": self.heads,
            "temporal_kernel": self.temporal_kernel,
            "graph_radius": self.graph_radius,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if self.temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be odd")
        for name in ("temporal_dropout", "attention_dropout", "projection_dropout"):
            value = getattr(self, name)
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if not 0 <= self.target_momentum < 1:
            raise ValueError("target_momentum must be in [0, 1)")
        if self.score_epsilon <= 0:
            raise ValueError("score_epsilon must be positive")
