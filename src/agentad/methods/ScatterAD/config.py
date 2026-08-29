"""Configuration for ScatterAD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self


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
    window_length: int = 110
    learning_rate: float = 1e-4
    epochs: int = 2
    batch_size: int = 128
    anomaly_ratio: float = 1.0
    random_seed: int = 2

    def __post_init__(self) -> None:
        positive = {
            "input_features": self.input_features,
            "hidden_dim": self.hidden_dim,
            "graph_layers": self.graph_layers,
            "heads": self.heads,
            "temporal_kernel": self.temporal_kernel,
            "graph_radius": self.graph_radius,
            "window_length": self.window_length,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
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
        if self.learning_rate <= 0 or not 0 <= self.anomaly_ratio <= 100:
            raise ValueError("learning_rate must be positive and anomaly_ratio in [0,100]")

    @classmethod
    def _original(cls, input_features: int) -> Self:
        return cls(input_features=input_features)

    @classmethod
    def original_msl(cls) -> Self:
        return cls._original(55)

    @classmethod
    def original_swat(cls) -> Self:
        return cls._original(51)

    @classmethod
    def original_psm(cls) -> Self:
        return cls._original(26)

    @classmethod
    def original_wadi(cls) -> Self:
        return cls._original(123)

    @classmethod
    def original_nips_water(cls) -> Self:
        # The original Water loader connects each step to its ±2 neighborhood.
        return cls(input_features=9, graph_radius=2)

    @classmethod
    def original_nips_swan(cls) -> Self:
        return cls._original(38)

    @classmethod
    def original_configs(cls) -> dict[str, Self]:
        return {
            "MSL": cls.original_msl(),
            "SWaT": cls.original_swat(),
            "PSM": cls.original_psm(),
            "WADI": cls.original_wadi(),
            "NIPS_TS_Water": cls.original_nips_water(),
            "NIPS_TS_Swan": cls.original_nips_swan(),
        }
