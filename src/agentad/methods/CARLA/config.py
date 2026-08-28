"""Configuration for CARLA."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CARLAConfig:
    input_features: int
    window_length: int = 200
    mid_channels: int = 4
    projection_dim: int = 128
    clusters: int = 10
    cluster_heads: int = 1
    temperature: float = 0.4
    pretext_margin: float = 1.0
    margin_adjustment: float = 0.1
    entropy_weight: float = 2.0
    inconsistency_weight: float = 0.0
    anomaly_min_fraction: float = 0.1
    anomaly_max_fraction: float = 0.9

    def __post_init__(self) -> None:
        for name in (
            "input_features",
            "window_length",
            "mid_channels",
            "projection_dim",
            "clusters",
            "cluster_heads",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.pretext_margin < 0 or self.margin_adjustment < 0:
            raise ValueError("margin values must be non-negative")
        if self.entropy_weight < 0 or self.inconsistency_weight < 0:
            raise ValueError("classification loss weights must be non-negative")
        if not 0 < self.anomaly_min_fraction <= self.anomaly_max_fraction < 1:
            raise ValueError("anomaly fractions must satisfy 0 < min <= max < 1")
