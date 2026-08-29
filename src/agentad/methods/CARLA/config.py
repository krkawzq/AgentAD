"""Configuration for CARLA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True, slots=True)
class CARLAConfig:
    input_features: int
    window_length: int = 200
    mid_channels: int = 4
    projection_dim: int = 4
    clusters: int = 10
    cluster_heads: int = 1
    temperature: float = 0.4
    pretext_margin: float = 1.0
    margin_adjustment: float = 0.1
    entropy_weight: float = 2.0
    inconsistency_weight: float = 0.0
    anomaly_min_fraction: float = 0.1
    anomaly_max_fraction: float = 0.9
    window_stride: int = 1
    neighbors: int = 5
    noise_sigma: float = 0.01
    pretext_epochs: int = 30
    pretext_batch_size: int = 50
    pretext_learning_rate: float = 1e-3
    pretext_weight_decay: float = 0.01
    pretext_cosine_decay_rate: float = 0.01
    classification_epochs: int = 50
    classification_batch_size: int = 50
    classification_learning_rate: float = 0.01
    classification_weight_decay: float = 0.001

    def __post_init__(self) -> None:
        for name in (
            "input_features",
            "window_length",
            "mid_channels",
            "projection_dim",
            "clusters",
            "cluster_heads",
            "window_stride",
            "neighbors",
            "pretext_epochs",
            "pretext_batch_size",
            "classification_epochs",
            "classification_batch_size",
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
        if self.noise_sigma < 0:
            raise ValueError("noise_sigma must be non-negative")
        for name in (
            "pretext_learning_rate",
            "pretext_cosine_decay_rate",
            "classification_learning_rate",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.pretext_weight_decay < 0 or self.classification_weight_decay < 0:
            raise ValueError("weight decay must be non-negative")

    @classmethod
    def _original(
        cls,
        *,
        input_features: int,
        window_length: int = 200,
        window_stride: int = 1,
        inconsistency_weight: float = 0.0,
    ) -> Self:
        return cls(
            input_features=input_features,
            window_length=window_length,
            window_stride=window_stride,
            inconsistency_weight=inconsistency_weight,
        )

    @classmethod
    def original_kpi(cls) -> Self:
        return cls._original(input_features=1, window_stride=5)

    @classmethod
    def original_msl(cls) -> Self:
        return cls._original(input_features=55)

    @classmethod
    def original_smap(cls) -> Self:
        return cls._original(input_features=25)

    @classmethod
    def original_smd(cls) -> Self:
        return cls._original(input_features=38, window_stride=5)

    @classmethod
    def original_swat(cls) -> Self:
        return cls._original(
            input_features=51,
            window_stride=10,
            inconsistency_weight=1.0,
        )

    @classmethod
    def original_yahoo(cls) -> Self:
        return cls._original(input_features=1, window_length=250)

    @classmethod
    def original_configs(cls) -> dict[str, Self]:
        return {
            "kpi": cls.original_kpi(),
            "msl": cls.original_msl(),
            "smap": cls.original_smap(),
            "smd": cls.original_smd(),
            "swat": cls.original_swat(),
            "yahoo": cls.original_yahoo(),
        }
