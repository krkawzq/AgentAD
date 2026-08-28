"""Configuration for Time-RCD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class TimeRCDConfig:
    input_features: int
    model_dim: int = 512
    projection_dim: int = 256
    patch_length: int = 16
    layers: int = 8
    heads: int = 8
    dropout: float = 0.1
    activation: Literal["relu", "gelu"] = "gelu"
    mask_ratio: float = 0.15
    reconstruction_weight: float = 1.0
    classification_weight: float = 1.0
    inference_window: int = 512
    epsilon: float = 1e-5

    def __post_init__(self) -> None:
        for name in (
            "input_features",
            "model_dim",
            "projection_dim",
            "patch_length",
            "layers",
            "heads",
            "inference_window",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.model_dim % self.heads:
            raise ValueError("model_dim must be divisible by heads")
        if self.projection_dim < 2:
            raise ValueError("projection_dim must be at least two")
        if self.model_dim % 2:
            raise ValueError("model_dim must be even for rotary embeddings")
        if not 0 <= self.dropout < 1 or not 0 < self.mask_ratio < 1:
            raise ValueError("dropout must be in [0,1) and mask_ratio in (0,1)")
        if self.reconstruction_weight < 0 or self.classification_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
