"""Configuration for TSPulse reconstruction and anomaly scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class TSPulseConfig:
    input_features: int
    context_length: int = 64
    patch_length: int = 8
    model_dim: int = 16
    layers: int = 3
    decoder_layers: int = 3
    expansion_factor: int = 2
    channel_mode: Literal["common", "mix"] = "common"
    register_tokens: int = 8
    dropout: float = 0.2
    mask_ratio: float = 0.4
    aggregation_length: int = 32
    smoothing_window: int = 8
    score_modes: tuple[Literal["time", "frequency", "forecast"], ...] = (
        "time",
        "frequency",
        "forecast",
    )
    forecast_length: int = 1
    time_loss_weight: float = 1.0
    frequency_loss_weight: float = 1.0
    forecast_loss_weight: float = 1.0
    revin_affine: bool = True
    epsilon: float = 1e-5

    def __post_init__(self) -> None:
        for name in (
            "input_features",
            "context_length",
            "patch_length",
            "model_dim",
            "layers",
            "decoder_layers",
            "expansion_factor",
            "aggregation_length",
            "smoothing_window",
            "forecast_length",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.context_length % self.patch_length:
            raise ValueError("context_length must be divisible by patch_length")
        if (
            self.aggregation_length > self.context_length
            or self.aggregation_length % self.patch_length
        ):
            raise ValueError(
                "aggregation_length must fit context_length and divide into patches"
            )
        if self.register_tokens < 0:
            raise ValueError("register_tokens must be non-negative")
        if not 0 <= self.dropout < 1 or not 0 < self.mask_ratio < 1:
            raise ValueError("dropout must be in [0,1) and mask_ratio in (0,1)")
        if not self.score_modes:
            raise ValueError("score_modes cannot be empty")
        supported_modes = {"time", "frequency", "forecast"}
        if any(mode not in supported_modes for mode in self.score_modes):
            raise ValueError(
                f"score_modes must be selected from {sorted(supported_modes)}"
            )
        if len(set(self.score_modes)) != len(self.score_modes):
            raise ValueError("score_modes cannot contain duplicates")
        if any(
            weight < 0
            for weight in (
                self.time_loss_weight,
                self.frequency_loss_weight,
                self.forecast_loss_weight,
            )
        ):
            raise ValueError("loss weights must be non-negative")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
