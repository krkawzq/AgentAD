"""Configuration for Time-RCD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self


@dataclass(frozen=True, slots=True)
class TimeRCDConfig:
    input_features: int
    model_dim: int = 512
    projection_dim: int = 256
    patch_length: int = 4
    layers: int = 8
    heads: int = 8
    dropout: float = 0.1
    activation: Literal["relu", "gelu"] = "gelu"
    mask_ratio: float = 0.15
    reconstruction_weight: float = 1.0
    classification_weight: float = 1.0
    inference_window: int = 5_000
    epsilon: float = 1e-5
    batch_size: int = 3
    learning_rate: float = 1e-4
    epochs: int = 1_000
    accumulation_steps: int = 1
    weight_decay: float = 1e-5
    train_encoder: bool = False
    random_seed: int = 72

    def __post_init__(self) -> None:
        for name in (
            "input_features",
            "model_dim",
            "projection_dim",
            "patch_length",
            "layers",
            "heads",
            "inference_window",
            "batch_size",
            "epochs",
            "accumulation_steps",
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
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")

    @classmethod
    def original_pretraining(cls, *, input_features: int) -> Self:
        return cls(input_features=input_features)

    @classmethod
    def original_pretrained_univariate(cls) -> Self:
        return cls(input_features=1, patch_length=16, batch_size=64)

    @classmethod
    def original_pretrained_multivariate(cls, *, input_features: int) -> Self:
        if input_features <= 1:
            raise ValueError("multivariate Time-RCD requires more than one feature")
        return cls(input_features=input_features, patch_length=16, batch_size=1)

    @classmethod
    def original_configs(cls, *, input_features: int) -> dict[str, Self]:
        configs = {
            "pretraining": cls.original_pretraining(input_features=input_features),
            "pretrained_univariate": cls.original_pretrained_univariate(),
        }
        if input_features > 1:
            configs["pretrained_multivariate"] = cls.original_pretrained_multivariate(
                input_features=input_features
            )
        return configs
