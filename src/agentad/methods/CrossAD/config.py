"""Configuration for CrossAD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class CrossADConfig:
    sequence_length: int = 96
    patch_length: int = 6
    scale_kernels: tuple[int, ...] = (16, 8, 4, 2)
    scale_method: Literal["average", "interval"] = "average"
    top_frequencies: int = 10
    query_count: int = 5
    query_length: int = 5
    context_size: int = 32
    context_decay: float = 0.95
    context_epsilon: float = 1e-5
    encoder_layers: int = 2
    decoder_layers: int = 2
    extractor_layers: int = 2
    heads: int = 4
    model_dim: int = 128
    feedforward_dim: int | None = None
    attention_dropout: float = 0.1
    projection_dropout: float = 0.1
    feedforward_dropout: float = 0.1
    normalization: Literal["layer", "batch"] = "layer"
    activation: Literal["relu", "gelu"] = "gelu"
    context_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        positive = {
            "sequence_length": self.sequence_length,
            "patch_length": self.patch_length,
            "top_frequencies": self.top_frequencies,
            "query_count": self.query_count,
            "query_length": self.query_length,
            "context_size": self.context_size,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "extractor_layers": self.extractor_layers,
            "heads": self.heads,
            "model_dim": self.model_dim,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.scale_kernels or any(kernel <= 1 for kernel in self.scale_kernels):
            raise ValueError("scale_kernels must contain integers greater than one")
        if any(left <= right for left, right in zip(self.scale_kernels, self.scale_kernels[1:])):
            raise ValueError("scale_kernels must be ordered from coarse to fine")
        if self.model_dim % self.heads:
            raise ValueError("model_dim must be divisible by heads")
        if self.feedforward_dim is not None and self.feedforward_dim <= 0:
            raise ValueError("feedforward_dim must be positive when provided")
        for name in ("attention_dropout", "projection_dropout", "feedforward_dropout"):
            value = getattr(self, name)
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if not 0 <= self.context_decay < 1:
            raise ValueError("context_decay must be in [0, 1)")
        if self.context_epsilon <= 0:
            raise ValueError("context_epsilon must be positive")
        if self.context_loss_weight < 0:
            raise ValueError("context_loss_weight must be non-negative")
