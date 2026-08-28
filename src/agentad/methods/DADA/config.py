"""Configuration for DADA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class DADAConfig:
    sequence_length: int = 100
    hidden_dim: int = 64
    representation_dim: int = 256
    bottleneck_dims: tuple[int, ...] = (16, 32, 64, 128, 192, 256)
    experts_per_input: int = 3
    patch_length: int = 5
    mask_mode: Literal["complementary", "random", "none"] = "complementary"
    encoder_depth: int = 10
    copies: int = 10
    representation_dropout: float = 0.1
    bottleneck_dropout: float = 0.1
    noisy_gating: bool = True
    load_balance_weight: float = 1e-2
    gradient_reverse_max: float = 0.5
    gradient_reverse_steps: int = 100_000

    def __post_init__(self) -> None:
        positive = {
            "sequence_length": self.sequence_length,
            "hidden_dim": self.hidden_dim,
            "representation_dim": self.representation_dim,
            "experts_per_input": self.experts_per_input,
            "patch_length": self.patch_length,
            "encoder_depth": self.encoder_depth,
            "copies": self.copies,
            "gradient_reverse_steps": self.gradient_reverse_steps,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.bottleneck_dims or any(
            width <= 0 for width in self.bottleneck_dims
        ):
            raise ValueError("bottleneck_dims must contain positive widths")
        if self.experts_per_input > len(self.bottleneck_dims):
            raise ValueError(
                "experts_per_input cannot exceed the number of bottleneck experts"
            )
        if self.mask_mode == "complementary" and self.copies % 2:
            raise ValueError("complementary masking requires an even number of copies")
        for name in ("representation_dropout", "bottleneck_dropout"):
            value = getattr(self, name)
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.load_balance_weight < 0:
            raise ValueError("load_balance_weight must be non-negative")
        if self.gradient_reverse_max < 0:
            raise ValueError("gradient_reverse_max must be non-negative")
