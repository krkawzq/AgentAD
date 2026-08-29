"""Configuration for PaAno."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True, slots=True)
class PaAnoConfig:
    input_features: int = 1
    patch_length: int = 64
    convolution_widths: tuple[int, ...] = (128, 256, 128, 64)
    kernel_sizes: tuple[int, ...] = (7, 5, 3, 3)
    projection_dim: int = 256
    use_revin: bool = False
    revin_affine: bool = False
    revin_epsilon: float = 1e-5
    revin_min_scale: float = 1e-5
    positive_radius: int = 2
    triplet_margin: float = 0.1
    triplet_scale: float = 0.1
    pretext_fraction: float = 0.2
    pretext_weight: float = 1.0
    random_negatives: int = 5
    top_k: int = 3
    memory_coreset_fraction: float | None = 0.1
    training_iterations: int = 200
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    final_learning_rate_factor: float = 0.1
    random_seed: int = 2000

    def __post_init__(self) -> None:
        positive = {
            "input_features": self.input_features,
            "patch_length": self.patch_length,
            "projection_dim": self.projection_dim,
            "positive_radius": self.positive_radius,
            "random_negatives": self.random_negatives,
            "top_k": self.top_k,
            "training_iterations": self.training_iterations,
            "batch_size": self.batch_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.convolution_widths or len(self.convolution_widths) != len(
            self.kernel_sizes
        ):
            raise ValueError(
                "convolution_widths and kernel_sizes must have equal non-zero length"
            )
        if any(width <= 0 for width in self.convolution_widths):
            raise ValueError("convolution widths must be positive")
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in self.kernel_sizes):
            raise ValueError("kernel sizes must be positive odd integers")
        if self.revin_epsilon <= 0 or self.revin_min_scale <= 0:
            raise ValueError("RevIN epsilon and minimum scale must be positive")
        if self.triplet_margin < 0 or self.triplet_scale <= 0:
            raise ValueError(
                "triplet margin must be non-negative and triplet_scale positive"
            )
        if not 0 <= self.pretext_fraction <= 1:
            raise ValueError("pretext_fraction must be in [0, 1]")
        if self.pretext_weight < 0:
            raise ValueError("pretext_weight must be non-negative")
        if self.memory_coreset_fraction is not None and not (
            0 < self.memory_coreset_fraction <= 1
        ):
            raise ValueError(
                "memory_coreset_fraction must be in (0, 1] when provided"
            )
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0 < self.final_learning_rate_factor <= 1:
            raise ValueError("final_learning_rate_factor must be in (0, 1]")

    @classmethod
    def original_default(cls, *, input_features: int = 1) -> Self:
        """Defaults from ``forks/PaAno/main.py``."""
        return cls(input_features=input_features)

    @classmethod
    def original_univariate(cls) -> Self:
        """Configuration from ``script/run_uni.sh``."""
        return cls(
            input_features=1,
            patch_length=96,
            use_revin=True,
            training_iterations=100,
            batch_size=512,
            learning_rate=1e-4,
            random_seed=2027,
        )

    @classmethod
    def original_multivariate(cls, *, input_features: int) -> Self:
        """Configuration from ``script/run_mul.sh``."""
        return cls(
            input_features=input_features,
            patch_length=96,
            use_revin=True,
            training_iterations=100,
            batch_size=512,
            learning_rate=1e-4,
            random_seed=2027,
        )

    @classmethod
    def original_configs(cls, *, input_features: int = 1) -> dict[str, Self]:
        return {
            "default": cls.original_default(input_features=input_features),
            "univariate": cls.original_univariate(),
            "multivariate": cls.original_multivariate(input_features=input_features),
        }
