"""Configuration for the official xLSTMAD encoder-decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self


@dataclass(frozen=True, slots=True)
class XLSTMADConfig:
    input_features: int
    window_length: int = 20
    embedding_dim: int = 40
    blocks: int = 3
    scalar_memory_blocks: tuple[int, ...] = (0, 1)
    heads: int = 4
    scalar_kernel: int = 4
    matrix_kernel: int = 8
    matrix_qkv_projection_block_size: int = 5
    matrix_round_projection_up: bool = False
    matrix_projection_multiple: int = 5
    feedforward_factor: float = 1.3
    feedforward_activation: Literal["gelu"] = "gelu"
    scalar_bias_initialization: Literal["powerlaw_blockdependent"] = (
        "powerlaw_blockdependent"
    )
    scalar_backend: Literal["cuda", "vanilla"] = "cuda"
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 3
    validation_fraction: float = 0.2
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4

    def __post_init__(self) -> None:
        for name in (
            "input_features",
            "window_length",
            "embedding_dim",
            "blocks",
            "heads",
            "scalar_kernel",
            "matrix_kernel",
            "matrix_qkv_projection_block_size",
            "matrix_projection_multiple",
            "batch_size",
            "epochs",
            "early_stopping_patience",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if any(
            index < 0 or index >= self.blocks for index in self.scalar_memory_blocks
        ):
            raise ValueError("scalar_memory_blocks contains an invalid block index")
        if self.feedforward_factor <= 0 or self.learning_rate <= 0:
            raise ValueError("feedforward_factor and learning_rate must be positive")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        if self.embedding_dim % self.heads != 0:
            raise ValueError("embedding_dim must be a multiple of heads")

    @classmethod
    def original_default(
        cls,
        *,
        input_features: int,
        window_length: int,
        scalar_backend: Literal["cuda", "vanilla"] = "cuda",
    ) -> Self:
        return cls(
            input_features=input_features,
            window_length=window_length,
            scalar_backend=scalar_backend,
        )

    @classmethod
    def original_example(
        cls,
        *,
        input_features: int,
        scalar_backend: Literal["cuda", "vanilla"] = "cuda",
    ) -> Self:
        return cls(
            input_features=input_features,
            window_length=20,
            embedding_dim=40,
            scalar_backend=scalar_backend,
        )

    @classmethod
    def original_configs(
        cls,
        *,
        input_features: int,
        window_length: int = 20,
        scalar_backend: Literal["cuda", "vanilla"] = "cuda",
    ) -> dict[str, Self]:
        return {
            "default": cls.original_default(
                input_features=input_features,
                window_length=window_length,
                scalar_backend=scalar_backend,
            ),
            "example": cls.original_example(
                input_features=input_features,
                scalar_backend=scalar_backend,
            ),
        }
