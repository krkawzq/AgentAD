"""Configuration for xLSTMAD."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class XLSTMADConfig:
    input_features: int
    window_length: int = 128
    embedding_dim: int = 64
    blocks: int = 3
    scalar_memory_blocks: tuple[int, ...] = (0, 1)
    heads: int = 4
    scalar_kernel: int = 4
    matrix_kernel: int = 8
    feedforward_factor: float = 1.3
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "input_features",
            "window_length",
            "embedding_dim",
            "blocks",
            "heads",
            "scalar_kernel",
            "matrix_kernel",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.embedding_dim % self.heads:
            raise ValueError("embedding_dim must be divisible by heads")
        if any(
            index < 0 or index >= self.blocks for index in self.scalar_memory_blocks
        ):
            raise ValueError("scalar_memory_blocks contains an invalid block index")
        if self.feedforward_factor <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("feedforward_factor must be positive and dropout in [0,1)")
