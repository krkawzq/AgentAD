"""Configuration for multidimensional matrix-profile anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class MMPADConfig:
    subsequence_length: int
    dimensions: int | float | None = None
    neighbors: int = 1
    exclusion_fraction: float = 0.5
    alignment: Literal["start", "center", "mean"] = "mean"
    query_chunk_size: int = 64
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.subsequence_length <= 1:
            raise ValueError("subsequence_length must be greater than one")
        if self.neighbors <= 0 or self.query_chunk_size <= 0:
            raise ValueError("neighbors and query_chunk_size must be positive")
        if not 0 <= self.exclusion_fraction <= 1:
            raise ValueError("exclusion_fraction must be in [0, 1]")
        if self.dimensions is not None and float(self.dimensions) <= 0:
            raise ValueError("dimensions must be positive when provided")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
