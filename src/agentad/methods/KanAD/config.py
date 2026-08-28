"""Configuration for KAN-AD."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KANADConfig:
    window: int = 96
    order: int = 2

    def __post_init__(self) -> None:
        if self.window <= 1:
            raise ValueError("window must be greater than one")
        if self.order <= 0:
            raise ValueError("order must be positive")
