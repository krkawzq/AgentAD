"""Configuration for KAN-AD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True, slots=True)
class KANADConfig:
    window: int = 96
    order: int = 2
    batch_size: int = 1024
    epochs: int = 100
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    learning_rate: float = 0.01
    scheduler_step_size: int = 5
    scheduler_gamma: float = 0.75

    def __post_init__(self) -> None:
        if self.window <= 1:
            raise ValueError("window must be greater than one")
        if self.order <= 0:
            raise ValueError("order must be positive")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        if self.learning_rate <= 0 or self.scheduler_step_size <= 0:
            raise ValueError("learning_rate and scheduler_step_size must be positive")
        if not 0 < self.scheduler_gamma <= 1:
            raise ValueError("scheduler_gamma must be in (0, 1]")

    @classmethod
    def original_default(cls) -> Self:
        """Configuration from ``forks/KAN-AD/kanad/config.toml``.

        The toml additionally declares ``[Data_Params]`` — first-order
        differencing followed by z-score standardization fitted on the
        training split — that the original EasyTSAD data factory applied
        outside the method; the caller must preprocess equivalently, so
        the fields are not carried here.
        """
        return cls()

    @classmethod
    def original_configs(cls) -> dict[str, Self]:
        return {"default": cls.original_default()}
