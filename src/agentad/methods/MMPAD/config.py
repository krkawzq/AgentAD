"""Configuration for multidimensional matrix-profile anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from typing import Self


@dataclass(frozen=True, slots=True)
class MMPADConfig:
    """MMPAD configuration.

    ``subsequence_length=None`` mirrors the original submission, which infers
    the window length from the first channel's autocorrelation.
    """

    subsequence_length: int | None = None
    dimensions: int | float | None = None
    neighbors: int = 1
    exclusion_fraction: float = 0.5
    alignment: Literal["start", "center", "mean"] = "mean"
    query_chunk_size: int = 64
    epsilon: float = 1e-6
    sorting_place: Literal["pre"] = "pre"
    mode: Literal["discord"] = "discord"
    flat_mode: Literal["invalid"] = "invalid"
    use_train_reference: bool = False

    def __post_init__(self) -> None:
        if self.subsequence_length is not None and self.subsequence_length <= 1:
            raise ValueError("subsequence_length must be greater than one")
        if self.neighbors <= 0 or self.query_chunk_size <= 0:
            raise ValueError("neighbors and query_chunk_size must be positive")
        if not 0 <= self.exclusion_fraction <= 1:
            raise ValueError("exclusion_fraction must be in [0, 1]")
        if self.dimensions is not None and float(self.dimensions) <= 0:
            raise ValueError("dimensions must be positive when provided")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

    @classmethod
    def original_default(cls, *, subsequence_length: int | None) -> Self:
        return cls(subsequence_length=subsequence_length)

    @classmethod
    def original_univariate_hp0(cls, *, subsequence_length: int | None) -> Self:
        return cls(subsequence_length=subsequence_length, neighbors=5)

    @classmethod
    def original_univariate_hp1(cls, *, subsequence_length: int | None) -> Self:
        return cls(subsequence_length=subsequence_length, neighbors=10)

    @classmethod
    def original_multivariate_hp0(cls, *, subsequence_length: int | None) -> Self:
        return cls(
            subsequence_length=subsequence_length,
            dimensions=0.7,
            neighbors=15,
        )

    @classmethod
    def original_multivariate_hp1(cls, *, subsequence_length: int | None) -> Self:
        return cls(
            subsequence_length=subsequence_length,
            dimensions=0.5,
            neighbors=15,
        )

    @classmethod
    def original_configs(cls, *, subsequence_length: int | None) -> dict[str, Self]:
        return {
            "default": cls.original_default(subsequence_length=subsequence_length),
            "univariate_hp0": cls.original_univariate_hp0(
                subsequence_length=subsequence_length
            ),
            "univariate_hp1": cls.original_univariate_hp1(
                subsequence_length=subsequence_length
            ),
            "multivariate_hp0": cls.original_multivariate_hp0(
                subsequence_length=subsequence_length
            ),
            "multivariate_hp1": cls.original_multivariate_hp1(
                subsequence_length=subsequence_length
            ),
        }
