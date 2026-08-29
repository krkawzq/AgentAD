"""Configuration for CrossAD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self


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
    train_epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-4
    optimizer: Literal["adam", "adamw"] = "adam"
    learning_rate_schedule: Literal["type1", "type2"] = "type1"
    patience: int = 3

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
            "train_epochs": self.train_epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.scale_kernels or any(kernel <= 1 for kernel in self.scale_kernels):
            raise ValueError("scale_kernels must contain integers greater than one")
        if any(
            left <= right
            for left, right in zip(self.scale_kernels, self.scale_kernels[1:])
        ):
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
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    @classmethod
    def _original(cls, **overrides: object) -> Self:
        return cls(**overrides)

    @classmethod
    def original_gecco(cls) -> Self:
        return cls._original(
            sequence_length=128,
            patch_length=4,
            scale_kernels=(32, 16, 8, 4),
        )

    @classmethod
    def original_msl(cls) -> Self:
        return cls()

    @classmethod
    def original_psm(cls) -> Self:
        return cls._original(
            sequence_length=192,
            patch_length=6,
            scale_kernels=(32, 16, 8, 4, 2),
            learning_rate_schedule="type2",
        )

    @classmethod
    def original_smap(cls) -> Self:
        return cls._original(
            sequence_length=192,
            patch_length=8,
            scale_kernels=(24, 12, 6),
            context_decay=0.9,
            encoder_layers=1,
            decoder_layers=1,
            extractor_layers=1,
            heads=8,
            model_dim=256,
            attention_dropout=0.0,
            projection_dropout=0.0,
            feedforward_dropout=0.0,
        )

    @classmethod
    def original_smd(cls) -> Self:
        return cls._original(
            sequence_length=192,
            patch_length=6,
            scale_kernels=(32, 16, 8, 4, 2),
        )

    @classmethod
    def original_swan(cls) -> Self:
        return cls._original(
            sequence_length=192,
            patch_length=3,
            scale_kernels=(64, 32, 16, 8, 4, 2),
        )

    @classmethod
    def original_swat(cls) -> Self:
        return cls._original(
            sequence_length=192,
            patch_length=6,
            scale_kernels=(32, 16, 8, 4, 2),
            context_decay=0.9,
            heads=8,
            model_dim=256,
            attention_dropout=0.2,
            projection_dropout=0.2,
            feedforward_dropout=0.2,
        )

    @classmethod
    def _original_tsb(cls, **overrides: object) -> Self:
        values: dict[str, object] = {
            "sequence_length": 96,
            "patch_length": 3,
            "scale_kernels": (32, 16, 8, 4, 2),
            "encoder_layers": 1,
            "decoder_layers": 2,
            "extractor_layers": 1,
            "attention_dropout": 0.0,
            "projection_dropout": 0.0,
            "feedforward_dropout": 0.0,
            "batch_size": 32,
        }
        values.update(overrides)
        return cls._original(**values)

    @classmethod
    def original_tsb_1(cls) -> Self:
        return cls._original_tsb(query_count=3, context_size=8)

    @classmethod
    def original_tsb_2(cls) -> Self:
        return cls._original_tsb()

    @classmethod
    def original_tsb_3(cls) -> Self:
        return cls._original_tsb(encoder_layers=2, extractor_layers=2)

    @classmethod
    def original_tsb_4(cls) -> Self:
        return cls._original_tsb(sequence_length=32, patch_length=1)

    @classmethod
    def original_tsb_5(cls) -> Self:
        return cls._original_tsb(
            encoder_layers=3,
            decoder_layers=3,
            extractor_layers=3,
            model_dim=32,
        )

    @classmethod
    def original_tsb_6(cls) -> Self:
        return cls._original_tsb(
            sequence_length=32,
            patch_length=1,
            encoder_layers=3,
            decoder_layers=3,
            extractor_layers=3,
            model_dim=32,
        )

    @classmethod
    def original_ucr_1(cls) -> Self:
        return cls._original(
            sequence_length=512,
            patch_length=32,
            scale_kernels=(32, 16, 8, 4, 2),
            query_count=3,
            context_size=16,
            encoder_layers=1,
            decoder_layers=1,
            extractor_layers=1,
            train_epochs=1,
        )

    @classmethod
    def original_ucr_2(cls) -> Self:
        return cls._original(
            sequence_length=96,
            patch_length=4,
            scale_kernels=(2,),
            query_count=3,
            context_size=16,
            encoder_layers=1,
            decoder_layers=1,
            extractor_layers=1,
            train_epochs=1,
        )

    @classmethod
    def original_configs(cls) -> dict[str, Self]:
        return {
            "GECCO": cls.original_gecco(),
            "MSL": cls.original_msl(),
            "PSM": cls.original_psm(),
            "SMAP": cls.original_smap(),
            "SMD": cls.original_smd(),
            "SWAN": cls.original_swan(),
            "SWAT": cls.original_swat(),
            "TSB-AD-U-1": cls.original_tsb_1(),
            "TSB-AD-U-2": cls.original_tsb_2(),
            "TSB-AD-U-3": cls.original_tsb_3(),
            "TSB-AD-U-4": cls.original_tsb_4(),
            "TSB-AD-U-5": cls.original_tsb_5(),
            "TSB-AD-U-6": cls.original_tsb_6(),
            "UCR-1": cls.original_ucr_1(),
            "UCR-2": cls.original_ucr_2(),
        }
