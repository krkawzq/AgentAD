"""Complete TSPulse reconstruction and anomaly-scoring configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Self


@dataclass(frozen=True, slots=True)
class TSPulseConfig:
    input_features: int = 1
    context_length: int = 64
    patch_length: int = 8
    patch_stride: int | None = None
    model_dim: int = 16
    layers: int = 3
    decoder_layers: int = 3
    decoder_model_dim: int = 16
    expansion_factor: int = 2
    channel_mode: Literal["common", "mix"] = "common"
    decoder_channel_mode: Literal["common", "mix"] = "common"
    gated_attention: bool = True
    normalization: Literal["LayerNorm", "BatchNorm"] = "LayerNorm"
    self_attention: bool = False
    self_attention_heads: int = 1
    positional_encoding: bool = False
    positional_encoding_type: Literal["random", "sincos"] = "sincos"
    scaling: Literal["mean", "std", "revin"] | None = "revin"
    loss: Literal["mse", "mae"] = "mse"
    dropout: float = 0.2
    head_dropout: float = 0.2
    reconstruction_type: Literal["patchwise", "full"] = "patchwise"
    mask_ratio: float | None = None
    mask_type: Literal["block", "user", "hybrid", "var_hybrid", "random"] = "block"
    mask_block_length: int | None = None
    loss_apply_mode: Literal["mask", "full", "mask_and_full"] = "mask"
    use_learnable_mask_token: bool = True
    minimum_scale: float = 1e-5
    fuse_fft: bool = False
    fft_weight: float = 1.0
    fft_probability_weight: float = 1.0
    fft_probability_length: int | None = None
    fft_probability_mode: Literal["plain", "log"] = "plain"
    enable_fft_probability_loss: bool = True
    fft_signal_loss_weight: float = 1.0
    fft_forecast_loss: bool = False
    fft_forecast_loss_weight: float = 1.0
    fft_time_consistent_masking: bool = True
    fft_mask_ratio: float | None = None
    fft_mask_strategy: Literal["magnitude", "random"] = "magnitude"
    fft_remove_component: Literal["last", "dc"] = "last"
    fft_applied_on: Literal["raw_ts", "scaled_ts"] = "scaled_ts"
    forecast_length: int = 96
    channel_consistent_masking: bool = True
    register_tokens: int = 8
    free_channel_flow: bool = True
    channel_mix_initialization: Literal["identity", "zero"] = "identity"
    revin_affine: bool = True
    batch_aware_masking: bool = False
    reconstruction_loss_weight: float = 1.0
    masked_reconstruction_loss_weight: float = 1.0
    aggregation_length: int = 32
    smoothing_window: int = 8
    score_modes: tuple[Literal["time", "frequency", "forecast"], ...] = (
        "time",
    )
    score_exponent: float = 1.0
    least_significant_scale: float = 0.0
    least_significant_score: float = 1.0
    pretrained_model_name: str | None = None
    finetune_epochs: int = 20
    finetune_validation: float = 0.2
    finetune_learning_rate: float = 1e-4
    finetune_batch_size: int = 256
    finetune_seed: int = 42
    finetune_freeze_backbone: bool = False
    finetune_weight_decay: float = 0.0
    one_cycle_pct_start: float = 0.3

    def __post_init__(self) -> None:
        for name in (
            "input_features",
            "context_length",
            "patch_length",
            "model_dim",
            "layers",
            "decoder_layers",
            "decoder_model_dim",
            "expansion_factor",
            "self_attention_heads",
            "forecast_length",
            "aggregation_length",
            "smoothing_window",
            "finetune_epochs",
            "finetune_batch_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        stride = self.patch_stride or self.patch_length
        if stride != self.patch_length:
            raise ValueError("the original TSPulse does not support overlapping patches")
        if self.context_length % self.patch_length:
            raise ValueError("context_length must be divisible by patch_length")
        if self.register_tokens < 0:
            raise ValueError("register_tokens must be non-negative")
        for name in ("dropout", "head_dropout"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.mask_ratio is not None and not 0 <= self.mask_ratio <= 1:
            raise ValueError("mask_ratio must be in [0, 1] when provided")
        if self.fft_mask_ratio is not None and not 0 <= self.fft_mask_ratio <= 1:
            raise ValueError("fft_mask_ratio must be in [0, 1] when provided")
        if not self.score_modes or len(set(self.score_modes)) != len(self.score_modes):
            raise ValueError("score_modes must be non-empty and unique")
        if not 0 <= self.finetune_validation < 1:
            raise ValueError("finetune_validation must be in [0, 1)")
        if self.finetune_learning_rate <= 0 or self.finetune_weight_decay < 0:
            raise ValueError("invalid fine-tuning optimizer parameters")
        if not 0 < self.one_cycle_pct_start < 1:
            raise ValueError("one_cycle_pct_start must be in (0, 1)")

    def reference_kwargs(self) -> dict[str, Any]:
        """Translate the AgentAD config without dropping any core TSPulse option."""
        return {
            "context_length": self.context_length,
            "patch_length": self.patch_length,
            "patch_stride": self.patch_stride or self.patch_length,
            "num_input_channels": self.input_features,
            "d_model": self.model_dim,
            "num_layers": self.layers,
            "decoder_num_layers": self.decoder_layers,
            "decoder_d_model": self.decoder_model_dim,
            "expansion_factor": self.expansion_factor,
            "mode": f"{self.channel_mode}_channel",
            "decoder_mode": f"{self.decoder_channel_mode}_channel",
            "gated_attn": self.gated_attention,
            "norm_mlp": self.normalization,
            "self_attn": self.self_attention,
            "self_attn_heads": self.self_attention_heads,
            "use_positional_encoding": self.positional_encoding,
            "positional_encoding_type": self.positional_encoding_type,
            "scaling": self.scaling,
            "loss": self.loss,
            "dropout": self.dropout,
            "head_dropout": self.head_dropout,
            "reconstruction_type": self.reconstruction_type,
            "mask_ratio": self.mask_ratio,
            "mask_type": self.mask_type,
            "mask_block_length": self.mask_block_length or self.patch_length,
            "loss_apply_mode": self.loss_apply_mode,
            "use_learnable_mask_token": self.use_learnable_mask_token,
            "minimum_scale": self.minimum_scale,
            "fuse_fft": self.fuse_fft,
            "fft_weight": self.fft_weight,
            "fft_prob_weight": self.fft_probability_weight,
            "fft_prob_length": self.fft_probability_length,
            "fft_prob_mode": self.fft_probability_mode,
            "enable_fft_prob_loss": self.enable_fft_probability_loss,
            "fft_original_signal_loss_weight": self.fft_signal_loss_weight,
            "fft_time_add_forecasting_pt_loss": self.fft_forecast_loss,
            "fft_time_add_forecasting_pt_loss_weight": self.fft_forecast_loss_weight,
            "prediction_length": self.forecast_length,
            "channel_consistent_masking": self.channel_consistent_masking,
            "patch_register_tokens": self.register_tokens,
            "free_channel_flow": self.free_channel_flow,
            "fft_time_consistent_masking": self.fft_time_consistent_masking,
            "fft_mask_ratio": self.fft_mask_ratio,
            "fft_mask_strategy": self.fft_mask_strategy,
            "fft_remove_component": self.fft_remove_component,
            "channel_mix_init": self.channel_mix_initialization,
            "batch_aware_masking": self.batch_aware_masking,
            "revin_affine": self.revin_affine,
            "reconstruction_loss_weight": self.reconstruction_loss_weight,
            "masked_reconstruction_loss_weight": self.masked_reconstruction_loss_weight,
            "fft_applied_on": self.fft_applied_on,
        }

    @classmethod
    def original_default(cls, *, input_features: int = 1) -> Self:
        return cls(input_features=input_features)

    @classmethod
    def original_tsb_zero_shot(cls, *, input_features: int = 1) -> Self:
        channel_mode: Literal["common", "mix"] = (
            "common" if input_features == 1 else "mix"
        )
        return cls(
            input_features=input_features,
            decoder_channel_mode=channel_mode,
            mask_type="user",
            aggregation_length=96,
            pretrained_model_name="ibm-granite/granite-timeseries-tspulse-r1",
        )

    @classmethod
    def original_tsb_finetune(cls, *, input_features: int = 1) -> Self:
        return cls.original_tsb_zero_shot(input_features=input_features)

    @classmethod
    def original_configs(cls, *, input_features: int = 1) -> dict[str, Self]:
        return {
            "default": cls.original_default(input_features=input_features),
            "tsb_zero_shot": cls.original_tsb_zero_shot(
                input_features=input_features
            ),
            "tsb_finetune": cls.original_tsb_finetune(input_features=input_features),
        }
