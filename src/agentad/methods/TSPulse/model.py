"""Exact adapter around the official Granite TSPulse reconstruction core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .._utils import evaluation_mode, sliding_windows, validate_series
from .config import TSPulseConfig


def _reference_types() -> tuple[type[Any], type[Any]]:
    try:
        from tsfm_public.models.tspulse.configuration_tspulse import (
            TSPulseConfig as ReferenceConfig,
        )
        from tsfm_public.models.tspulse.modeling_tspulse import (
            TSPulseForReconstruction as ReferenceModel,
        )
    except ImportError as exc:
        raise ImportError(
            "TSPulse requires the pinned granite-tsfm dependency. Run `uv sync`."
        ) from exc
    return ReferenceConfig, ReferenceModel


def _stitch_function() -> Any:
    try:
        from tsfm_public.models.tspulse.utils.helpers import (
            patchwise_stitched_reconstruction,
        )
    except ImportError as exc:
        raise ImportError(
            "TSPulse scoring requires the pinned granite-tsfm dependency."
        ) from exc
    return patchwise_stitched_reconstruction


@dataclass(frozen=True, slots=True)
class TSPulseOutput:
    time_reconstruction: Tensor
    frequency_reconstruction: Tensor | None
    forecast: Tensor | None
    masked_points: Tensor
    raw_output: Any


@dataclass(frozen=True, slots=True)
class TSPulseLoss:
    total: Tensor
    time_reconstruction: Tensor
    masked_time_reconstruction: Tensor
    frequency_reconstruction: Tensor
    frequency_time_reconstruction: Tensor
    frequency_probability: Tensor
    forecast: Tensor


class TSPulse(nn.Module):
    """AgentAD interface backed by the complete official TSPulse model."""

    def __init__(self, config: TSPulseConfig, *, core: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        if core is None:
            ReferenceConfig, ReferenceModel = _reference_types()
            core = ReferenceModel(ReferenceConfig(**config.reference_kwargs()))
        self.core = core
        core_channels = int(self.core.config.num_input_channels)
        if core_channels != config.input_features:
            raise ValueError(
                "TSPulse core feature count differs from the AgentAD configuration"
            )

    @classmethod
    def from_pretrained(
        cls,
        config: TSPulseConfig,
        model_name_or_path: str | None = None,
        **kwargs: Any,
    ) -> "TSPulse":
        _, ReferenceModel = _reference_types()
        source = model_name_or_path or config.pretrained_model_name
        if source is None:
            raise ValueError("a TSPulse pretrained model name or path is required")
        decoder_mode = f"{config.decoder_channel_mode}_channel"
        if config.input_features == 1:
            decoder_mode = "common_channel"
        core = ReferenceModel.from_pretrained(
            source,
            num_input_channels=config.input_features,
            decoder_mode=decoder_mode,
            scaling=config.scaling,
            mask_type="user",
            **kwargs,
        )
        return cls(config, core=core)

    @property
    def context_length(self) -> int:
        return int(self.core.config.context_length)

    @property
    def patch_length(self) -> int:
        return int(self.core.config.patch_length)

    def _validate_windows(self, series: Tensor) -> Tensor:
        series = validate_series(series, length=self.context_length)
        if series.shape[2] != self.config.input_features:
            raise ValueError("series feature count does not match TSPulse")
        return series

    def forward(
        self,
        series: Tensor,
        observed_mask: Tensor | None = None,
        *,
        future_values: Tensor | None = None,
        return_loss: bool = False,
    ) -> TSPulseOutput:
        series = self._validate_windows(series)
        if observed_mask is not None and observed_mask.shape != series.shape:
            raise ValueError("observed_mask must match the input series")
        raw = self.core(
            past_values=series,
            future_values=future_values,
            past_observed_mask=observed_mask,
            return_loss=return_loss,
            return_dict=True,
        )
        mask = raw.mask
        if mask is None:
            mask = torch.zeros_like(series, dtype=torch.bool)
        return TSPulseOutput(
            raw.reconstruction_outputs,
            raw.reconstructed_ts_from_fft,
            raw.forecast_output,
            mask.bool(),
            raw,
        )

    @staticmethod
    def _loss_scalar(value: Tensor | None, reference: Tensor) -> Tensor:
        if value is None:
            return reference.new_zeros(())
        return value.mean()

    def compute_loss(
        self,
        series: Tensor,
        *,
        future_values: Tensor | None = None,
        observed_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> TSPulseLoss:
        devices = []
        if series.is_cuda and series.device.index is not None:
            devices = [series.device.index]
        with torch.random.fork_rng(devices=devices):
            if generator is not None:
                seed = generator.initial_seed()
                torch.manual_seed(seed)
                if series.is_cuda:
                    torch.cuda.manual_seed(seed)
            output = self(
                series,
                observed_mask,
                future_values=future_values,
                return_loss=True,
            )
        raw = output.raw_output
        total = self._loss_scalar(raw.loss, series)
        return TSPulseLoss(
            total,
            self._loss_scalar(raw.reconstruction_loss, series),
            self._loss_scalar(raw.masked_reconstruction_loss, series),
            self._loss_scalar(raw.fft_loss, series),
            self._loss_scalar(raw.reconstructed_ts_from_fft_loss, series),
            self._loss_scalar(raw.fft_prob_loss, series),
            self._loss_scalar(raw.forecast_loss, series),
        )

    @staticmethod
    def _normalize_score(score: Tensor) -> Tensor:
        minimum = score.amin(1, keepdim=True)
        maximum = score.amax(1, keepdim=True)
        return (score - minimum) / (maximum - minimum).clamp_min(1e-12)

    def _align_reconstruction_scores(self, score: Tensor, length: int) -> Tensor:
        aggregation = min(self.config.aggregation_length, self.context_length)
        left = self.context_length - aggregation // 2
        right = length - left - score.shape[1]
        return torch.cat(
            (
                score[:, :1].expand(-1, left),
                score,
                score[:, -1:].expand(-1, max(0, right)),
            ),
            dim=1,
        )[:, :length]

    def _reconstruction_window_score(
        self, windows: Tensor, *, frequency: bool
    ) -> Tensor:
        aggregation = min(self.config.aggregation_length, self.context_length)
        if aggregation < self.patch_length or aggregation % self.patch_length:
            raise ValueError(
                "aggregation_length must be at least and divisible by patch_length"
            )
        key = "reconstructed_ts_from_fft" if frequency else "reconstruction_outputs"
        stitched = _stitch_function()(
            model=self.core,
            past_values=windows,
            patch_size=self.patch_length,
            keys_to_stitch=[key],
            keys_to_aggregate=[],
            reconstruct_start=self.context_length - aggregation,
            reconstruct_end=self.context_length,
            debug=False,
        )
        if isinstance(stitched, tuple):
            stitched = stitched[0]
        reconstructed = stitched[key]
        start = self.context_length - aggregation
        return (reconstructed[:, start:] - windows[:, start:]).square().mean((1, 2))

    @torch.inference_mode()
    def score(self, series: Tensor, *, batch_size: int = 128) -> Tensor:
        series = validate_series(series, min_length=self.context_length)
        if series.shape[2] != self.config.input_features:
            raise ValueError("series feature count does not match TSPulse")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        with evaluation_mode(self):
            windows = sliding_windows(series, self.context_length)
            flattened = windows.flatten(0, 1)
            mode_scores: list[Tensor] = []
            for mode in self.config.score_modes:
                if mode == "forecast":
                    forecast_windows = sliding_windows(
                        series[:, :-1], self.context_length
                    ).flatten(0, 1)
                    targets = series[:, self.context_length :].flatten(0, 1)
                    values: list[Tensor] = []
                    offset = 0
                    for chunk in forecast_windows.split(batch_size):
                        prediction = self(chunk).forecast
                        if prediction is None:
                            raise RuntimeError("TSPulse core has no forecast head")
                        values.append(
                            (prediction[:, 0] - targets[offset : offset + len(chunk)])
                            .square()
                            .mean(1)
                        )
                        offset += len(chunk)
                    score = torch.cat(values).reshape(series.shape[0], -1)
                    score = torch.cat(
                        (score[:, :1].expand(-1, self.context_length), score), dim=1
                    )[:, : series.shape[1]]
                else:
                    values = [
                        self._reconstruction_window_score(
                            chunk, frequency=mode == "frequency"
                        )
                        for chunk in flattened.split(batch_size)
                    ]
                    score = torch.cat(values).reshape(series.shape[0], -1)
                    score = self._align_reconstruction_scores(score, series.shape[1])
                mode_scores.append(self._normalize_score(score))
            combined = torch.stack(mode_scores).mean(0).pow(self.config.score_exponent)
            if self.config.smoothing_window > 1:
                kernel = combined.new_full(
                    (1, 1, self.config.smoothing_window),
                    1 / self.config.smoothing_window,
                )
                left = (self.config.smoothing_window - 1) // 2
                right = self.config.smoothing_window - 1 - left
                combined = F.conv1d(
                    F.pad(combined[:, None], (left, right), mode="replicate"),
                    kernel,
                )[:, 0]
            return combined


class TSPulseZeroShot(TSPulse):
    def __init__(self, config: TSPulseConfig, *, core: nn.Module | None = None) -> None:
        if core is None:
            raise ValueError("TSPulseZeroShot must be created with from_pretrained()")
        super().__init__(config, core=core)


class TSPulseFineTune(TSPulse):
    pass
