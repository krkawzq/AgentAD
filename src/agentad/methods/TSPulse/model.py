"""Exact adapter around the official Granite TSPulse reconstruction core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from collections.abc import Mapping
import lightning as L
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

from .._utils import evaluation_mode, sliding_windows, validate_series
from .config import TSPulseConfig
from .._utils import ValidationEarlyStopping


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
            "TSPulse requires the pinned granite-tsfm dependency. "
            "Run `uv sync --extra tspulse`."
        ) from exc
    return ReferenceConfig, ReferenceModel


def _stitch_function() -> Any:
    try:
        from tsfm_public.models.tspulse.utils.helpers import (
            patchwise_stitched_reconstruction,
        )
    except ImportError as exc:
        raise ImportError(
            "TSPulse scoring requires the pinned granite-tsfm dependency. "
            "Run `uv sync --extra tspulse`."
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

    # tsfm_public ships no type stubs; treat the wrapped core as untyped.
    core: Any

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

    def _generate_training_mask(self, series: Tensor) -> Tensor:
        """Hide one random patch inside the aggregation window per sample.

        The reference fine-tuning wraps every window in
        ``PatchMaskingDatasetWrapper``, which enumerates one sample per patch
        of the aggregation window so each patch is hidden exactly once per
        epoch. Passing an ``observed_mask`` built from such an enumerated
        dataset reproduces that recipe; this fallback instead draws one
        random patch per window and epoch, which keeps the masking
        distribution but shrinks each epoch by that factor.
        """
        batch, time, channels = series.shape
        mask = torch.ones(batch, time, channels, dtype=torch.bool, device=series.device)
        window_start = max(0, time - self.config.aggregation_length)
        count = (time - window_start) // self.patch_length
        if count < 1:
            raise ValueError("the aggregation window must contain one patch")
        choice = torch.randint(count, (batch,), device=series.device)
        starts = window_start + choice * self.patch_length
        offsets = torch.arange(self.patch_length, device=series.device)
        rows = torch.arange(batch, device=series.device)[:, None]
        mask[rows, starts[:, None] + offsets[None, :]] = False
        return mask

    def compute_loss(
        self,
        series: Tensor,
        *,
        future_values: Tensor | None = None,
        observed_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> TSPulseLoss:
        series = self._validate_windows(series)
        if observed_mask is not None:
            if observed_mask.shape != series.shape:
                raise ValueError("observed_mask must match the input series")
            hidden = ~observed_mask.bool()
            if not hidden.any(dim=(1, 2)).all():
                raise ValueError(
                    "observed_mask must hide at least one time point of every "
                    "window; the masked reconstruction loss is otherwise empty"
                )
        devices = []
        if series.is_cuda and series.device.index is not None:
            devices = [series.device.index]
        with torch.random.fork_rng(devices=devices):
            if generator is not None:
                seed = generator.initial_seed()
                torch.manual_seed(seed)
                if series.is_cuda:
                    torch.cuda.manual_seed(seed)
            if observed_mask is None and self.config.mask_type == "user":
                observed_mask = self._generate_training_mask(series)
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

    @staticmethod
    def _standardize_series(series: Tensor) -> Tensor:
        """Whole-series z-score per channel, as the reference scoring pipeline."""
        mean = series.mean(1, keepdim=True)
        deviation = series.std(1, keepdim=True, unbiased=False)
        deviation = torch.where(deviation == 0, torch.ones_like(deviation), deviation)
        return (series - mean) / deviation

    def _moving_average(self, score: Tensor) -> Tensor:
        """Zero-padded moving average matching ``np.convolve(..., mode="same")``."""
        window = self.config.smoothing_window
        kernel = score.new_full((1, 1, window), 1 / window)
        # np.convolve "same" left-pads by w // 2, which shifts even windows
        # one position left of a symmetric kernel.
        left = window // 2
        right = (window - 1) // 2
        return F.conv1d(F.pad(score[:, None], (left, right)), kernel)[:, 0]

    def _align_reconstruction_scores(self, score: Tensor, length: int) -> Tensor:
        # Reference padding: context - aggregation//2 replicated values on the
        # left and aggregation//2 on the right, applied to the T - context
        # windows that exclude the final context window.
        aggregation = self.config.aggregation_length
        left = max(0, self.context_length - aggregation // 2)
        right = aggregation // 2
        if left + score.shape[1] + right < length:
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
        aggregation = self.config.aggregation_length
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
        # The raw start keeps the reference's slicing semantics, including the
        # negative index when aggregation exceeds the context length.
        start = self.context_length - aggregation
        return (reconstructed[:, start:] - windows[:, start:]).square().mean((1, 2))

    def _least_significant_gate(
        self, score: Tensor, series: Tensor, *, predictive: bool
    ) -> tuple[Tensor, Tensor]:
        """Apply the reference least-significant-score gating.

        The reference scales scores above ``least_significant_scale`` times the
        squared NaN-standard-deviation of the series' first differences by
        ``1 / least_significant_score`` and otherwise rescales the whole
        series; the TSB-AD defaults make both operations identities.
        """
        difference = series[:, 1:] - series[:, :-1]
        # NaN-aware standard deviation with ddof=0, as np.nanstd in the
        # reference.
        finite = torch.isfinite(difference)
        count = finite.sum(dim=1, keepdim=True)
        filled = torch.where(finite, difference, torch.zeros_like(difference))
        mean = filled.sum(dim=1, keepdim=True) / count.clamp_min(1)
        centered = torch.where(finite, difference - mean, torch.zeros_like(difference))
        variance = centered.square().sum(dim=1, keepdim=True) / count.clamp_min(1)
        minimum = self.config.least_significant_scale * variance[:, 0]
        if minimum.shape[-1] != 1:
            minimum = minimum.amax(-1, keepdim=True)
        if predictive:
            minimum = minimum * (2.0**0.5)
        else:
            minimum = minimum * (1 + self.patch_length)
        above = score > minimum
        adjusted = torch.where(
            above, score / self.config.least_significant_score, score
        )
        scale = torch.where(
            above.any(1, keepdim=True),
            score.new_ones(()),
            score.new_full((), self.config.least_significant_score),
        )
        return adjusted, scale

    @torch.inference_mode()
    def score(self, series: Tensor, *, batch_size: int = 128) -> Tensor:
        series = validate_series(series, min_length=self.context_length)
        if series.shape[2] != self.config.input_features:
            raise ValueError("series feature count does not match TSPulse")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        with evaluation_mode(self):
            # The reference pipeline imputes NaN inputs with zero before the
            # channel-wise standardization.
            series = self._standardize_series(torch.nan_to_num(series, nan=0.0))
            windows = sliding_windows(series, self.context_length)
            # The reference dataset keeps the T - context windows that leave
            # room for one forecast target, so reconstruction modes exclude
            # the final context window as well.
            scored = windows[:, :-1] if windows.shape[1] > 1 else windows
            flattened = scored.flatten(0, 1)
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
                # Reference order: least-significant gate, exponent, min-max
                # with the gate's scale, then per-mode smoothing.
                score, scale = self._least_significant_gate(
                    score, series, predictive=mode == "forecast"
                )
                score = (
                    self._normalize_score(score.pow(self.config.score_exponent)) * scale
                )
                if mode != "forecast" and self.config.smoothing_window > 1:
                    score = self._moving_average(score)
                mode_scores.append(score)
            combined = torch.stack(mode_scores).max(0).values
            # The reference divides by np.nanmax per series, so NaN entries
            # stay NaN and an all-NaN series stays NaN as well.
            finite = combined.masked_fill(torch.isnan(combined), -torch.inf)
            return combined / (finite.amax(1, keepdim=True) + 1e-5)


class TSPulseZeroShot(TSPulse):
    def __init__(self, config: TSPulseConfig, *, core: nn.Module | None = None) -> None:
        if core is None:
            raise ValueError("TSPulseZeroShot must be created with from_pretrained()")
        super().__init__(config, core=core)


class TSPulseFineTune(TSPulse):
    pass


class TSPulseLightningModule(ValidationEarlyStopping, L.LightningModule):
    """Fine-tuning module.

    Data contract: batches should carry an ``observed_mask`` from an
    enumerated patch-masking dataset to reproduce the reference recipe
    (see ``TSPulse._generate_training_mask``) and windows standardized
    per channel as the reference fine-tuning dataset does. The reference
    pipeline also caps fine-tuning at five epochs when the series has
    fewer than 3000 points and drops the batch size to eight when the
    enumerated training set has fewer than 500 samples, so scale
    ``finetune_epochs``/``finetune_batch_size`` down accordingly.
    """

    def __init__(
        self,
        config: TSPulseConfig,
        model: TSPulseFineTune | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model or TSPulseFineTune(config)
        if config.finetune_freeze_backbone:
            for parameter in self.model.core.backbone.parameters():
                parameter.requires_grad_(False)
        self._init_validation_early_stopping(
            monitor="val/loss",
            patience=config.finetune_early_stopping_patience,
            min_delta=config.finetune_early_stopping_threshold,
        )

    def forward(
        self,
        series: Tensor,
        observed_mask: Tensor | None = None,
        future_values: Tensor | None = None,
    ):
        return self.model(
            series,
            observed_mask,
            future_values=future_values,
            return_loss=False,
        )

    @staticmethod
    def _batch(
        batch: object,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if isinstance(batch, Tensor):
            return batch, None, None
        if isinstance(batch, Mapping):
            series = batch.get("series", batch.get("past_values"))
            future = batch.get("future_values")
            observed = batch.get("observed_mask", batch.get("past_observed_mask"))
        elif isinstance(batch, (tuple, list)) and batch:
            series = batch[0]
            future = batch[1] if len(batch) > 1 else None
            observed = batch[2] if len(batch) > 2 else None
        else:
            series = future = observed = None
        if not isinstance(series, Tensor):
            raise TypeError("TSPulse batches must provide past_values/series")
        if future is not None and not isinstance(future, Tensor):
            raise TypeError("TSPulse future_values must be a tensor")
        if observed is not None and not isinstance(observed, Tensor):
            raise TypeError("TSPulse observed mask must be a tensor")
        return series, future, observed

    def _step(self, batch: object, stage: str) -> Tensor:
        series, future, observed = self._batch(batch)
        losses = self.model.compute_loss(
            series,
            future_values=future,
            observed_mask=observed,
        )
        names = (
            "total",
            "time_reconstruction",
            "masked_time_reconstruction",
            "frequency_reconstruction",
            "frequency_time_reconstruction",
            "frequency_probability",
            "forecast",
        )
        for name in names:
            metric = "loss" if name == "total" else name
            self.log(
                f"{stage}/{metric}",
                getattr(losses, name),
                on_step=stage == "train" and name == "total",
                on_epoch=True,
                prog_bar=name == "total",
                sync_dist=True,
                batch_size=series.shape[0],
            )
        return losses.total

    def training_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def predict_step(
        self, batch: object, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        series, _, _ = self._batch(batch)
        return self.model.score(series, batch_size=self.config.finetune_batch_size)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.finetune_learning_rate,
            weight_decay=self.config.finetune_weight_decay,
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.finetune_learning_rate,
            total_steps=total_steps,
            pct_start=self.config.one_cycle_pct_start,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
