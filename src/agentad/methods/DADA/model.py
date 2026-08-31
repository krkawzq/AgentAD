"""DADA masked-copy inference and adaptive bottleneck architecture."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.autograd import Function
from torch.nn import functional as F
from collections.abc import Mapping
import lightning as L

from .._utils import evaluation_mode, validate_series
from .config import DADAConfig


@dataclass(frozen=True, slots=True)
class DADAOutput:
    reconstructions: Tensor
    balance_loss: Tensor
    adversarial_reconstruction: Tensor | None = None


class _SamePadConv(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, kernel_size: int, dilation: int = 1
    ) -> None:
        super().__init__()
        receptive_field = (kernel_size - 1) * dilation + 1
        self.remove_last = receptive_field % 2 == 0
        self.conv = nn.Conv1d(
            input_dim,
            output_dim,
            kernel_size,
            padding=receptive_field // 2,
            dilation=dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return x[..., :-1] if self.remove_last else x


class _ResidualConvBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, final: bool = False) -> None:
        super().__init__()
        self.conv1 = _SamePadConv(input_dim, output_dim, 3)
        self.conv2 = _SamePadConv(output_dim, output_dim, 3)
        self.projector = (
            nn.Conv1d(input_dim, output_dim, 1)
            if input_dim != output_dim or final
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x if self.projector is None else self.projector(x)
        x = self.conv1(F.gelu(x))
        x = self.conv2(F.gelu(x))
        return x + residual


class _FeatureExtractor(nn.Module):
    """Typed container keeping the released checkpoint's `feature_extractor.net` keys."""

    def __init__(self, net: nn.Sequential) -> None:
        super().__init__()
        self.net = net


class _DilatedEncoder(nn.Module):
    def __init__(
        self, hidden_dim: int, output_dim: int, depth: int, dropout: float
    ) -> None:
        super().__init__()
        widths = [hidden_dim] * depth + [output_dim]
        self.feature_extractor = _FeatureExtractor(
            nn.Sequential(
                *[
                    _ResidualConvBlock(
                        hidden_dim if index == 0 else widths[index - 1],
                        width,
                        final=index == len(widths) - 1,
                    )
                    for index, width in enumerate(widths)
                ]
            )
        )
        self.repr_dropout = nn.Dropout(dropout)

    def forward(self, patches: Tensor) -> Tensor:
        encoded = self.feature_extractor.net(patches.transpose(1, 2))
        return self.repr_dropout(encoded).transpose(1, 2)


class _BottleneckExpert(nn.Module):
    def __init__(
        self,
        patch_count: int,
        representation_dim: int,
        bottleneck_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_count = patch_count
        self.representation_dim = representation_dim
        self.net = nn.Sequential(
            nn.Linear(representation_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, representation_dim),
        )

    def forward(self, representation: Tensor) -> Tensor:
        residual = representation
        shaped = representation.reshape(-1, self.patch_count, self.representation_dim)
        return residual + self.net(shaped).flatten(1)


class _AdaptiveBottleneck(nn.Module):
    def __init__(self, config: DADAConfig, patch_count: int) -> None:
        super().__init__()
        self.k = config.experts_per_input
        self.noisy_gating = config.noisy_gating
        self.loss_weight = config.load_balance_weight
        input_dim = patch_count * config.representation_dim
        self.bottlenecks = nn.ModuleList(
            _BottleneckExpert(
                patch_count,
                config.representation_dim,
                width,
                config.bottleneck_dropout,
            )
            for width in config.bottleneck_dims
        )
        self.w_gate = nn.Parameter(torch.zeros(input_dim, len(self.bottlenecks)))
        self.w_noise = nn.Parameter(torch.zeros(input_dim, len(self.bottlenecks)))
        # These buffers retain compatibility with the released DADA checkpoint.
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

    @staticmethod
    def _cv_squared(values: Tensor) -> Tensor:
        if values.numel() <= 1:
            return values.new_zeros(())
        values = values.float()
        return values.var() / (values.mean().square() + 1e-10)

    def _gates(self, x: Tensor) -> tuple[Tensor, Tensor]:
        clean_logits = x @ self.w_gate
        noisy_logits: Tensor | None = None
        noise_std: Tensor | None = None
        if self.noisy_gating and self.training:
            noise_std = F.softplus(x @ self.w_noise) + 1e-2
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_std
            logits = noisy_logits
        else:
            logits = clean_logits
        top_count = min(self.k + 1, logits.shape[1])
        top_values, top_indices = logits.topk(top_count, dim=1)
        values = top_values[:, : self.k]
        indices = top_indices[:, : self.k]
        weights = values.softmax(dim=1)
        gates = torch.zeros_like(logits).scatter(1, indices, weights)
        if (
            noisy_logits is not None
            and noise_std is not None
            and self.k < logits.shape[1]
        ):
            threshold_in = top_values[:, self.k : self.k + 1]
            threshold_out = top_values[:, self.k - 1 : self.k]
            is_in = noisy_logits > threshold_in
            standardized_in = (clean_logits - threshold_in) / noise_std
            standardized_out = (clean_logits - threshold_out) / noise_std
            cdf_in = 0.5 * (1 + torch.erf(standardized_in / math.sqrt(2)))
            cdf_out = 0.5 * (1 + torch.erf(standardized_out / math.sqrt(2)))
            load = torch.where(is_in, cdf_in, cdf_out).sum(0)
        else:
            load = (gates > 0).sum(0).to(gates.dtype)
        balance = (
            self._cv_squared(gates.sum(0)) + self._cv_squared(load)
        ) * self.loss_weight
        return gates, balance

    def forward(
        self, router_input: Tensor, representation: Tensor
    ) -> tuple[Tensor, Tensor]:
        gates, balance = self._gates(router_input)
        combined = torch.zeros_like(representation)
        for expert_index, expert in enumerate(self.bottlenecks):
            rows = torch.nonzero(gates[:, expert_index] > 0, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            expert_output = expert(representation.index_select(0, rows))
            weights = gates.index_select(0, rows)[:, expert_index, None]
            combined.index_add_(0, rows, expert_output * weights)
        return combined, balance


class _Decoder(nn.Module):
    def __init__(self, representation_dim: int, patch_length: int) -> None:
        super().__init__()
        hidden = representation_dim * 2
        self.net = nn.Sequential(
            nn.GELU(),
            nn.Linear(representation_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, patch_length),
        )
        self.flatten = nn.Flatten(-2)

    def forward(self, representation: Tensor) -> Tensor:
        return self.flatten(self.net(representation))


class _ReverseGradient(Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, coefficient: Tensor) -> Tensor:
        ctx.coefficient = coefficient
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor, None]:
        return -ctx.coefficient * gradient, None


class _WarmGradientReversal(nn.Module):
    steps: Tensor

    def __init__(self, maximum: float, total_steps: int) -> None:
        super().__init__()
        self.maximum = maximum
        self.total_steps = total_steps
        self.register_buffer(
            "steps", torch.zeros((), dtype=torch.long), persistent=False
        )

    def forward(self, x: Tensor) -> Tensor:
        # The original constructs the layer with auto_step=True: every forward
        # pass advances the counter, and the progress is never clamped.
        progress = self.steps.to(x.dtype) / self.total_steps
        coefficient = self.maximum * (2 / (1 + torch.exp(-progress)) - 1)
        self.steps.add_(1)
        return _ReverseGradient.apply(x, coefficient)


class DADA(nn.Module):
    """Released DADA architecture and stochastic reconstruction score.

    The original project's pretraining objective is not public. ``forward`` exposes
    both decoder paths and the MoE balance loss so a caller can reproduce a licensed
    training procedure without this package silently inventing one.
    """

    def __init__(self, config: DADAConfig = DADAConfig()) -> None:
        super().__init__()
        self.config = config
        self.patch_count = math.ceil(config.sequence_length / config.patch_length)
        self.input_embed = nn.Linear(config.patch_length, config.hidden_dim)
        self.encoder = _DilatedEncoder(
            config.hidden_dim,
            config.representation_dim,
            config.encoder_depth,
            config.representation_dropout,
        )
        self.adaptive_bottleneck = _AdaptiveBottleneck(config, self.patch_count)
        self.decoder = _Decoder(config.representation_dim, config.patch_length)
        self.adv_decoder = _Decoder(config.representation_dim, config.patch_length)
        self.grl = _WarmGradientReversal(
            config.gradient_reverse_max,
            config.gradient_reverse_steps,
        )

    def _masked_patches(
        self,
        patches: Tensor,
        copies: int,
        mode: str,
        generator: torch.Generator | None,
    ) -> Tensor:
        if mode == "none":
            if copies != 1:
                raise ValueError("mask_mode='none' requires copies=1")
            return patches
        repeated = patches.repeat(copies, 1, 1)
        if mode == "complementary":
            if copies % 2:
                raise ValueError(
                    "complementary masking requires an even number of copies"
                )
            first = (
                torch.rand(
                    patches.shape[0] * (copies // 2),
                    self.patch_count,
                    device=patches.device,
                    generator=generator,
                )
                < 0.5
            )
            mask = torch.cat((first, ~first), dim=0)
        elif mode == "random":
            mask = (
                torch.rand(
                    patches.shape[0] * copies,
                    self.patch_count,
                    device=patches.device,
                    generator=generator,
                )
                < 0.5
            )
        else:
            raise ValueError(f"unknown mask mode: {mode!r}")
        return repeated.masked_fill(mask.unsqueeze(-1), 0)

    def _represent(
        self,
        x: Tensor,
        *,
        copies: int,
        mask_mode: str,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor]:
        batch, time, features = x.shape
        independent = x.transpose(1, 2).reshape(batch * features, time)
        padded_length = self.patch_count * self.config.patch_length
        if padded_length != time:
            independent = F.pad(independent, (0, padded_length - time))
        patches = independent.unfold(
            -1, self.config.patch_length, self.config.patch_length
        )
        patches = self.input_embed(patches)
        patches = self._masked_patches(patches, copies, mask_mode, generator)
        representation = self.encoder(patches).flatten(1)
        representation, balance = self.adaptive_bottleneck(
            representation, representation
        )
        return representation.reshape(
            -1, self.patch_count, self.config.representation_dim
        ), balance

    def _decode(
        self,
        representation: Tensor,
        decoder: nn.Module,
        batch: int,
        features: int,
        copies: int,
    ) -> Tensor:
        reconstructed = decoder(representation)[..., : self.config.sequence_length]
        reconstructed = reconstructed.reshape(
            copies, batch, features, self.config.sequence_length
        )
        return reconstructed.permute(0, 1, 3, 2)

    def forward(
        self,
        x: Tensor,
        *,
        copies: int | None = None,
        mask_mode: str | None = None,
        normalize: bool = False,
        adversarial: bool = False,
        generator: torch.Generator | None = None,
    ) -> DADAOutput:
        x = validate_series(x, length=self.config.sequence_length)
        copies = self.config.copies if copies is None else copies
        mode = self.config.mask_mode if mask_mode is None else mask_mode
        if copies <= 0:
            raise ValueError("copies must be positive")
        if mode == "none":
            copies = 1

        mean: Tensor | None = None
        scale: Tensor | None = None
        if normalize:
            mean = x.mean(1, keepdim=True).detach()
            scale = x.var(1, keepdim=True, unbiased=False).add(1e-5).sqrt()
            model_input = (x - mean) / scale
        else:
            model_input = x
        representation, balance = self._represent(
            model_input,
            copies=copies,
            mask_mode=mode,
            generator=generator,
        )
        reconstruction = self._decode(
            representation, self.decoder, x.shape[0], x.shape[2], copies
        )
        adversarial_reconstruction: Tensor | None = None
        if adversarial:
            reversed_representation = self.grl(representation)
            adversarial_reconstruction = self._decode(
                reversed_representation,
                self.adv_decoder,
                x.shape[0],
                x.shape[2],
                copies,
            )
        if mean is not None and scale is not None:
            reconstruction = reconstruction * scale.unsqueeze(0) + mean.unsqueeze(0)
            if adversarial_reconstruction is not None:
                adversarial_reconstruction = (
                    adversarial_reconstruction * scale.unsqueeze(0) + mean.unsqueeze(0)
                )
        return DADAOutput(reconstruction, balance, adversarial_reconstruction)

    @torch.inference_mode()
    def score(
        self,
        x: Tensor,
        *,
        normalize: bool = False,
        variance_weight: float = 1.0,
        generator: torch.Generator | None = None,
        error_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> Tensor:
        """Return masked-copy uncertainty, optionally mixed with reconstruction error.

        ``error_fn`` replaces the default per-point squared error of the mixed
        path, mirroring the original's injectable ``anomaly_criterion``; its
        output is averaged over the channel dimension like the original's
        ``torch.mean(recon_score, dim=-1)``.
        """
        if not 0 <= variance_weight <= 1:
            raise ValueError("variance_weight must be in [0, 1]")
        if self.config.copies < 2:
            raise ValueError("DADA variance scoring requires at least two copies")
        with evaluation_mode(self):
            output = self(x, normalize=normalize, generator=generator)
            uncertainty = output.reconstructions.var(dim=0, unbiased=True).mean(dim=2)
            if variance_weight == 1:
                return uncertainty
            mean_reconstruction = output.reconstructions.mean(0)
            if error_fn is None:
                reconstruction = (mean_reconstruction - x).square().mean(dim=2)
            else:
                reconstruction = error_fn(mean_reconstruction, x).mean(dim=2)
            return (
                variance_weight * uncertainty + (1 - variance_weight) * reconstruction
            )

    def load_reference_checkpoint(
        self, path: str | Path, *, strict: bool = True
    ) -> None:
        """Load the published Hugging Face checkpoint without depending on transformers."""
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise TypeError("checkpoint does not contain a state dictionary")
        converted = {
            (key[6:] if key.startswith("model.") else key): value
            for key, value in state.items()
        }
        self.load_state_dict(converted, strict=strict)

    @classmethod
    def from_official_checkpoint(cls, path: str | Path) -> "DADA":
        """Build a DADA with the published architecture and load its weights.

        The released checkpoint stores the Hugging Face AutoModel form, whose
        keys carry a ``model.`` prefix; :meth:`load_reference_checkpoint`
        strips it, so the published weights load without any renaming.
        """
        model = cls(DADAConfig.original_pretrained())
        model.load_reference_checkpoint(path)
        return model


class DADALightningModule(L.LightningModule):
    """DADA has no public training objective; this module never invents one."""

    def __init__(self, config: DADAConfig | None = None) -> None:
        super().__init__()
        self.config = config or DADAConfig.original_pretrained()
        self.model = DADA(self.config)
        if self.config.reference_checkpoint is not None:
            self.model.load_reference_checkpoint(self.config.reference_checkpoint)
        self.automatic_optimization = False

    def forward(self, series: Tensor, **kwargs: object):
        return self.model(series, **kwargs)

    @staticmethod
    def _series(batch: object) -> Tensor:
        series: Tensor | None
        if isinstance(batch, Tensor):
            series = batch
        elif isinstance(batch, Mapping):
            series = batch.get("series")
        elif isinstance(batch, (tuple, list)) and batch:
            series = batch[0]
        else:
            series = None
        if not isinstance(series, Tensor):
            raise TypeError("DADA batches must provide a series tensor")
        return series

    def training_step(self, batch: object, batch_idx: int) -> None:
        raise RuntimeError(
            "The released DADA project provides pretrained zero-shot inference but "
            "does not publish a training objective. Load its reference checkpoint."
        )

    def validation_step(self, batch: object, batch_idx: int) -> Tensor:
        score = self.model.score(
            self._series(batch),
            normalize=self.config.normalization,
            variance_weight=self.config.variance_weight,
        )
        value = score.mean()
        self.log("val/mean_score", value, sync_dist=True, batch_size=score.shape[0])
        return value

    def predict_step(
        self, batch: object, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        return self.model.score(
            self._series(batch),
            normalize=self.config.normalization,
            variance_weight=self.config.variance_weight,
        )

    def configure_optimizers(self) -> None:
        return None
