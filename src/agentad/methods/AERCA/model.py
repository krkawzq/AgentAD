"""Adaptive error reconstruction with dynamic Granger coefficients."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from collections.abc import Mapping
import lightning as L
from torch import Tensor

from .._utils import evaluation_mode, sliding_windows, validate_series
from .config import AERCAConfig
from .._utils import ValidationEarlyStopping


class _SelfExplainingGranger(nn.Module):
    """Amortize one directed coefficient matrix for every lag and sample."""

    def __init__(self, config: AERCAConfig) -> None:
        super().__init__()
        self.features = config.input_features
        self.order = config.window
        networks: list[nn.Module] = []
        for _ in range(config.window):
            layers: list[nn.Module] = [
                nn.Linear(config.input_features, config.hidden_dim),
                nn.ReLU(),
            ]
            for _ in range(config.hidden_layers - 1):
                layers.extend(
                    (nn.Linear(config.hidden_dim, config.hidden_dim), nn.ReLU())
                )
            layers.extend(
                (
                    nn.Linear(config.hidden_dim, config.input_features**2),
                    nn.Tanh(),
                )
            )
            networks.append(nn.Sequential(*layers))
        self.coefficient_networks = nn.ModuleList(networks)

    def forward(self, histories: Tensor) -> tuple[Tensor, Tensor]:
        if histories.ndim != 3 or histories.shape[1:] != (self.order, self.features):
            raise ValueError(
                f"histories must have shape [sample, {self.order}, {self.features}]"
            )
        coefficients = torch.stack(
            [
                network(histories[:, lag]).reshape(-1, self.features, self.features)
                for lag, network in enumerate(self.coefficient_networks)
            ],
            dim=1,
        )
        prediction = torch.einsum("nkij,nkj->ni", coefficients, histories)
        return prediction, coefficients


@dataclass(frozen=True, slots=True)
class AERCAOutput:
    reconstruction: Tensor
    target: Tensor
    encoder_coefficients: Tensor
    decoder_coefficients: Tensor
    previous_coefficients: Tensor
    residuals: Tensor
    kl_divergence: Tensor


@dataclass(frozen=True, slots=True)
class AERCALoss:
    total: Tensor
    reconstruction: Tensor
    encoder_sparsity: Tensor
    decoder_sparsity: Tensor
    encoder_smoothness: Tensor
    decoder_smoothness: Tensor
    kl_divergence: Tensor


class AERCA(nn.Module):
    """AERCA detector with causal-graph and root-cause outputs."""

    def __init__(self, config: AERCAConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = _SelfExplainingGranger(config)
        self.residual_decoder = _SelfExplainingGranger(config)
        self.history_decoder = _SelfExplainingGranger(config)
        self.register_buffer("root_cause_center", torch.zeros(config.input_features))
        self.register_buffer("root_cause_scale", torch.ones(config.input_features))
        self.register_buffer(
            "root_cause_calibrated", torch.tensor(False), persistent=False
        )

    def _kl_standard_normal(self, residuals: Tensor) -> Tensor:
        # The original fits the residual distribution one series at a time,
        # so every batch member gets its own KL term, averaged over the batch.
        count, features = residuals.shape[1:]
        mean = residuals.mean(1)
        centered = residuals - mean.unsqueeze(1)
        covariance = (
            torch.einsum("bti,btj->bij", centered, centered)
            / max(count - 1, 1)
        )
        # The original regularizes by the condition number so that stiff
        # covariance matrices receive proportionally stronger shrinkage.
        eigenvalues = torch.linalg.eigvalsh(covariance)
        condition = (
            eigenvalues.max(-1).values
            / eigenvalues.clamp_min(1e-9).min(-1).values
        )
        covariance = covariance + torch.eye(
            features,
            device=residuals.device,
            dtype=residuals.dtype,
        ) * (self.config.covariance_epsilon * condition).unsqueeze(-1).unsqueeze(-1)
        sign, logdet = torch.linalg.slogdet(covariance)
        if torch.any(sign <= 0):
            raise RuntimeError("residual covariance is not positive definite")
        kl = (
            covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
            + mean.square().sum(-1)
            - features
            + logdet
        )
        return kl.mean()

    def _validate_input(self, x: Tensor) -> Tensor:
        minimum = 2 * self.config.window + 2
        x = validate_series(x, min_length=minimum)
        if x.shape[2] != self.config.input_features:
            raise ValueError("x feature count does not match the AERCA config")
        return x

    def _encode_series(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch = x.shape[0]
        encoder_windows = sliding_windows(x, self.config.window + 1)
        histories = encoder_windows[:, :, :-1]
        next_values = encoder_windows[:, :, -1]
        encoder_prediction, encoder_coefficients = self.encoder(histories.flatten(0, 1))
        encoder_prediction = encoder_prediction.reshape(
            batch, -1, self.config.input_features
        )
        encoder_coefficients = encoder_coefficients.reshape(
            batch,
            -1,
            self.config.window,
            self.config.input_features,
            self.config.input_features,
        )
        residuals = encoder_prediction - next_values
        return histories, next_values, encoder_coefficients, residuals

    def _decode_series(
        self,
        histories: Tensor,
        next_values: Tensor,
        residuals: Tensor,
        *,
        include_current_residual: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch = histories.shape[0]
        residual_windows = sliding_windows(residuals, self.config.window + 1)
        residual_histories = residual_windows[:, :, :-1]
        current_residual = residual_windows[:, :, -1]
        decoder_count = residual_histories.shape[1]
        original_histories = histories[:, :decoder_count]

        residual_prediction, decoder_coefficients = self.residual_decoder(
            residual_histories.flatten(0, 1)
        )
        history_prediction, previous_coefficients = self.history_decoder(
            original_histories.flatten(0, 1)
        )
        residual_prediction = residual_prediction.reshape(
            batch, decoder_count, self.config.input_features
        )
        history_prediction = history_prediction.reshape_as(residual_prediction)
        reconstruction = residual_prediction + history_prediction
        if include_current_residual:
            reconstruction = reconstruction + current_residual

        coefficient_shape = (
            batch,
            decoder_count,
            self.config.window,
            self.config.input_features,
            self.config.input_features,
        )
        target = next_values[:, self.config.window :]
        return (
            reconstruction,
            target,
            decoder_coefficients.reshape(coefficient_shape),
            previous_coefficients.reshape(coefficient_shape),
        )

    def forward(
        self, x: Tensor, *, include_current_residual: bool = True
    ) -> AERCAOutput:
        x = self._validate_input(x)
        histories, next_values, encoder_coefficients, residuals = self._encode_series(x)
        reconstruction, target, decoder_coefficients, previous_coefficients = (
            self._decode_series(
                histories,
                next_values,
                residuals,
                include_current_residual=include_current_residual,
            )
        )
        return AERCAOutput(
            reconstruction,
            target,
            encoder_coefficients,
            decoder_coefficients,
            previous_coefficients,
            residuals,
            self._kl_standard_normal(residuals),
        )

    @staticmethod
    def _sparsity(coefficients: Tensor, alpha: float) -> Tensor:
        l2 = torch.linalg.vector_norm(coefficients, ord=2, dim=2).mean()
        l1 = torch.linalg.vector_norm(coefficients, ord=1, dim=2).mean()
        return (1 - alpha) * l2 + alpha * l1

    @staticmethod
    def _smoothness(coefficients: Tensor) -> Tensor:
        # Smoothness is measured across adjacent lags within each window's
        # coefficient set, matching the original GVAR regularizer; it vanishes
        # for order-one (window=1) models.
        if coefficients.shape[2] < 2:
            return coefficients.new_zeros(())
        difference = coefficients[:, :, 1:] - coefficients[:, :, :-1]
        return torch.linalg.vector_norm(difference, ord=2, dim=2).mean()

    def compute_loss(self, x: Tensor) -> AERCALoss:
        output = self(x)
        reconstruction = F.mse_loss(output.reconstruction, output.target)
        encoder_sparsity = self._sparsity(
            output.encoder_coefficients, self.config.encoder_alpha
        )
        decoder_sparsity = self._sparsity(
            output.decoder_coefficients, self.config.decoder_alpha
        ) + self._sparsity(output.previous_coefficients, self.config.decoder_alpha)
        encoder_smoothness = self._smoothness(output.encoder_coefficients)
        decoder_smoothness = self._smoothness(
            output.decoder_coefficients
        ) + self._smoothness(output.previous_coefficients)
        total = (
            reconstruction
            + self.config.encoder_sparsity_weight * encoder_sparsity
            + self.config.decoder_sparsity_weight * decoder_sparsity
            + self.config.encoder_smoothness_weight * encoder_smoothness
            + self.config.decoder_smoothness_weight * decoder_smoothness
            + self.config.kl_weight * output.kl_divergence
        )
        return AERCALoss(
            total,
            reconstruction,
            encoder_sparsity,
            decoder_sparsity,
            encoder_smoothness,
            decoder_smoothness,
            output.kl_divergence,
        )

    def _feature_score(self, x: Tensor) -> Tensor:
        """Return decoder reconstruction errors aligned to input time points."""
        x = self._validate_input(x)
        histories, next_values, _, residuals = self._encode_series(x)
        reconstruction, target, _, _ = self._decode_series(
            histories,
            next_values,
            residuals,
            include_current_residual=False,
        )
        error = (reconstruction - target).square()
        return F.pad(error.transpose(1, 2), (2 * self.config.window, 0)).transpose(1, 2)

    @torch.inference_mode()
    def feature_score(self, x: Tensor) -> Tensor:
        with evaluation_mode(self):
            return self._feature_score(x)

    @torch.inference_mode()
    def score(self, x: Tensor) -> Tensor:
        with evaluation_mode(self):
            return self._feature_score(x).mean(dim=2)

    @torch.inference_mode()
    def calibrate_root_causes(self, normal_series: Tensor) -> None:
        with evaluation_mode(self):
            normal_series = self._validate_input(normal_series)
            _, _, _, residuals = self._encode_series(normal_series)
        flattened = residuals.flatten(0, 1)
        self.root_cause_center.copy_(flattened.median(0).values)
        self.root_cause_scale.copy_(
            flattened.std(0, unbiased=False).clamp_min(self.config.covariance_epsilon)
        )
        self.root_cause_calibrated.fill_(True)

    @torch.inference_mode()
    def root_cause_score(self, x: Tensor) -> Tensor:
        """Return signed encoder-residual z-scores aligned as ``[B, T, C]``."""
        if not self.root_cause_calibrated:
            raise RuntimeError("call calibrate_root_causes() before root_cause_score()")
        with evaluation_mode(self):
            x = self._validate_input(x)
            _, _, _, residuals = self._encode_series(x)
            residuals = residuals[:, self.config.window :]
        score = -(residuals - self.root_cause_center) / self.root_cause_scale
        return F.pad(score.transpose(1, 2), (2 * self.config.window, 0)).transpose(1, 2)

    @torch.inference_mode()
    def causal_graph(self, x: Tensor) -> Tensor:
        """Aggregate dynamic coefficients into ``[B, effect, cause]``."""
        with evaluation_mode(self):
            x = self._validate_input(x)
            _, _, coefficients, _ = self._encode_series(x)
            coefficients = coefficients.abs().median(1).values
        return coefficients.max(1).values


class AERCALightningModule(ValidationEarlyStopping, L.LightningModule):
    def __init__(self, config: AERCAConfig) -> None:
        super().__init__()
        self.config = config
        self.model = AERCA(config)
        self._init_validation_early_stopping(
            monitor="val/loss",
            patience=config.patience,
        )

    def forward(self, series: Tensor, *, include_current_residual: bool = True):
        return self.model(
            series, include_current_residual=include_current_residual
        )

    @staticmethod
    def _series(batch: object) -> Tensor:
        if isinstance(batch, Tensor):
            return batch
        if isinstance(batch, Mapping):
            value = batch.get("series")
        elif isinstance(batch, (tuple, list)) and batch:
            value = batch[0]
        else:
            value = None
        if not isinstance(value, Tensor):
            raise TypeError("AERCA batches must provide a series tensor")
        return value

    def _step(self, batch: object, stage: str) -> Tensor:
        series = self._series(batch)
        losses: AERCALoss = self.model.compute_loss(series)
        self.log(
            f"{stage}/loss",
            losses.total,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=series.shape[0],
        )
        for name in (
            "reconstruction",
            "encoder_sparsity",
            "decoder_sparsity",
            "encoder_smoothness",
            "decoder_smoothness",
            "kl_divergence",
        ):
            self.log(
                f"{stage}/{name}",
                getattr(losses, name),
                on_step=False,
                on_epoch=True,
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
        return self.model.score(self._series(batch))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
