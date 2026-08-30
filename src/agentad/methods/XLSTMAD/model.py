"""Exact xLSTMAD encoder-decoder using the official xLSTM block stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from collections.abc import Mapping
import lightning as L

from .._utils import evaluation_mode, sliding_windows, validate_series
from .config import XLSTMADConfig
from .._utils import ValidationEarlyStopping


def _build_stack_config(config: XLSTMADConfig) -> Any:
    try:
        from xlstm import (
            FeedForwardConfig,
            mLSTMBlockConfig,
            mLSTMLayerConfig,
            sLSTMBlockConfig,
            sLSTMLayerConfig,
            xLSTMBlockStackConfig,
        )
    except ImportError as exc:
        raise ImportError(
            "xLSTMAD requires xlstm==2.0.5. Run `uv sync --extra xlstmad`."
        ) from exc
    return xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=config.matrix_kernel,
                qkv_proj_blocksize=config.matrix_qkv_projection_block_size,
                num_heads=config.heads,
                round_proj_up_dim_up=config.matrix_round_projection_up,
                round_proj_up_to_multiple_of=config.matrix_projection_multiple,
                embedding_dim=config.embedding_dim,
            )
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend=config.scalar_backend,
                num_heads=config.heads,
                conv1d_kernel_size=config.scalar_kernel,
                bias_init=config.scalar_bias_initialization,
            ),
            feedforward=FeedForwardConfig(
                proj_factor=config.feedforward_factor,
                act_fn=config.feedforward_activation,
                embedding_dim=config.embedding_dim,
            ),
        ),
        context_length=config.window_length,
        num_blocks=config.blocks,
        embedding_dim=config.embedding_dim,
        slstm_at=list(config.scalar_memory_blocks),
    )


def _stack(config: XLSTMADConfig) -> nn.Module:
    try:
        from xlstm import xLSTMBlockStack
    except ImportError as exc:
        raise ImportError(
            "xLSTMAD requires xlstm==2.0.5. Run `uv sync --extra xlstmad`."
        ) from exc
    return xLSTMBlockStack(_build_stack_config(config))


@dataclass(frozen=True, slots=True)
class XLSTMADOutput:
    reconstruction: Tensor


@dataclass(frozen=True, slots=True)
class XLSTMADLoss:
    total: Tensor
    reconstruction: Tensor


class XLSTMAD(nn.Module):
    def __init__(self, config: XLSTMADConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = _stack(config)
        self.decoder = _stack(config)
        self.input_projection = nn.Linear(config.input_features, config.embedding_dim)
        self.output_projection = nn.Linear(config.embedding_dim, config.input_features)
        self.activation = nn.GELU()

    def forward(self, windows: Tensor) -> XLSTMADOutput:
        windows = validate_series(
            windows, length=self.config.window_length, name="windows"
        )
        if windows.shape[2] != self.config.input_features:
            raise ValueError("window feature count does not match xLSTMAD")
        encoded = self.encoder(self.input_projection(windows))
        decoded = self.decoder(encoded)
        return XLSTMADOutput(self.output_projection(self.activation(decoded)))

    def compute_loss(self, windows: Tensor) -> XLSTMADLoss:
        reconstruction = F.mse_loss(self(windows).reconstruction, windows)
        return XLSTMADLoss(reconstruction, reconstruction)

    @torch.inference_mode()
    def window_score(self, windows: Tensor) -> Tensor:
        with evaluation_mode(self):
            reconstruction = self(windows).reconstruction
            return (reconstruction - windows).square().mean((1, 2))

    @torch.inference_mode()
    def score(self, series: Tensor, *, batch_size: int = 256) -> Tensor:
        """Return per-window scores aligned to each window's last time step.

        The first ``window_length - 1`` points are zero padding; the original
        produces no scores for them.
        """
        series = validate_series(
            series, min_length=self.config.window_length, name="series"
        )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        with evaluation_mode(self):
            windows = sliding_windows(series, self.config.window_length)
            flattened = windows.flatten(0, 1)
            scores = torch.cat(
                [self.window_score(chunk) for chunk in flattened.split(batch_size)]
            ).reshape(series.shape[0], -1)
            return F.pad(scores, (self.config.window_length - 1, 0))


class XLSTMADLightningModule(ValidationEarlyStopping, L.LightningModule):
    def __init__(self, config: XLSTMADConfig) -> None:
        super().__init__()
        self.config = config
        self.model = XLSTMAD(config)
        self._init_validation_early_stopping(
            monitor="val/loss",
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
        )

    def forward(self, windows: Tensor):
        return self.model(windows)

    @staticmethod
    def _batch(batch: object) -> tuple[Tensor, Tensor | None]:
        if isinstance(batch, Tensor):
            return batch, None
        if isinstance(batch, Mapping):
            windows = batch.get("windows", batch.get("series"))
            labels = batch.get("labels")
        elif isinstance(batch, (tuple, list)) and batch:
            windows = batch[0]
            labels = batch[1] if len(batch) > 1 else None
        else:
            windows = labels = None
        if not isinstance(windows, Tensor):
            raise TypeError("xLSTMAD batches must provide windows")
        if labels is not None and not isinstance(labels, Tensor):
            raise TypeError("xLSTMAD labels must be tensors when provided")
        return windows, labels

    def _step(self, batch: object, stage: str) -> Tensor:
        windows, _ = self._batch(batch)
        loss = self.model.compute_loss(windows).total
        self.log(
            f"{stage}/loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=windows.shape[0],
        )
        return loss

    def training_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def predict_step(
        self, batch: object, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        # The original predict_step compares the reconstruction against the
        # label tensor and cannot run; scoring follows its test_step, which
        # reconstructs the input windows.
        windows, _ = self._batch(batch)
        return self.model.window_score(windows)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
