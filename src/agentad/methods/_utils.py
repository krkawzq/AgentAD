"""Shared tensor utilities for AgentAD neural methods."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import math

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class ValidationEarlyStopping:
    """Mixin for Lightning modules replicating the original early-stopping loops.

    Tracks the monitored validation metric per epoch, requests the trainer to
    stop after ``patience`` epochs without an improvement of at least
    ``min_delta``, and reloads the best weights when training ends. Host
    modules must not define the same hooks without calling ``super()``.
    """

    def _init_validation_early_stopping(
        self, *, monitor: str, patience: int, min_delta: float = 0.0
    ) -> None:
        self._monitor = monitor
        self._early_stopping_patience = patience
        self._early_stopping_min_delta = min_delta
        self._best_metric: float | None = None
        self._stale_epochs = 0
        self._best_state: dict[str, Tensor] | None = None

    def on_validation_epoch_end(self) -> None:
        trainer = getattr(self, "trainer", None)
        if trainer is None or trainer.sanity_checking:
            return
        metric = trainer.callback_metrics.get(self._monitor)
        if metric is None:
            raise RuntimeError(
                f"validation early stopping requires the {self._monitor!r} metric"
            )
        value = float(metric)
        if (
            self._best_metric is None
            or value < self._best_metric - self._early_stopping_min_delta
        ):
            self._best_metric = value
            self._stale_epochs = 0
            self._best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in self.state_dict().items()
            }
        else:
            self._stale_epochs += 1
            if self._stale_epochs >= self._early_stopping_patience:
                trainer.should_stop = True

    def on_train_end(self) -> None:
        if self._best_state is not None:
            self.load_state_dict(self._best_state)


@contextmanager
def evaluation_mode(module: nn.Module) -> Iterator[None]:
    """Temporarily use deterministic evaluation behavior and restore the caller state."""
    states = [(submodule, submodule.training) for submodule in module.modules()]
    module.eval()
    try:
        yield
    finally:
        for submodule, was_training in states:
            submodule.training = was_training


def validate_series(
    x: Tensor,
    *,
    length: int | None = None,
    min_length: int = 1,
    name: str = "x",
) -> Tensor:
    """Validate the package-wide ``[batch, time, feature]`` convention."""
    if not isinstance(x, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.ndim != 3:
        raise ValueError(
            f"{name} must have shape [batch, time, feature], got {tuple(x.shape)}"
        )
    if not x.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype, got {x.dtype}")
    if x.shape[0] == 0 or x.shape[2] == 0:
        raise ValueError(f"{name} must contain at least one batch item and feature")
    if x.shape[1] < min_length:
        raise ValueError(f"{name} must contain at least {min_length} time steps")
    if length is not None and x.shape[1] != length:
        raise ValueError(
            f"{name} must contain exactly {length} time steps, got {x.shape[1]}"
        )
    return x


def sliding_windows(x: Tensor, window: int, stride: int = 1) -> Tensor:
    """Return a zero-copy ``[batch, windows, window, feature]`` view."""
    validate_series(x, min_length=window)
    if window <= 0:
        raise ValueError("window must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    # Tensor.unfold appends the extracted dimension at the end.
    return x.unfold(1, window, stride).permute(0, 1, 3, 2)


def overlap_average(
    patch_scores: Tensor,
    *,
    patch_length: int,
    output_length: int,
    stride: int = 1,
) -> Tensor:
    """Average patch scores over all time points covered by each patch."""
    if patch_scores.ndim != 2:
        raise ValueError("patch_scores must have shape [batch, patches]")
    if patch_length <= 0 or stride <= 0 or output_length <= 0:
        raise ValueError("patch_length, stride, and output_length must be positive")

    batch, count = patch_scores.shape
    expected = (output_length - patch_length) // stride + 1
    if output_length < patch_length or count != expected:
        raise ValueError(
            f"{count} patch scores cannot cover length {output_length} "
            f"with patch_length={patch_length} and stride={stride}"
        )
    covered_length = (count - 1) * stride + patch_length
    if stride > patch_length or covered_length != output_length:
        raise ValueError(
            "patch layout leaves uncovered output points; choose a stride that "
            "covers the complete output length"
        )

    starts = torch.arange(count, device=patch_scores.device) * stride
    ends = starts + patch_length
    starts = starts.unsqueeze(0).expand(batch, -1)
    ends = ends.unsqueeze(0).expand(batch, -1)

    # Difference arrays avoid materializing one index and value per covered point.
    sums = patch_scores.new_zeros(batch, output_length + 1)
    sums.scatter_add_(1, starts, patch_scores)
    sums.scatter_add_(1, ends, -patch_scores)
    counts = patch_scores.new_zeros(batch, output_length + 1)
    counts.scatter_add_(1, starts, torch.ones_like(patch_scores))
    counts.scatter_add_(1, ends, -torch.ones_like(patch_scores))
    sums = sums[:, :-1].cumsum(1)
    counts = counts[:, :-1].cumsum(1)
    return sums / counts.clamp_min_(1)


def topk_cosine_distance(
    queries: Tensor,
    bank: Tensor,
    *,
    k: int,
    query_chunk_size: int = 2048,
    bank_chunk_size: int = 16384,
    bank_is_normalized: bool = False,
) -> Tensor:
    """Mean top-k cosine distance with bounded temporary memory."""
    if queries.ndim != 2 or bank.ndim != 2 or queries.shape[1] != bank.shape[1]:
        raise ValueError(
            "queries and bank must be 2-D tensors with equal feature width"
        )
    if bank.shape[0] == 0:
        raise RuntimeError("the normal memory bank is empty")
    if queries.shape[0] == 0:
        return queries.new_empty(0)
    if k <= 0:
        raise ValueError("k must be positive")
    if query_chunk_size <= 0 or bank_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")

    k = min(k, bank.shape[0])
    normalized_bank = (
        bank if bank_is_normalized else F.normalize(bank, dim=1, eps=1e-12)
    )
    distances: list[Tensor] = []
    for q_start in range(0, queries.shape[0], query_chunk_size):
        query = F.normalize(
            queries[q_start : q_start + query_chunk_size], dim=1, eps=1e-12
        )
        best = query.new_full((query.shape[0], k), -torch.inf)
        for b_start in range(0, normalized_bank.shape[0], bank_chunk_size):
            similarities = (
                query @ normalized_bank[b_start : b_start + bank_chunk_size].T
            )
            chunk_k = min(k, similarities.shape[1])
            chunk_best = similarities.topk(
                chunk_k, dim=1, largest=True, sorted=False
            ).values
            candidates = torch.cat((best, chunk_best), dim=1)
            best = candidates.topk(k, dim=1, largest=True, sorted=False).values
        distances.append((1 - best).mean(dim=1))
    return torch.cat(distances, dim=0)


def sinusoidal_encoding(
    length: int, width: int, *, dtype: torch.dtype = torch.float32
) -> Tensor:
    """Create a standard sinusoidal position table of shape ``[1, length, width]``."""
    positions = torch.arange(length, dtype=dtype).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, width, 2, dtype=dtype) * (-math.log(10_000.0) / width)
    )
    table = torch.zeros(length, width, dtype=dtype)
    table[:, 0::2] = torch.sin(positions * frequencies)
    if width > 1:
        table[:, 1::2] = torch.cos(positions * frequencies[: table[:, 1::2].shape[1]])
    return table.unsqueeze(0)
