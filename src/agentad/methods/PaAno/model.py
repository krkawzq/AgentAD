"""Patch-based representation learning and normal-memory scoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .._utils import (
    evaluation_mode,
    overlap_average,
    sliding_windows,
    topk_cosine_distance,
    validate_series,
)
from .config import PaAnoConfig


class _RevIN(nn.Module):
    def __init__(
        self, features: int, *, epsilon: float, min_scale: float, affine: bool
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.min_scale = min_scale
        if affine:
            self.weight = nn.Parameter(torch.ones(1, features, 1))
            self.bias = nn.Parameter(torch.zeros(1, features, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdim=True)
        scale = x.var(-1, keepdim=True, unbiased=False).add(self.epsilon).sqrt()
        normalized = (x - mean) / scale.clamp_min(self.min_scale)
        if self.weight is not None:
            normalized = normalized * self.weight + self.bias
        return normalized


class PatchEncoder(nn.Module):
    """The lightweight PaAno 1-D convolutional patch encoder."""

    def __init__(self, config: PaAnoConfig = PaAnoConfig()) -> None:
        super().__init__()
        self.config = config
        self.revin = (
            _RevIN(
                config.input_features,
                epsilon=config.revin_epsilon,
                min_scale=config.revin_min_scale,
                affine=config.revin_affine,
            )
            if config.use_revin
            else nn.Identity()
        )
        blocks: list[nn.Module] = []
        input_width = config.input_features
        for output_width, kernel in zip(config.convolution_widths, config.kernel_sizes):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        input_width,
                        output_width,
                        kernel,
                        padding=kernel // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(output_width),
                    nn.ReLU(inplace=True),
                )
            )
            input_width = output_width
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        embedding_dim = config.convolution_widths[-1]
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, config.projection_dim),
            nn.ReLU(),
            nn.Linear(config.projection_dim, config.projection_dim),
        )
        self.pretext_head = nn.Linear(2 * embedding_dim, 1)

    @property
    def embedding_dim(self) -> int:
        return self.config.convolution_widths[-1]

    def forward(self, patches: Tensor) -> Tensor:
        if patches.ndim != 3:
            raise ValueError("patches must have shape [patch, feature, time]")
        if patches.shape[1:] != (self.config.input_features, self.config.patch_length):
            raise ValueError(
                "patches must have shape "
                f"[patch, {self.config.input_features}, {self.config.patch_length}]"
            )
        return self.pool(self.blocks(self.revin(patches))).flatten(1)

    def project(self, embeddings: Tensor) -> Tensor:
        return self.projection_head(embeddings)


@dataclass(frozen=True, slots=True)
class PaAnoLoss:
    total: Tensor
    triplet: Tensor
    pretext: Tensor
    pretext_weight: float


class PaAno(nn.Module):
    """PaAno encoder, self-supervised losses, and bounded-memory scoring."""

    def __init__(self, config: PaAnoConfig = PaAnoConfig()) -> None:
        super().__init__()
        self.config = config
        self.encoder = PatchEncoder(config)
        self.register_buffer(
            "normal_memory",
            torch.empty(0, self.encoder.embedding_dim),
            persistent=False,
        )

    def forward(self, patches: Tensor) -> Tensor:
        return self.encoder(patches)

    @staticmethod
    def _random_neighbor_indices(
        indices: Tensor,
        total: int,
        radius: int,
        generator: torch.Generator | None,
    ) -> Tensor:
        offsets = torch.cat(
            (
                torch.arange(-radius, 0, device=indices.device),
                torch.arange(1, radius + 1, device=indices.device),
            )
        )
        candidates = indices[:, None] + offsets[None, :]
        valid = (candidates >= 0) & (candidates < total)
        draws = torch.rand(
            candidates.shape,
            device=indices.device,
            generator=generator,
        ).masked_fill(~valid, -1)
        selected = candidates.gather(1, draws.argmax(1, keepdim=True)).squeeze(1)
        return torch.where(valid.any(1), selected, indices)

    def compute_loss(
        self,
        all_patches: Tensor,
        anchor_indices: Tensor,
        *,
        iteration: int,
        total_iterations: int,
        pretext_step: int | None = None,
        generator: torch.Generator | None = None,
    ) -> PaAnoLoss:
        """Compute PaAno's local triplet and early temporal-pretext losses."""
        if all_patches.ndim != 3 or all_patches.shape[0] < 2:
            raise ValueError(
                "all_patches must contain at least two [feature, time] patches"
            )
        if anchor_indices.ndim != 1 or anchor_indices.numel() < 2:
            raise ValueError(
                "anchor_indices must be a 1-D tensor with at least two entries"
            )
        if anchor_indices.dtype != torch.long:
            raise TypeError("anchor_indices must use torch.long")
        anchor_indices = anchor_indices.to(all_patches.device)
        if iteration < 0 or total_iterations <= 0:
            raise ValueError(
                "iteration must be non-negative and total_iterations positive"
            )
        if anchor_indices.min() < 0 or anchor_indices.max() >= all_patches.shape[0]:
            raise IndexError("anchor index is outside all_patches")

        pretext_step = (
            self.config.patch_length if pretext_step is None else pretext_step
        )
        if pretext_step <= 0:
            raise ValueError("pretext_step must be positive")
        anchors = all_patches.index_select(0, anchor_indices)
        positive_indices = self._random_neighbor_indices(
            anchor_indices,
            all_patches.shape[0],
            self.config.positive_radius,
            generator,
        )
        positives = all_patches.index_select(0, positive_indices)
        anchor_embeddings, positive_embeddings = self.encoder(
            torch.cat((anchors, positives), dim=0)
        ).chunk(2)
        anchor_projection = F.normalize(self.encoder.project(anchor_embeddings), dim=1)
        positive_projection = F.normalize(
            self.encoder.project(positive_embeddings), dim=1
        )

        similarities = anchor_projection @ positive_projection.T
        positive_distance = 1 - similarities.diagonal()
        negative_similarities = similarities.masked_fill(
            torch.eye(
                similarities.shape[0], device=similarities.device, dtype=torch.bool
            ),
            torch.inf,
        )
        selected_negative_distance = (1 - negative_similarities).max(1).values
        triplet = (
            F.relu(
                positive_distance
                - selected_negative_distance
                + self.config.triplet_margin
            ).mean()
            * self.config.triplet_scale
        )

        pretext_horizon = max(1, round(total_iterations * self.config.pretext_fraction))
        if iteration < pretext_horizon and self.config.pretext_weight > 0:
            current_weight = self.config.pretext_weight * (
                1 - iteration / pretext_horizon
            )
            previous_indices = anchor_indices - pretext_step
            valid = previous_indices >= 0
            if valid.any():
                previous = self.encoder(
                    all_patches.index_select(0, previous_indices[valid])
                )
                adjacent = torch.cat((anchor_embeddings[valid], previous), dim=1)
                adjacent_logits = self.encoder.pretext_head(adjacent).squeeze(1)
            else:
                adjacent_logits = anchor_embeddings.new_empty(0)

            count = anchor_embeddings.shape[0]
            anchors_repeated = anchor_embeddings.repeat_interleave(
                self.config.random_negatives, dim=0
            )
            offsets = torch.randint(
                1,
                count,
                (count * self.config.random_negatives,),
                device=anchor_embeddings.device,
                generator=generator,
            )
            base = torch.arange(
                count, device=anchor_embeddings.device
            ).repeat_interleave(self.config.random_negatives)
            unrelated = anchor_embeddings[(base + offsets) % count]
            unrelated_logits = self.encoder.pretext_head(
                torch.cat((anchors_repeated, unrelated), dim=1)
            ).squeeze(1)
            positive_loss = (
                F.binary_cross_entropy_with_logits(
                    adjacent_logits, torch.ones_like(adjacent_logits)
                )
                if adjacent_logits.numel()
                else triplet.new_zeros(())
            )
            negative_loss = F.binary_cross_entropy_with_logits(
                unrelated_logits, torch.zeros_like(unrelated_logits)
            )
            pretext = positive_loss + negative_loss
        else:
            current_weight = 0.0
            pretext = triplet.new_zeros(())
        return PaAnoLoss(
            triplet + current_weight * pretext, triplet, pretext, current_weight
        )

    def _select_coreset(self, embeddings: Tensor) -> Tensor:
        """Keep the sample closest to each MiniBatchKMeans cluster center.

        This replicates the original ``create_memory_bank`` coreset: at least
        ``min(500, N - 1)`` clusters (or a fraction of the patch count when
        larger), then one representative embedding per cluster.
        """
        fraction = self.config.memory_coreset_fraction
        count = embeddings.shape[0]
        if fraction is None or count < 2:
            return embeddings
        target = int(round(fraction * count))
        minimum = min(500, max(1, count - 1))
        clusters = max(minimum, min(target, count - 1))
        if clusters >= count:
            return embeddings
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise ImportError(
                "PaAno coreset selection requires scikit-learn. Run `uv sync`."
            ) from exc
        normalized = F.normalize(embeddings, dim=1, eps=1e-12)
        clustering = MiniBatchKMeans(
            n_clusters=clusters,
            init="k-means++",
            random_state=42,
            batch_size=max(8192, clusters),
            max_iter=50,
            n_init=1,
            reassignment_ratio=0.01,
        )
        clustering.fit(normalized.detach().cpu().numpy())
        centers = torch.as_tensor(
            clustering.cluster_centers_,
            dtype=normalized.dtype,
            device=normalized.device,
        )
        # One representative per cluster: the sample closest to each center.
        core_indices = torch.cdist(normalized, centers).argmin(0)
        return embeddings.index_select(0, core_indices)

    @torch.inference_mode()
    def build_memory_bank(self, patches: Tensor, *, batch_size: int = 2048) -> Tensor:
        """Encode normal patches and retain a coreset as the scoring bank."""
        if patches.ndim != 3 or patches.shape[0] == 0:
            raise ValueError(
                "patches must be a non-empty [patch, feature, time] tensor"
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        with evaluation_mode(self):
            embeddings = [self.encoder(chunk) for chunk in patches.split(batch_size)]
            selected = self._select_coreset(torch.cat(embeddings))
            memory = F.normalize(selected, dim=1, eps=1e-12).detach()
            self.normal_memory = memory
            return memory

    @torch.inference_mode()
    def fit_memory(self, normal_series: Tensor, *, batch_size: int = 2048) -> Tensor:
        normal_series = validate_series(
            normal_series,
            min_length=self.config.patch_length,
            name="normal_series",
        )
        if normal_series.shape[2] != self.config.input_features:
            raise ValueError(
                "normal_series feature count does not match the PaAno config"
            )
        windows = sliding_windows(normal_series, self.config.patch_length)
        patches = windows.flatten(0, 1).transpose(1, 2).contiguous()
        return self.build_memory_bank(patches, batch_size=batch_size)

    @torch.inference_mode()
    def score(
        self,
        x: Tensor,
        *,
        query_chunk_size: int = 2048,
        bank_chunk_size: int = 16384,
        embedding_batch_size: int = 2048,
    ) -> Tensor:
        """Return overlap-averaged normal-memory distance as ``[B, T]``."""
        x = validate_series(x, min_length=self.config.patch_length)
        if x.shape[2] != self.config.input_features:
            raise ValueError("x feature count does not match the PaAno config")
        if self.normal_memory.shape[0] == 0:
            raise RuntimeError(
                "call fit_memory() or build_memory_bank() before score()"
            )
        if embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be positive")
        with evaluation_mode(self):
            windows = sliding_windows(x, self.config.patch_length)
            patches = windows.flatten(0, 1).transpose(1, 2).contiguous()
            score_chunks = [
                topk_cosine_distance(
                    self.encoder(chunk),
                    self.normal_memory,
                    k=self.config.top_k,
                    query_chunk_size=query_chunk_size,
                    bank_chunk_size=bank_chunk_size,
                    bank_is_normalized=True,
                )
                for chunk in patches.split(embedding_batch_size)
            ]
            patch_scores = torch.cat(score_chunks).reshape(x.shape[0], windows.shape[1])
            return overlap_average(
                patch_scores,
                patch_length=self.config.patch_length,
                output_length=x.shape[1],
            )
