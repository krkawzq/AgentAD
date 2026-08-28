"""Two-stage contrastive and neighbor-classification CARLA model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .._utils import evaluation_mode, overlap_average, sliding_windows, validate_series
from .config import CARLAConfig


class _SameConv1d(nn.Conv1d):
    def forward(self, x: Tensor) -> Tensor:
        kernel = self.weight.shape[-1]
        total = self.dilation[0] * (kernel - 1)
        left = total // 2
        return F.conv1d(
            F.pad(x, (left, total - left)),
            self.weight,
            self.bias,
            self.stride,
            0,
            self.dilation,
            self.groups,
        )


class _ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_channels
        for kernel in (8, 5, 3):
            layers.extend(
                (
                    _SameConv1d(width, output_channels, kernel),
                    nn.BatchNorm1d(output_channels),
                    nn.ReLU(),
                )
            )
            width = output_channels
        self.layers = nn.Sequential(*layers)
        self.residual = (
            nn.Sequential(
                _SameConv1d(input_channels, output_channels, 1),
                nn.BatchNorm1d(output_channels),
            )
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.layers(x) + self.residual(x))


class _RepresentationNet(nn.Module):
    def __init__(self, config: CARLAConfig) -> None:
        super().__init__()
        middle = config.mid_channels
        self.layers = nn.Sequential(
            _ResidualBlock(config.input_features, middle),
            _ResidualBlock(middle, 2 * middle),
            _ResidualBlock(2 * middle, 2 * middle),
        )
        self.output_dim = 2 * middle

    def forward(self, windows: Tensor) -> Tensor:
        return self.layers(windows.transpose(1, 2)).mean(dim=2)


@dataclass(frozen=True, slots=True)
class CARLAOutput:
    features: Tensor
    logits: tuple[Tensor, ...]


@dataclass(frozen=True, slots=True)
class CARLALoss:
    total: Tensor
    consistency: Tensor
    inconsistency: Tensor
    entropy: Tensor


class CARLA(nn.Module):
    """CARLA pretext projection and self-supervised clustering stages."""

    def __init__(self, config: CARLAConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = _RepresentationNet(config)
        self.projection_head = nn.Sequential(
            nn.Linear(self.backbone.output_dim, self.backbone.output_dim),
            nn.ReLU(),
            nn.Linear(self.backbone.output_dim, config.projection_dim),
        )
        self.cluster_heads = nn.ModuleList(
            nn.Linear(self.backbone.output_dim, config.clusters)
            for _ in range(config.cluster_heads)
        )
        self.register_buffer("pretext_margin", torch.tensor(config.pretext_margin))
        self.register_buffer(
            "normal_clusters",
            torch.full((config.cluster_heads,), -1, dtype=torch.long),
            persistent=False,
        )

    def _validate_windows(self, windows: Tensor) -> Tensor:
        windows = validate_series(windows, min_length=2, name="windows")
        if windows.shape[2] != self.config.input_features:
            raise ValueError("window feature count does not match the CARLA config")
        return windows

    def encode(self, windows: Tensor) -> Tensor:
        return self.backbone(self._validate_windows(windows))

    def project(self, windows: Tensor) -> Tensor:
        return F.normalize(self.projection_head(self.encode(windows)), dim=1)

    def forward(self, windows: Tensor) -> CARLAOutput:
        features = self.encode(windows)
        return CARLAOutput(
            features, tuple(head(features) for head in self.cluster_heads)
        )

    def compute_pretext_loss(
        self,
        anchors: Tensor,
        positives: Tensor,
        injected_anomalies: Tensor,
        *,
        previous_loss: float | Tensor | None = None,
    ) -> Tensor:
        """Contrast nearby windows against the hardest injected anomaly."""
        if (
            anchors.shape != positives.shape
            or anchors.shape != injected_anomalies.shape
        ):
            raise ValueError(
                "anchors, positives, and injected_anomalies must have equal shape"
            )
        if previous_loss is not None:
            value = (
                float(previous_loss.detach())
                if isinstance(previous_loss, Tensor)
                else previous_loss
            )
            with torch.no_grad():
                self.pretext_margin.sub_(self.config.margin_adjustment * value).clamp_(
                    min=0.01
                )
        projected = self.project(torch.cat((anchors, positives, injected_anomalies)))
        anchor, positive, negative = projected.chunk(3)
        positive_distance = (anchor - positive).square().sum(
            1
        ) / self.config.temperature
        negative_distance = (anchor[:, None, :] - negative[None, :, :]).square().sum(
            2
        ) / self.config.temperature
        hardest_negative = negative_distance.min(1).values
        return F.relu(self.pretext_margin + positive_distance - hardest_negative).mean()

    @staticmethod
    def _entropy(probabilities: Tensor) -> Tensor:
        mean_probability = probabilities.mean(0).clamp_min(1e-8)
        return -(mean_probability * mean_probability.log()).sum()

    def compute_classification_loss(
        self,
        anchors: Tensor,
        nearest: Tensor,
        furthest: Tensor,
    ) -> CARLALoss:
        """Optimize neighbor consistency, furthest-neighbor separation, and entropy."""
        if anchors.shape != nearest.shape or anchors.shape != furthest.shape:
            raise ValueError("anchors, nearest, and furthest must have equal shape")
        outputs = self(torch.cat((anchors, nearest, furthest))).logits
        consistency_terms: list[Tensor] = []
        inconsistency_terms: list[Tensor] = []
        entropy_terms: list[Tensor] = []
        batch = anchors.shape[0]
        for logits in outputs:
            anchor_logits, nearest_logits, furthest_logits = logits.split(batch)
            anchor_probability = anchor_logits.softmax(1)
            nearest_probability = nearest_logits.softmax(1)
            furthest_probability = furthest_logits.softmax(1)
            nearest_similarity = (anchor_probability * nearest_probability).sum(1)
            furthest_similarity = (anchor_probability * furthest_probability).sum(1)
            consistency_terms.append(
                F.binary_cross_entropy(
                    nearest_similarity, torch.ones_like(nearest_similarity)
                )
            )
            inconsistency_terms.append(
                F.binary_cross_entropy(
                    furthest_similarity, torch.zeros_like(furthest_similarity)
                )
            )
            entropy_terms.append(self._entropy(anchor_probability))
        consistency = torch.stack(consistency_terms).mean()
        inconsistency = torch.stack(inconsistency_terms).mean()
        entropy = torch.stack(entropy_terms).mean()
        total = (
            consistency
            + self.config.inconsistency_weight * inconsistency
            - self.config.entropy_weight * entropy
        )
        return CARLALoss(total, consistency, inconsistency, entropy)

    def _encode_batches(self, windows: Tensor, batch_size: int) -> Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return torch.cat([self.encode(chunk) for chunk in windows.split(batch_size)])

    @torch.inference_mode()
    def mine_neighbors(
        self,
        windows: Tensor,
        *,
        k: int = 5,
        query_chunk_size: int = 2048,
        bank_chunk_size: int = 16384,
        encoding_batch_size: int = 2048,
    ) -> tuple[Tensor, Tensor]:
        """Return per-window nearest and furthest indices using bounded memory."""
        if k <= 0 or query_chunk_size <= 0 or bank_chunk_size <= 0:
            raise ValueError("k and chunk sizes must be positive")
        with evaluation_mode(self):
            features = F.normalize(
                self._encode_batches(windows, encoding_batch_size), dim=1
            )
        if features.shape[0] <= k:
            raise ValueError("neighbor mining requires more than k windows")
        nearest: list[Tensor] = []
        furthest: list[Tensor] = []
        for query_start in range(0, features.shape[0], query_chunk_size):
            query_stop = min(query_start + query_chunk_size, features.shape[0])
            query = features[query_start:query_stop]
            nearest_values = query.new_full((query.shape[0], k), -torch.inf)
            furthest_values = query.new_full((query.shape[0], k), torch.inf)
            nearest_indices = torch.zeros(
                query.shape[0], k, device=query.device, dtype=torch.long
            )
            furthest_indices = nearest_indices.clone()
            for bank_start in range(0, features.shape[0], bank_chunk_size):
                bank_stop = min(bank_start + bank_chunk_size, features.shape[0])
                similarity = query @ features[bank_start:bank_stop].T
                overlap_start = max(query_start, bank_start)
                overlap_stop = min(query_stop, bank_stop)
                if overlap_start < overlap_stop:
                    diagonal = torch.arange(
                        overlap_start, overlap_stop, device=features.device
                    )
                    similarity[
                        diagonal - query_start,
                        diagonal - bank_start,
                    ] = torch.nan
                indices = torch.arange(
                    bank_start, bank_stop, device=features.device
                ).expand(query.shape[0], -1)
                near_values = torch.cat(
                    (nearest_values, similarity.nan_to_num(nan=-torch.inf)), dim=1
                )
                near_indices = torch.cat((nearest_indices, indices), dim=1)
                nearest_values, selected = near_values.topk(k, dim=1)
                nearest_indices = near_indices.gather(1, selected)

                far_values = torch.cat(
                    (furthest_values, similarity.nan_to_num(nan=torch.inf)), dim=1
                )
                far_indices = torch.cat((furthest_indices, indices), dim=1)
                furthest_values, selected = far_values.topk(k, dim=1, largest=False)
                furthest_indices = far_indices.gather(1, selected)
            nearest.append(nearest_indices)
            furthest.append(furthest_indices)
        return torch.cat(nearest), torch.cat(furthest)

    @torch.inference_mode()
    def calibrate_normal_clusters(
        self, normal_windows: Tensor, *, batch_size: int = 2048
    ) -> Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        with evaluation_mode(self):
            counts = torch.zeros(
                self.config.cluster_heads,
                self.config.clusters,
                device=normal_windows.device,
                dtype=torch.long,
            )
            for windows in normal_windows.split(batch_size):
                for index, logits in enumerate(self(windows).logits):
                    counts[index] += logits.argmax(1).bincount(
                        minlength=self.config.clusters
                    )
            clusters = counts.argmax(1)
            self.normal_clusters.copy_(clusters)
            return clusters

    def _window_score(self, windows: Tensor, *, head: int) -> Tensor:
        probability = self(windows).logits[head].softmax(1)
        return 1 - probability[:, int(self.normal_clusters[head])]

    @torch.inference_mode()
    def window_score(self, windows: Tensor, *, head: int = 0) -> Tensor:
        """Return ``1 - P(normal cluster)`` for every input window."""
        if not 0 <= head < self.config.cluster_heads:
            raise IndexError("head is outside the configured cluster heads")
        normal = int(self.normal_clusters[head])
        if normal < 0:
            raise RuntimeError("call calibrate_normal_clusters() before score()")
        with evaluation_mode(self):
            return self._window_score(windows, head=head)

    @torch.inference_mode()
    def score(
        self,
        x: Tensor,
        *,
        head: int = 0,
        stride: int = 1,
        batch_size: int = 2048,
    ) -> Tensor:
        """Score overlapping windows and distribute scores back to time points."""
        x = validate_series(x, min_length=self.config.window_length)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 <= head < self.config.cluster_heads:
            raise IndexError("head is outside the configured cluster heads")
        if int(self.normal_clusters[head]) < 0:
            raise RuntimeError("call calibrate_normal_clusters() before score()")
        with evaluation_mode(self):
            windows = sliding_windows(x, self.config.window_length, stride)
            flattened = windows.flatten(0, 1)
            scores = torch.cat(
                [
                    self._window_score(chunk, head=head)
                    for chunk in flattened.split(batch_size)
                ]
            ).reshape(x.shape[0], -1)
            return overlap_average(
                scores,
                patch_length=self.config.window_length,
                output_length=x.shape[1],
                stride=stride,
            )
