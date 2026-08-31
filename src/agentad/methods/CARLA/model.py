"""Two-stage contrastive and neighbor-classification CARLA model.

Stage assembly is the caller's job: train ``CARLAPretextLightningModule``
first, then hand its weights to ``CARLAClassificationLightningModule``
(the original transfers every pretext weight except the projection head)
and mine each anchor's nearest/furthest neighbors in the pretext
projection space before classification, mirroring the hand-off between
the original pretext and classification entry points.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from collections.abc import Mapping
import lightning as L
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

from .._utils import evaluation_mode, overlap_average, sliding_windows, validate_series
from .config import CARLAConfig


def inject_anomalies(
    windows: Tensor,
    *,
    min_fraction: float = 0.1,
    max_fraction: float = 0.9,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Inject one of CARLA's five anomaly families into every input window."""
    windows = validate_series(windows, min_length=4, name="windows")
    if not 0 < min_fraction <= max_fraction < 1:
        raise ValueError("fractions must satisfy 0 < min <= max < 1")
    result = windows.clone()
    _, time, features = result.shape
    minimum = max(1, round(time * min_fraction))
    maximum = max(minimum + 1, round(time * max_fraction))

    def randint(low: int, high: int) -> int:
        return int(
            torch.randint(
                low,
                high,
                (),
                device=result.device,
                generator=generator,
            )
        )

    def inject_family(
        source: Tensor,
        *,
        start: int,
        length: int,
        compression: int | None = None,
        scale: float = 1.0,
        trend: bool = False,
        trend_end: bool = False,
        shapelet: bool = False,
    ) -> Tensor:
        output = source.clone()
        stop = time if trend_end else min(start + length, time)
        segment = output[start:stop]
        compression = randint(2, 5) if compression is None else compression
        segment = segment.repeat(compression, 1)[::compression][: stop - start]
        segment = segment * scale
        if trend:
            magnitude = 1 + 0.5 * torch.randn(
                (), device=result.device, dtype=result.dtype, generator=generator
            )
            direction = -1 if randint(0, 2) == 0 else 1
            segment = segment + direction * magnitude
        if shapelet:
            segment = output[start] + 0.1 * torch.rand(
                output[start].shape,
                device=result.device,
                dtype=result.dtype,
                generator=generator,
            )
        output[start:stop] = segment
        return output

    for batch_index in range(result.shape[0]):
        length = randint(minimum, min(maximum, time))
        start = randint(0, time - length)
        if features == 1:
            affected = [0]
        else:
            low = max(1, int(features / 10))
            high = max(low + 1, int(features / 2))
            affected = [randint(0, features) for _ in range(randint(low, high))]
        family = randint(0, 5)
        for feature in affected:
            source = windows[batch_index, :, feature : feature + 1]
            if family == 0:
                changed = inject_family(source, start=start, length=length)
            elif family == 1:
                changed = inject_family(
                    source,
                    start=start,
                    length=length,
                    compression=1,
                    trend=True,
                    trend_end=True,
                )
            elif family == 2:
                changed = inject_family(
                    source,
                    start=start,
                    length=3 if features == 1 else 2,
                    compression=1,
                    scale=8,
                )
            elif family == 3:
                changed = inject_family(
                    source,
                    start=start,
                    length=5 if features == 1 else 4,
                    compression=1,
                    scale=3,
                )
            else:
                changed = inject_family(
                    source,
                    start=start,
                    length=length,
                    compression=1,
                    shapelet=True,
                )
            result[batch_index, :, feature : feature + 1] = changed
    return result


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
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        output = self.layers(x)
        if self.residual is not None:
            output = output + self.residual(x)
        return output


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

    pretext_margin: Tensor
    normal_clusters: Tensor

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
        anchor_probabilities: list[Tensor] | None = None,
    ) -> CARLALoss:
        """Optimize neighbor consistency, furthest-neighbor separation, and entropy.

        ``anchor_probabilities``, when provided, is filled with the detached
        per-head anchor softmax outputs so callers can track cluster
        assignments without an extra forward pass.
        """
        if anchors.shape != nearest.shape or anchors.shape != furthest.shape:
            raise ValueError("anchors, nearest, and furthest must have equal shape")
        # The original classification stage runs three separate forward passes
        # so that batch normalization only sees one neighbor role at a time.
        anchor_outputs = self(anchors).logits
        nearest_outputs = self(nearest).logits
        furthest_outputs = self(furthest).logits
        consistency_terms: list[Tensor] = []
        inconsistency_terms: list[Tensor] = []
        entropy_terms: list[Tensor] = []
        for anchor_logits, nearest_logits, furthest_logits in zip(
            anchor_outputs, nearest_outputs, furthest_outputs
        ):
            anchor_probability = anchor_logits.softmax(1)
            nearest_probability = nearest_logits.softmax(1)
            furthest_probability = furthest_logits.softmax(1)
            if anchor_probabilities is not None:
                anchor_probabilities.append(anchor_probability.detach())
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
        consistency = torch.stack(consistency_terms).sum(0)
        inconsistency = torch.stack(inconsistency_terms).sum(0)
        entropy = torch.stack(entropy_terms).sum(0)
        total = (
            consistency
            + self.config.inconsistency_weight * inconsistency
            - self.config.entropy_weight * entropy
        )
        return CARLALoss(total, consistency, inconsistency, entropy)

    def _project_batches(self, windows: Tensor, batch_size: int) -> Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        windows = self._validate_windows(windows)
        return torch.cat([self.project(chunk) for chunk in windows.split(batch_size)])

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
        """Return per-window nearest and furthest indices using bounded memory.

        Mining happens in the pretext projection space, matching the original
        repository fill/mining pipeline.
        """
        if k <= 0 or query_chunk_size <= 0 or bank_chunk_size <= 0:
            raise ValueError("k and chunk sizes must be positive")
        with evaluation_mode(self):
            features = self._project_batches(windows, encoding_batch_size)
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
                chunk_k = min(k, bank_stop - bank_start)
                chunk_nearest_values, chunk_nearest_indices = similarity.nan_to_num(
                    nan=-torch.inf
                ).topk(chunk_k, dim=1)
                chunk_nearest_indices = chunk_nearest_indices + bank_start
                near_values = torch.cat((nearest_values, chunk_nearest_values), dim=1)
                near_indices = torch.cat(
                    (nearest_indices, chunk_nearest_indices), dim=1
                )
                nearest_values, selected = near_values.topk(k, dim=1)
                nearest_indices = near_indices.gather(1, selected)

                chunk_furthest_values, chunk_furthest_indices = similarity.nan_to_num(
                    nan=torch.inf
                ).topk(chunk_k, dim=1, largest=False)
                chunk_furthest_indices = chunk_furthest_indices + bank_start
                far_values = torch.cat((furthest_values, chunk_furthest_values), dim=1)
                far_indices = torch.cat(
                    (furthest_indices, chunk_furthest_indices), dim=1
                )
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
        normal_windows = self._validate_windows(normal_windows)
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


class CARLAPretextLightningModule(L.LightningModule):
    def __init__(self, config: CARLAConfig, model: CARLA | None = None) -> None:
        super().__init__()
        self.config = config
        self.model = model or CARLA(config)
        self._previous_loss: Tensor | None = None

    def forward(self, windows: Tensor) -> Tensor:
        return self.model.project(windows)

    def _batch(self, batch: object) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(batch, Mapping):
            raise TypeError(
                "CARLA pretext batches must map anchors/positives/injected_anomalies"
            )
        anchors = batch.get("anchors", batch.get("ts_org"))
        positives = batch.get("positives", batch.get("ts_w_augment"))
        anomalies = batch.get("injected_anomalies", batch.get("ts_ss_augment"))
        if not isinstance(anchors, Tensor):
            raise TypeError("CARLA pretext anchors must be a tensor")
        if positives is None:
            positives = self._positives(batch, anchors)
        if anomalies is None:
            anomalies = inject_anomalies(
                anchors,
                min_fraction=self.config.anomaly_min_fraction,
                max_fraction=self.config.anomaly_max_fraction,
            )
        if not isinstance(positives, Tensor) or not isinstance(anomalies, Tensor):
            raise TypeError("CARLA positives and injected anomalies must be tensors")
        return anchors, positives, anomalies

    def _positives(self, batch: Mapping, anchors: Tensor) -> Tensor:
        """Build pretext positives the way the original dataset does.

        With a window pool and per-anchor indices, each positive is one of the
        ten preceding windows of the same series; the first ten windows and
        any batch without a pool fall back to additive noise.
        """
        pool = batch.get("window_pool", batch.get("ts_org"))
        indices = batch.get("anchor_indices", batch.get("indices"))
        if not isinstance(pool, Tensor) or not isinstance(indices, Tensor):
            return anchors + self.config.noise_sigma * torch.randn_like(anchors)
        if pool.ndim != 3 or pool.shape[1:] != anchors.shape[1:]:
            raise TypeError("window_pool must be [windows, length, features]")
        if indices.shape != (anchors.shape[0],):
            raise TypeError("anchor_indices must have one entry per anchor")
        positions = indices.to(torch.long)
        offsets = torch.randint(1, 11, positions.shape, device=positions.device)
        neighbors = (positions - offsets).clamp_min(0)
        gathered = pool.to(anchors)[neighbors]
        early = (positions <= 10).unsqueeze(-1).unsqueeze(-1)
        noise = anchors + self.config.noise_sigma * torch.randn_like(anchors)
        return torch.where(early, noise, gathered)

    def _step(self, batch: object, stage: str) -> Tensor:
        anchors, positives, anomalies = self._batch(batch)
        previous = self._previous_loss if stage == "train" else None
        loss = self.model.compute_pretext_loss(
            anchors,
            positives,
            anomalies,
            previous_loss=previous,
        )
        if stage == "train":
            self._previous_loss = loss.detach()
        self.log(
            f"{stage}/loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=anchors.shape[0],
        )
        return loss

    def training_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def on_train_epoch_start(self) -> None:
        # The original passes its never-updated prev_loss=None to every epoch,
        # so the first batch of each epoch skips the margin adjustment.
        self._previous_loss = None

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.pretext_learning_rate
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.pretext_epochs,
            eta_min=self.config.pretext_learning_rate
            * self.config.pretext_cosine_decay_rate**3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class CARLAClassificationLightningModule(L.LightningModule):
    """Classification stage with the original best-F1 model selection.

    The original re-predicts the train windows every epoch (eval mode,
    end-of-epoch weights) to refresh the majority (normal) cluster, scores
    validation windows as ``1 - P(majority cluster)``, and reloads the
    checkpoint of the epoch with the best validation F1. This module
    reproduces the selection whenever validation batches carry a ``labels``
    entry with window-level anomaly labels; without labels the final weights
    are kept and ``calibrate_normal_clusters`` remains the scoring setup.
    """

    def __init__(self, config: CARLAConfig, model: CARLA | None = None) -> None:
        super().__init__()
        self.config = config
        self.model = model or CARLA(config)
        self._epoch_counts: Tensor | None = None
        self._majority_clusters: Tensor | None = None
        self._val_scores: list[Tensor] = []
        self._val_labels: list[Tensor] = []
        self._best_f1: float | None = None
        self._best_state: dict[str, Tensor] | None = None
        self._best_majority: Tensor | None = None

    def forward(self, windows: Tensor):
        return self.model(windows)

    @staticmethod
    def _batch(batch: object) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(batch, Mapping):
            raise TypeError(
                "CARLA classification batches must map anchors/nearest/furthest"
            )
        anchors = batch.get("anchors", batch.get("anchor"))
        nearest = batch.get("nearest", batch.get("NNeighbor"))
        furthest = batch.get("furthest", batch.get("FNeighbor"))
        if not all(isinstance(value, Tensor) for value in (anchors, nearest, furthest)):
            raise TypeError("CARLA classification batch values must be tensors")
        return anchors, nearest, furthest

    def _step(self, batch: object, stage: str) -> Tensor:
        anchors, nearest, furthest = self._batch(batch)
        probabilities: list[Tensor] = []
        losses = self.model.compute_classification_loss(
            anchors, nearest, furthest, anchor_probabilities=probabilities
        )
        if stage == "train":
            if self._epoch_counts is None:
                self._epoch_counts = torch.zeros(
                    self.config.cluster_heads,
                    self.config.clusters,
                    dtype=torch.long,
                )
            for head, probability in enumerate(probabilities):
                self._epoch_counts[head] += (
                    probability.argmax(1).cpu().bincount(minlength=self.config.clusters)
                )
        elif (
            self._majority_clusters is not None
            and isinstance(batch, Mapping)
            and isinstance(batch.get("labels"), Tensor)
        ):
            # Sanity-check validation runs before any training epoch, so
            # _majority_clusters is None there and this branch stays inactive.
            labels = batch["labels"]
            if labels.shape != (anchors.shape[0],):
                raise ValueError(
                    "labels must provide one window-level label per anchor"
                )
            # The original scores validation windows as 1 - P(majority).
            normal = int(self._majority_clusters[0])
            self._val_scores.append((1 - probabilities[0][:, normal]).detach().cpu())
            self._val_labels.append((labels == 1).cpu())
        for name in ("total", "consistency", "inconsistency", "entropy"):
            metric = "loss" if name == "total" else name
            self.log(
                f"{stage}/{metric}",
                getattr(losses, name),
                on_step=stage == "train" and name == "total",
                on_epoch=True,
                prog_bar=name == "total",
                sync_dist=True,
                batch_size=anchors.shape[0],
            )
        return losses.total

    def training_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: object, batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def on_validation_epoch_start(self) -> None:
        self._val_scores = []
        self._val_labels = []
        # Lightning runs the epoch's validation before on_train_epoch_end, so
        # the majority refresh belongs here: the weights are already the
        # end-of-epoch ones, exactly when the original re-predicts the train
        # windows. The sanity check runs before any training epoch and stays
        # inactive.
        if self._trainer is not None and self._trainer.sanity_checking:
            return
        counts = self._repredict_cluster_counts()
        if counts is None:
            counts = self._epoch_counts
        if counts is not None:
            self._majority_clusters = counts.argmax(1)
        self._epoch_counts = None

    def _repredict_cluster_counts(self) -> Tensor | None:
        loader = None
        try:
            loader = self.trainer.train_dataloader
            if isinstance(loader, (list, tuple)):
                loader = loader[0] if loader else None
        except Exception:  # pragma: no cover - trainer internals vary by setup
            loader = None
        if loader is None:
            return None
        counts = torch.zeros(
            self.config.cluster_heads,
            self.config.clusters,
            dtype=torch.long,
        )
        found = False
        with evaluation_mode(self.model), torch.inference_mode():
            for batch in loader:
                try:
                    anchors, _, _ = self._batch(batch)
                except TypeError:
                    continue
                found = True
                for head, logits in enumerate(self.model(anchors).logits):
                    counts[head] += (
                        logits.argmax(1).cpu().bincount(minlength=self.config.clusters)
                    )
        return counts if found else None

    def on_validation_epoch_end(self) -> None:
        if not self._val_scores or self._majority_clusters is None:
            return
        scores = torch.cat(self._val_scores)
        labels = torch.cat(self._val_labels)
        sorted_scores, order = scores.sort(descending=True)
        sorted_labels = labels[order].to(scores.dtype)
        true_positives = sorted_labels.cumsum(0)
        positions = torch.arange(1, scores.numel() + 1)
        # A threshold only changes where the sorted score strictly decreases,
        # which matches scikit-learn's unique-score thresholds.
        thresholds = torch.ones_like(sorted_scores, dtype=torch.bool)
        thresholds[:-1] = sorted_scores[1:] != sorted_scores[:-1]
        precision = true_positives[thresholds] / positions[thresholds]
        positives = sorted_labels.sum()
        recall = true_positives[thresholds] / positives.clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall)
        best = float(torch.nan_to_num(f1, nan=0.0).max())
        if self._best_f1 is None or best > self._best_f1:
            self._best_f1 = best
            self._best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in self.state_dict().items()
            }
            self._best_majority = self._majority_clusters.clone()

    def on_train_end(self) -> None:
        # The original saves its normal_label from a misspelled variable and
        # therefore always reloads the initial value (zero); wiring the best
        # epoch's majority cluster implements its evident intent.
        if self._best_state is not None and self._best_majority is not None:
            self.load_state_dict(self._best_state)
            self.model.normal_clusters.copy_(
                self._best_majority.to(self.model.normal_clusters.device)
            )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config.classification_learning_rate,
            weight_decay=self.config.classification_weight_decay,
        )
