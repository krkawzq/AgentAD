"""Independent Lightning modules for CARLA's two published training stages."""

from __future__ import annotations

from collections.abc import Mapping

import lightning as L
import torch
from torch import Tensor

from .._utils import evaluation_mode
from .config import CARLAConfig
from .model import CARLA, inject_anomalies


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

    def configure_optimizers(self) -> dict[str, object]:
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
                    probability.argmax(1).cpu().bincount(
                        minlength=self.config.clusters
                    )
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
            self._val_scores.append(
                (1 - probabilities[0][:, normal]).detach().cpu()
            )
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
                    counts[head] += logits.argmax(1).cpu().bincount(
                        minlength=self.config.clusters
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
        if self._best_state is not None:
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
