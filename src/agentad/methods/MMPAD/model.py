"""Chunked multidimensional matrix profile for anomaly detection."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .._utils import evaluation_mode, overlap_average, sliding_windows, validate_series
from .config import MMPADConfig


class MMPAD(nn.Module):
    """MMPAD using exact local correlations and bounded query memory."""

    def __init__(self, config: MMPADConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer(
            "reference_patches",
            torch.empty(0, 0, 0),
            persistent=False,
        )

    def _patches(self, series: Tensor) -> Tensor:
        windows = sliding_windows(series, self.config.subsequence_length)
        mean = windows.mean(2, keepdim=True)
        scale = windows.var(2, keepdim=True, unbiased=False).sqrt()
        return (windows - mean) / scale.clamp_min(self.config.epsilon)

    def _dimension_count(self, features: int) -> int:
        value = self.config.dimensions
        if value is None:
            return features
        if float(value) < 1:
            return max(1, min(features, math.ceil(float(value) * features)))
        return max(1, min(features, int(value)))

    @torch.inference_mode()
    def fit(self, normal_series: Tensor) -> "MMPAD":
        normal_series = validate_series(
            normal_series,
            min_length=self.config.subsequence_length,
            name="normal_series",
        )
        self.reference_patches = self._patches(normal_series).flatten(0, 1).detach()
        return self

    def _window_scores(
        self,
        query: Tensor,
        reference: Tensor,
        *,
        self_join: bool,
    ) -> Tensor:
        dimensions = self._dimension_count(query.shape[-1])
        exclusion = max(1, int(self.config.exclusion_fraction * self.config.subsequence_length))
        reference_positions = torch.arange(reference.shape[0], device=query.device)
        chunks: list[Tensor] = []
        for start in range(0, query.shape[0], self.config.query_chunk_size):
            stop = min(start + self.config.query_chunk_size, query.shape[0])
            correlation = torch.einsum(
                "qlc,rlc->qrc", query[start:stop], reference
            ) / self.config.subsequence_length
            similarity = correlation.kthvalue(dimensions, dim=2).values
            if self_join:
                query_positions = torch.arange(start, stop, device=query.device)
                similarity.masked_fill_(
                    (query_positions[:, None] - reference_positions[None]).abs()
                    < exclusion,
                    -torch.inf,
                )

            selected = similarity.new_full((similarity.shape[0],), -torch.inf)
            candidates = similarity.clone()
            for _ in range(self.config.neighbors):
                value, index = candidates.max(1)
                selected = torch.where(torch.isfinite(value), value, selected)
                candidates.masked_fill_(
                    (reference_positions[None] - index[:, None]).abs() < exclusion,
                    -torch.inf,
                )
            chunks.append(selected)
        similarity = torch.cat(chunks)
        anomaly = -torch.nan_to_num(similarity, nan=0.0, neginf=0.0, posinf=0.0)
        minimum = anomaly.min()
        maximum = anomaly.max()
        return (anomaly - minimum) / (maximum - minimum).clamp_min(self.config.epsilon)

    def _align(self, window_scores: Tensor, output_length: int) -> Tensor:
        length = self.config.subsequence_length
        if self.config.alignment == "mean":
            return overlap_average(
                window_scores[None],
                patch_length=length,
                output_length=output_length,
            )[0]
        if self.config.alignment == "start":
            return F.pad(window_scores, (0, length - 1))
        left = (length - 1) // 2
        return F.pad(window_scores, (left, length - 1 - left))

    @torch.inference_mode()
    def score(self, series: Tensor) -> Tensor:
        series = validate_series(
            series,
            min_length=self.config.subsequence_length,
            name="series",
        )
        with evaluation_mode(self):
            query_batches = self._patches(series)
            outputs: list[Tensor] = []
            if self.reference_patches.numel():
                if self.reference_patches.shape[-1] != series.shape[-1]:
                    raise ValueError("series feature count differs from the fitted reference")
                reference = self.reference_patches
                for query in query_batches:
                    outputs.append(
                        self._align(
                            self._window_scores(query, reference, self_join=False),
                            series.shape[1],
                        )
                    )
            else:
                for query in query_batches:
                    outputs.append(
                        self._align(
                            self._window_scores(query, query, self_join=True),
                            series.shape[1],
                        )
                    )
            return torch.stack(outputs)

    def forward(self, series: Tensor) -> Tensor:
        return self.score(series)
