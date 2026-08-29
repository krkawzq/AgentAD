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
        self.register_buffer(
            "reference_valid",
            torch.empty(0, 0),
            persistent=False,
        )

    def _patches(self, series: Tensor) -> tuple[Tensor, Tensor]:
        """Return z-normalized patches and per-dimension validity flags.

        A subsequence is invalid on a channel when its standard deviation is
        below ``epsilon`` or it contains non-finite values, matching the
        original ``skip_loc`` marking.
        """
        windows = sliding_windows(series, self.config.subsequence_length)
        mean = windows.mean(2, keepdim=True)
        scale = windows.var(2, keepdim=True, unbiased=False).sqrt()
        valid = scale >= self.config.epsilon
        normalized = (windows - mean) / scale.clamp_min(self.config.epsilon)
        return normalized, valid.squeeze(2)

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
        patches, valid = self._patches(normal_series)
        self.reference_patches = patches.flatten(0, 1).detach()
        self.reference_valid = valid.flatten(0, 1).detach()
        return self

    def _window_scores(
        self,
        query: Tensor,
        query_valid: Tensor,
        reference: Tensor,
        reference_valid: Tensor,
        *,
        self_join: bool,
    ) -> Tensor:
        dimensions = self._dimension_count(query.shape[-1])
        exclusion = int(self.config.exclusion_fraction * self.config.subsequence_length)
        reference_positions = torch.arange(reference.shape[0], device=query.device)
        chunks: list[Tensor] = []
        for start in range(0, query.shape[0], self.config.query_chunk_size):
            stop = min(start + self.config.query_chunk_size, query.shape[0])
            correlation = (
                torch.einsum("qlc,rlc->qrc", query[start:stop], reference)
                / self.config.subsequence_length
            )
            # Invalid pairs rank last in the ascending-dimension selection,
            # exactly like the original invalid marking before the sort.
            invalid = (
                ~query_valid[start:stop, None, :]
                | ~reference_valid[None, :, :]
                | ~torch.isfinite(correlation)
            )
            similarity = correlation.masked_fill(invalid, torch.inf)
            if self_join:
                query_positions = torch.arange(start, stop, device=query.device)
                exclusion_zone = (
                    reference_positions[None]
                    >= query_positions[:, None] - exclusion
                ) & (
                    reference_positions[None]
                    < query_positions[:, None] + exclusion
                )
                similarity.masked_fill_(exclusion_zone.unsqueeze(-1), -torch.inf)
            finite = torch.isfinite(similarity)
            selected_k = similarity.kthvalue(dimensions, dim=2).values
            # With fewer than ``dimensions`` valid channels the original
            # forward-fills from the previous sorted slot, which is the largest
            # valid correlation; with none it stays invalid.
            largest_valid = similarity.masked_fill(~finite, -torch.inf).amax(dim=2)
            valid_count = finite.sum(dim=2)
            value = torch.where(
                valid_count >= dimensions,
                selected_k,
                torch.where(
                    valid_count >= 1,
                    largest_valid,
                    selected_k.new_full((), -torch.inf),
                ),
            )

            selected = value.new_full((value.shape[0],), -torch.inf)
            candidates = value.clone()
            for _ in range(self.config.neighbors):
                best, index = candidates.max(1)
                selected = torch.where(torch.isfinite(best), best, selected)
                candidates.masked_fill_(
                    (reference_positions[None] >= index[:, None] - exclusion)
                    & (reference_positions[None] < index[:, None] + exclusion),
                    -torch.inf,
                )
            chunks.append(selected)
        anomaly = -torch.cat(chunks)
        # Only finite scores take part in the min-max scaling; the original
        # maps the remaining non-finite entries to zero afterwards.
        finite = torch.isfinite(anomaly)
        if not finite.any():
            return anomaly.new_zeros(anomaly.shape)
        valid_values = anomaly[finite]
        minimum = valid_values.min()
        maximum = valid_values.max()
        if maximum > minimum:
            scaled = (anomaly - minimum) / (maximum - minimum)
            return torch.where(finite, scaled, anomaly.new_zeros(()))
        return anomaly.new_zeros(anomaly.shape)

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
            query_patches, query_valid = self._patches(series)
            outputs: list[Tensor] = []
            if self.reference_patches.numel():
                if self.reference_patches.shape[-1] != series.shape[-1]:
                    raise ValueError(
                        "series feature count differs from the fitted reference"
                    )
                reference = self.reference_patches.to(query_patches)
                reference_valid = self.reference_valid.to(query_patches.device)
                for query, valid in zip(query_patches, query_valid):
                    outputs.append(
                        self._align(
                            self._window_scores(
                                query,
                                valid,
                                reference,
                                reference_valid,
                                self_join=False,
                            ),
                            series.shape[1],
                        )
                    )
            else:
                for query, valid in zip(query_patches, query_valid):
                    outputs.append(
                        self._align(
                            self._window_scores(
                                query,
                                valid,
                                query,
                                valid,
                                self_join=True,
                            ),
                            series.shape[1],
                        )
                    )
            return torch.stack(outputs)

    def forward(self, series: Tensor) -> Tensor:
        return self.score(series)
