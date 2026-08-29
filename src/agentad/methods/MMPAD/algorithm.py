"""Chunked multidimensional matrix profile for anomaly detection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .._utils import overlap_average, sliding_windows, validate_series
from .config import MMPADConfig


def _infer_subsequence_length(series: Tensor) -> int:
    """Infer the window length from the first channel, as the original does.

    The original z-scores the first channel of the first series, picks the
    strongest autocorrelation local maximum past lag 3, falls back to 125 when
    no peak exists or the peak position leaves [3, 300], then clamps the
    result to at least 64 and at most the series length.
    """
    length = series.shape[1]
    values = series[0, :, 0].to(torch.float64)[:20_000]
    centered = values - values.mean()
    energy = centered.square().sum()
    # The original calls acf with nlags=400, which needs at least 401 points;
    # anything shorter (or a flat channel) takes the fallback period.
    if centered.numel() < 401 or not bool(energy > 0):
        return max(1, min(125, length))
    size = 1
    while size < 2 * centered.numel():
        size *= 2
    spectrum = torch.fft.rfft(centered, n=size)
    autocorrelation = torch.fft.irfft(spectrum * spectrum.conj(), n=size)
    lags = min(400, centered.numel() - 1)
    window = autocorrelation[3 : lags + 1] / energy
    peaks = (
        (window[1:-1] > window[:-2]) & (window[1:-1] > window[2:])
    ).nonzero().squeeze(1) + 1
    if peaks.numel() == 0:
        return max(1, min(125, length))
    position = int(peaks[window[peaks].argmax()])
    if position < 3 or position > 300:
        return max(1, min(125, length))
    return max(1, min(max(64, position + 3), length))


def _resolved_length(config: MMPADConfig, series: Tensor) -> int:
    if config.subsequence_length is not None:
        return config.subsequence_length
    return _infer_subsequence_length(series)


def _patches(
    config: MMPADConfig, series: Tensor, length: int
) -> tuple[Tensor, Tensor]:
    """Return z-normalized patches and per-dimension validity flags.

    A subsequence is invalid on a channel when its standard deviation is
    below ``epsilon`` or it contains non-finite values, matching the
    ``flat_mode='invalid'`` marking of the original mstomp code.
    """
    windows = sliding_windows(series, length)
    mean = windows.mean(2, keepdim=True)
    scale = windows.var(2, keepdim=True, unbiased=False).sqrt()
    valid = scale >= config.epsilon
    normalized = (windows - mean) / scale.clamp_min(config.epsilon)
    return normalized, valid.squeeze(2)


def _dimension_count(config: MMPADConfig, features: int) -> int:
    value = config.dimensions
    if value is None:
        return features
    if float(value) < 1:
        return max(1, min(features, math.ceil(float(value) * features)))
    return max(1, min(features, int(value)))


def _window_scores(
    config: MMPADConfig,
    query: Tensor,
    query_valid: Tensor,
    reference: Tensor,
    reference_valid: Tensor,
    *,
    self_join: bool,
    length: int,
) -> Tensor:
    dimensions = _dimension_count(config, query.shape[-1])
    exclusion = int(config.exclusion_fraction * length)
    reference_positions = torch.arange(reference.shape[0], device=query.device)
    chunks: list[Tensor] = []
    for start in range(0, query.shape[0], config.query_chunk_size):
        stop = min(start + config.query_chunk_size, query.shape[0])
        correlation = (
            torch.einsum("qlc,rlc->qrc", query[start:stop], reference) / length
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
        # Ascending dimension order per pair: the original sorts with the
        # invalid entries sinking to the tail and then maps them to -inf.
        ordered = similarity.sort(dim=2).values
        ordered = torch.where(torch.isfinite(ordered), ordered, -torch.inf)

        # The original picks neighbors independently per dimension slot
        # (find_knn, argmax greedy with a per-slot exclusion zone), then
        # forward-fills across the dimension slots and the neighbor ranks
        # (get_score) and keeps the last entry of each axis. A pair only
        # competes in the slots covered by its own valid-channel count.
        slots = min(dimensions, ordered.shape[2])
        picked = ordered.new_full(
            (ordered.shape[0], slots, config.neighbors), -torch.inf
        )
        for slot in range(slots):
            candidates = ordered[:, :, slot].clone()
            for rank in range(config.neighbors):
                best, index = candidates.max(1)
                picked[:, slot, rank] = best
                candidates.masked_fill_(
                    (reference_positions[None] >= index[:, None] - exclusion)
                    & (reference_positions[None] < index[:, None] + exclusion),
                    -torch.inf,
                )
        for slot in range(1, slots):
            picked[:, slot] = torch.where(
                torch.isfinite(picked[:, slot]),
                picked[:, slot],
                picked[:, slot - 1],
            )
        ranks = picked[:, -1].clone()
        for rank in range(1, config.neighbors):
            ranks[:, rank] = torch.where(
                torch.isfinite(ranks[:, rank]),
                ranks[:, rank],
                ranks[:, rank - 1],
            )
        chunks.append(ranks[:, -1])
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


def _align(
    config: MMPADConfig,
    window_scores: Tensor,
    output_length: int,
    length: int,
) -> Tensor:
    if config.alignment == "mean":
        return overlap_average(
            window_scores[None],
            patch_length=length,
            output_length=output_length,
        )[0]
    if config.alignment == "start":
        return F.pad(window_scores, (0, length - 1))
    left = (length - 1) // 2
    return F.pad(window_scores, (left, length - 1 - left))


@dataclass(frozen=True, slots=True)
class MMPADReference:
    """Reference library built from a normal series by ``build_reference``.

    ``patches`` and ``valid`` are the flattened z-normalized training
    patches and their per-dimension validity flags.
    """

    patches: Tensor
    valid: Tensor
    length: int


@torch.inference_mode()
def build_reference(
    normal_series: Tensor, config: MMPADConfig
) -> MMPADReference:
    length = _resolved_length(config, normal_series)
    normal_series = validate_series(
        normal_series,
        min_length=length,
        name="normal_series",
    )
    patches, valid = _patches(config, normal_series, length)
    return MMPADReference(patches.flatten(0, 1), valid.flatten(0, 1), length)


@torch.inference_mode()
def score(
    series: Tensor,
    config: MMPADConfig,
    reference: MMPADReference | None = None,
) -> Tensor:
    if reference is not None:
        # A fitted reference fixes the window length for every later
        # score call, because query and reference patches must agree.
        length = reference.length
    else:
        length = _resolved_length(config, series)
    series = validate_series(series, min_length=length, name="series")
    query_patches, query_valid = _patches(config, series, length)
    outputs: list[Tensor] = []
    if reference is not None:
        if reference.patches.shape[-1] != series.shape[-1]:
            raise ValueError(
                "series feature count differs from the fitted reference"
            )
        ref_patches = reference.patches.to(query_patches)
        ref_valid = reference.valid.to(query_patches.device)
        for query, valid in zip(query_patches, query_valid):
            outputs.append(
                _align(
                    config,
                    _window_scores(
                        config,
                        query,
                        valid,
                        ref_patches,
                        ref_valid,
                        self_join=False,
                        length=length,
                    ),
                    series.shape[1],
                    length,
                )
            )
    else:
        for query, valid in zip(query_patches, query_valid):
            outputs.append(
                _align(
                    config,
                    _window_scores(
                        config,
                        query,
                        valid,
                        query,
                        valid,
                        self_join=True,
                        length=length,
                    ),
                    series.shape[1],
                    length,
                )
            )
    return torch.stack(outputs)
