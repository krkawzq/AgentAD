"""Synthetic anomaly families used by CARLA's pretext stage."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .._utils import validate_series


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
    maximum = max(minimum, round(time * max_fraction))

    for batch_index in range(result.shape[0]):
        length = int(
            torch.randint(
                minimum,
                maximum + 1,
                (),
                device=result.device,
                generator=generator,
            )
        )
        start = int(
            torch.randint(
                0,
                time - length + 1,
                (),
                device=result.device,
                generator=generator,
            )
        )
        affected_count = max(1, features // 3)
        affected = torch.randperm(features, device=result.device, generator=generator)[
            :affected_count
        ]
        family = int(torch.randint(5, (), device=result.device, generator=generator))
        segment = result[batch_index, start : start + length, affected]
        local_scale = (
            windows[batch_index, :, affected].std(0, unbiased=False).clamp_min(1e-3)
        )

        if family == 0:  # seasonal/frequency change
            frequency = int(
                torch.randint(2, 6, (), device=result.device, generator=generator)
            )
            phase = torch.arange(length, device=result.device, dtype=result.dtype)
            modulation = 1 + 0.5 * torch.sin(2 * math.pi * frequency * phase / length)
            segment = segment * modulation[:, None]
        elif family == 1:  # trend
            direction = torch.where(
                torch.rand((), device=result.device, generator=generator) < 0.5,
                -result.new_ones(()),
                result.new_ones(()),
            )
            ramp = torch.linspace(
                0, 3, length, device=result.device, dtype=result.dtype
            )
            segment = segment + direction * ramp[:, None] * local_scale
        elif family == 2:  # global amplitude
            segment = segment * 8
        elif family == 3:  # contextual amplitude
            center = windows[batch_index, :, affected].mean(0)
            segment = center + 3 * (segment - center)
        else:  # shapelet replacement
            noise = torch.randn(
                segment.shape,
                device=result.device,
                dtype=result.dtype,
                generator=generator,
            )
            segment = segment[:1].expand_as(segment) + 0.1 * local_scale * noise
        result[batch_index, start : start + length, affected] = segment
    return result
