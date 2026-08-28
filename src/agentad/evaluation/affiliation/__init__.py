"""Affiliation metrics for time-series anomaly detection.

Ported from ``TSB_AD/evaluation/affiliation/`` (TheDatumOrg/TSB-AD, Apache
License 2.0), which packages the affiliation metrics of Hwang, Lyu,
Paoletti et al. (upstream https://github.com/Horea201/Affiliation-F1Score).

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

from ._generics import (
    convert_vector_to_events,
    f1_func,
    has_point_anomalies,
    infer_Trange,
)
from ._metrics import pr_from_events, test_events

__license__ = "Apache-2.0"
__source__ = "TSB_AD/evaluation/affiliation/ @ TheDatumOrg/TSB-AD"

__all__ = [
    "convert_vector_to_events",
    "f1_func",
    "has_point_anomalies",
    "infer_Trange",
    "pr_from_events",
    "test_events",
]
