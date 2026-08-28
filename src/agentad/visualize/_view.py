"""Lazy JSON-ready views over :class:`agentad.series.SeriesData`."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from ..series import SeriesData, SeriesItem
from ._normalize import NORMALIZATIONS, _normalize_inplace

DEFAULT_MAX_POINTS = 4_000
HARD_MAX_POINTS = 50_000
MAX_SELECTED_FEATURES = 128
MAX_RESPONSE_VALUES = 1_000_000


def _display(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="backslashreplace")
    return str(value)


def _is_missing_scalar(value: Any) -> bool:
    """Return whether one metadata value is missing without expanding arrays."""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _display_names(values: Sequence[Any]) -> list[str]:
    names: list[str] = []
    seen: dict[str, int] = {}
    for position, value in enumerate(values):
        base = _display(value)
        duplicate = seen.get(base, 0)
        seen[base] = duplicate + 1
        names.append(base if duplicate == 0 else f"{base} [{position}]")
    return names


def _feature_records(features: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    names = _display_names(features.index.tolist())
    columns = _display_names(features.columns.tolist())
    for position, (_, row) in enumerate(features.iterrows()):
        metadata = {
            column: _display(value)
            for column, value in zip(columns, row.tolist())
            if not _is_missing_scalar(value)
        }
        records.append(
            {"index": position, "name": names[position], "metadata": metadata}
        )
    return records


def _binary_label_records(labels: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    names = _display_names(labels.columns.tolist())
    for position in range(labels.shape[1]):
        values = labels.iloc[:, position]
        non_missing = values[~values.isna()]
        try:
            unique = set(non_missing.unique().tolist())
        except TypeError:
            continue
        if unique.issubset({False, True, 0, 1, 0.0, 1.0}):
            records.append({"index": position, "name": names[position]})
    return records


def _downsample_indices(values: np.ndarray, max_points: int) -> np.ndarray:
    """Select bounded ordered points while retaining envelope extrema."""
    rows = values.shape[0]
    if rows <= max_points:
        return np.arange(rows, dtype=np.int64)
    if max_points == 2:
        return np.array([0, rows - 1], dtype=np.int64)

    interior_budget = max_points - 2
    buckets = max(1, (interior_budget + 1) // 2)
    boundaries = np.linspace(1, rows - 1, buckets + 1, dtype=np.int64)
    selected: list[int] = [0]
    remaining = interior_budget
    for bucket, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if upper <= lower:
            continue
        remaining_buckets = buckets - bucket - 1
        slots = min(2, remaining - remaining_buckets)
        chunk = values[lower:upper]
        finite = np.isfinite(chunk)
        if not finite.any():
            selected.append(int(lower))
            remaining -= 1
            continue
        row_min = np.min(chunk, axis=1, where=finite, initial=np.inf)
        row_max = np.max(chunk, axis=1, where=finite, initial=-np.inf)
        low = int(lower + np.argmin(row_min))
        high = int(lower + np.argmax(row_max))
        if slots == 1:
            low_score = abs(float(row_min[low - lower]))
            high_score = abs(float(row_max[high - lower]))
            selected.append(low if low_score >= high_score else high)
            remaining -= 1
        else:
            candidates = (low, high) if low <= high else (high, low)
            selected.extend(candidates)
            remaining -= len(set(candidates))
    selected.append(rows - 1)
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def _finite_json_column(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def _timestamp_labels(item: SeriesItem, indices: np.ndarray) -> list[str]:
    encoding = item.meta.get("timestamp_encoding", {})
    representation = (
        encoding.get("original_values") if isinstance(encoding, dict) else None
    )
    if representation == "labels.timestamp" and "timestamp" in item.labels:
        source = item.labels["timestamp"].to_numpy()[indices]
        return [_display(value) for value in source]

    packed = item.timestamps[indices]
    if representation in {"datetime", "formatted"}:
        try:
            values = pd.Series(pd.to_datetime(packed, unit="ns", utc=True))
            timezone = encoding.get("source_timezone")
            if representation == "datetime" and timezone is not None:
                values = values.dt.tz_convert(str(timezone))
            elif representation == "datetime":
                values = values.dt.tz_localize(None)
            source_format = encoding.get("source_format")
            if representation == "formatted" and isinstance(source_format, str):
                if source_format == "slash-unpadded-ymd-minute":
                    return [
                        f"{value.year}/{value.month}/{value.day} "
                        f"{value.hour:02d}:{value.minute:02d}"
                        for value in values
                    ]
                return values.dt.strftime(source_format).tolist()
            return [_display(value) for value in values]
        except (TypeError, ValueError, OverflowError):
            pass
    return [str(int(value)) for value in packed]


def _label_runs(
    item: SeriesItem,
    label_index: int | None,
    start: int,
    stop: int,
) -> list[dict[str, str | int]]:
    if label_index is None:
        return []
    if label_index < 0 or label_index >= item.labels.shape[1]:
        raise ValueError(f"label index out of range: {label_index}")
    values = item.labels.iloc[start:stop, label_index]
    if values.empty:
        return []
    non_missing = values[~values.isna()]
    try:
        unique = set(non_missing.unique().tolist())
    except TypeError as error:
        raise ValueError("selected label column is not binary") from error
    if not unique.issubset({False, True, 0, 1, 0.0, 1.0}):
        raise ValueError("selected label column is not binary")

    mask = values.astype("boolean").fillna(False).to_numpy(dtype=bool)
    timestamps = item.timestamps[start:stop]
    edges = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    run_starts = np.flatnonzero(edges == 1)
    run_stops = np.flatnonzero(edges == -1) - 1
    return [
        {
            "start": str(int(timestamps[run_start])),
            "stop": str(int(timestamps[run_stop])),
            "start_index": int(start + run_start),
            "stop_index": int(start + run_stop),
        }
        for run_start, run_stop in zip(run_starts, run_stops)
    ]


class SeriesDataView:
    """Validate queries and expose only the selected part of a collection."""

    def __init__(self, sdata: SeriesData) -> None:
        if not isinstance(sdata, SeriesData):
            raise TypeError("sdata must be a SeriesData")
        self.sdata = sdata
        self.features = _feature_records(sdata.features)
        self.binary_labels = _binary_label_records(sdata.labels)
        self._binary_label_indices = frozenset(
            record["index"] for record in self.binary_labels
        )
        self._folded_ids = tuple(series_id.casefold() for series_id in sdata.ids)

    def overview(self, title: str) -> dict[str, Any]:
        lengths = np.diff(self.sdata.offsets)
        return {
            "title": title,
            "series_count": len(self.sdata),
            "point_count": self.sdata.total_points,
            "feature_count": self.sdata.n_features,
            "label_count": self.sdata.labels.shape[1],
            "min_length": int(lengths.min()) if len(lengths) else 0,
            "max_length": int(lengths.max()) if len(lengths) else 0,
            "features": [
                {**record, "metadata": dict(record["metadata"])}
                for record in self.features
            ],
            "binary_labels": [dict(record) for record in self.binary_labels],
            "normalizations": [dict(record) for record in NORMALIZATIONS],
            "manifest": dict(self.sdata.manifest),
            "limits": {
                "default_max_points": DEFAULT_MAX_POINTS,
                "hard_max_points": HARD_MAX_POINTS,
                "max_selected_features": MAX_SELECTED_FEATURES,
                "max_response_values": MAX_RESPONSE_VALUES,
            },
            "geometry": "polyline-2d-v1",
        }

    def items(
        self, query: str = "", offset: int = 0, limit: int = 50
    ) -> dict[str, Any]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if limit < 1 or limit > 200:
            raise ValueError("limit must be between 1 and 200")
        needle = query.casefold().strip()
        if not needle:
            total = len(self.sdata)
            end = min(total, offset + limit)
            page = [
                {
                    "index": index,
                    "id": self.sdata.ids[index],
                    "points": int(
                        self.sdata.offsets[index + 1] - self.sdata.offsets[index]
                    ),
                }
                for index in range(min(offset, total), end)
            ]
            return {
                "items": page,
                "offset": offset,
                "total": total,
                "has_more": end < total,
            }
        page: list[dict[str, Any]] = []
        total = 0
        for index, folded_id in enumerate(self._folded_ids):
            if needle and needle not in folded_id:
                continue
            if offset <= total < offset + limit:
                page.append(
                    {
                        "index": index,
                        "id": self.sdata.ids[index],
                        "points": int(
                            self.sdata.offsets[index + 1] - self.sdata.offsets[index]
                        ),
                    }
                )
            total += 1
        return {
            "items": page,
            "offset": offset,
            "total": total,
            "has_more": offset + len(page) < total,
        }

    def data(
        self,
        *,
        series_index: int,
        feature_indices: Sequence[int] | None = None,
        normalization: str = "none",
        scope: str = "feature",
        start: int = 0,
        stop: int | None = None,
        max_points: int = DEFAULT_MAX_POINTS,
        label_index: int | None = None,
    ) -> dict[str, Any]:
        if series_index < 0 or series_index >= len(self.sdata):
            raise ValueError(f"series index out of range: {series_index}")
        item = self.sdata[series_index]
        if stop is None:
            stop = len(item)
        if start < 0 or stop < start or stop > len(item):
            raise ValueError(
                f"invalid row window [{start}, {stop}) for {len(item)} points"
            )
        if max_points < 2 or max_points > HARD_MAX_POINTS:
            raise ValueError(f"max_points must be between 2 and {HARD_MAX_POINTS}")

        if label_index is not None:
            if label_index < 0 or label_index >= item.labels.shape[1]:
                raise ValueError(f"label index out of range: {label_index}")
            if label_index not in self._binary_label_indices:
                raise ValueError("selected label column is not binary")

        if feature_indices is None:
            feature_indices = tuple(range(min(item.n_features, 8)))
        else:
            feature_indices = tuple(feature_indices)
        if not feature_indices and item.n_features:
            raise ValueError("at least one feature must be selected")
        if len(feature_indices) > MAX_SELECTED_FEATURES:
            raise ValueError(
                f"at most {MAX_SELECTED_FEATURES} features may be selected"
            )
        if len(set(feature_indices)) != len(feature_indices):
            raise ValueError("feature indices must be unique")
        if any(index < 0 or index >= item.n_features for index in feature_indices):
            raise ValueError("feature index out of range")

        returned_points = min(stop - start, max_points)
        if returned_points * len(feature_indices) > MAX_RESPONSE_VALUES:
            raise ValueError(
                "requested chart contains too many values; reduce max_points "
                "or the selected feature count"
            )

        source = item.data[start:stop]
        if np.iscomplexobj(source):
            raise TypeError("complex series values cannot be visualized")
        normalized = np.empty(
            (stop - start, len(feature_indices)),
            dtype=np.float64,
            order="C",
        )
        for column, feature_index in enumerate(feature_indices):
            normalized[:, column] = source[:, feature_index]
        _normalize_inplace(normalized, normalization, scope)
        sampled = _downsample_indices(normalized, max_points)
        absolute = sampled + start
        selected = normalized[sampled]
        timestamp_values = item.timestamps[absolute]

        return {
            "series": {
                "index": series_index,
                "id": item.id,
                "points": len(item),
                "meta": deepcopy(item.meta),
            },
            "window": {"start": start, "stop": stop, "points": stop - start},
            "normalization": normalization,
            "scope": scope,
            "geometry": "polyline-2d-v1",
            "sampled_points": len(sampled),
            "indices": [int(value) for value in absolute],
            "timestamps": [str(int(value)) for value in timestamp_values],
            "timestamp_labels": _timestamp_labels(item, absolute),
            "features": [
                {
                    **self.features[index],
                    "metadata": dict(self.features[index]["metadata"]),
                    "values": _finite_json_column(selected[:, column]),
                }
                for column, index in enumerate(feature_indices)
            ],
            "label_runs": _label_runs(item, label_index, start, stop),
        }
