"""Load a :class:`SeriesData` from a zarr-backed package."""

from __future__ import annotations

import gzip
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
import zarr
import zarr.storage

from ._data import SeriesData, _validate_json_basic
from ._write import (
    ARRAY_DATA,
    ARRAY_OFFSETS,
    ARRAY_TIMESTAMPS,
    FILE_FEATURES,
    FILE_LABELS,
    FILE_META,
    FORMAT,
)

__all__ = ["read"]


def read(path: str | os.PathLike) -> SeriesData:
    """Load a collection written by :func:`agentad.series.write`.

    ``path`` accepts the packaged archive (``zarr.zip``) or the unpacked
    directory written with ``zipped=False``.
    """
    path = os.fspath(path)
    if os.path.isdir(path):

        def member(name: str) -> bytes:
            with open(os.path.join(path, name), "rb") as handle:
                return handle.read()

        payload = _read_payload(member)
        group = zarr.open_group(store=path, mode="r")
        features = pd.read_parquet(os.path.join(path, FILE_FEATURES))
        labels = pd.read_parquet(os.path.join(path, FILE_LABELS))
        return _load(group, payload, features, labels)

    with zipfile.ZipFile(path) as archive:
        payload = _read_payload(archive.read)
        store = zarr.storage.ZipStore(path, mode="r")
        try:
            group = zarr.open_group(store, mode="r")
            features = pd.read_parquet(io.BytesIO(archive.read(FILE_FEATURES)))
            labels = pd.read_parquet(io.BytesIO(archive.read(FILE_LABELS)))
            return _load(group, payload, features, labels)
        finally:
            store.close()


def _read_payload(member) -> dict:
    payload = json.loads(gzip.decompress(member(FILE_META)))
    found_format = payload.get("format") if isinstance(payload, dict) else None
    if found_format != FORMAT:
        raise ValueError(f"unsupported series package format: {found_format!r}")
    return payload


def _restore_empty_categoricals(
    frame: pd.DataFrame, schema: object, where: str
) -> pd.DataFrame:
    if schema is None:
        return frame
    if len(frame) != 0:
        raise ValueError(f"{where} empty categorical metadata requires zero rows")
    if not isinstance(schema, list) or len(schema) != frame.shape[1]:
        raise ValueError(
            f"{where} empty categorical metadata must align with its columns"
        )
    attrs = frame.attrs.copy()
    for position, spec in enumerate(schema):
        if spec is None:
            continue
        if not isinstance(spec, dict):
            raise TypeError(
                f"{where} categorical metadata at column {position} must be a mapping"
            )
        categories = spec.get("categories")
        ordered = spec.get("ordered")
        if not isinstance(categories, list) or not isinstance(ordered, bool):
            raise TypeError(
                f"{where} categorical metadata at column {position} is invalid"
            )
        _validate_json_basic(categories, f"{where} categorical column {position}")
        dtype = pd.CategoricalDtype(categories=categories, ordered=ordered)
        frame.isetitem(
            position,
            pd.Series(pd.Categorical([], dtype=dtype), index=frame.index),
        )
    frame.attrs = attrs
    return frame


def _restore_duplicate_columns(
    frame: pd.DataFrame, metadata: object, where: str
) -> pd.DataFrame:
    if metadata is None:
        return frame
    if not isinstance(metadata, dict) or set(metadata) != {"values", "name"}:
        raise TypeError(f"{where} duplicate column metadata is invalid")
    values = metadata["values"]
    name = metadata["name"]
    if (
        not isinstance(values, list)
        or len(values) != frame.shape[1]
        or not values
        or not all(type(value) is str for value in values)
        or len(set(values)) == len(values)
        or (name is not None and type(name) is not str)
    ):
        raise ValueError(f"{where} duplicate column metadata is inconsistent")
    frame.columns = pd.Index(values, dtype=object, name=name)
    return frame


def _group_array(group: zarr.Group, name: str) -> zarr.Array:
    node = group[name]
    if not isinstance(node, zarr.Array):
        raise TypeError(f"stored {name!r} must be an array, got {type(node).__name__}")
    return node


def _load(
    group: zarr.Group,
    payload: dict,
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> SeriesData:
    label_columns_in_index = payload.get("label_columns_in_index", False)
    if not isinstance(label_columns_in_index, bool):
        raise TypeError("label column-axis metadata must be a bool")
    label_columns_range = payload.get("label_columns_range")
    if label_columns_range is not None:
        if not label_columns_in_index:
            raise ValueError(
                "label RangeIndex metadata requires indexed column-axis metadata"
            )
        if (
            not isinstance(label_columns_range, dict)
            or set(label_columns_range) != {"start", "stop", "step", "name"}
            or any(
                type(label_columns_range[field]) is not int
                for field in ("start", "stop", "step")
            )
        ):
            raise TypeError("label RangeIndex metadata is invalid")
        _validate_json_basic(label_columns_range["name"], "label RangeIndex name")
    if label_columns_in_index:
        if len(labels) != 0 or labels.shape[1] != 0:
            raise ValueError("label column-axis metadata requires an empty table")
        if label_columns_range is None:
            label_columns = labels.index.copy()
        else:
            label_columns = pd.RangeIndex(**label_columns_range)
    else:
        label_columns = None
    duplicate_columns = payload.get("duplicate_columns", {})
    if not isinstance(duplicate_columns, dict):
        raise TypeError("duplicate column metadata must be a mapping")
    if set(duplicate_columns) - {"features", "labels"}:
        raise ValueError("duplicate column metadata has unknown tables")
    features = _restore_duplicate_columns(
        features, duplicate_columns.get("features"), "features"
    )
    labels = _restore_duplicate_columns(
        labels, duplicate_columns.get("labels"), "labels"
    )
    empty_categoricals = payload.get("empty_categoricals", {})
    if not isinstance(empty_categoricals, dict):
        raise TypeError("empty categorical metadata must be a mapping")
    features = _restore_empty_categoricals(
        features, empty_categoricals.get("features"), "features"
    )
    labels = _restore_empty_categoricals(
        labels, empty_categoricals.get("labels"), "labels"
    )
    offsets = np.asarray(_group_array(group, ARRAY_OFFSETS)[:])
    if offsets.ndim != 1:
        raise ValueError(f"stored offsets must be 1-D, got shape {offsets.shape}")
    if offsets.dtype != np.dtype(np.int64):
        raise TypeError(f"stored offsets must have dtype int64, got {offsets.dtype}")
    if len(offsets) == 0:
        raise ValueError("stored offsets must contain at least the initial zero")
    if offsets[0] != 0:
        raise ValueError("stored offsets must start at 0")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("stored offsets must be non-decreasing")
    point_count = int(offsets[-1])
    label_rows = payload.get("label_rows")
    if label_rows is not None and (
        isinstance(label_rows, bool)
        or not isinstance(label_rows, int)
        or label_rows != point_count
    ):
        raise ValueError(
            f"label row metadata must equal point count {point_count}, "
            f"got {label_rows!r}"
        )
    if labels.shape[1] == 0:
        if label_rows is None and len(labels) != point_count:
            raise ValueError(
                f"zero-column labels has {len(labels)} rows, expected {point_count}"
            )
        label_attrs = labels.attrs.copy()
        labels = pd.DataFrame(index=pd.RangeIndex(point_count))
        if label_columns is not None:
            labels.columns = label_columns
        labels.attrs = label_attrs
    expected_values = point_count * len(features)
    stored_data = _group_array(group, ARRAY_DATA)
    if stored_data.ndim != 1 or stored_data.size != expected_values:
        raise ValueError(
            "stored data must be a flat array with "
            f"{expected_values} values, got shape {stored_data.shape}"
        )
    stored_timestamps = _group_array(group, ARRAY_TIMESTAMPS)
    if (
        stored_timestamps.ndim != 1
        or stored_timestamps.shape[0] != point_count
        or np.dtype(stored_timestamps.dtype) != np.dtype(np.int64)
    ):
        raise ValueError(
            "stored timestamps must be a 1-D int64 array with "
            f"{point_count} values, got shape {stored_timestamps.shape} "
            f"and dtype {stored_timestamps.dtype}"
        )
    flat_data = np.asarray(stored_data[:])
    data = flat_data.reshape(point_count, len(features))
    timestamps = np.asarray(stored_timestamps[:])
    return SeriesData(
        features=features,
        labels=labels,
        data=data,
        timestamps=timestamps,
        offsets=offsets,
        ids=payload["ids"],
        metas=payload["metas"],
        manifest=payload["manifest"],
    )
