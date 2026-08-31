"""Load contract-based CSV files for the visualization service."""

from __future__ import annotations

import csv
import os
from typing import Any

import numpy as np
import pandas as pd

from ..series import SeriesData, read

CSV_SERIES_COLUMN = "series_id"
CSV_TIMESTAMP_COLUMN = "timestamp"
CSV_FEATURE_PREFIX = "feature."
CSV_LABEL_PREFIX = "label."
CSV_META_PREFIX = "meta."
CSV_FORMAT = "agentad.visualize.csv.v1"
DEFAULT_MAX_CSV_BYTES = 512 * 1024 * 1024
MAX_CSV_COLUMNS = 2_048
MAX_SERIES_ID_LENGTH = 4_096


def _header(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), None)
    except (csv.Error, UnicodeDecodeError) as error:
        raise ValueError(f"invalid UTF-8 CSV header: {error}") from error
    if header is None:
        raise ValueError("CSV file is empty")
    if not header or any(not name for name in header):
        raise ValueError("CSV column names must be non-empty")
    if len(header) > MAX_CSV_COLUMNS:
        raise ValueError(f"CSV may contain at most {MAX_CSV_COLUMNS} columns")
    if len(set(header)) != len(header):
        raise ValueError("CSV column names must be unique")
    return header


def _contract(
    header: list[str],
) -> tuple[list[str], list[str], list[str]]:
    required = {CSV_SERIES_COLUMN, CSV_TIMESTAMP_COLUMN}
    missing = sorted(required.difference(header))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    feature_columns: list[str] = []
    label_columns: list[str] = []
    meta_columns: list[str] = []
    unknown: list[str] = []
    for name in header:
        if name in required:
            continue
        if name.startswith(CSV_FEATURE_PREFIX):
            feature_columns.append(name)
        elif name.startswith(CSV_LABEL_PREFIX):
            label_columns.append(name)
        elif name.startswith(CSV_META_PREFIX):
            meta_columns.append(name)
        else:
            unknown.append(name)
    if unknown:
        preview = ", ".join(unknown[:8])
        suffix = " ..." if len(unknown) > 8 else ""
        raise ValueError(
            f"CSV contains columns outside the contract: {preview}{suffix}"
        )
    if not feature_columns:
        raise ValueError(f"CSV requires at least one {CSV_FEATURE_PREFIX}<name> column")

    for columns, prefix, kind in (
        (feature_columns, CSV_FEATURE_PREFIX, "feature"),
        (label_columns, CSV_LABEL_PREFIX, "label"),
        (meta_columns, CSV_META_PREFIX, "metadata"),
    ):
        names = [column.removeprefix(prefix) for column in columns]
        if any(not name for name in names):
            raise ValueError(f"CSV {kind} names must be non-empty")
        if len(set(names)) != len(names):
            raise ValueError(f"CSV {kind} names must be unique")
    return feature_columns, label_columns, meta_columns


def _timestamps(values: pd.Series) -> tuple[np.ndarray, dict[str, str] | None]:
    if values.isna().any():
        raise ValueError("CSV timestamp values must not be missing")
    int64_info = np.iinfo(np.int64)
    if pd.api.types.is_integer_dtype(values.dtype):
        if pd.api.types.is_unsigned_integer_dtype(values.dtype) and (
            len(values) and int(values.max()) > int64_info.max
        ):
            raise ValueError("CSV timestamp integer is outside the signed int64 range")
        return np.ascontiguousarray(values.to_numpy(dtype=np.int64, copy=False)), None
    if pd.api.types.is_float_dtype(values.dtype):
        original = values.to_numpy(dtype=np.float64, copy=False)
        if (
            np.isfinite(original).all()
            and np.equal(original, np.trunc(original)).all()
            and (np.abs(original) <= 2**53).all()
        ):
            return np.ascontiguousarray(original.astype(np.int64)), None
        raise ValueError("CSV numeric timestamps must be exact signed int64 integers")
    elif pd.api.types.is_object_dtype(values.dtype) or isinstance(
        values.dtype, pd.StringDtype
    ):
        strings = values.astype("string")
        integer_strings = strings.str.fullmatch(r"[+-]?\d+")
        if integer_strings.any() and not integer_strings.all():
            raise ValueError("CSV must use one timestamp encoding per file")
        if integer_strings.all():
            integers = [int(value) for value in strings]
            if any(
                value < int64_info.min or value > int64_info.max for value in integers
            ):
                raise ValueError(
                    "CSV timestamp integer is outside the signed int64 range"
                )
            return np.asarray(integers, dtype=np.int64), None

    try:
        parsed = pd.to_datetime(values, errors="raise", format="mixed", utc=True)
        # pandas >= 3 parses to microsecond resolution; the packed timestamp
        # contract stores epoch nanoseconds.
        timestamps = (
            parsed.dt.as_unit("ns").astype("int64").to_numpy(dtype=np.int64, copy=False)
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "CSV timestamp values must be signed int64 or ISO-8601 datetimes"
        ) from error
    return np.ascontiguousarray(timestamps), {
        "original_values": "datetime",
        "source_timezone": "UTC",
    }


def _json_scalar(value: Any) -> bool | int | float | str | None:
    if value is pd.NA or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _series_layout(series: pd.Series) -> tuple[list[str], np.ndarray]:
    if series.isna().any():
        raise ValueError("CSV series_id values must not be missing")
    ids_as_string = series.astype("string")
    if (ids_as_string.str.len() == 0).any():
        raise ValueError("CSV series_id values must not be empty")
    if (ids_as_string.str.len() > MAX_SERIES_ID_LENGTH).any():
        raise ValueError(
            f"CSV series_id values may contain at most {MAX_SERIES_ID_LENGTH} characters"
        )
    changed = ids_as_string.ne(ids_as_string.shift(fill_value=pd.NA)).fillna(True)
    starts = np.flatnonzero(changed.to_numpy(dtype=bool))
    ids = ids_as_string.iloc[starts].tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("rows for each CSV series_id must form one contiguous block")
    offsets = np.concatenate(
        (starts.astype(np.int64, copy=False), np.array([len(series)], dtype=np.int64))
    )
    return ids, offsets


def read_csv(
    path: str | os.PathLike[str],
    *,
    max_bytes: int = DEFAULT_MAX_CSV_BYTES,
) -> SeriesData:
    """Read a CSV that follows the ``agentad.visualize.csv.v1`` contract.

    Required columns are ``series_id`` and ``timestamp``. Numeric feature
    columns use ``feature.<name>``; optional point labels use ``label.<name>``;
    optional per-series values use ``meta.<name>`` and must be constant within
    each contiguous series block.
    """
    target = os.path.realpath(os.fspath(path))
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise TypeError("max_bytes must be an integer")
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    try:
        size = os.stat(target).st_size
    except OSError as error:
        raise ValueError(f"cannot stat CSV file: {error}") from error
    if size > max_bytes:
        raise ValueError(
            f"CSV file is {size:,} bytes; the configured limit is {max_bytes:,} bytes"
        )

    header = _header(target)
    feature_columns, label_columns, meta_columns = _contract(header)
    dtypes: dict[str, Any] = {
        CSV_SERIES_COLUMN: "string",
        **{column: np.float64 for column in feature_columns},
    }
    try:
        frame = pd.read_csv(
            target,
            dtype=dtypes,
            encoding="utf-8-sig",
            keep_default_na=False,
            low_memory=False,
            memory_map=True,
            na_values={
                **{column: [""] for column in feature_columns},
                **{column: [""] for column in label_columns},
                **{column: [""] for column in meta_columns},
                CSV_TIMESTAMP_COLUMN: [""],
            },
        )
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        raise ValueError(f"failed to parse CSV: {error}") from error
    if frame.columns.tolist() != header:
        raise ValueError("CSV columns changed while the file was being read")
    try:
        final_size = os.stat(target).st_size
    except OSError as error:
        raise ValueError(f"cannot stat CSV file after parsing: {error}") from error
    if final_size != size:
        raise ValueError("CSV file changed while it was being read")

    ids, offsets = _series_layout(frame[CSV_SERIES_COLUMN])
    timestamps, timestamp_encoding = _timestamps(frame[CSV_TIMESTAMP_COLUMN])
    data = np.ascontiguousarray(
        frame.loc[:, feature_columns].to_numpy(dtype=np.float64, copy=False)
    )

    label_names = [column.removeprefix(CSV_LABEL_PREFIX) for column in label_columns]
    labels = frame.loc[:, label_columns].copy(deep=False)
    labels.columns = label_names
    feature_names = [
        column.removeprefix(CSV_FEATURE_PREFIX) for column in feature_columns
    ]
    features = pd.DataFrame(
        {"source_column": feature_columns},
        index=pd.Index(feature_names, name="feature"),
    )

    metas: list[dict[str, Any]] = []
    for series_index, series_id in enumerate(ids):
        start = int(offsets[series_index])
        stop = int(offsets[series_index + 1])
        meta: dict[str, Any] = {}
        for column in meta_columns:
            values = frame[column].iloc[start:stop].dropna().unique().tolist()
            if len(values) > 1:
                name = column.removeprefix(CSV_META_PREFIX)
                raise ValueError(
                    f"CSV metadata {name!r} is not constant for series {series_id!r}"
                )
            meta[column.removeprefix(CSV_META_PREFIX)] = (
                None if not values else _json_scalar(values[0])
            )
        if timestamp_encoding is not None:
            meta["timestamp_encoding"] = dict(timestamp_encoding)
        metas.append(meta)

    return SeriesData(
        features=features,
        labels=labels,
        data=data,
        timestamps=timestamps,
        offsets=offsets,
        ids=ids,
        metas=metas,
        manifest={
            "format": CSV_FORMAT,
            "source_file": os.path.basename(target),
            "columns": {
                "series": CSV_SERIES_COLUMN,
                "timestamp": CSV_TIMESTAMP_COLUMN,
                "features": feature_columns,
                "labels": label_columns,
                "metadata": meta_columns,
            },
        },
    )


def read_source(
    path: str | os.PathLike[str],
    *,
    max_csv_bytes: int = DEFAULT_MAX_CSV_BYTES,
) -> SeriesData:
    """Read a supported visualization source from disk."""
    target = os.fspath(path)
    if target.casefold().endswith(".csv"):
        return read_csv(target, max_bytes=max_csv_bytes)
    return read(target)
