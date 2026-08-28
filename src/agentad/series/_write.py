"""Persist a :class:`SeriesData` as a zarr-backed package."""

from __future__ import annotations

import gzip
import io
import json
import os
import tempfile
import warnings
import zipfile

import numpy as np
import pandas as pd
import zarr

from ._data import SeriesData, _validate_json_basic

__all__ = ["write"]

# Stable identity of a serialized SeriesData package.
FORMAT = "agentad.series"

ARRAY_DATA = "data"
ARRAY_TIMESTAMPS = "timestamps"
ARRAY_OFFSETS = "offsets"
FILE_FEATURES = "features.parquet.gzip"
FILE_LABELS = "labels.parquet.gzip"
FILE_META = "series_data.json.gzip"
_TARGET_CHUNK_BYTES = 16 * 1024 * 1024
_ZARR_DATA_DTYPES = frozenset(
    np.dtype(dtype)
    for dtype in (
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    )
)
_LOSSLESS_PARQUET_OBJECT_KINDS = frozenset(
    {"bytes", "date", "decimal", "empty", "string"}
)


def _chunk_shape(
    values: np.ndarray, *, target_bytes: int = _TARGET_CHUNK_BYTES
) -> tuple[int, ...]:
    if target_bytes <= 0:
        raise ValueError("target chunk bytes must be positive")
    if values.ndim == 0:
        return ()

    remaining = max(1, target_bytes // max(1, values.dtype.itemsize))
    chunks = [1] * values.ndim
    for axis in range(values.ndim - 1, -1, -1):
        size = max(1, int(values.shape[axis]))
        chunks[axis] = min(size, remaining)
        remaining = max(1, remaining // chunks[axis])
    return tuple(chunks)


def _create_array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    array = group.create_array(
        name,
        shape=values.shape,
        dtype=values.dtype,
        chunks=_chunk_shape(values),
    )
    if values.size:
        array[...] = values


def _parquet_round_trip(sample: pd.DataFrame, *, index: bool | None) -> pd.DataFrame:
    sample = sample.copy(deep=False)
    sample.attrs = {}
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sample.to_parquet(buffer, compression=None, index=index)
    buffer.seek(0)
    return pd.read_parquet(buffer)


def _is_plain_string_axis(axis: pd.Index) -> bool:
    return (
        not isinstance(axis, pd.MultiIndex)
        and axis.dtype == np.dtype(object)
        and (axis.name is None or type(axis.name) is str)
        and all(type(value) is str for value in axis)
    )


def _validate_parquet_axes(
    frame: pd.DataFrame,
    where: str,
    *,
    index: bool | None,
    validate_index: bool = False,
) -> None:
    """Reject axes that the active pandas/pyarrow pair would alter."""
    columns_safe = frame.shape[1] == 0 or (
        frame.columns.is_unique and _is_plain_string_axis(frame.columns)
    )
    index_safe = (
        not validate_index
        or len(frame.index) == 0
        or _is_plain_string_axis(frame.index)
    )
    if validate_index and index_safe:
        index_field = frame.index.name or "__index_level_0__"
        cross_axis_safe = index_field not in frame.columns
    else:
        cross_axis_safe = not validate_index

    if not columns_safe or not cross_axis_safe:
        try:
            schema = _parquet_round_trip(frame.iloc[:0], index=index)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"{where} schema cannot be represented losslessly in Parquet"
            ) from error
        if frame.shape[1] and not frame.columns.identical(schema.columns):
            raise TypeError(
                f"{where} columns cannot be represented losslessly in Parquet"
            )
    if validate_index and not index_safe:
        try:
            restored = _parquet_round_trip(pd.DataFrame(index=frame.index), index=True)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"{where} index cannot be represented losslessly in Parquet"
            ) from error
        if not frame.index.identical(restored.index):
            raise TypeError(
                f"{where} index cannot be represented losslessly in Parquet"
            )


def _validate_empty_index_round_trip(index: pd.Index, where: str) -> None:
    try:
        restored = _parquet_round_trip(
            pd.DataFrame(index=index), index=True
        ).index
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{where} cannot be represented losslessly in Parquet"
        ) from error
    if not index.identical(restored):
        raise TypeError(f"{where} cannot be represented losslessly in Parquet")


def _validate_object_values(values: pd.Series, where: str) -> None:
    inferred = pd.api.types.infer_dtype(values, skipna=True)
    if inferred not in _LOSSLESS_PARQUET_OBJECT_KINDS:
        raise TypeError(
            f"{where} contains object values of kind {inferred!r}, which "
            "cannot be represented losslessly in Parquet"
        )
    missing = values[values.isna()]
    if any(value is not None for value in missing.array):
        raise TypeError(
            f"{where} contains non-None missing object values, which cannot "
            "be represented losslessly in Parquet"
        )


def _validate_parquet_values(frame: pd.DataFrame, where: str) -> None:
    """Reject pandas object values whose Python representation Parquet alters."""
    for position, dtype in enumerate(frame.dtypes):
        if pd.api.types.is_object_dtype(dtype):
            _validate_object_values(
                frame.iloc[:, position], f"{where} column {position}"
            )
        elif isinstance(dtype, pd.CategoricalDtype) and pd.api.types.is_object_dtype(
            dtype.categories.dtype
        ):
            _validate_object_values(
                pd.Series(dtype.categories, dtype=object),
                f"{where} categorical column {position}",
            )


def _validate_zarr_data_dtype(dtype: np.dtype) -> None:
    if dtype not in _ZARR_DATA_DTYPES:
        raise TypeError(f"data dtype {dtype} cannot be represented losslessly in Zarr")


def _encode_duplicate_columns(
    frame: pd.DataFrame, where: str
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    if frame.columns.is_unique:
        return frame, None
    if not _is_plain_string_axis(frame.columns):
        raise TypeError(
            f"{where} duplicate columns must be plain strings for persistence"
        )
    metadata = {
        "values": frame.columns.tolist(),
        "name": frame.columns.name,
    }
    _validate_json_basic(metadata, f"{where} duplicate columns")
    encoded = frame.copy(deep=False)
    encoded.attrs = frame.attrs.copy()
    encoded.columns = [
        f"__agentad_{where}_column_{position}__" for position in range(frame.shape[1])
    ]
    return encoded, metadata


def _empty_categorical_schema(
    frame: pd.DataFrame, where: str
) -> list[dict[str, object] | None] | None:
    if len(frame) != 0:
        return None
    schema: list[dict[str, object] | None] = []
    found = False
    for position, dtype in enumerate(frame.dtypes):
        if isinstance(dtype, pd.CategoricalDtype):
            categories = dtype.categories.tolist()
            _validate_json_basic(categories, f"{where} categorical column {position}")
            schema.append({"categories": categories, "ordered": bool(dtype.ordered)})
            found = True
        else:
            schema.append(None)
    return schema if found else None


def write(
    sdata: SeriesData,
    path: str | os.PathLike,
    *,
    zipped: bool = True,
    create_parents: bool = False,
) -> None:
    """Persist a collection as a zarr-backed package.

    Dirty item labels are folded in first, so assigning ``item.labels``
    before writing is enough.  ``data`` is stored flattened to 1-D in C
    order; ``timestamps`` and ``offsets`` are 1-D int64 zarr arrays.
    ``features`` and ``labels`` go to ``.parquet.gzip`` files; ids,
    per-series ``metas`` and ``manifest`` share one ``series_data.json.gzip``.
    Object columns are limited to scalar strings, bytes, dates, decimals and
    ``None`` so their Python values survive the Parquet round trip unchanged.

    With ``zipped=True`` (default) the build directory is packaged into a
    ZIP_STORED archive at ``path`` — packaging only, no compression; the
    recommended file name is ``zarr.zip``.  With ``zipped=False`` the
    unpacked package is written to the (non-existing) directory ``path``.
    ``create_parents=True`` creates missing output parent directories.
    """
    if not isinstance(sdata, SeriesData):
        raise TypeError("sdata must be a SeriesData")
    if not isinstance(zipped, bool):
        raise TypeError("zipped must be a bool")
    if not isinstance(create_parents, bool):
        raise TypeError("create_parents must be a bool")
    sdata = sdata.materialize()
    _validate_zarr_data_dtype(sdata.data.dtype)
    _validate_json_basic(sdata.features.attrs, "features attrs")
    _validate_json_basic(sdata.labels.attrs, "labels attrs")
    _validate_parquet_values(sdata.features, "features")
    _validate_parquet_values(sdata.labels, "labels")
    empty_categoricals = {}
    feature_categoricals = _empty_categorical_schema(sdata.features, "features")
    label_categoricals = _empty_categorical_schema(sdata.labels, "labels")
    if feature_categoricals is not None:
        empty_categoricals["features"] = feature_categoricals
    if label_categoricals is not None:
        empty_categoricals["labels"] = label_categoricals
    duplicate_columns = {}
    features, feature_columns = _encode_duplicate_columns(sdata.features, "features")
    labels, label_columns = _encode_duplicate_columns(sdata.labels, "labels")
    if feature_columns is not None:
        duplicate_columns["features"] = feature_columns
    if label_columns is not None:
        duplicate_columns["labels"] = label_columns
    features_index = True
    _validate_parquet_axes(
        features,
        "features",
        index=features_index,
        validate_index=True,
    )
    label_columns_in_index = labels.shape[1] == 0
    label_columns_range = None
    if label_columns_in_index:
        stored_labels = pd.DataFrame(index=labels.columns.copy())
        stored_labels.attrs = labels.attrs.copy()
        labels_index = True
        if isinstance(labels.columns, pd.RangeIndex):
            _validate_json_basic(labels.columns.name, "labels column axis name")
            label_columns_range = {
                "start": labels.columns.start,
                "stop": labels.columns.stop,
                "step": labels.columns.step,
                "name": labels.columns.name,
            }
        else:
            _validate_empty_index_round_trip(
                labels.columns, "labels column axis"
            )
        _validate_parquet_axes(
            stored_labels,
            "labels column axis",
            index=labels_index,
            validate_index=True,
        )
    else:
        stored_labels = labels
        labels_index = None
        _validate_parquet_axes(stored_labels, "labels", index=labels_index)
    path = os.path.abspath(os.fspath(path))
    parent = os.path.dirname(path)
    if create_parents:
        os.makedirs(parent, exist_ok=True)
    elif not os.path.isdir(parent):
        raise FileNotFoundError(f"output parent directory does not exist: {parent}")
    with tempfile.TemporaryDirectory(prefix="agentad-series-", dir=parent) as staging:
        # A child created with ordinary mkdir permissions honors the caller's
        # umask. Moving TemporaryDirectory itself would publish it as 0700.
        build = os.path.join(staging, "package")
        os.mkdir(build)
        features.to_parquet(
            os.path.join(build, FILE_FEATURES),
            compression="gzip",
            index=features_index,
        )
        stored_labels.to_parquet(
            os.path.join(build, FILE_LABELS),
            compression="gzip",
            index=labels_index,
        )
        # Parquet cannot encode a row count without a column. The physical
        # zero-row table carries the logical column axis in its index, while
        # label_rows restores the packed row count without a large sentinel.
        group = zarr.group(store=build)
        _create_array(group, ARRAY_DATA, np.ascontiguousarray(sdata.data).reshape(-1))
        _create_array(group, ARRAY_TIMESTAMPS, sdata.timestamps)
        _create_array(group, ARRAY_OFFSETS, sdata.offsets)
        payload = {
            "format": FORMAT,
            "ids": list(sdata.ids),
            "metas": [dict(meta) for meta in sdata.metas],
            "manifest": dict(sdata.manifest),
            "label_rows": len(sdata.labels),
            "empty_categoricals": empty_categoricals,
        }
        if label_columns_in_index:
            payload["label_columns_in_index"] = True
        if label_columns_range is not None:
            payload["label_columns_range"] = label_columns_range
        if duplicate_columns:
            payload["duplicate_columns"] = duplicate_columns
        with gzip.open(
            os.path.join(build, FILE_META), "wt", encoding="utf-8"
        ) as handle:
            json.dump(payload, handle, allow_nan=False)
        if zipped:
            with tempfile.TemporaryDirectory(
                prefix="agentad-series-publish-", dir=parent
            ) as publish:
                staged = os.path.join(publish, "package.zip")
                with zipfile.ZipFile(
                    staged, "w", compression=zipfile.ZIP_STORED
                ) as archive:
                    for root, dirs, files in os.walk(build):
                        dirs.sort()
                        for file_name in sorted(files):
                            full = os.path.join(root, file_name)
                            archive.write(full, os.path.relpath(full, build))
                with open(staged, "rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(staged, path)
        else:
            if os.path.lexists(path):
                raise FileExistsError(f"output path already exists: {path}")
            os.replace(build, path)
