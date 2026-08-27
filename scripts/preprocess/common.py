"""Shared preprocessing primitives that pack every source into SeriesData.

Each dataset becomes one :class:`SeriesData` package per (split, feature
schema) group: ``data`` keeps the dense values, ``labels`` carries the
point-wise label column, and per-series provenance lives in the package
metadata. Source timestamps use int64 only when their original values are
reconstructible from an explicit encoding descriptor; other series keep 0-based
offsets and store the original timestamps as a ``timestamp`` column of the labels
frame.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import tempfile
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from agentad.series import SeriesData, SeriesItem
from agentad.series import write as _write_package


SPLIT_NAMES = {"train", "val", "test", "eval", "data"}


def jsonable(value: Any) -> Any:
    """Convert nested NumPy/pandas values to strict JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, range):
        return list(value)
    if isinstance(value, (Path, os.PathLike)):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        converted = {}
        for key, item in value.items():
            string_key = str(key)
            if string_key in converted:
                raise ValueError(
                    f"metadata keys collide after string conversion: {string_key!r}"
                )
            converted[string_key] = jsonable(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [jsonable(item) for item in sorted(value, key=lambda item: repr(item))]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"cannot convert {type(value).__name__} to JSON")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_signature(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


@lru_cache(maxsize=None)
def _sha256_for_signature(
    path: str, signature: tuple[int, int, int, int, int]
) -> str:
    del signature
    return sha256(Path(path))


def safe_name(value: str) -> str:
    """Return a stable path component without discarding readable identifiers."""
    name = re.sub(r"[\\/\x00-\x1f]", "_", str(value)).strip()
    if name in {"", ".", ".."}:
        raise ValueError(f"invalid empty path identifier: {value!r}")
    return name


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


@contextmanager
def staged_source_directory(output_root: Path, source: str) -> Iterator[Path]:
    """Build one source in isolation and publish it only after success."""
    safe_source = safe_name(source)
    if safe_source != source:
        raise ValueError(f"unsafe source namespace: {source!r}")
    output_root = Path(output_root)
    if output_root.resolve() == Path(output_root.anchor):
        raise ValueError("output_root must not be a filesystem root")
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / source
    staging = Path(tempfile.mkdtemp(prefix=f".{source}.staging-", dir=output_root))
    backup = output_root / f".{source}.previous-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        yield staging
        had_target = os.path.lexists(target)
        if had_target:
            os.replace(target, backup)
        try:
            os.replace(staging, target)
        except BaseException:
            if had_target and not os.path.lexists(target):
                os.replace(backup, target)
            raise
        if had_target:
            try:
                _remove_path(backup)
            except OSError as error:
                warnings.warn(
                    f"published {target}, but could not remove {backup}: {error}"
                )
    except BaseException:
        _remove_path(staging)
        raise


def source_fingerprint(path: Path, root: Path | None = None) -> dict[str, Any]:
    resolved = Path(path).resolve()
    before = resolved.stat()
    signature = _stat_signature(before)
    digest = _sha256_for_signature(str(resolved), signature)
    after = resolved.stat()
    if _stat_signature(after) != signature:
        raise RuntimeError(f"source file changed while hashing: {resolved}")
    display = resolved
    if root is not None:
        try:
            display = display.relative_to(Path(root).resolve())
        except ValueError:
            pass
    return {
        "path": str(display),
        "size_bytes": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "sha256": digest,
    }


def strict_int(value: Any, *, name: str) -> int:
    """Parse an integer without truncating floats or accepting booleans."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if not isinstance(missing, (bool, np.bool_)) or bool(missing):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)) or not float(value).is_integer():
            raise ValueError(f"{name} must be an integer, got {value!r}")
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value)
    raise ValueError(f"{name} must be an integer, got {value!r}")


@dataclass(slots=True)
class SplitData:
    values: pd.DataFrame | np.ndarray | Sequence[Any]
    timestamps: Sequence[Any] | np.ndarray | pd.Series | None = None
    labels: Sequence[Any] | np.ndarray | pd.Series | None = None
    feature_names: Sequence[str] | None = None
    timestamp_kind: str = "index"


def long_to_split(df: pd.DataFrame, *, require_label: bool = True) -> SplitData:
    """Convert canonical long ``date,data,cols`` input without silent aggregation."""
    required = {"date", "data", "cols"}
    if not required <= set(df.columns):
        raise ValueError(
            f"missing long-format columns: {sorted(required - set(df.columns))}"
        )
    data = df.loc[:, ["date", "data", "cols"]].copy()
    data["cols"] = data["cols"].map(str)
    if data[["date", "cols"]].duplicated().any():
        duplicate = data.loc[
            data[["date", "cols"]].duplicated(), ["date", "cols"]
        ].iloc[0]
        raise ValueError(f"duplicate date/feature pair: {duplicate.to_dict()}")

    column_order = [str(value) for value in pd.unique(data["cols"])]
    has_label = "label" in column_order
    if require_label and not has_label:
        raise ValueError("long-format dataset has no label rows")
    features = [name for name in column_order if name != "label"]
    if not features:
        raise ValueError("long-format dataset has no feature rows")
    wide = data.pivot(index="date", columns="cols", values="data")
    wide = wide.sort_index()
    missing_features = [name for name in features if name not in wide]
    if missing_features:
        raise ValueError(f"features disappeared during pivot: {missing_features}")
    labels = None
    if has_label:
        if wide["label"].isna().any():
            raise ValueError("label rows are missing for one or more timestamps")
        labels = wide["label"].to_numpy()
    return SplitData(
        values=wide.loc[:, features].reset_index(drop=True),
        timestamps=wide.index.to_numpy(),
        labels=labels,
        feature_names=features,
        timestamp_kind="source",
    )


def _feature_matrix(
    values: Any, feature_names: Sequence[str] | None
) -> tuple[np.ndarray, list[str]]:
    if isinstance(values, pd.DataFrame):
        matrix = values.to_numpy()
        default_names = [str(column) for column in values.columns]
    else:
        matrix = np.asarray(values)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        default_names = []
    if matrix.ndim != 2:
        raise ValueError(f"values must be 1D or 2D, got shape {matrix.shape}")
    if matrix.shape[1] == 0:
        raise ValueError("values must contain at least one feature")
    if not default_names:
        default_names = [f"feat_{index}" for index in range(matrix.shape[1])]
    if isinstance(feature_names, (str, bytes)):
        raise TypeError("feature_names must be a sequence, not one string")
    names = (
        [str(name) for name in feature_names]
        if feature_names is not None
        else default_names
    )
    if len(names) != matrix.shape[1]:
        raise ValueError("feature_names length does not match feature count")
    if any(not name for name in names):
        raise ValueError("feature names must be non-empty")
    if len(set(names)) != len(names):
        raise ValueError("feature names must be unique")
    if not (
        np.issubdtype(matrix.dtype, np.number) or np.issubdtype(matrix.dtype, np.bool_)
    ):
        raise TypeError(
            f"feature values must be numeric or boolean, got {matrix.dtype}; "
            "encode non-numeric columns before packing"
        )
    return matrix, names


def _label_vector(labels: Any, rows: int) -> np.ndarray | None:
    if labels is None:
        return None
    series = pd.Series(np.asarray(labels).reshape(-1))
    if len(series) != rows:
        raise ValueError(f"label length {len(series)} != value rows {rows}")
    if series.isna().any():
        raise ValueError("labels contain missing values")
    try:
        numeric = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as error:
        raise TypeError("labels must be numeric or boolean") from error
    values = numeric.to_numpy()
    if np.issubdtype(values.dtype, np.complexfloating):
        raise TypeError("labels must be real-valued")
    if np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.bool_):
        if len(values) == 0:
            return values.astype(np.int64)
        lower = int(values.min())
        upper = int(values.max())
        limits = np.iinfo(np.int64)
        if limits.min <= lower and upper <= limits.max:
            return values.astype(np.int64)
        return np.array(values, copy=True)
    real = values.astype(np.float64, copy=False)
    if not np.isfinite(real).all():
        raise ValueError("labels contain non-finite values")
    if len(real) == 0 or (
        np.equal(real, np.floor(real)).all()
        and (real >= -(2**63)).all()
        and (real < 2**63).all()
    ):
        return real.astype(np.int64)
    return real


def _timestamp_encoding(
    source_dtype: str | None,
    conversion: str,
    original_values: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "source_dtype": source_dtype,
        "conversion": conversion,
        "original_values": original_values,
        **extra,
    }


def _offset_fallback(
    series: pd.Series, rows: int
) -> tuple[np.ndarray, pd.Series, str, dict[str, Any]]:
    return (
        np.arange(rows, dtype=np.int64),
        series,
        "offset",
        _timestamp_encoding(str(series.dtype), "offset", "labels.timestamp"),
    )


def _integer_timestamps(values: np.ndarray) -> np.ndarray | None:
    if len(values) == 0:
        return np.empty(0, dtype=np.int64)
    lower = int(values.min())
    upper = int(values.max())
    limits = np.iinfo(np.int64)
    if lower < limits.min or upper > limits.max:
        return None
    return values.astype(np.int64)


def _render_datetime_strings(
    packed: np.ndarray, source_format: str
) -> np.ndarray:
    values = pd.Series(pd.to_datetime(packed, unit="ns", utc=True)).dt.tz_localize(None)
    if source_format == "slash-unpadded-ymd-minute":
        return values.map(
            lambda value: (
                f"{value.year}/{value.month}/{value.day} "
                f"{value.hour:02d}:{value.minute:02d}"
            )
        ).to_numpy(dtype=object)
    return values.dt.strftime(source_format).to_numpy(dtype=object)


def _reversible_datetime_format(
    source: pd.Series, packed: np.ndarray
) -> str | None:
    strings = source.astype(str).to_numpy(dtype=object)
    if len(strings) == 0:
        return None
    first = str(strings[0])
    candidates: list[str] = []
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", first):
        candidates.append("%Y-%m-%d")
    elif re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", first):
        candidates.append("%Y-%m-%d %H:%M")
    elif re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", first):
        candidates.append("%Y-%m-%d %H:%M:%S")
    elif re.fullmatch(r"\d{4}/\d{1,2}/\d{1,2} \d{2}:\d{2}", first):
        candidates.extend(("%Y/%m/%d %H:%M", "slash-unpadded-ymd-minute"))
    for candidate in candidates:
        rendered = _render_datetime_strings(packed, candidate)
        if np.array_equal(rendered, strings):
            return candidate
    return None


def decode_source_timestamps(
    packed: np.ndarray,
    labels: pd.DataFrame,
    encoding: Mapping[str, Any],
) -> np.ndarray:
    """Reconstruct the source timestamp values represented by one package item."""
    representation = encoding.get("original_values")
    if representation == "labels.timestamp":
        if "timestamp" not in labels:
            raise ValueError("timestamp sidecar is missing from labels")
        values = labels["timestamp"].to_numpy()
        source_dtype = encoding.get("source_dtype")
        if isinstance(source_dtype, str) and source_dtype not in {"object", "string"}:
            try:
                return values.astype(np.dtype(source_dtype))
            except (TypeError, ValueError):
                pass
        return values
    if representation == "formatted":
        source_format = encoding.get("source_format")
        if not isinstance(source_format, str) or not source_format:
            raise ValueError("formatted timestamps have no source_format")
        return _render_datetime_strings(packed, source_format)
    if representation == "datetime":
        timezone = encoding.get("source_timezone")
        values = pd.Series(pd.to_datetime(packed, unit="ns", utc=True))
        if timezone is None:
            values = values.dt.tz_localize(None)
        else:
            values = values.dt.tz_convert(str(timezone))
        return values.to_numpy()
    if representation == "cast":
        dtype = encoding.get("source_dtype")
        if not isinstance(dtype, str) or dtype == "object":
            return packed.astype(object)
        return packed.astype(np.dtype(dtype), copy=False)
    if representation == "packed":
        return packed
    raise ValueError(f"unsupported timestamp representation: {representation!r}")


def _encode_timestamps(
    timestamps: Any, rows: int
) -> tuple[np.ndarray, pd.Series, str, dict[str, Any]]:
    """Pack timestamps to int64 when they convert cleanly.

    Integer values keep their magnitude, exactly reversible datetime strings use
    epoch nanoseconds plus a format descriptor, and everything else falls back to
    0-based offsets while the original representation is returned for the labels
    frame.
    """
    if timestamps is None:
        packed = np.arange(rows, dtype=np.int64)
        return (
            packed,
            pd.Series(packed),
            "index",
            _timestamp_encoding(None, "generated_index", "packed"),
        )

    series = pd.Series(timestamps).reset_index(drop=True)
    if len(series) != rows:
        raise ValueError(f"timestamp length {len(series)} != value rows {rows}")
    if series.isna().any():
        raise ValueError("timestamps contain missing values")
    if rows == 0:
        return (
            np.empty(0, dtype=np.int64),
            series,
            "index",
            _timestamp_encoding(str(series.dtype), "empty", "packed"),
        )

    if pd.api.types.is_integer_dtype(series.dtype):
        packed = _integer_timestamps(series.to_numpy())
        if packed is not None:
            return (
                packed,
                series,
                "int64",
                _timestamp_encoding(str(series.dtype), "integer_identity", "packed"),
            )
        return _offset_fallback(series, rows)
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        packed = pd.to_datetime(series, utc=True).astype("int64").to_numpy()
        timezone = getattr(series.dt, "tz", None)
        return (
            packed,
            series,
            "epoch_ns",
            _timestamp_encoding(
                str(series.dtype),
                "datetime_epoch_ns",
                "datetime",
                source_timezone=None if timezone is None else str(timezone),
            ),
        )
    if pd.api.types.is_float_dtype(series.dtype):
        values = series.to_numpy(dtype=np.float64, copy=False)
        if (
            np.isfinite(values).all()
            and np.equal(values, np.floor(values)).all()
            and (values >= -(2**63)).all()
            and (values < 2**63).all()
        ):
            return (
                values.astype(np.int64),
                series,
                "int64",
                _timestamp_encoding(str(series.dtype), "integer_cast", "cast"),
            )
        return _offset_fallback(series, rows)

    string_values = series.map(lambda value: isinstance(value, str))
    all_strings = bool(string_values.all())
    if all_strings and series.astype(str).str.strip().str.fullmatch(r"[+-]?\d+").any():
        return _offset_fallback(series, rows)
    try:
        numeric = None if all_strings else pd.to_numeric(series)
    except (TypeError, ValueError):
        numeric = None
    if numeric is not None and pd.api.types.is_integer_dtype(numeric.dtype):
        packed = _integer_timestamps(numeric.to_numpy())
        if packed is not None:
            return (
                packed,
                series,
                "int64",
                _timestamp_encoding(str(series.dtype), "integer_cast", "cast"),
            )
    elif numeric is not None and pd.api.types.is_float_dtype(numeric.dtype):
        values = numeric.to_numpy(dtype=np.float64, copy=False)
        if (
            np.isfinite(values).all()
            and np.equal(values, np.floor(values)).all()
            and (values >= -(2**63)).all()
            and (values < 2**63).all()
        ):
            return (
                values.astype(np.int64),
                series,
                "int64",
                _timestamp_encoding(str(series.dtype), "integer_cast", "cast"),
            )
    try:
        parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError, OverflowError):
        parsed = None
    if parsed is not None and not parsed.isna().any():
        packed = parsed.astype("int64").to_numpy()
        source_format = _reversible_datetime_format(series, packed)
        if source_format is not None:
            return (
                packed,
                series,
                "epoch_ns",
                _timestamp_encoding(
                    str(series.dtype),
                    "formatted_datetime_epoch_ns",
                    "formatted",
                    source_format=source_format,
                ),
            )
    return _offset_fallback(series, rows)


def label_runs(
    labels: np.ndarray | None, timestamps: np.ndarray, split: str
) -> list[dict[str, Any]]:
    if labels is None or len(labels) == 0:
        return []
    array = np.asarray(labels).reshape(-1)
    nonzero = array != 0
    if not nonzero.any():
        return []
    starts = np.flatnonzero(nonzero & np.r_[True, array[1:] != array[:-1]])
    ends = np.flatnonzero(nonzero & np.r_[array[:-1] != array[1:], True])
    return [
        {
            "split": split,
            "start": jsonable(timestamps[start]),
            "end": jsonable(timestamps[end]),
            "start_index": int(start),
            "end_index": int(end),
            "label": jsonable(array[start]),
        }
        for start, end in zip(starts, ends, strict=True)
    ]


def split_interval_annotations(
    start: int, end: int, split_at: int, *, label: int = 1
) -> list[dict[str, Any]]:
    """Split one global half-open interval into train/test-local annotations."""
    if not isinstance(split_at, int) or isinstance(split_at, bool) or split_at < 0:
        raise ValueError(f"invalid split point: {split_at!r}")
    if not 0 <= start <= end:
        raise ValueError(f"invalid half-open interval: {(start, end)}")
    annotations = []
    for split, lower, upper in (
        ("train", 0, split_at),
        ("test", split_at, end),
    ):
        overlap_start = max(start, lower)
        overlap_end = min(end, upper)
        if overlap_start >= overlap_end:
            continue
        annotations.append(
            {
                "split": split,
                "start": overlap_start,
                "end": overlap_end - 1,
                "start_index": overlap_start - lower,
                "end_index": overlap_end - lower - 1,
                "label": label,
                "source_start_index": start,
                "source_end_index": end - 1,
                "source_end_is_exclusive": True,
            }
        )
    return annotations


@dataclass(slots=True)
class _SplitBuffer:
    matrix: np.ndarray
    names: list[str]
    labels: np.ndarray | None
    packed_timestamps: np.ndarray
    original_timestamps: pd.Series | None
    encoded_timestamp_kind: str
    source_timestamp_kind: str
    timestamp_encoding: dict[str, Any]


@dataclass(slots=True)
class _SeriesEntry:
    series_id: str
    splits: dict[str, _SplitBuffer]
    source_files: list[dict[str, Any]]
    source_metadata: dict[str, Any]
    annotations: list[dict[str, Any]]


class DatasetWriter:
    """Buffers per-series splits and packs each (split, feature-schema) group
    into one SeriesData package when finalized.

    Series whose timestamps cannot be packed to int64 use 0-based offsets and
    contribute their original timestamps to a shared ``timestamp`` column of
    the labels frame, so every package stays internally consistent.
    """

    def __init__(
        self,
        dataset_dir: Path,
        *,
        source: str,
        dataset: str,
        task: str,
        project_root: Path,
    ) -> None:
        if safe_name(source) != source or safe_name(dataset) != dataset:
            raise ValueError(f"unsafe source/dataset identity: {source!r}/{dataset!r}")
        if Path(dataset_dir).name != dataset:
            raise ValueError(
                f"dataset directory name {Path(dataset_dir).name!r} != {dataset!r}"
            )
        if not isinstance(task, str) or not task:
            raise ValueError("task must be non-empty")
        self._dataset_dir = Path(dataset_dir)
        self._source = source
        self._dataset = dataset
        self._task = task
        self._project_root = Path(project_root)
        self._entries: list[_SeriesEntry] = []
        self._series_ids: set[str] = set()

    def add_series(
        self,
        *,
        series_id: str,
        splits: Mapping[str, SplitData],
        source_files: Iterable[Path] = (),
        source_metadata: Mapping[str, Any] | None = None,
        annotations: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not splits:
            raise ValueError(f"series {series_id!r} has no splits")
        if not isinstance(series_id, str):
            raise TypeError("series_id must be a string")
        if not series_id:
            raise ValueError("series_id must be non-empty")
        if series_id in self._series_ids:
            raise ValueError(f"duplicate series_id: {series_id!r}")
        if any(not isinstance(name, str) for name in splits):
            raise TypeError("split names must be strings")
        unknown = set(splits) - SPLIT_NAMES
        if unknown:
            raise ValueError(f"unknown split names: {sorted(unknown)}")

        canonical: list[str] | None = None
        buffers: dict[str, _SplitBuffer] = {}
        for split_name, split in splits.items():
            if not isinstance(split, SplitData):
                raise TypeError(f"split {split_name!r} must be SplitData")
            if not isinstance(split.timestamp_kind, str) or not split.timestamp_kind:
                raise ValueError(
                    f"timestamp_kind for split {split_name!r} must be non-empty"
                )
            matrix, names = _feature_matrix(split.values, split.feature_names)
            if canonical is None:
                canonical = names
            elif names != canonical:
                raise ValueError(f"feature names differ between splits for {series_id}")
            labels = _label_vector(split.labels, len(matrix))
            if (
                self._task == "anomaly_detection"
                and labels is not None
                and not np.isin(labels, [0, 1]).all()
            ):
                raise ValueError(
                    f"anomaly labels for {series_id!r}/{split_name} must be binary"
                )
            (
                packed_timestamps,
                original_timestamps,
                encoded_timestamp_kind,
                timestamp_encoding,
            ) = _encode_timestamps(split.timestamps, len(matrix))
            buffers[split_name] = _SplitBuffer(
                matrix=matrix,
                names=names,
                labels=labels,
                packed_timestamps=packed_timestamps,
                original_timestamps=(
                    original_timestamps
                    if encoded_timestamp_kind == "offset" or labels is not None
                    else None
                ),
                encoded_timestamp_kind=encoded_timestamp_kind,
                source_timestamp_kind=split.timestamp_kind,
                timestamp_encoding=timestamp_encoding,
            )

        entry = _SeriesEntry(
            series_id=series_id,
            splits=buffers,
            source_files=[
                source_fingerprint(path, self._project_root) for path in source_files
            ],
            source_metadata=jsonable(dict(source_metadata or {})),
            annotations=[jsonable(dict(item)) for item in (annotations or [])],
        )
        self._entries.append(entry)
        self._series_ids.add(series_id)
        return {
            "series_id": entry.series_id,
            "splits": {name: len(buffer.matrix) for name, buffer in buffers.items()},
        }

    def finalize(
        self, *, source_metadata: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Pack every buffered (split, feature-schema) group into one package.

        Dataset-level ``source_metadata`` rides in each package manifest;
        everything else about the dataset is derivable from the packages
        themselves, so no sidecar files are written next to them.
        """
        if not self._entries:
            raise ValueError(f"dataset {self._dataset!r} has no series")
        self._dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_source_metadata = jsonable(dict(source_metadata or {}))

        group_key = tuple[tuple[str, ...], str, bool, bool]
        groups: dict[str, dict[group_key, list[_SeriesEntry]]] = {}
        for entry in self._entries:
            for split_name, buffer in entry.splits.items():
                groups.setdefault(split_name, {}).setdefault(
                    (
                        tuple(buffer.names),
                        buffer.matrix.dtype.str,
                        buffer.labels is not None,
                        buffer.encoded_timestamp_kind == "offset",
                    ),
                    [],
                ).append(entry)

        for split_name in sorted(groups):
            schema_groups = groups[split_name]
            ordered = sorted(schema_groups)
            for schema_index, key in enumerate(ordered):
                names = key[0]
                suffix = "" if len(ordered) == 1 else f"__schema{schema_index}"
                self._pack_group(
                    split_name,
                    f"{split_name}{suffix}",
                    list(names),
                    schema_groups[key],
                    dataset_source_metadata,
                )
        return {"num_series": len(self._entries)}

    def _pack_group(
        self,
        split_name: str,
        package_name: str,
        names: list[str],
        entries: list[_SeriesEntry],
        dataset_source_metadata: dict[str, Any],
    ) -> None:
        prepared = [(entry, entry.splits[split_name]) for entry in entries]
        kinds = sorted({buffer.encoded_timestamp_kind for _, buffer in prepared})
        # Original timestamps stay recoverable whenever any series in the
        # package had to fall back to offsets.
        needs_original = "offset" in kinds
        common_dtype = prepared[0][1].matrix.dtype

        items: list[SeriesItem] = []
        for entry, buffer in prepared:
            columns: dict[str, Any] = {}
            if buffer.labels is not None:
                columns["label"] = buffer.labels
            if needs_original:
                if buffer.original_timestamps is None:
                    raise AssertionError("offset timestamps lost their source values")
                columns["timestamp"] = [
                    str(value) for value in buffer.original_timestamps.tolist()
                ]
            if columns:
                frame = pd.DataFrame(
                    {
                        name: pd.Series(values).reset_index(drop=True)
                        for name, values in columns.items()
                    }
                )
            else:
                frame = pd.DataFrame(index=pd.RangeIndex(len(buffer.packed_timestamps)))
            source_events = [
                item
                for item in entry.annotations
                if item.get("split") in (None, split_name, "cross_split")
            ]
            items.append(
                SeriesItem(
                    entry.series_id,
                    buffer.matrix.astype(common_dtype, copy=False),
                    buffer.packed_timestamps,
                    frame,
                    meta={
                        "source": self._source,
                        "dataset": self._dataset,
                        "series_id": entry.series_id,
                        "task": self._task,
                        "split": split_name,
                        "rows": int(len(buffer.packed_timestamps)),
                        "timestamp_kind": buffer.encoded_timestamp_kind,
                        "source_timestamp_kind": buffer.source_timestamp_kind,
                        "timestamp_encoding": buffer.timestamp_encoding,
                        "source_files": entry.source_files,
                        "source_metadata": entry.source_metadata,
                        "annotations": {
                            "source_events": source_events,
                            "derived_label_runs": label_runs(
                                buffer.labels,
                                decode_source_timestamps(
                                    buffer.packed_timestamps,
                                    frame,
                                    buffer.timestamp_encoding,
                                ),
                                split_name,
                            ),
                        },
                    },
                )
            )

        features = pd.DataFrame(
            {"dtype": [str(common_dtype)] * len(names)}, index=names
        )
        manifest = {
            "source": self._source,
            "dataset": self._dataset,
            "task": self._task,
            "split": split_name,
        }
        if dataset_source_metadata:
            manifest["source_metadata"] = dataset_source_metadata
        sdata = SeriesData.from_items(items, features=features, manifest=manifest)
        target = self._dataset_dir / f"{safe_name(package_name)}.zarr.zip"
        _write_package(sdata, target)
