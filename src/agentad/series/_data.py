"""Compact packed storage for variable-length series.

The storage contract is deliberately small:

- ``data`` is a two-dimensional numeric or boolean NumPy array;
- ``timestamps`` is a one-dimensional ``np.int64`` array whose values are
  opaque to this module;
- ``features`` and ``labels`` are pandas DataFrames with unique columns;
- ``offsets`` positionally align every series across the packed containers;
- series ids are unique strings.

The NumPy containers are write-through: a :class:`SeriesItem` holds plain
slices of the packed arrays, so writes through an item land in the parent
``SeriesData`` immediately.  ``pandas.DataFrame`` has no stable slice
write-through contract, which makes labels the one exception: live item frames
are weakly tracked against a compact baseline, and assigning ``item.labels``
explicitly installs an override.  :meth:`SeriesData.materialize` resolves
changed label frames into a packed snapshot.  ``meta`` is
write-through as well: assigning ``item.meta`` replaces the entry in the
parent's meta storage.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import weakref
from collections.abc import Iterable, Iterator, Mapping
from copy import deepcopy
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, DTypeLike

__all__ = ["SeriesData", "SeriesItem"]


_INT64 = np.dtype(np.int64)


def _pickle_signature(frame: pd.DataFrame) -> bytes | None:
    try:
        values = frame.to_numpy(copy=True)
        payload = pickle.dumps(values, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as error:
        if isinstance(error, MemoryError):
            raise
        # Explicit ``item.labels = frame`` overrides remain supported for
        # values that can be neither hashed nor snapshotted.
        return None
    digest = hashlib.blake2b(digest_size=16)
    digest.update(payload)
    _update_schema_digest(digest, frame)
    return digest.digest()


def _pickle_schema(value: Any) -> bytes:
    try:
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as error:
        if isinstance(error, MemoryError):
            raise
        return repr(value).encode("utf-8", errors="backslashreplace")


def _attrs_signature(attrs: dict[str, Any]) -> bytes:
    try:
        return json.dumps(
            attrs,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return _pickle_schema(attrs)


def _update_schema_digest(digest: "hashlib.blake2b", frame: pd.DataFrame) -> None:
    digest.update(_pickle_schema((frame.columns, tuple(frame.dtypes))))
    digest.update(_attrs_signature(frame.attrs))
    digest.update(str(frame.shape).encode("ascii"))


def _frame_signature(frame: pd.DataFrame) -> bytes | None:
    """Return a compact value/schema fingerprint for a tracked label frame.

    Numeric and extension frames fingerprint through a single C-level pass.
    Object frames use serialization because their Python scalar types are
    part of the label value contract.
    """
    # pandas hashes object scalars by their value representation, so distinct
    # Python values such as ``b"a"`` and ``"a"`` can collide deliberately.
    # Pickle preserves those scalar types and keeps dirty-label detection exact.
    if any(pd.api.types.is_object_dtype(dtype) for dtype in frame.dtypes):
        return _pickle_signature(frame)
    try:
        row_hashes = pd.util.hash_pandas_object(
            frame, index=False, categorize=True
        ).to_numpy(dtype=np.uint64, copy=False)
    except Exception as error:
        if isinstance(error, MemoryError):
            raise
        return _pickle_signature(frame)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(row_hashes.tobytes())
    _update_schema_digest(digest, frame)
    return digest.digest()


def _frames_equal_positionally(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return (
        left.columns.identical(right.columns)
        and _pickle_schema(tuple(left.dtypes)) == _pickle_schema(tuple(right.dtypes))
        and _attrs_signature(left.attrs) == _attrs_signature(right.attrs)
        and left.reset_index(drop=True).equals(right.reset_index(drop=True))
    )


def _validate_json_basic(value: Any, where: str) -> None:
    """Meta containers persist to JSON, so only values that survive a JSON
    round-trip unchanged are accepted."""
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where} contains a non-finite float")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_basic(item, where)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{where} keys must be strings, got {type(key).__name__}"
                )
            _validate_json_basic(item, where)
        return
    raise TypeError(
        f"{where} values must be JSON basic types "
        f"(None, bool, int, float, str, list, dict), got {type(value).__name__}"
    )


def _copy_json_mapping(value: Mapping[str, Any], where: str) -> dict[str, Any]:
    unpacked = dict(value)
    _validate_json_basic(unpacked, where)
    return deepcopy(unpacked)


def _validate_data_dtype(dtype: np.dtype[Any], where: str) -> None:
    if not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
        raise TypeError(f"{where} must have a numeric or boolean dtype, got {dtype}")


def _native_contiguous_data(array: np.ndarray, where: str) -> np.ndarray:
    _validate_data_dtype(array.dtype, where)
    if not array.dtype.isnative:
        array = array.astype(array.dtype.newbyteorder("="), copy=True)
    return np.ascontiguousarray(array)


def _validate_unique_columns(frame: pd.DataFrame, where: str) -> None:
    if not frame.columns.is_unique:
        raise ValueError(f"{where} columns must be unique")


class SeriesData:
    """Positionally aligned packed storage for multiple series.

    The constructor adopts native-endian contiguous ``data`` and contiguous
    ``timestamps`` inputs without copying them.  Callers should therefore
    treat ownership as transferred.  Everything else is read-only by
    convention apart from the write-through channels described in the module
    docstring.

    Items and their label frames are weakly cached: repeated access reuses live
    objects without retaining one Python object per series after iteration.  A
    persistence layer should call :meth:`materialize` and write the returned
    snapshot so explicit item label overrides are folded back in.
    """

    __slots__ = (
        "__weakref__",
        "_data",
        "_features",
        "_id_to_index",
        "_ids",
        "_item_label_baselines",
        "_item_label_overrides",
        "_item_labels",
        "_items",
        "_labels",
        "_manifest",
        "_metas",
        "_offsets",
        "_timestamps",
    )

    def __init__(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        data: ArrayLike,
        timestamps: ArrayLike,
        offsets: ArrayLike,
        ids: Iterable[str],
        metas: Iterable[Mapping[str, Any]] | None = None,
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        packed_data = np.asarray(data)
        if packed_data.ndim == 1:
            packed_data = packed_data.reshape(-1, 1)
        if packed_data.ndim != 2:
            raise ValueError(f"data must be 1-D or 2-D, got shape {packed_data.shape}")
        packed_data = _native_contiguous_data(packed_data, "data")

        packed_timestamps = np.asarray(timestamps)
        if packed_timestamps.ndim != 1:
            raise ValueError(
                f"timestamps must be 1-D, got shape {packed_timestamps.shape}"
            )
        if packed_timestamps.dtype != _INT64:
            raise TypeError(
                f"timestamps must have dtype int64, got {packed_timestamps.dtype}"
            )
        packed_timestamps = np.ascontiguousarray(packed_timestamps)

        raw_offsets = np.asarray(offsets)
        if raw_offsets.ndim != 1:
            raise ValueError(f"offsets must be 1-D, got shape {raw_offsets.shape}")
        if raw_offsets.dtype != _INT64:
            raise TypeError(f"offsets must have dtype int64, got {raw_offsets.dtype}")
        # Offsets define the structure rather than values, so SeriesData owns a
        # small independent copy even when the packed value arrays are adopted.
        packed_offsets = np.array(raw_offsets, dtype=np.int64, copy=True, order="C")
        packed_offsets.flags.writeable = False

        if isinstance(ids, str):
            raise TypeError("ids must be an iterable of strings, not one string")
        packed_ids = tuple(ids)
        for index, series_id in enumerate(packed_ids):
            if not isinstance(series_id, str):
                raise TypeError(
                    f"ids[{index}] must be a string, got {type(series_id).__name__}"
                )
        if len(set(packed_ids)) != len(packed_ids):
            raise ValueError("series ids must be unique")

        n_points, n_features = packed_data.shape
        n_series = len(packed_ids)
        if len(packed_timestamps) != n_points:
            raise ValueError(
                "data and timestamps must have the same number of points: "
                f"{n_points} != {len(packed_timestamps)}"
            )
        if len(packed_offsets) != n_series + 1:
            raise ValueError(
                f"offsets has length {len(packed_offsets)}, expected {n_series + 1}"
            )
        if packed_offsets[0] != 0:
            raise ValueError("offsets must start at 0")
        if np.any(packed_offsets[1:] < packed_offsets[:-1]):
            raise ValueError("offsets must be non-decreasing")
        if packed_offsets[-1] != n_points:
            raise ValueError(
                f"offsets must end at the point count {n_points}, "
                f"got {int(packed_offsets[-1])}"
            )
        if manifest is not None:
            if not isinstance(manifest, Mapping):
                raise TypeError("manifest must be a mapping or None")
            packed_manifest = _copy_json_mapping(manifest, "manifest")
        else:
            packed_manifest = {}

        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        _validate_unique_columns(features, "features")
        if len(features) != n_features:
            raise ValueError(
                f"features has {len(features)} rows, expected {n_features}"
            )
        self._features = features.copy(deep=False)

        if not isinstance(labels, pd.DataFrame):
            raise TypeError("labels must be a pandas DataFrame")
        _validate_unique_columns(labels, "labels")
        if len(labels) != n_points:
            raise ValueError(f"labels has {len(labels)} rows, expected {n_points}")
        packed_labels = labels.copy(deep=False)
        # Alignment is positional.  A global RangeIndex exposes the corresponding
        # packed row number without making it part of the stored data.
        packed_labels.index = pd.RangeIndex(n_points)
        self._labels = packed_labels

        if metas is None:
            packed_metas: list[dict[str, Any]] = [{} for _ in range(n_series)]
        else:
            if isinstance(metas, Mapping):
                raise TypeError(
                    "metas must be an iterable of mappings, not one mapping"
                )
            packed_metas = []
            for index, meta in enumerate(metas):
                if not isinstance(meta, Mapping):
                    raise TypeError(
                        f"metas[{index}] must be a mapping, got {type(meta).__name__}"
                    )
                copied = _copy_json_mapping(meta, f"metas[{index}]")
                packed_metas.append(copied)
            if len(packed_metas) != n_series:
                raise ValueError(
                    f"metas has {len(packed_metas)} entries, expected {n_series}"
                )

        self._data = packed_data
        self._timestamps = packed_timestamps
        self._offsets = packed_offsets
        self._ids = packed_ids
        self._id_to_index = {
            series_id: index for index, series_id in enumerate(packed_ids)
        }
        self._metas = packed_metas
        self._manifest = packed_manifest
        self._items: weakref.WeakValueDictionary[int, SeriesItem] = (
            weakref.WeakValueDictionary()
        )
        self._item_labels: weakref.WeakValueDictionary[int, pd.DataFrame] = (
            weakref.WeakValueDictionary()
        )
        self._item_label_baselines: dict[
            int, tuple[weakref.ReferenceType[pd.DataFrame], bytes | None]
        ] = {}
        self._item_label_overrides: dict[int, pd.DataFrame] = {}

    @classmethod
    def empty(
        cls,
        *,
        features: pd.DataFrame,
        labels: pd.DataFrame | None = None,
        data_dtype: DTypeLike = np.float64,
        manifest: Mapping[str, Any] | None = None,
    ) -> SeriesData:
        """Create an empty collection while preserving table schemas.

        ``labels`` is an optional zero-row schema frame. Its columns, dtypes
        and attrs are retained, which makes this constructor preferable to
        inferring an empty schema from an empty item iterable.
        """
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        if labels is None:
            labels = pd.DataFrame(index=pd.RangeIndex(0))
        elif not isinstance(labels, pd.DataFrame):
            raise TypeError("labels must be a pandas DataFrame or None")
        elif len(labels) != 0:
            raise ValueError("empty labels schema must have zero rows")
        dtype = np.dtype(data_dtype)
        return cls(
            features=features.copy(deep=True),
            labels=labels.copy(deep=True),
            data=np.empty((0, len(features)), dtype=dtype),
            timestamps=np.empty(0, dtype=np.int64),
            offsets=np.array([0], dtype=np.int64),
            ids=(),
            metas=(),
            manifest=manifest,
        )

    @classmethod
    def from_items(
        cls,
        items: Iterable[SeriesItem],
        *,
        features: pd.DataFrame,
        manifest: Mapping[str, Any] | None = None,
    ) -> SeriesData:
        """Pack standalone items or existing views into a new collection.

        Every packed container is freshly constructed: ``data`` and
        ``timestamps`` are preallocated and filled from the items, ``labels``
        are concatenated into a new frame, and ``features`` is deep-copied.
        Item storage is never adopted, so later changes to the input items
        cannot leak into the collection.  All items must agree on the feature
        count, the data dtype and the label columns.
        """
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        _validate_unique_columns(features, "features")
        materialized = tuple(items)
        if not materialized:
            return cls.empty(
                features=features,
                manifest=manifest,
            )

        for index, item in enumerate(materialized):
            if not isinstance(item, SeriesItem):
                raise TypeError(
                    f"items[{index}] must be a SeriesItem, got {type(item).__name__}"
                )
            item._validate_current()

        first = materialized[0]
        n_features = first.n_features
        if len(features) != n_features:
            raise ValueError(
                f"features has {len(features)} rows, expected {n_features}"
            )
        data_dtype = first.data.dtype
        label_frames = tuple(item.labels for item in materialized)
        label_columns = label_frames[0].columns.copy()
        label_attrs = _attrs_signature(label_frames[0].attrs)
        for item, labels in zip(materialized, label_frames):
            if item.n_features != n_features:
                raise ValueError("all items must have the same feature count")
            if item.data.dtype != data_dtype:
                raise ValueError("all items must have the same data dtype")
            if not labels.columns.identical(label_columns):
                raise ValueError("all items must have the same label columns")
            if _attrs_signature(labels.attrs) != label_attrs:
                raise ValueError("all items must have the same label attrs")

        item_ids = tuple(item.id for item in materialized)
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("series ids must be unique")

        lengths = np.fromiter(
            (len(item) for item in materialized),
            dtype=np.int64,
            count=len(materialized),
        )
        offsets = np.zeros(len(materialized) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])

        # Copy semantics: preallocate the packed containers and fill them from
        # the items instead of adopting item-owned arrays.
        packed_data = np.empty((int(offsets[-1]), n_features), dtype=data_dtype)
        packed_timestamps = np.empty(int(offsets[-1]), dtype=np.int64)
        for item, lo, hi in zip(materialized, offsets[:-1], offsets[1:]):
            lo = int(lo)
            hi = int(hi)
            packed_data[lo:hi] = item.data
            packed_timestamps[lo:hi] = item.timestamps

        return cls(
            features=features.copy(deep=True),
            labels=pd.concat(label_frames, ignore_index=True),
            data=packed_data,
            timestamps=packed_timestamps,
            offsets=offsets,
            ids=item_ids,
            metas=(item.meta for item in materialized),
            manifest=manifest,
        )

    @property
    def features(self) -> pd.DataFrame:
        """Feature annotations; the index identifies feature rows."""
        return self._features

    @property
    def labels(self) -> pd.DataFrame:
        """Packed point labels with global packed row numbers as the index."""
        return self._labels

    @property
    def data(self) -> np.ndarray:
        """Packed ``(n_points, n_features)`` values."""
        return self._data

    @property
    def timestamps(self) -> np.ndarray:
        """Packed opaque int64 coordinates."""
        return self._timestamps

    @property
    def offsets(self) -> np.ndarray:
        """Read-only series boundary offsets."""
        return self._offsets

    @property
    def ids(self) -> tuple[str, ...]:
        return self._ids

    @property
    def metas(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._metas)

    @property
    def manifest(self) -> Mapping[str, Any]:
        """A read-only snapshot of collection-level JSON metadata."""
        return MappingProxyType(deepcopy(self._manifest))

    @property
    def total_points(self) -> int:
        return self._data.shape[0]

    @property
    def n_features(self) -> int:
        return self._data.shape[1]

    @property
    def live_items(self) -> int:
        """Number of item objects currently retained by callers."""
        return len(self._items)

    @property
    def tracked_label_frames(self) -> int:
        """Number of per-item DataFrames currently tracked by the collection."""
        return len(self._item_labels)

    def __len__(self) -> int:
        return len(self._ids)

    def __iter__(self) -> Iterator[SeriesItem]:
        return (self[index] for index in range(len(self)))

    def keys(self) -> Iterator[str]:
        """Iterate over series ids in packed order."""
        return iter(self._ids)

    def values(self) -> Iterator[SeriesItem]:
        """Iterate over series items in packed order."""
        return iter(self)

    def items(self) -> Iterator[tuple[str, SeriesItem]]:
        """Iterate over ``(series_id, item)`` pairs in packed order."""
        return ((series_id, self[index]) for index, series_id in enumerate(self._ids))

    def __contains__(self, series_id: object) -> bool:
        return isinstance(series_id, str) and series_id in self._id_to_index

    def get(self, key: int | str, default: Any = None) -> Any:
        """Return an item, or ``default`` when an id/index is absent."""
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    def index(self, key: int | str) -> int:
        """Resolve a series id or position to its normalized position."""
        return self._resolve_index(key)

    def select(self, keys: Iterable[int | str]) -> SeriesData:
        """Return an independent packed subset in the requested order.

        Keys must be unique. An empty selection retains the feature and label
        schemas, data dtype and manifest of the source collection.
        """
        if isinstance(keys, (str, bytes)):
            raise TypeError("keys must be an iterable of series keys")
        order = tuple(self._resolve_index(key) for key in keys)
        if len(set(order)) != len(order):
            raise ValueError("keys must identify each selected series once")
        return self._take_indices(order)

    def _resolve_index(self, key: int | str) -> int:
        if isinstance(key, str):
            try:
                return self._id_to_index[key]
            except KeyError:
                raise KeyError(f"unknown series id: {key!r}") from None
        if isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_)):
            index = int(key)
            if index < 0:
                index += len(self)
            if 0 <= index < len(self):
                return index
            raise IndexError(f"series index out of range: {key}")
        raise TypeError(f"series key must be int or str, got {type(key).__name__}")

    def _bounds(self, index: int) -> tuple[int, int]:
        return int(self._offsets[index]), int(self._offsets[index + 1])

    def _validate_table_alignment(self, *, check_offset_order: bool = False) -> None:
        if self._data.ndim != 2:
            raise ValueError(f"data must remain 2-D, got shape {self._data.shape}")
        _validate_data_dtype(self._data.dtype, "data")
        if not self._data.dtype.isnative:
            raise TypeError(f"data must have native byte order, got {self._data.dtype}")
        if self._timestamps.ndim != 1:
            raise ValueError(
                f"timestamps must remain 1-D, got shape {self._timestamps.shape}"
            )
        if self._timestamps.dtype != _INT64:
            raise TypeError(
                f"timestamps must have dtype int64, got {self._timestamps.dtype}"
            )
        if len(self._timestamps) != self.total_points:
            raise ValueError(
                "data and timestamps must have the same number of points: "
                f"{self.total_points} != {len(self._timestamps)}"
            )
        if self._offsets.ndim != 1 or self._offsets.dtype != _INT64:
            raise TypeError("offsets must remain a 1-D int64 array")
        if len(self._offsets) != len(self) + 1:
            raise ValueError(
                f"offsets has length {len(self._offsets)}, expected {len(self) + 1}"
            )
        if self._offsets[0] != 0:
            raise ValueError("offsets must start at 0")
        if self._offsets[-1] != self.total_points:
            raise ValueError(
                f"offsets must end at the point count {self.total_points}, "
                f"got {int(self._offsets[-1])}"
            )
        if check_offset_order and np.any(self._offsets[1:] < self._offsets[:-1]):
            raise ValueError("offsets must be non-decreasing")
        _validate_unique_columns(self._features, "features")
        _validate_unique_columns(self._labels, "labels")
        if len(self._features) != self.n_features:
            raise ValueError(
                f"features has {len(self._features)} rows, expected {self.n_features}"
            )
        if len(self._labels) != self.total_points:
            raise ValueError(
                f"labels has {len(self._labels)} rows, expected {self.total_points}"
            )

    def _register_item_labels(
        self,
        index: int,
        labels: pd.DataFrame,
        *,
        override: bool,
    ) -> None:
        self._item_labels[index] = labels
        if override:
            # Explicit assignments belong to the collection, so retain their
            # frames independently of the temporary SeriesItem used to assign
            # them. Clean views remain weakly held.
            self._item_label_overrides[index] = labels
            self._item_label_baselines.pop(index, None)
        else:
            self._item_label_overrides.pop(index, None)
            self._set_item_label_baseline(index, labels)

    def _set_item_label_baseline(self, index: int, labels: pd.DataFrame) -> None:
        owner_ref = weakref.ref(self)

        def remove_baseline(
            frame_ref: weakref.ReferenceType[pd.DataFrame],
        ) -> None:
            owner = owner_ref()
            if owner is None:
                return
            tracked = owner._item_label_baselines.get(index)
            if tracked is not None and tracked[0] is frame_ref:
                owner._item_label_baselines.pop(index, None)

        frame_ref = weakref.ref(labels, remove_baseline)
        self._item_label_baselines[index] = (
            frame_ref,
            _frame_signature(labels),
        )

    def _item_label_baseline(self, index: int, labels: pd.DataFrame) -> bytes | None:
        tracked = self._item_label_baselines.get(index)
        if tracked is None or tracked[0]() is not labels:
            return None
        return tracked[1]

    def _refresh_item_labels(
        self, index: int, labels: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, bool]:
        """Refresh an unchanged cached frame after parent labels have moved.

        pandas can replace a DataFrame's backing blocks without updating an
        existing shallow slice.  A locally edited frame is retained; a frame
        that still matches its baseline is rebound to the current packed rows.
        """
        override = self._item_label_overrides.get(index)
        if override is not None:
            return override, True

        lo, hi = self._bounds(index)
        base = self._labels.iloc[lo:hi]
        if labels is None:
            labels = self._item_labels.get(index)
            if labels is None:
                labels = base.copy(deep=False)
                self._register_item_labels(index, labels, override=False)
                return labels, False
        if _frames_equal_positionally(labels, base):
            self._set_item_label_baseline(index, labels)
            return labels, False

        baseline = self._item_label_baseline(index, labels)
        if baseline is not None and _frame_signature(labels) != baseline:
            return labels, True

        refreshed = base.copy(deep=False)
        self._register_item_labels(index, refreshed, override=False)
        return refreshed, False

    def __getitem__(self, key: int | str) -> SeriesItem:
        self._validate_table_alignment()
        index = self._resolve_index(key)
        item = self._items.get(index)
        if item is None:
            item = SeriesItem._from_parent(self, index)
            self._items[index] = item
        return item

    def materialize(self) -> SeriesData:
        """Return the packed snapshot that should be persisted.

        Clean items read through to the packed containers.  Explicit label
        overrides are revalidated against the current packed schema and win
        over the corresponding packed rows.  The snapshot shares the NumPy
        arrays; the original collection and its live items remain valid.
        """
        self._validate_table_alignment(check_offset_order=True)
        _validate_json_basic(self._manifest, "manifest")

        # Meta dictionaries are intentionally mutable, so persistence is the
        # transaction boundary at which their JSON contract is revalidated.
        for index, meta in enumerate(self._metas):
            _validate_json_basic(meta, f"metas[{index}]")

        if not self._item_labels and not self._item_label_overrides:
            return self

        candidate_indices = set(self._item_label_overrides)
        candidate_indices.update(tuple(self._item_labels.keys()))
        replacements: dict[int, pd.DataFrame] = {}
        for index in sorted(candidate_indices):
            lo, hi = self._bounds(index)
            base = self._labels.iloc[lo:hi]
            labels = self._item_label_overrides.get(index)
            explicit_override = labels is not None
            if labels is None:
                labels = self._item_labels.get(index)
            baseline = (
                None if labels is None else self._item_label_baseline(index, labels)
            )
            frame_changed = (
                labels is not None
                and baseline is not None
                and _frame_signature(labels) != baseline
            )
            if labels is not None and (explicit_override or frame_changed):
                if _attrs_signature(labels.attrs) != _attrs_signature(
                    self._labels.attrs
                ):
                    raise ValueError(
                        f"series {self._ids[index]!r}: label attrs do not "
                        "match the packed schema"
                    )
                if not labels.columns.identical(self._labels.columns):
                    raise ValueError(
                        f"series {self._ids[index]!r}: label columns do not "
                        "match the packed schema"
                    )
                if len(labels) != hi - lo:
                    raise ValueError(
                        f"series {self._ids[index]!r}: labels has {len(labels)} "
                        f"rows, expected {hi - lo}"
                    )
                # The per-item index is presentation metadata; persistence
                # aligns rows positionally and rebuilds a global RangeIndex.
                if not _frames_equal_positionally(labels, base):
                    replacements[index] = labels
        if not replacements:
            return self

        sources: list[pd.DataFrame] = []
        cursor = 0
        for index, labels in replacements.items():
            lo, hi = self._bounds(index)
            if cursor < lo:
                sources.append(self._labels.iloc[cursor:lo])
            sources.append(labels)
            cursor = hi
        if cursor < self.total_points:
            sources.append(self._labels.iloc[cursor:])
        return SeriesData(
            features=self._features,
            labels=pd.concat(sources, ignore_index=True),
            data=self._data,
            timestamps=self._timestamps,
            offsets=self._offsets,
            ids=self._ids,
            metas=self._metas,
            manifest=self._manifest,
        )

    def _take_indices(self, order: tuple[int, ...]) -> SeriesData:
        base = self.materialize()
        if not order:
            return SeriesData.empty(
                features=base._features,
                labels=base._labels.iloc[:0],
                data_dtype=base._data.dtype,
                manifest=base._manifest,
            )

        lengths = np.fromiter(
            (int(base._offsets[index + 1] - base._offsets[index]) for index in order),
            dtype=np.int64,
            count=len(order),
        )
        offsets = np.zeros(len(order) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        packed_data = np.empty(
            (int(offsets[-1]), base.n_features), dtype=base._data.dtype
        )
        packed_timestamps = np.empty(int(offsets[-1]), dtype=np.int64)
        label_parts: list[pd.DataFrame] = []

        position = 0
        while position < len(order):
            run_end = position + 1
            while run_end < len(order) and order[run_end] == order[run_end - 1] + 1:
                run_end += 1
            source_lo = int(base._offsets[order[position]])
            source_hi = int(base._offsets[order[run_end - 1] + 1])
            destination_lo = int(offsets[position])
            destination_hi = int(offsets[run_end])
            packed_data[destination_lo:destination_hi] = base._data[source_lo:source_hi]
            packed_timestamps[destination_lo:destination_hi] = base._timestamps[
                source_lo:source_hi
            ]
            label_parts.append(base._labels.iloc[source_lo:source_hi])
            position = run_end

        return SeriesData(
            features=base._features.copy(deep=True),
            labels=pd.concat(label_parts, ignore_index=True),
            data=packed_data,
            timestamps=packed_timestamps,
            offsets=offsets,
            ids=[base._ids[index] for index in order],
            metas=[base._metas[index] for index in order],
            manifest=base._manifest,
        )

    def reorder(self, indices: Iterable[int | str]) -> SeriesData:
        """Return the collection with the series packed in a new order.

        ``indices`` is a complete permutation of series keys (positions or
        ids); ``indices[k]`` names the series that ends up at position ``k``.
        Reordering triggers :meth:`materialize` first, so dirty item labels
        are folded into the packed storage before the new packing is built
        from it.  The result does not share NumPy arrays with ``self``.
        """
        if isinstance(indices, (str, bytes)):
            raise TypeError("indices must be an iterable of series keys")
        order = tuple(self._resolve_index(key) for key in indices)
        if len(order) != len(self) or set(order) != set(range(len(self))):
            raise ValueError("indices must contain every series exactly once")
        if not order:
            return self
        return self._take_indices(order)

    def __repr__(self) -> str:
        return (
            f"SeriesData(series={len(self)}, points={self.total_points}, "
            f"features={self.n_features}, label_columns={self._labels.shape[1]})"
        )


class SeriesItem:
    """One series, either standalone or an item of a parent ``SeriesData``.

    ``data`` and ``timestamps`` have no setter: for a parented item they are
    slices of the packed containers and writes go straight through to the
    parent.  ``labels`` is the one assignable frame, because pandas has no
    slice write-through; assigning it marks the item dirty for
    :meth:`SeriesData.materialize`, while a clean item reads straight through
    to the packed labels.  ``meta`` writes through to the parent's meta
    storage on assignment.
    """

    __slots__ = (
        "__weakref__",
        "_data",
        "_id",
        "_index",
        "_labels",
        "_labels_dirty",
        "_meta",
        "_owner",
        "_timestamps",
    )

    def __init__(
        self,
        series_id: str,
        data: ArrayLike,
        timestamps: ArrayLike,
        labels: pd.DataFrame,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(series_id, str):
            raise TypeError("series_id must be a string")
        if not isinstance(labels, pd.DataFrame):
            raise TypeError("labels must be a pandas DataFrame")
        if meta is not None:
            if not isinstance(meta, Mapping):
                raise TypeError("meta must be a mapping or None")
            packed_meta = _copy_json_mapping(meta, "item meta")
        else:
            packed_meta = {}

        array = np.asarray(data)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError(f"data must be 1-D or 2-D, got shape {array.shape}")
        self._data = _native_contiguous_data(array, "item data")

        stamps = np.asarray(timestamps)
        if stamps.ndim != 1:
            raise ValueError(f"timestamps must be 1-D, got shape {stamps.shape}")
        if stamps.dtype != _INT64:
            raise TypeError(f"timestamps must have dtype int64, got {stamps.dtype}")
        self._timestamps = np.ascontiguousarray(stamps)

        self._owner: SeriesData | None = None
        self._index: int | None = None
        self._id = series_id
        self._labels = labels
        self._meta = packed_meta
        self._labels_dirty = False
        self._validate_current()

    @classmethod
    def _from_parent(
        cls,
        owner: SeriesData,
        index: int,
    ) -> SeriesItem:
        item = cls.__new__(cls)
        item._owner = owner
        item._index = index
        item._id = owner._ids[index]
        lo, hi = owner._bounds(index)
        item._data = owner._data[lo:hi]
        item._timestamps = owner._timestamps[lo:hi]
        item._labels = None
        item._labels_dirty = index in owner._item_label_overrides
        return item

    def _validate_current(self) -> None:
        if self._data.ndim != 2:
            raise ValueError(f"item data must be 2-D, got shape {self._data.shape}")
        _validate_data_dtype(self._data.dtype, "item data")
        if not self._data.dtype.isnative:
            raise TypeError(
                f"item data must have native byte order, got {self._data.dtype}"
            )
        if self._timestamps.ndim != 1 or self._timestamps.dtype != _INT64:
            raise TypeError("item timestamps must be a 1-D int64 array")
        length = self._data.shape[0]
        if len(self._timestamps) != length:
            raise ValueError(
                "item data and timestamps must have equal lengths: "
                f"{length}, {len(self._timestamps)}"
            )
        if self._labels is None:
            if self._owner is None:
                raise TypeError("standalone item labels must be a pandas DataFrame")
            return
        if not isinstance(self._labels, pd.DataFrame):
            raise TypeError("item labels must be a pandas DataFrame")
        _validate_unique_columns(self._labels, "item labels")
        if len(self._labels) != length:
            raise ValueError(
                "item data, timestamps and labels must have equal lengths: "
                f"{length}, {len(self._timestamps)}, {len(self._labels)}"
            )

    @property
    def id(self) -> str:
        return self._id

    @property
    def sdata(self) -> SeriesData | None:
        """Parent collection, or ``None`` for a standalone item."""
        return self._owner

    @property
    def index(self) -> int | None:
        """Position in the parent collection, or ``None`` when standalone."""
        return self._index

    @property
    def row_slice(self) -> slice:
        """Rows occupied in the parent packed containers."""
        if self._owner is None or self._index is None:
            return slice(0, len(self))
        lo, hi = self._owner._bounds(self._index)
        return slice(lo, hi)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def timestamps(self) -> np.ndarray:
        return self._timestamps

    @property
    def n_features(self) -> int:
        return self._data.shape[1]

    @property
    def labels(self) -> pd.DataFrame:
        if self._owner is not None and self._index is not None:
            self._labels, self._labels_dirty = self._owner._refresh_item_labels(
                self._index, self._labels
            )
        if self._labels is None:
            raise RuntimeError("item labels are unavailable")
        return self._labels

    @labels.setter
    def labels(self, value: pd.DataFrame) -> None:
        if not isinstance(value, pd.DataFrame):
            raise TypeError("item labels must be a pandas DataFrame")
        _validate_unique_columns(value, "item labels")
        if len(value) != len(self):
            raise ValueError(f"item labels has {len(value)} rows, expected {len(self)}")
        if self._owner is not None and not value.columns.identical(
            self._owner._labels.columns
        ):
            raise ValueError("item label columns must match the packed schema")
        if self._owner is not None and _attrs_signature(
            value.attrs
        ) != _attrs_signature(self._owner._labels.attrs):
            raise ValueError("item label attrs must match the packed schema")
        self._labels = value
        self._labels_dirty = True
        if self._owner is not None and self._index is not None:
            self._owner._register_item_labels(self._index, value, override=True)

    @property
    def meta(self) -> dict[str, Any]:
        if self._owner is not None:
            assert self._index is not None
            return self._owner._metas[self._index]
        return self._meta

    @meta.setter
    def meta(self, value: Mapping[str, Any]) -> None:
        if not isinstance(value, Mapping):
            raise TypeError("item meta must be a mapping")
        copied = _copy_json_mapping(value, "item meta")
        if self._owner is not None:
            assert self._index is not None
            self._owner._metas[self._index] = copied
        else:
            self._meta = copied

    def detach_labels(self, *, deep: bool = True) -> pd.DataFrame:
        """Install and return an editable label copy tracked by the parent."""
        self.labels = self.labels.copy(deep=deep)
        return self._labels

    def copy(self) -> SeriesItem:
        """Return an independent standalone copy of this series."""
        labels = self.labels
        self._validate_current()
        return SeriesItem(
            self._id,
            self._data.copy(order="C"),
            self._timestamps.copy(),
            labels.copy(deep=True),
            meta=self.meta,
        )

    def __len__(self) -> int:
        return self._data.shape[0]

    def __repr__(self) -> str:
        suffix = ", labels-dirty" if self._labels_dirty else ""
        return (
            f"SeriesItem(id={self._id!r}, points={len(self)}, "
            f"features={self.n_features}, "
            f"label_columns={self.labels.shape[1]}{suffix})"
        )
