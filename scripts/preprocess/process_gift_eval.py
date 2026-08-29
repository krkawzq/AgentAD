"""Convert Hugging Face GIFT-Eval Arrow datasets to SeriesData packages."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

if __package__:
    from .common import (
        DatasetWriter,
        SplitData,
        jsonable,
        safe_name,
        source_fingerprint,
        staged_source_directory,
    )
else:
    from common import (
        DatasetWriter,
        SplitData,
        jsonable,
        safe_name,
        source_fingerprint,
        staged_source_directory,
    )


SOURCE = "gift-eval"


def arrow_rows(path: Path) -> Iterator[dict[str, Any]]:
    with pa.memory_map(str(path), "r") as source:
        with ipc.open_stream(source) as reader:
            for batch in reader:
                yield from batch.to_pylist()


def time_aligned_matrix(value: Any, rows: int, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 1:
        if len(array) != rows:
            raise ValueError(f"{name} length {len(array)} != target rows {rows}")
        array = array.reshape(rows, 1)
    elif array.ndim == 2:
        if array.shape[1] == rows:
            array = array.T
        elif array.shape[0] != rows:
            raise ValueError(f"{name} shape {array.shape} does not align to {rows}")
    else:
        raise ValueError(f"{name} must be 1D or 2D, got {array.shape}")
    if not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise TypeError(f"{name} must be numeric, got {array.dtype}")
    return array


def timestamps_from_start(start: Any, freq: str, rows: int) -> tuple[Any, str]:
    try:
        timestamps = pd.date_range(start=pd.Timestamp(start), periods=rows, freq=freq)
    except (OverflowError, ValueError, TypeError):
        return np.arange(rows, dtype=np.int64), "offset_with_start_and_frequency"
    return timestamps, "generated_from_start_and_frequency"


def gift_record(
    row: Mapping[str, Any], index: int
) -> tuple[str, SplitData, dict[str, Any]]:
    required = {"start", "freq", "target"}
    missing = required - set(row)
    if missing:
        raise ValueError(f"GIFT-Eval row is missing {sorted(missing)}")
    target_raw = np.asarray(row["target"])
    if target_raw.ndim == 1:
        rows = len(target_raw)
    elif target_raw.ndim == 2:
        rows = target_raw.shape[1]
    else:
        raise ValueError(f"invalid GIFT-Eval target shape: {target_raw.shape}")
    target = time_aligned_matrix(target_raw, rows, name="target")
    matrices = [target]
    feature_names = [f"target_{position}" for position in range(target.shape[1])]
    static_metadata: dict[str, Any] = {}
    for name, value in row.items():
        if name in {"item_id", "start", "freq", "target"}:
            continue
        if "dynamic" not in name:
            static_metadata[name] = jsonable(value)
            continue
        try:
            dynamic = time_aligned_matrix(value, rows, name=name)
        except (TypeError, ValueError):
            static_metadata[name] = jsonable(value)
            continue
        matrices.append(dynamic)
        feature_names.extend(
            f"{name}_{position}" for position in range(dynamic.shape[1])
        )
    values = np.concatenate(matrices, axis=1)
    freq = str(row["freq"])
    timestamps, timestamp_kind = timestamps_from_start(row["start"], freq, rows)
    series_id = str(row.get("item_id", index))
    if not series_id:
        raise ValueError("GIFT-Eval item_id must be non-empty")
    return (
        series_id,
        SplitData(
            values=values,
            timestamps=timestamps,
            feature_names=feature_names,
            timestamp_kind=timestamp_kind,
        ),
        {
            "item_id": series_id,
            "start": jsonable(row["start"]),
            "frequency": freq,
            "target_features": target.shape[1],
            "dynamic_covariate_features": values.shape[1] - target.shape[1],
            **static_metadata,
        },
    )


def dataset_name(input_root: Path, arrow_path: Path) -> tuple[str, str]:
    relative = arrow_path.parent.relative_to(input_root)
    source_name = relative.as_posix()
    safe = safe_name(source_name.replace("/", "__"))
    if not safe or safe in {".", ".."}:
        raise ValueError(f"invalid GIFT-Eval dataset path: {source_name!r}")
    return safe, source_name


def process_dataset(
    input_root: Path,
    arrow_paths: Path | Sequence[Path],
    source_dir: Path,
    project_root: Path,
) -> None:
    paths = [arrow_paths] if isinstance(arrow_paths, Path) else list(arrow_paths)
    if not paths:
        raise ValueError("GIFT-Eval dataset has no Arrow shards")
    parents = {path.parent for path in paths}
    if len(parents) != 1:
        raise ValueError("GIFT-Eval Arrow shards must share one dataset directory")
    paths.sort()
    dataset, source_name = dataset_name(input_root, paths[0])
    info_path = paths[0].parent / "dataset_info.json"
    state_path = paths[0].parent / "state.json"
    if not info_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(
            f"GIFT-Eval metadata is incomplete beside {paths[0]}"
        )
    with info_path.open(encoding="utf-8") as handle:
        dataset_info = json.load(handle)
    writer = DatasetWriter(
        source_dir / dataset,
        source=SOURCE,
        dataset=dataset,
        task="forecasting",
        project_root=project_root,
    )
    count = 0
    seen_ids: set[str] = set()
    source_files = [*paths, info_path, state_path]
    for arrow_path in paths:
        for row in arrow_rows(arrow_path):
            series_id, split, metadata = gift_record(row, count)
            if series_id in seen_ids:
                raise ValueError(
                    f"duplicate GIFT-Eval item_id in {source_name}: {series_id}"
                )
            seen_ids.add(series_id)
            writer.add_series(
                series_id=series_id,
                splits={"data": split},
                source_metadata=metadata,
            )
            count += 1
    writer.finalize(
        source_metadata={
            "gift_eval_dataset": source_name,
            "dataset_info": jsonable(dataset_info),
            "source_files": [
                source_fingerprint(path, project_root) for path in source_files
            ],
            "split_contract": (
                "GIFT-Eval derives train/validation/test chronologically from "
                "the prediction length and evaluation term"
            ),
        }
    )
    print(f"{SOURCE}/{dataset}: {count} series")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/raw/GiftEval")
    )
    parser.add_argument(
        "--dataset",
        nargs="*",
        help="optional source paths such as m4_yearly or electricity/15T",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    input_root = args.input_dir.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    arrow_paths = sorted(
        path
        for path in input_root.rglob("data-*.arrow")
        if ".cache" not in path.parts
    )
    if args.dataset:
        selected = {value.strip("/") for value in args.dataset}
        arrow_paths = [
            path
            for path in arrow_paths
            if path.parent.relative_to(input_root).as_posix() in selected
        ]
        found = {
            path.parent.relative_to(input_root).as_posix() for path in arrow_paths
        }
        if found != selected:
            raise FileNotFoundError(
                f"missing requested GIFT-Eval datasets: {sorted(selected - found)}"
            )
    if not arrow_paths:
        raise FileNotFoundError(f"no GIFT-Eval Arrow payloads under {input_root}")
    datasets: dict[Path, list[Path]] = {}
    for path in arrow_paths:
        datasets.setdefault(path.parent, []).append(path)

    with staged_source_directory(args.output_dir.resolve(), SOURCE) as source_dir:
        for paths in datasets.values():
            process_dataset(
                input_root, paths, source_dir, project_root
            )


if __name__ == "__main__":
    sys.exit(main())
