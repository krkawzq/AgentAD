"""Convert ordinary LangTime forecasting datasets without changing time axes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__:
    from .common import (
        DatasetWriter,
        SplitData,
        jsonable,
        source_fingerprint,
        staged_source_directory,
    )
else:
    from common import (
        DatasetWriter,
        SplitData,
        jsonable,
        source_fingerprint,
        staged_source_directory,
    )


SOURCE = "langtime"


def load_config(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"LangTime config root is not an object: {path}")
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for name, config in raw.items():
        if not isinstance(config, dict) or "data_path" not in config:
            raise ValueError(f"invalid LangTime config entry {name!r}")
        data_path = str(config["data_path"])
        configs_for_path = result.setdefault(data_path, {})
        if name in configs_for_path:
            raise ValueError(f"duplicate LangTime config name {name!r}")
        configs_for_path[name] = jsonable(config)
    return result


def split_ranges(name: str, rows: int) -> dict[str, list[int]]:
    if name.startswith("ETTh"):
        train_end = 12 * 30 * 24
        val_end = train_end + 4 * 30 * 24
        test_end = val_end + 4 * 30 * 24
    elif name.startswith("ETTm"):
        train_end = 12 * 30 * 24 * 4
        val_end = train_end + 4 * 30 * 24 * 4
        test_end = val_end + 4 * 30 * 24 * 4
    else:
        train_end = int(rows * 0.7)
        test_rows = int(rows * 0.2)
        val_end = rows - test_rows
        test_end = rows
    if test_end > rows:
        raise ValueError(f"official split exceeds rows for {name}: {test_end}>{rows}")
    ranges = {
        "train": [0, train_end],
        "val": [train_end, val_end],
        "test": [val_end, test_end],
    }
    if test_end < rows:
        ranges["unassigned_source_tail"] = [test_end, rows]
    return ranges


def process_standard_csvs(
    input_root: Path,
    source_dir: Path,
    project_root: Path,
    configs: dict[str, dict[str, dict[str, Any]]],
) -> None:
    excluded = {"m4", "PEMS", "solar"}
    seen_config_paths: set[str] = set()
    for raw_dataset_dir in sorted(
        path
        for path in input_root.iterdir()
        if path.is_dir() and path.name not in excluded
    ):
        files = sorted(raw_dataset_dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"no CSV files under {raw_dataset_dir}")
        for path in files:
            frame = pd.read_csv(path)
            if frame.empty or frame.shape[1] < 2:
                raise ValueError(f"invalid forecasting CSV: {path}")
            timestamp_col = str(frame.columns[0])
            feature_names = [str(column) for column in frame.columns[1:]]
            relative = str(path.relative_to(input_root))
            source_configs = configs.get(relative)
            if source_configs is None:
                raise KeyError(f"no LangTime config for {relative}")
            seen_config_paths.add(relative)
            for config_name, config in source_configs.items():
                target = config.get("target")
                if target is not None and target not in feature_names:
                    raise ValueError(
                        f"target {target!r} from {config_name!r} absent from {path}"
                    )
            collection = raw_dataset_dir.name if len(files) > 1 else None
            dataset = path.stem if collection is not None else raw_dataset_dir.name
            dataset_dir = (
                source_dir / collection / dataset
                if collection is not None
                else source_dir / dataset
            )
            writer = DatasetWriter(
                dataset_dir,
                source=SOURCE,
                collection=collection,
                dataset=dataset,
                task="forecasting",
                project_root=project_root,
            )
            writer.add_series(
                series_id=path.stem,
                splits={
                    "data": SplitData(
                        values=frame.iloc[:, 1:],
                        timestamps=frame.iloc[:, 0],
                        feature_names=feature_names,
                        timestamp_kind="source",
                    )
                },
                source_files=[path],
                source_metadata={
                    "timestamp_column": timestamp_col,
                    "configs": source_configs,
                    "recommended_split_ranges": split_ranges(path.stem, len(frame)),
                },
            )
            writer.finalize()
            print(f"{SOURCE}/{dataset}: 1 series")
    if seen_config_paths != set(configs):
        raise FileNotFoundError(
            f"LangTime config paths have no input CSV: "
            f"{sorted(set(configs) - seen_config_paths)}"
        )


def pack_ragged(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if raw.ndim != 1 or raw.dtype != object:
        raise ValueError(f"expected 1D object array, got {raw.shape} {raw.dtype}")
    lengths = np.fromiter(
        (len(np.asarray(item).reshape(-1)) for item in raw),
        dtype=np.int64,
        count=len(raw),
    )
    offsets = np.empty(len(raw) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    values = np.empty(int(offsets[-1]), dtype=np.float64)
    for index, item in enumerate(raw):
        values[offsets[index] : offsets[index + 1]] = np.asarray(
            item, dtype=np.float64
        ).reshape(-1)
    return values, offsets


def process_m4(
    input_root: Path, source_dir: Path, project_root: Path
) -> None:
    raw_dir = input_root / "m4"
    info_path = raw_dir / "M4-info.csv"
    train_path = raw_dir / "training.npz"
    test_path = raw_dir / "test.npz"
    info = pd.read_csv(info_path)
    train_raw = np.load(train_path, allow_pickle=True)
    test_raw = np.load(test_path, allow_pickle=True)
    if len(info) != len(train_raw) or len(info) != len(test_raw):
        raise ValueError("M4 metadata/train/test counts differ")
    train_values, train_offsets = pack_ragged(train_raw)
    test_values, test_offsets = pack_ragged(test_raw)
    del train_raw, test_raw

    id_column = "M4id"
    if id_column not in info.columns:
        raise ValueError(f"M4 metadata has no series id column: {info_path}")
    if info[id_column].isna().any() or info[id_column].astype(str).duplicated().any():
        raise ValueError(f"M4 metadata has empty or duplicate series ids: {info_path}")
    if "SP" not in info.columns:
        raise ValueError(f"M4 metadata has no seasonal-pattern column: {info_path}")
    source_files = [
        source_fingerprint(path, project_root)
        for path in (info_path, train_path, test_path)
    ]
    for pattern in sorted(info["SP"].astype(str).unique()):
        artifact = f"M4-{pattern}"
        writer = DatasetWriter(
            source_dir / "M4" / artifact,
            source=SOURCE,
            collection="M4",
            dataset=artifact,
            task="forecasting",
            project_root=project_root,
        )
        indices = np.flatnonzero(info["SP"].astype(str).to_numpy() == pattern)
        for index in indices:
            train = train_values[train_offsets[index] : train_offsets[index + 1]]
            test = test_values[test_offsets[index] : test_offsets[index + 1]]
            series_id = str(info.iloc[index][id_column])
            writer.add_series(
                series_id=series_id,
                splits={
                    "train": SplitData(
                        train.reshape(-1, 1),
                        np.arange(len(train)),
                        feature_names=["value"],
                    ),
                    "test": SplitData(
                        test.reshape(-1, 1),
                        np.arange(len(train), len(train) + len(test)),
                        feature_names=["value"],
                    ),
                },
                source_metadata={"m4_info": jsonable(info.iloc[index].to_dict())},
            )
        writer.finalize(
            source_metadata={
                "seasonal_pattern": pattern,
                "source_files": source_files,
            }
        )
        print(f"{SOURCE}/M4/{artifact}: {len(indices)} series")


def process_pems(
    input_root: Path, source_dir: Path, project_root: Path
) -> None:
    raw_dir = input_root / "PEMS"
    files = sorted(raw_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no PEMS archives under {raw_dir}")
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ["data"]:
                raise ValueError(f"unexpected PEMS arrays in {path}: {archive.files}")
            data = archive["data"]
        if data.ndim != 3:
            raise ValueError(f"expected time,node,channel PEMS array: {path}")
        rows, nodes, channels = data.shape
        values = data.reshape(rows, nodes * channels)
        feature_names = [
            f"sensor_{node}_channel_{channel}"
            for node in range(nodes)
            for channel in range(channels)
        ]
        dataset = path.stem
        dataset_dir = source_dir / dataset
        writer = DatasetWriter(
            dataset_dir,
            source=SOURCE,
            dataset=dataset,
            task="forecasting",
            project_root=project_root,
        )
        writer.add_series(
            series_id="default",
            splits={
                "data": SplitData(values, np.arange(rows), feature_names=feature_names)
            },
            source_files=[path],
            source_metadata={
                "source_shape": list(data.shape),
                "axis_order": ["time", "sensor", "channel"],
                "recommended_split_ranges": split_ranges(dataset, rows),
            },
        )
        writer.finalize()
        print(f"{SOURCE}/{dataset}: {data.shape}")


def process_solar(
    input_root: Path, source_dir: Path, project_root: Path
) -> None:
    path = input_root / "solar" / "solar_AL.txt"
    values = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError(f"invalid solar array in {path}: {values.shape}")
    dataset = "solar_AL"
    dataset_dir = source_dir / dataset
    feature_names = [f"plant_{index}" for index in range(values.shape[1])]
    writer = DatasetWriter(
        dataset_dir,
        source=SOURCE,
        dataset=dataset,
        task="forecasting",
        project_root=project_root,
    )
    writer.add_series(
        series_id="default",
        splits={
            "data": SplitData(
                values, np.arange(len(values)), feature_names=feature_names
            )
        },
        source_files=[path],
        source_metadata={
            "recommended_split_ranges": split_ranges(dataset, len(values))
        },
    )
    writer.finalize()
    print(f"{SOURCE}/{dataset}: {values.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/raw/LangTime/dataset")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("forks/LangTime/configs/datasets_config.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    input_root = args.input_dir.resolve()
    configs = load_config(args.config.resolve())
    with staged_source_directory(args.output_dir.resolve(), SOURCE) as source_dir:
        process_standard_csvs(input_root, source_dir, project_root, configs)
        process_m4(input_root, source_dir, project_root)
        process_pems(input_root, source_dir, project_root)
        process_solar(input_root, source_dir, project_root)


if __name__ == "__main__":
    sys.exit(main())
