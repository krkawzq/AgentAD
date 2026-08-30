"""Convert every CrossAD dataset to canonical SeriesData packages."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .common import (  # type: ignore[import-not-found]
        DatasetWriter,
        SplitData,
        long_to_split,
        split_interval_annotations,
        staged_source_directory,
        strict_int,
    )
else:
    from common import (
        DatasetWriter,
        SplitData,
        long_to_split,
        split_interval_annotations,
        staged_source_directory,
        strict_int,
    )


SOURCE = "crossad"


def _slice(split: SplitData, start: int, end: int) -> SplitData:
    return SplitData(
        values=split.values.iloc[start:end].reset_index(drop=True),
        timestamps=np.asarray(split.timestamps)[start:end],
        labels=None if split.labels is None else np.asarray(split.labels)[start:end],
        feature_names=split.feature_names,
        timestamp_kind=split.timestamp_kind,
    )


def read_train_lengths(meta_path: Path) -> dict[str, int]:
    if not meta_path.exists():
        raise FileNotFoundError(f"CrossAD split metadata not found: {meta_path}")
    meta = pd.read_csv(meta_path)
    required = {"file_name", "train_lens"}
    if not required <= set(meta.columns):
        raise ValueError(f"invalid CrossAD metadata: {meta_path}")
    if meta["file_name"].isna().any() or meta["file_name"].duplicated().any():
        raise ValueError(
            f"CrossAD metadata has empty or duplicate file names: {meta_path}"
        )
    return {
        str(row.file_name): strict_int(
            row.train_lens, name=f"train_lens for {row.file_name}"
        )
        for row in meta.itertuples()
    }


def parse_ucr_name(path: Path) -> tuple[int, int, int]:
    parts = path.stem.rsplit("_", 3)
    if len(parts) != 4:
        raise ValueError(f"UCR filename has no split/annotation suffix: {path.name}")
    try:
        values = tuple(int(value) for value in parts[-3:])
    except ValueError as error:
        raise ValueError(f"invalid UCR boundaries in {path.name}") from error
    return values[0], values[1], values[2]


def process_long_datasets(
    dataset_root: Path,
    source_dir: Path,
    train_lengths: dict[str, int],
    project_root: Path,
    meta_path: Path,
) -> None:
    data_dir = dataset_root / "data"
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CrossAD long CSV files under {data_dir}")
    raw_names = {path.name for path in files}
    if set(train_lengths) != raw_names:
        raise ValueError(
            "CrossAD metadata/raw file mismatch: "
            f"missing_files={sorted(set(train_lengths) - raw_names)}, "
            f"missing_metadata={sorted(raw_names - set(train_lengths))}"
        )

    for csv_path in files:
        raw = pd.read_csv(csv_path, dtype={"cols": "string"})
        full = long_to_split(raw)
        split_at = train_lengths.get(csv_path.name)
        if split_at is None:
            raise KeyError(f"missing official train length for {csv_path.name}")
        if not 0 < split_at < len(full.values):
            raise ValueError(f"invalid train length {split_at} for {csv_path.name}")
        dataset = csv_path.stem
        dataset_dir = source_dir / dataset
        writer = DatasetWriter(
            dataset_dir,
            source=SOURCE,
            dataset=dataset,
            task="anomaly_detection",
            project_root=project_root,
        )
        record = writer.add_series(
            series_id="default",
            splits={
                "train": _slice(full, 0, split_at),
                "test": _slice(full, split_at, len(full.values)),
            },
            source_files=[csv_path, meta_path],
            source_metadata={
                "official_train_length": split_at,
                "input_format": "long-date-data-cols",
            },
        )
        writer.finalize()
        print(f"{SOURCE}/{dataset}: {record['splits']}")


def process_ucr(
    dataset_root: Path, source_dir: Path, project_root: Path
) -> None:
    ucr_dir = dataset_root / "UCR_Anomaly_FullData"
    files = sorted(ucr_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no CrossAD UCR files under {ucr_dir}")
    dataset = "UCR_Anomaly_FullData"
    dataset_dir = source_dir / dataset
    num_series = 0
    writer = DatasetWriter(
        dataset_dir,
        source=SOURCE,
        dataset=dataset,
        task="anomaly_detection",
        project_root=project_root,
    )
    for path in files:
        values = np.loadtxt(path, dtype=np.float64)
        values = np.asarray(values).reshape(-1, 1)
        train_len, anomaly_start, anomaly_end = parse_ucr_name(path)
        if not (
            0 < train_len < len(values)
            and 0 <= anomaly_start <= anomaly_end <= len(values)
        ):
            raise ValueError(f"invalid UCR boundaries for {path.name}")
        labels = np.zeros(len(values), dtype=np.int8)
        labels[anomaly_start:anomaly_end] = 1
        writer.add_series(
            series_id=path.stem,
            splits={
                "train": SplitData(
                    values[:train_len],
                    np.arange(train_len),
                    labels[:train_len],
                    ["value"],
                ),
                "test": SplitData(
                    values[train_len:],
                    np.arange(train_len, len(values)),
                    labels[train_len:],
                    ["value"],
                ),
            },
            source_files=[path],
            source_metadata={
                "official_train_length": train_len,
                "anomaly_start": anomaly_start,
                "anomaly_end_exclusive": anomaly_end,
            },
            annotations=split_interval_annotations(
                anomaly_start, anomaly_end, train_len
            ),
        )
        num_series += 1
    writer.finalize()
    print(f"{SOURCE}/{dataset}: {num_series} series")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/raw/CrossAD/dataset")
    )
    parser.add_argument(
        "--meta-path", type=Path, default=Path("forks/CrossAD/dataset/DETECT_META.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    dataset_root = args.input_dir.resolve()
    if dataset_root.name == "data":
        dataset_root = dataset_root.parent
    meta_path = args.meta_path.resolve()
    train_lengths = read_train_lengths(meta_path)
    with staged_source_directory(args.output_dir.resolve(), SOURCE) as source_dir:
        process_long_datasets(
            dataset_root, source_dir, train_lengths, project_root, meta_path
        )
        process_ucr(dataset_root, source_dir, project_root)


if __name__ == "__main__":
    main()
