"""Convert TSB-AD-U/M while retaining feature names and benchmark partitions."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .common import DatasetWriter, SplitData, staged_source_directory
else:
    from common import DatasetWriter, SplitData, staged_source_directory


SOURCE = "tsb-ad"
FILENAME_PATTERN = re.compile(
    r"^(?P<index>\d+)_(?P<source_dataset>.+?)_id_(?P<source_id>\d+)_"
    r"(?P<domain>.+)_tr_(?P<train_length>\d+)_1st_(?P<first_anomaly>\d+)$"
)


def parse_filename(path: Path) -> dict[str, object]:
    match = FILENAME_PATTERN.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"invalid TSB-AD filename: {path.name}")
    values = match.groupdict()
    return {
        "index": int(values["index"]),
        "source_dataset": values["source_dataset"],
        "source_id": int(values["source_id"]),
        "domain": values["domain"],
        "train_length": int(values["train_length"]),
        "first_anomaly": int(values["first_anomaly"]),
    }


def read_name_list(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    names = {row[0].strip() for row in rows if row and row[0].strip()}
    names.discard("file_name")
    return names


def load_partitions(list_dir: Path, key: str) -> dict[str, tuple[Path, set[str]]]:
    prefix = f"TSB-AD-{key}"
    paths = sorted(list_dir.glob(f"{prefix}*.csv"))
    if not paths:
        raise FileNotFoundError(f"no TSB-AD file lists under {list_dir}")
    result = {}
    for path in paths:
        name = path.stem.removeprefix(prefix).lstrip("-") or "all"
        if name in result:
            raise ValueError(f"duplicate TSB-AD partition name {name!r}")
        result[name] = (path, read_name_list(path))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/TSB-AD"))
    parser.add_argument(
        "--file-list-dir", type=Path, default=Path("forks/TSB-AD/Datasets/File_List")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--only", choices=["U", "M"])
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    input_root = args.input_dir.resolve()
    list_dir = args.file_list_dir.resolve()
    with staged_source_directory(args.output_dir.resolve(), SOURCE) as source_dir:
        for key in [args.only] if args.only else ["U", "M"]:
            dataset = f"TSB-AD-{key}"
            raw_dir = input_root / dataset
            files = sorted(raw_dir.glob("*.csv"))
            if not files:
                raise FileNotFoundError(f"no CSV files under {raw_dir}")
            partitions = load_partitions(list_dir, key)
            all_partition_names = set().union(
                *(names for _, names in partitions.values())
            )
            raw_names = {path.name for path in files}
            unknown_names = sorted(all_partition_names - raw_names)
            if unknown_names:
                print(
                    f"{SOURCE}/{dataset}: {len(unknown_names)} file-list entries "
                    "have no raw CSV"
                )
            unlisted_names = sorted(raw_names - all_partition_names)
            if unlisted_names:
                print(
                    f"{SOURCE}/{dataset}: {len(unlisted_names)} raw CSV files "
                    "are in no file list"
                )

            dataset_dir = source_dir / dataset
            num_series = 0
            writer = DatasetWriter(
                dataset_dir,
                source=SOURCE,
                dataset=dataset,
                task="anomaly_detection",
                project_root=project_root,
            )
            first_anomaly_mismatches = []
            for path in files:
                frame = pd.read_csv(path)
                if frame.empty or frame.shape[1] < 2:
                    raise ValueError(f"invalid TSB-AD CSV: {path}")
                label_col = str(frame.columns[-1])
                if label_col.lower() != "label":
                    raise ValueError(f"last TSB-AD column is not Label: {path}")
                feature_names = [str(column) for column in frame.columns[:-1]]
                filename_meta = parse_filename(path)
                train_len = int(filename_meta["train_length"])
                if not 0 < train_len < len(frame):
                    raise ValueError(f"invalid train length for {path.name}")
                label_series = pd.to_numeric(frame.iloc[:, -1], errors="raise")
                if label_series.isna().any() or not set(label_series.unique()) <= {
                    0,
                    1,
                }:
                    raise ValueError(f"TSB-AD labels must be binary in {path}")
                labels = label_series.to_numpy(dtype=np.int8)
                nonzero = np.flatnonzero(labels)
                first_anomaly = int(filename_meta["first_anomaly"])
                observed_first = None if len(nonzero) == 0 else int(nonzero[0])
                first_anomaly_matches = observed_first == first_anomaly
                if not first_anomaly_matches:
                    first_anomaly_mismatches.append(path.name)
                memberships = sorted(
                    name
                    for name, (_, names) in partitions.items()
                    if path.name in names
                )
                writer.add_series(
                    series_id=path.stem,
                    splits={
                        "train": SplitData(
                            frame.iloc[:train_len, :-1],
                            np.arange(train_len),
                            labels[:train_len],
                            feature_names,
                        ),
                        "test": SplitData(
                            frame.iloc[train_len:, :-1],
                            np.arange(train_len, len(frame)),
                            labels[train_len:],
                            feature_names,
                        ),
                    },
                    source_files=[path, *[partitions[name][0] for name in memberships]],
                    source_metadata={
                        **filename_meta,
                        "benchmark_partitions": memberships,
                        "label_column": label_col,
                        "observed_first_anomaly": observed_first,
                        "filename_first_anomaly_matches": first_anomaly_matches,
                    },
                )
                num_series += 1
            writer.finalize(
                source_metadata={
                    "file_list_dir": str(list_dir),
                    "first_anomaly_mismatches": first_anomaly_mismatches,
                }
            )
            if first_anomaly_mismatches:
                print(
                    f"{SOURCE}/{dataset}: filename first-anomaly metadata disagrees "
                    f"with labels for {len(first_anomaly_mismatches)} series"
                )
            print(f"{SOURCE}/{dataset}: {num_series} series")


if __name__ == "__main__":
    sys.exit(main())
