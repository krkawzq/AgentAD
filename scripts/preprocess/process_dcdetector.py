"""Convert DCDetector/ScatterAD archives while preserving split pairing."""

from __future__ import annotations

import argparse
import filecmp
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .common import (
        DatasetWriter,
        SplitData,
        split_interval_annotations,
        staged_source_directory,
    )
else:
    from common import (
        DatasetWriter,
        SplitData,
        split_interval_annotations,
        staged_source_directory,
    )


SOURCE = "dcdetector"


def series_key_and_role(path: Path) -> tuple[str, str | None]:
    stem = path.stem
    for role in ("train", "test"):
        suffix = f".{role}"
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)], role
    return stem, None


def parse_ucr_name(name: str) -> tuple[int, int, int] | None:
    parts = name.rsplit("_", 3)
    if len(parts) != 4:
        return None
    try:
        return int(parts[-3]), int(parts[-2]), int(parts[-1])
    except ValueError:
        return None


def choose_duplicate(paths: list[Path]) -> tuple[Path, list[Path]]:
    ordered = sorted(paths, key=lambda path: (path.suffix != ".out", str(path)))
    chosen = ordered[0]
    for other in ordered[1:]:
        if not filecmp.cmp(chosen, other, shallow=False):
            raise ValueError(f"conflicting duplicate inputs: {chosen}, {other}")
    return chosen, ordered


def read_series(path: Path, timestamp_start: int = 0) -> SplitData:
    frame = pd.read_csv(path, header=None)
    if frame.shape[1] != 2:
        raise ValueError(
            f"expected value,label columns in {path}, got {frame.shape[1]}"
        )
    return SplitData(
        values=frame.iloc[:, [0]],
        timestamps=np.arange(timestamp_start, timestamp_start + len(frame)),
        labels=frame.iloc[:, 1],
        feature_names=["value"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/raw/DCDetector/benchmark")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    input_root = args.input_dir.resolve()
    with staged_source_directory(args.output_dir.resolve(), SOURCE) as source_dir:
        dataset_dirs = sorted(
            path
            for path in input_root.iterdir()
            if path.is_dir() and not path.name.startswith("__")
        )
        if not dataset_dirs:
            raise FileNotFoundError(f"no DCDetector datasets under {input_root}")
        for raw_dataset_dir in dataset_dirs:
            files = sorted(
                {*raw_dataset_dir.rglob("*.out"), *raw_dataset_dir.rglob("*.csv")}
            )
            files = [path for path in files if "__MACOSX" not in path.parts]
            if not files:
                print(
                    f"{SOURCE}: skipped {raw_dataset_dir.name}: no supported files"
                )
                continue
            grouped: dict[str, dict[str | None, list[Path]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for path in files:
                key, role = series_key_and_role(path)
                grouped[key][role].append(path)

            dataset = raw_dataset_dir.name
            dataset_dir = source_dir / dataset
            num_series = 0
            writer = DatasetWriter(
                dataset_dir,
                source=SOURCE,
                dataset=dataset,
                task="anomaly_detection",
                project_root=project_root,
            )
            for series_id, role_paths in sorted(grouped.items()):
                selected: dict[str | None, Path] = {}
                source_files: list[Path] = []
                for role, candidates in role_paths.items():
                    selected[role], duplicates = choose_duplicate(candidates)
                    source_files.extend(duplicates)
                if None in selected and len(selected) != 1:
                    raise ValueError(
                        f"series {series_id!r} mixes unsplit and split inputs"
                    )

                splits: dict[str, SplitData]
                annotations: list[dict[str, object]] = []
                source_metadata: dict[str, object] = {
                    "input_roles": sorted(role or "unsplit" for role in selected),
                    "deduplicated_files": len(source_files) - len(selected),
                }
                ucr_bounds = parse_ucr_name(series_id) if None in selected else None
                if ucr_bounds is not None:
                    full = read_series(selected[None])
                    train_len, anomaly_start, anomaly_end = ucr_bounds
                    if not (
                        0 < train_len < len(full.values)
                        and 0 <= anomaly_start <= anomaly_end <= len(full.values)
                    ):
                        raise ValueError(f"invalid UCR boundaries for {series_id}")
                    label_series = pd.to_numeric(pd.Series(full.labels), errors="raise")
                    if label_series.isna().any() or not set(label_series.unique()) <= {
                        0,
                        1,
                    }:
                        raise ValueError(f"labels are not binary for {series_id}")
                    labels = label_series.to_numpy(dtype=np.int8)
                    expected = np.zeros(len(labels), dtype=np.int8)
                    expected[anomaly_start:anomaly_end] = 1
                    if not np.array_equal(labels, expected):
                        raise ValueError(
                            f"filename and labels disagree for {series_id}"
                        )
                    splits = {
                        "train": SplitData(
                            full.values.iloc[:train_len],
                            np.arange(train_len),
                            labels[:train_len],
                            ["value"],
                        ),
                        "test": SplitData(
                            full.values.iloc[train_len:],
                            np.arange(train_len, len(labels)),
                            labels[train_len:],
                            ["value"],
                        ),
                    }
                    source_metadata.update(
                        {
                            "official_train_length": train_len,
                            "anomaly_start": anomaly_start,
                            "anomaly_end_exclusive": anomaly_end,
                        }
                    )
                    annotations.extend(
                        split_interval_annotations(
                            anomaly_start, anomaly_end, train_len
                        )
                    )
                elif "train" in selected and "test" in selected:
                    train = read_series(selected["train"])
                    test = read_series(selected["test"], len(train.values))
                    splits = {"train": train, "test": test}
                elif "train" in selected:
                    splits = {"train": read_series(selected["train"])}
                elif "test" in selected:
                    splits = {"test": read_series(selected["test"])}
                else:
                    splits = {"data": read_series(selected[None])}

                writer.add_series(
                    series_id=series_id,
                    splits=splits,
                    source_files=source_files,
                    source_metadata=source_metadata,
                    annotations=annotations,
                )
                num_series += 1
            writer.finalize()
            print(f"{SOURCE}/{dataset}: {num_series} series")


if __name__ == "__main__":
    sys.exit(main())
