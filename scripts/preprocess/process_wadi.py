"""Convert authorized WADI train/test CSVs to a canonical SeriesData package."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__:
    from .common import DatasetWriter, SplitData, staged_source_directory
else:
    from common import DatasetWriter, SplitData, staged_source_directory


SOURCE = "wadi"
DATASET = "WADI"


def normalized_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def choose_files(input_dir: Path) -> tuple[Path, Path]:
    files = sorted(input_dir.rglob("*.csv"))
    train = [
        path
        for path in files
        if "14days" in path.name.lower()
        or ("normal" in path.name.lower() and "attack" not in path.name.lower())
    ]
    test = [
        path
        for path in files
        if "attackdata" in path.name.lower()
        or ("attack" in path.name.lower() and path not in train)
    ]

    def train_rank(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        return (0 if name == "wadi_14days.csv" else 1, name)

    def test_rank(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        return (0 if "lable" in name or "label" in name else 1, name)

    if not train or not test:
        raise FileNotFoundError(
            f"could not identify WADI train/test CSVs under {input_dir}"
        )
    return sorted(train, key=train_rank)[0], sorted(test, key=test_rank)[0]


def timestamp_values(frame: pd.DataFrame) -> tuple[np.ndarray, list[str], str]:
    by_name = {normalized_name(column): str(column) for column in frame.columns}
    date_col = by_name.get("date")
    time_col = by_name.get("time")
    if date_col is not None and time_col is not None:
        values = (
            frame[date_col].astype(str).str.strip()
            + " "
            + frame[time_col].astype(str).str.strip()
        ).to_numpy()
        return values, [date_col, time_col], "date+time"
    for candidate in ("timestamp", "datetime", "time", "date"):
        column = by_name.get(candidate)
        if column is not None:
            return frame[column].to_numpy(), [column], column
    return np.arange(len(frame), dtype=np.int64), [], "generated_index"


def label_column(frame: pd.DataFrame) -> str | None:
    candidates = [
        str(column)
        for column in frame.columns
        if "attack" in normalized_name(column) or "label" in normalized_name(column)
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda name: (
            0 if normalized_name(name) in {"attack", "label"} else 1,
            name,
        )
    )
    return candidates[0]


def canonical_labels(values: pd.Series) -> np.ndarray:
    if values.isna().any():
        raise ValueError("WADI attack labels contain missing values")
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        unique = set(numeric.astype(float).unique().tolist())
        if unique <= {-1.0, 1.0}:
            return (numeric.to_numpy() != 1).astype(np.int8)
        if unique <= {0.0, 1.0}:
            return numeric.to_numpy(dtype=np.int8)
        raise ValueError(f"unsupported numeric WADI label values: {sorted(unique)}")

    normalized = values.astype(str).map(normalized_name)
    normal = {"normal", "noattack", "noanomaly", "0"}
    anomalous = {"attack", "anomaly", "1", "1attack"}
    unknown = set(normalized.unique()) - normal - anomalous
    if unknown:
        raise ValueError(f"unsupported textual WADI labels: {sorted(unknown)}")
    return normalized.isin(anomalous).to_numpy(dtype=np.int8)


def numeric_features(
    frame: pd.DataFrame, excluded: set[str]
) -> tuple[pd.DataFrame, list[str]]:
    values: dict[str, pd.Series] = {}
    dropped_empty: list[str] = []
    for raw_name in frame.columns:
        name = str(raw_name)
        if name in excluded or name.startswith("Unnamed:"):
            continue
        raw = frame[raw_name]
        nonempty = raw.notna() & raw.astype(str).str.strip().ne("")
        numeric = pd.to_numeric(raw, errors="coerce")
        invalid = nonempty & numeric.isna()
        if invalid.any():
            sample = raw[invalid].iloc[0]
            raise ValueError(f"non-numeric WADI value in {name!r}: {sample!r}")
        if not numeric.notna().any():
            dropped_empty.append(name)
            continue
        if name in values:
            raise ValueError(f"duplicate WADI feature name: {name!r}")
        values[name] = numeric.reset_index(drop=True)
    if not values:
        raise ValueError("WADI frame has no numeric features")
    return pd.DataFrame(values), dropped_empty


def load_wadi_csv(
    path: Path, *, require_labels: bool
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError(f"empty WADI CSV: {path}")
    timestamps, timestamp_columns, timestamp_source = timestamp_values(frame)
    raw_label_name = label_column(frame)
    if require_labels and raw_label_name is None:
        raise ValueError(f"WADI test CSV has no attack label column: {path}")
    excluded = set(timestamp_columns)
    if raw_label_name is not None:
        excluded.add(raw_label_name)
    excluded.update(
        str(column)
        for column in frame.columns
        if normalized_name(column) in {"row", "index", "rowindex"}
    )
    features, dropped_empty = numeric_features(frame, excluded)
    if raw_label_name is None:
        labels = np.zeros(len(frame), dtype=np.int8)
        extra = pd.DataFrame(index=pd.RangeIndex(len(frame)))
    else:
        labels = canonical_labels(frame[raw_label_name])
        extra = pd.DataFrame(
            {"source_attack_label": frame[raw_label_name].reset_index(drop=True)}
        )
    metadata = {
        "timestamp_columns": timestamp_columns,
        "timestamp_source": timestamp_source,
        "source_label_column": raw_label_name,
        "excluded_columns": sorted(excluded),
        "dropped_all_empty_columns": dropped_empty,
    }
    return features, timestamps, labels, extra, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/WADI"))
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    input_dir = args.input_dir.resolve()
    if args.train_file is None and args.test_file is None:
        train_path, test_path = choose_files(input_dir)
    elif args.train_file is not None and args.test_file is not None:
        train_path, test_path = args.train_file.resolve(), args.test_file.resolve()
    else:
        parser.error("--train-file and --test-file must be provided together")
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError((train_path, test_path))

    train, train_ts, train_labels, train_extra, train_meta = load_wadi_csv(
        train_path, require_labels=False
    )
    test, test_ts, test_labels, test_extra, test_meta = load_wadi_csv(
        test_path, require_labels=True
    )
    if list(train.columns) != list(test.columns):
        raise ValueError(
            "WADI train/test feature schemas differ: "
            f"train_only={sorted(set(train) - set(test))}, "
            f"test_only={sorted(set(test) - set(train))}"
        )

    with staged_source_directory(args.output_dir.resolve(), SOURCE) as source_dir:
        writer = DatasetWriter(
            source_dir / DATASET,
            source=SOURCE,
            dataset=DATASET,
            task="anomaly_detection",
            project_root=project_root,
        )
        writer.add_series(
            series_id="default",
            splits={
                "train": SplitData(
                    train,
                    train_ts,
                    train_labels,
                    list(train.columns),
                    timestamp_kind="source",
                    label_columns=train_extra,
                ),
                "test": SplitData(
                    test,
                    test_ts,
                    test_labels,
                    list(test.columns),
                    timestamp_kind="source",
                    label_columns=test_extra,
                ),
            },
            source_files=[train_path, test_path],
            source_metadata={"train": train_meta, "test": test_meta},
        )
        writer.finalize(
            source_metadata={
                "access": "authorized iTrust WADI download",
                "official_url": "https://www.sutd.edu.sg/itrust/itrust-labs/datasets/",
            }
        )
    print(f"{SOURCE}/{DATASET}: train={len(train)} test={len(test)}")


if __name__ == "__main__":
    sys.exit(main())
