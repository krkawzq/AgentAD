"""Convert AERCA's MSDS concurrent-system dataset to SeriesData."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__:
    from .common import DatasetWriter, SplitData, staged_source_directory  # type: ignore[import-not-found]
else:
    from common import DatasetWriter, SplitData, staged_source_directory


SOURCE = "aerca"
MSDS_METRIC_FILES = (
    "wally122_metrics_concurrent.csv",
    "wally113_metrics_concurrent.csv",
    "wally123_metrics_concurrent.csv",
    "wally117_metrics_concurrent.csv",
    "wally124_metrics_concurrent.csv",
)
MSDS_DROPPED_COLUMNS = ("load.cpucore", "load.min1", "load.min5", "load.min15")


def find_unique(root: Path, name: str) -> Path:
    matches = sorted(path for path in root.rglob(name) if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one {name!r} under {root}, found {len(matches)}"
        )
    return matches[0]


def load_msds_frames(
    input_dir: Path, labels_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path], dict[str, Any]]:
    """Reproduce AERCA's MSDS preprocessing verbatim.

    Upstream trims each file to the common [start, end] window with two
    successive ``df.drop(positional_indices)`` calls. The second call runs on a
    gappy index, so it silently truncates the tail of every file (~488 common
    timestamps). The released ``labels.csv`` was produced against that exact
    output, so the quirk is reproduced here to keep the official labels
    aligned; it is recorded in the package metadata.
    """
    metric_paths = [find_unique(input_dir, name) for name in MSDS_METRIC_FILES]
    frames: list[pd.DataFrame] = []
    retained_columns: dict[str, list[str]] = {}
    raw_frames: list[tuple[Path, pd.DataFrame]] = []
    for path in metric_paths:
        raw = pd.read_csv(path)
        if "now" not in raw.columns:
            raise ValueError(f"MSDS metric file has no 'now' column: {path}")
        raw = raw.drop(
            columns=[
                column for column in MSDS_DROPPED_COLUMNS if column in raw.columns
            ]
        )
        raw_frames.append((path, raw))
    start = max(raw["now"].min() for _, raw in raw_frames)
    end = min(raw["now"].max() for _, raw in raw_frames)
    for path, raw in raw_frames:
        # AERCA's exact trim: same-second samples then aggregate through
        # melt + dropna + pivot_table, i.e. per-timestamp mean over the
        # non-missing samples.
        trimmed = raw.drop(np.argwhere(list(raw["now"] < start)).reshape(-1))
        trimmed = trimmed.drop(
            np.argwhere(list(trimmed["now"] > end)).reshape(-1)
        )
        melted = trimmed.melt(id_vars=["now"]).dropna()
        values = melted.pivot_table(index=["now"], columns="variable", values="value")
        values = values.apply(pd.to_numeric, errors="raise")
        feature_columns = [str(column) for column in values.columns]
        if not feature_columns:
            raise ValueError(f"MSDS metric file has no retained features: {path}")
        node = path.name.removesuffix("_metrics_concurrent.csv")
        values.columns = [f"{column}__{node}" for column in feature_columns]
        frames.append(values)
        retained_columns[node] = feature_columns

    merged = frames[0]
    for right in frames[1:]:
        merged = merged.merge(right, left_index=True, right_index=True)
    merged = merged.sort_index(kind="stable")
    if merged.empty:
        raise ValueError("MSDS metric files have no common timestamps")
    burn_in = round(len(merged) * 0.1)
    retained = merged.iloc[burn_in:]
    train_rows = round(len(retained) / 2)
    train = retained.iloc[:train_rows]
    test = retained.iloc[train_rows:]
    if train.empty or test.empty:
        raise ValueError("MSDS AERCA split produced an empty partition")

    root_causes = pd.read_csv(labels_path)
    if root_causes.columns[0].startswith("Unnamed"):
        root_causes = root_causes.iloc[:, 1:]
    root_causes = root_causes.apply(pd.to_numeric, errors="raise")
    # labels.csv rows align with AERCA's test[::5] view; expand each label to
    # the consecutive 5-row block it represents so the canonical package can
    # stay at full (non-downsampled) resolution.
    expected = math.ceil(len(test) / 5)
    if len(root_causes) != expected:
        raise ValueError(
            f"MSDS labels/downsampled-test length mismatch: "
            f"{len(root_causes)} != {expected}"
        )
    root_causes = root_causes.iloc[
        np.arange(len(test)) // 5
    ].reset_index(drop=True)
    if not np.isin(root_causes.to_numpy(), [0, 1]).all():
        raise ValueError("MSDS root-cause labels must be binary")
    root_causes.columns = [f"root_cause_{column}" for column in root_causes.columns]
    metadata = {
        "metric_files": list(MSDS_METRIC_FILES),
        "retained_columns": retained_columns,
        "dropped_columns": list(MSDS_DROPPED_COLUMNS),
        "timestamp_alignment": "chained index merge, stable sorted",
        "duplicate_timestamp_aggregation": "per-timestamp mean (melt+dropna+pivot)",
        "upstream_window_trim": (
            "AERCA's positional-drop tail truncation reproduced verbatim; "
            "the released labels.csv aligns only with that output"
        ),
        "label_alignment": (
            "labels.csv matches test[::5]; expanded to canonical rows in "
            "consecutive 5-row blocks"
        ),
        "burn_in_rows": burn_in,
        "split_after_burn_in": [len(train), len(test)],
        "aerca_model_downsample_factor": 5,
        "canonical_data_is_not_downsampled": True,
    }
    return train, test, root_causes, [*metric_paths, labels_path], metadata


def process_msds(
    input_dir: Path, labels_path: Path, source_dir: Path, project_root: Path
) -> None:
    train, test, root_causes, source_files, metadata = load_msds_frames(
        input_dir, labels_path
    )
    dataset = "MSDS"
    writer = DatasetWriter(
        source_dir / dataset,
        source=SOURCE,
        dataset=dataset,
        task="anomaly_detection",
        project_root=project_root,
    )
    zero_causes = pd.DataFrame(
        np.zeros((len(train), root_causes.shape[1]), dtype=np.int8),
        columns=root_causes.columns,
    )
    writer.add_series(
        series_id="default",
        splits={
            "train": SplitData(
                values=train.reset_index(drop=True),
                timestamps=train.index.to_numpy(),
                labels=np.zeros(len(train), dtype=np.int8),
                feature_names=list(train.columns),
                timestamp_kind="source",
                label_columns=zero_causes,
            ),
            "test": SplitData(
                values=test.reset_index(drop=True),
                timestamps=test.index.to_numpy(),
                labels=root_causes.max(axis=1).to_numpy(dtype=np.int8),
                feature_names=list(test.columns),
                timestamp_kind="source",
                label_columns=root_causes,
            ),
        },
        source_files=source_files,
        source_metadata=metadata,
    )
    writer.finalize()
    print(f"{SOURCE}/{dataset}: train={len(train)} test={len(test)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/AERCA"))
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("data/raw/AERCA/msds/labels.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    input_root = args.input_dir.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    labels_path = args.labels_path.resolve()
    msds_dir = input_root / "msds"
    if not msds_dir.is_dir() or not labels_path.is_file():
        raise FileNotFoundError(
            f"MSDS data not found under {input_root} (expected msds/ and "
            f"{labels_path})"
        )

    with staged_source_directory(args.output_dir.resolve(), SOURCE) as source_dir:
        process_msds(msds_dir, labels_path, source_dir, project_root)


if __name__ == "__main__":
    main()
