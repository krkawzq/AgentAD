"""Convert Granite/FoundTS ZafNoo data in either long or wide CSV form."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import pandas as pd

if __package__:
    from .common import DatasetWriter, SplitData, jsonable, staged_source_directory  # type: ignore[import-not-found]
else:
    from common import DatasetWriter, SplitData, jsonable, staged_source_directory


SOURCE = "granite-tsfm"
DATASET = "ZafNoo"


def recommended_split_ranges(rows: int) -> dict[str, list[int]]:
    train_end = int(rows * 0.7)
    test_rows = int(rows * 0.2)
    val_end = rows - test_rows
    return {
        "train": [0, train_end],
        "val": [train_end, val_end],
        "test": [val_end, rows],
    }


def zafnoo_series(
    path: Path,
) -> Iterator[tuple[str, SplitData, dict[str, object]]]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty ZafNoo CSV: {path}")
    if {"date", "data", "cols"} <= set(frame.columns):
        if frame[["date", "cols"]].isna().any().any():
            raise ValueError("ZafNoo long CSV has missing dates or series ids")
        if frame[["date", "cols"]].duplicated().any():
            raise ValueError("ZafNoo long CSV has duplicate date/cols pairs")
        for raw_id in pd.unique(frame["cols"]):
            subset = frame.loc[frame["cols"] == raw_id, ["date", "data"]]
            values = pd.to_numeric(subset["data"], errors="raise")
            series_id = str(raw_id)
            if not series_id:
                raise ValueError("ZafNoo has an empty cols identifier")
            yield (
                series_id,
                SplitData(
                    values.to_numpy().reshape(-1, 1),
                    subset["date"].to_numpy(),
                    feature_names=["value"],
                    timestamp_kind="source",
                ),
                {
                    "input_format": "long-date-data-cols",
                    "source_column_id": jsonable(raw_id),
                    "recommended_split_ranges": recommended_split_ranges(len(subset)),
                },
            )
        return

    timestamp_column = "date" if "date" in frame.columns else str(frame.columns[0])
    feature_columns = [
        str(column) for column in frame.columns if column != timestamp_column
    ]
    if not feature_columns:
        raise ValueError("ZafNoo wide CSV has no feature columns")
    values = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="raise")
    yield (
        "default",
        SplitData(
            values,
            frame[timestamp_column].to_numpy(),
            feature_names=feature_columns,
            timestamp_kind="source",
        ),
        {
            "input_format": "wide",
            "timestamp_column": timestamp_column,
            "recommended_split_ranges": recommended_split_ranges(len(frame)),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/raw/GraniteTSFM/ZafNoo.csv"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "forks/granite-tsfm/tsfm_public/resources/data_config/zafnoo.yaml"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    input_path = args.input_path.resolve()
    config_path = args.config.resolve()
    if not input_path.is_file() or not config_path.is_file():
        raise FileNotFoundError((input_path, config_path))

    with staged_source_directory(args.output_dir.resolve(), SOURCE) as source_dir:
        writer = DatasetWriter(
            source_dir / DATASET,
            source=SOURCE,
            dataset=DATASET,
            task="forecasting",
            project_root=project_root,
        )
        count = 0
        for series_id, split, metadata in zafnoo_series(input_path):
            writer.add_series(
                series_id=series_id,
                splits={"data": split},
                source_files=[input_path, config_path],
                source_metadata=metadata,
            )
            count += 1
        writer.finalize(
            source_metadata={
                "benchmark": "Granite TTM FoundTS",
                "frequency": "30min",
            }
        )
    print(f"{SOURCE}/{DATASET}: {count} series")


if __name__ == "__main__":
    main()
