"""Convert DADA evaluation and Monash pretraining data to SeriesData packages."""

from __future__ import annotations

import argparse
import re
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
        long_to_split,
        safe_name,
        staged_source_directory,
        strict_int,
    )
else:
    from common import (
        DatasetWriter,
        SplitData,
        jsonable,
        long_to_split,
        safe_name,
        staged_source_directory,
        strict_int,
    )


SOURCE = "dada"

MONASH_FAMILIES = {
    "australian_electricity_demand_dataset": "Australian-Electricity-Demand",
    "bitcoin_dataset_without_missing_values": "Bitcoin",
    "kdd_cup_2018_dataset_without_missing_values": "KDD-Cup-2018",
    "london_smart_meters_dataset_without_missing_values": "London-Smart-Meters",
    "saugeenday_dataset": "Saugeen-Day",
    "solar_10_minutes_dataset": "Solar-10min",
    "solar_4_seconds_dataset": "Solar-4sec",
    "sunspot_dataset_without_missing_values": "Sunspot",
    "us_births_dataset": "US-Births",
    "weather_dataset": "Weather",
    "wind_4_seconds_dataset": "Wind-4sec",
    "wind_farms_minutely_dataset_without_missing_values": "Wind-Farms-Minutely",
}


def monash_identity(path: Path) -> tuple[str, int | None]:
    stem = path.stem
    match = re.fullmatch(r"(.+?)(?:_downsample_(\d+))?", stem)
    if match is None:
        raise ValueError(f"invalid Monash series name: {path.name}")
    source_id = match.group(1)
    family_match = re.fullmatch(r"(.+)\(\d+\)", source_id)
    family_key = family_match.group(1) if family_match is not None else source_id
    try:
        family = MONASH_FAMILIES[family_key]
    except KeyError as error:
        raise ValueError(f"unknown Monash family in {path.name}") from error
    factor = None if match.group(2) is None else int(match.group(2))
    return family, factor


def _slice(split: SplitData, start: int, end: int) -> SplitData:
    return SplitData(
        values=split.values.iloc[start:end].reset_index(drop=True),
        timestamps=np.asarray(split.timestamps)[start:end],
        labels=None if split.labels is None else np.asarray(split.labels)[start:end],
        feature_names=split.feature_names,
        timestamp_kind=split.timestamp_kind,
    )


def load_eval_metadata(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = np.load(path, allow_pickle=True).item()
    if not isinstance(raw, dict) or "data_info" not in raw:
        raise ValueError(f"invalid DADA metadata: {path}")
    data_info = raw["data_info"]
    if not isinstance(data_info, dict):
        raise ValueError(f"DADA data_info is not a mapping: {path}")
    file_names = data_info.get("file_name", {})
    if not isinstance(file_names, dict):
        raise ValueError(f"DADA file_name metadata is not a mapping: {path}")
    records: dict[str, dict[str, Any]] = {}
    for index, file_name in file_names.items():
        record = {
            key: values.get(index)
            for key, values in data_info.items()
            if isinstance(values, dict)
        }
        record["meta_index"] = index
        name = str(file_name)
        if name in records:
            raise ValueError(f"duplicate DADA file metadata for {name!r}")
        records[name] = jsonable(record)
    return records, jsonable(raw.get("channel_info", {}))


def process_eval(
    input_dir: Path, source_dir: Path, project_root: Path
) -> None:
    meta_path = input_dir / "meta.npy"
    metadata, channel_info = load_eval_metadata(meta_path)
    files = sorted(input_dir.glob("*.csv"))
    if set(metadata) != {path.name for path in files}:
        missing_files = sorted(set(metadata) - {path.name for path in files})
        missing_meta = sorted({path.name for path in files} - set(metadata))
        raise ValueError(
            f"DADA metadata mismatch: missing_files={missing_files}, "
            f"missing_meta={missing_meta}"
        )

    grouped: dict[str, list[Path]] = {}
    for csv_path in files:
        source_meta = metadata[csv_path.name]
        dataset = str(source_meta.get("dataset_name") or csv_path.stem)
        if safe_name(dataset) != dataset:
            raise ValueError(f"unsafe DADA dataset name: {dataset!r}")
        grouped.setdefault(dataset, []).append(csv_path)

    for dataset, dataset_files in sorted(grouped.items()):
        selected_channels = channel_info.get(dataset)
        if selected_channels is None and len(dataset_files) == 1:
            selected_channels = channel_info.get(dataset_files[0].stem)
        dataset_dir = source_dir / dataset
        num_series = 0
        writer = DatasetWriter(
            dataset_dir,
            source=SOURCE,
            dataset=dataset,
            task="anomaly_detection",
            project_root=project_root,
        )
        for csv_path in dataset_files:
            source_meta = metadata[csv_path.name]
            split_at = strict_int(
                source_meta.get("train_lens"),
                name=f"DADA train_lens for {csv_path.name}",
            )
            full = long_to_split(pd.read_csv(csv_path, dtype={"cols": "string"}))
            if not 0 < split_at < len(full.values):
                raise ValueError(f"invalid DADA train length for {csv_path.name}")
            writer.add_series(
                series_id=("default" if len(dataset_files) == 1 else csv_path.stem),
                splits={
                    "train": _slice(full, 0, split_at),
                    "test": _slice(full, split_at, len(full.values)),
                },
                source_files=[csv_path, meta_path],
                source_metadata={
                    **source_meta,
                    "selected_channels": selected_channels,
                    "input_format": "long-date-data-cols",
                },
            )
            num_series += 1
        writer.finalize(source_metadata={"selected_channels": selected_channels})
        print(f"{SOURCE}/{dataset}: {num_series} series")


def process_monash(
    input_dir: Path, source_dir: Path, project_root: Path
) -> None:
    files = sorted(input_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no Monash CSV files under {input_dir}")
    stems = [path.stem for path in files]
    if len(stems) != len(set(stems)):
        raise ValueError("Monash contains duplicate series identifiers")

    grouped: dict[str, list[tuple[Path, int | None]]] = {}
    for path in files:
        family, downsample_factor = monash_identity(path)
        grouped.setdefault(family, []).append((path, downsample_factor))

    total = 0
    for family, family_files in sorted(grouped.items()):
        writer = DatasetWriter(
            source_dir / "Monash" / family,
            source=SOURCE,
            collection="Monash",
            dataset=family,
            task="pretraining",
            project_root=project_root,
        )
        for path, downsample_factor in family_files:
            frame = pd.read_csv(path)
            if frame.empty or frame.shape[1] < 2:
                raise ValueError(f"invalid Monash series: {path}")
            timestamp_col = str(frame.columns[0])
            source_feature_names = [str(column) for column in frame.columns[1:]]
            if len(source_feature_names) != 1:
                raise ValueError(f"Monash series is not univariate: {path}")
            writer.add_series(
                series_id=path.stem,
                splits={
                    "train": SplitData(
                        values=frame.iloc[:, 1:],
                        timestamps=frame.iloc[:, 0],
                        feature_names=["value"],
                        timestamp_kind="source",
                    )
                },
                source_files=[path],
                source_metadata={
                    "monash_family": family,
                    "timestamp_column": timestamp_col,
                    "source_feature_names": source_feature_names,
                    "downsample_factor": downsample_factor,
                },
            )
            total += 1
            if total % 1000 == 0:
                print(f"{SOURCE}/Monash: {total}/{len(files)}")
        writer.finalize(
            source_metadata={
                "monash_collection": "DADA pretraining",
                "family": family,
            }
        )
        print(f"{SOURCE}/Monash/{family}: {len(family_files)} series")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/DADA"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--only", choices=["eval", "monash"])
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    input_root = args.input_dir.resolve()
    with staged_source_directory(args.output_dir.resolve(), SOURCE) as source_dir:
        if args.only in (None, "eval"):
            process_eval(
                input_root / "eval" / "dataset" / "evaluation_dataset",
                source_dir,
                project_root,
            )
        if args.only in (None, "monash"):
            process_monash(
                input_root / "monash" / "dataset" / "Monash",
                source_dir,
                project_root,
            )


if __name__ == "__main__":
    sys.exit(main())
