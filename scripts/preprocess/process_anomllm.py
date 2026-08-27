"""Convert AnomLLM synthetic series without duplicating train/eval samples."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .common import (
        DatasetWriter,
        SplitData,
        staged_source_directory,
        strict_int,
    )
else:
    from common import (
        DatasetWriter,
        SplitData,
        staged_source_directory,
        strict_int,
    )


SOURCE = "anomllm"


def flatten_intervals(annotation: Any, length: int) -> list[tuple[int, int]]:
    if not isinstance(annotation, (list, tuple)):
        raise ValueError(f"invalid AnomLLM annotation: {annotation!r}")
    intervals: list[tuple[int, int]] = []
    for segment in annotation:
        if not isinstance(segment, (list, tuple)):
            raise ValueError(f"invalid AnomLLM segment: {segment!r}")
        for interval in segment:
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                raise ValueError(f"invalid AnomLLM interval: {interval!r}")
            start = strict_int(interval[0], name="AnomLLM interval start")
            end = strict_int(interval[1], name="AnomLLM interval end")
            if not 0 <= start < end <= length:
                raise ValueError(f"out-of-range AnomLLM interval: {(start, end)}")
            intervals.append((start, end))
    return intervals


def load_pickle(path: Path) -> tuple[list[Any], list[Any]]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict) or set(data) != {"series", "anom"}:
        raise ValueError(f"unexpected AnomLLM payload keys in {path}")
    series = data["series"]
    annotations = data["anom"]
    if len(series) != len(annotations):
        raise ValueError(f"series/annotation count mismatch in {path}")
    return series, annotations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/raw/anomllm/data/synthetic")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    input_dir = args.input_dir.resolve()
    with staged_source_directory(args.output_dir.resolve(), SOURCE) as source_dir:
        type_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
        if not type_dirs:
            raise FileNotFoundError(f"no AnomLLM types under {input_dir}")
        for type_dir in type_dirs:
            dataset = type_dir.name
            dataset_dir = source_dir / dataset
            num_series = 0
            writer = DatasetWriter(
                dataset_dir,
                source=SOURCE,
                dataset=dataset,
                task="anomaly_detection",
                project_root=project_root,
            )
            for source_split, output_split in (("train", "train"), ("eval", "eval")):
                pkl_path = type_dir / source_split / "data.pkl"
                if not pkl_path.is_file():
                    raise FileNotFoundError(pkl_path)
                series_list, annotation_list = load_pickle(pkl_path)
                for index, (raw_series, raw_annotation) in enumerate(
                    zip(series_list, annotation_list, strict=True)
                ):
                    values = np.asarray(raw_series)
                    if values.ndim == 1:
                        values = values.reshape(-1, 1)
                    if values.ndim != 2:
                        raise ValueError(
                            f"invalid series shape {values.shape} in {pkl_path}"
                        )
                    intervals = flatten_intervals(raw_annotation, len(values))
                    labels = np.zeros(len(values), dtype=np.int8)
                    for start, end in intervals:
                        labels[start:end] = 1
                    annotations = [
                        {
                            "split": output_split,
                            "start": start,
                            "end": end - 1,
                            "start_index": start,
                            "end_index": end - 1,
                            "label": 1,
                            "source_end_is_exclusive": True,
                        }
                        for start, end in intervals
                    ]
                    series_id = f"{source_split}-{index:04d}"
                    writer.add_series(
                        series_id=series_id,
                        splits={
                            output_split: SplitData(
                                values=values,
                                timestamps=np.arange(len(values)),
                                labels=labels,
                                feature_names=[
                                    f"value_{i}" for i in range(values.shape[1])
                                ],
                            )
                        },
                        source_files=[pkl_path],
                        source_metadata={
                            "source_split": source_split,
                            "source_index": index,
                        },
                        annotations=annotations,
                    )
                    num_series += 1
            writer.finalize(source_metadata={"domain": "synthetic"})
            print(f"{SOURCE}/{dataset}: {num_series} series")


if __name__ == "__main__":
    sys.exit(main())
