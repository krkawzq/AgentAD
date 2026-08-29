"""Convert trusted Time-RCD generated pickle corpora to SeriesData."""

from __future__ import annotations

import argparse
import pickle
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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


SOURCE = "time-rcd"


def time_matrix(value: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(value)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty time-by-feature array")
    if not (
        np.issubdtype(matrix.dtype, np.number)
        or np.issubdtype(matrix.dtype, np.bool_)
    ):
        raise TypeError(f"{name} must be numeric, got {matrix.dtype}")
    return matrix


def time_rcd_sample(
    sample: Mapping[str, Any], index: int
) -> tuple[str, SplitData, dict[str, Any]]:
    required = {"normal_time_series", "time_series", "labels", "attribute"}
    missing = required - set(sample)
    if missing:
        raise ValueError(f"Time-RCD sample is missing {sorted(missing)}")
    values = time_matrix(sample["time_series"], name="time_series")
    normal = time_matrix(sample["normal_time_series"], name="normal_time_series")
    if normal.shape != values.shape:
        raise ValueError(
            f"Time-RCD normal/anomalous shapes differ: {normal.shape} != {values.shape}"
        )
    labels = np.asarray(sample["labels"]).reshape(-1)
    if len(labels) != len(values) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Time-RCD labels must be a point-aligned binary vector")
    feature_names = [f"value_{position}" for position in range(values.shape[1])]
    normal_columns = pd.DataFrame(
        normal,
        columns=[f"normal_value_{position}" for position in range(normal.shape[1])],
    )
    return (
        f"sample-{index:08d}",
        SplitData(
            values=values,
            timestamps=np.arange(len(values), dtype=np.int64),
            labels=labels.astype(np.int8),
            feature_names=feature_names,
            label_columns=normal_columns,
        ),
        {"attribute": jsonable(sample["attribute"])},
    )


def load_trusted_pickle(path: Path) -> Sequence[Mapping[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise TypeError(f"Time-RCD pickle root must be a sequence: {path}")
    if any(not isinstance(sample, Mapping) for sample in payload):
        raise TypeError(f"Time-RCD pickle contains a non-mapping sample: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/raw/TimeRCD/datasets")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    project_root = Path.cwd().resolve()
    input_dir = args.input_dir.resolve()
    pickle_paths = sorted(input_dir.glob("*.pkl"))
    if not pickle_paths:
        raise FileNotFoundError(f"no trusted Time-RCD pickle files under {input_dir}")
    names = [safe_name(path.stem) for path in pickle_paths]
    if len(names) != len(set(names)):
        raise ValueError("Time-RCD pickle names collide after path normalization")

    with staged_source_directory(args.output_dir.resolve(), SOURCE) as source_dir:
        for dataset, path in zip(names, pickle_paths, strict=True):
            samples = load_trusted_pickle(path)
            writer = DatasetWriter(
                source_dir / dataset,
                source=SOURCE,
                dataset=dataset,
                task="anomaly_detection",
                project_root=project_root,
            )
            for index, sample in enumerate(samples):
                series_id, split, metadata = time_rcd_sample(sample, index)
                writer.add_series(
                    series_id=series_id,
                    splits={"data": split},
                    source_metadata=metadata,
                )
            writer.finalize(
                source_metadata={
                    "generator": "thu-sail-lab/Datasets-RCD",
                    "source_file": source_fingerprint(path, project_root),
                    "recommended_random_split": {
                        "train_ratio": 0.95,
                        "seed": 42,
                    },
                    "normal_time_series_storage": "labels.normal_value_*",
                }
            )
            print(f"{SOURCE}/{dataset}: {len(samples)} series")


if __name__ == "__main__":
    sys.exit(main())
