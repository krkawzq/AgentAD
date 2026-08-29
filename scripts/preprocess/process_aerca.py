"""Convert AERCA's MSDS and generated synthetic datasets to SeriesData."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__:
    from .common import DatasetWriter, SplitData, jsonable, staged_source_directory
else:
    from common import DatasetWriter, SplitData, jsonable, staged_source_directory


SOURCE = "aerca"
SYNTHETIC_DIRECTORIES = {
    "linear": ("linear",),
    "nonlinear": ("nonlinear",),
    "lorenz96": ("lorenz96",),
    "lotka_volterra": ("lotka_volterra", "lv"),
}
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
    metric_paths = [find_unique(input_dir, name) for name in MSDS_METRIC_FILES]
    frames: list[pd.DataFrame] = []
    retained_columns: dict[str, list[str]] = {}
    for path in metric_paths:
        raw = pd.read_csv(path)
        if "now" not in raw.columns:
            raise ValueError(f"MSDS metric file has no 'now' column: {path}")
        if raw["now"].isna().any() or raw["now"].duplicated().any():
            raise ValueError(f"MSDS timestamps are empty or duplicated: {path}")
        feature_columns = [
            str(column)
            for column in raw.columns
            if column != "now" and column not in MSDS_DROPPED_COLUMNS
        ]
        if not feature_columns:
            raise ValueError(f"MSDS metric file has no retained features: {path}")
        values = raw.loc[:, feature_columns].apply(pd.to_numeric, errors="raise")
        node = path.name.removesuffix("_metrics_concurrent.csv")
        values.columns = [f"{column}__{node}" for column in feature_columns]
        values.index = pd.Index(raw["now"].astype(str), name="now")
        frames.append(values)
        retained_columns[node] = feature_columns

    merged = pd.concat(frames, axis=1, join="inner", copy=False).sort_index(
        kind="stable"
    )
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
    if len(root_causes) != len(test):
        raise ValueError(
            f"MSDS labels/test length mismatch: {len(root_causes)} != {len(test)}"
        )
    if not np.isin(root_causes.to_numpy(), [0, 1]).all():
        raise ValueError("MSDS root-cause labels must be binary")
    root_causes.columns = [f"root_cause_{column}" for column in root_causes.columns]
    metadata = {
        "metric_files": list(MSDS_METRIC_FILES),
        "retained_columns": retained_columns,
        "dropped_columns": list(MSDS_DROPPED_COLUMNS),
        "timestamp_alignment": "stable sorted inner join",
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


def load_npy(path: Path, *, mmap: bool = True) -> np.ndarray:
    return np.load(path, allow_pickle=False, mmap_mode="r" if mmap else None)


def point_and_source_labels(
    raw: np.ndarray, *, rows: int, feature_count: int
) -> tuple[np.ndarray, pd.DataFrame]:
    labels = np.asarray(raw)
    if labels.ndim == 1:
        if len(labels) != rows:
            raise ValueError("synthetic label length mismatch")
        point = labels
        frame = pd.DataFrame({"source_label": labels})
    elif labels.ndim == 2:
        if labels.shape[0] != rows:
            raise ValueError("synthetic root-cause label length mismatch")
        point = np.any(labels != 0, axis=1).astype(np.int8)
        frame = pd.DataFrame(
            labels,
            columns=[f"root_cause_{index}" for index in range(labels.shape[1])],
        )
        if labels.shape[1] not in {1, feature_count}:
            raise ValueError(
                "synthetic root-cause width differs from the feature count"
            )
    else:
        raise ValueError(f"invalid synthetic label shape: {labels.shape}")
    point = np.asarray(point)
    if not np.isin(point, [0, 1]).all():
        raise ValueError("synthetic point labels must be binary")
    return point.astype(np.int8), frame


def process_synthetic(
    dataset: str,
    dataset_dir: Path,
    source_dir: Path,
    project_root: Path,
    training_size: int,
) -> None:
    required = {
        name: dataset_dir / f"{name}.npy"
        for name in ("x_n_list", "x_ab_list", "label_list")
    }
    missing = [path for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete AERCA {dataset}: {missing}")
    normal = load_npy(required["x_n_list"])
    anomalous = load_npy(required["x_ab_list"])
    raw_labels = load_npy(required["label_list"])
    if normal.ndim != 3 or anomalous.shape != normal.shape:
        raise ValueError(
            f"invalid AERCA {dataset} arrays: {normal.shape}, {anomalous.shape}"
        )
    if raw_labels.shape[:2] != normal.shape[:2]:
        raise ValueError(
            f"AERCA {dataset} label shape {raw_labels.shape} does not match "
            f"data {normal.shape}"
        )
    if not 0 < training_size < len(normal):
        raise ValueError(
            f"synthetic training size {training_size} is invalid for {len(normal)}"
        )
    rows, feature_count = normal.shape[1:]
    feature_names = [f"value_{index}" for index in range(feature_count)]

    noise_paths = {
        variant: dataset_dir / f"eps_{suffix}_list.npy"
        for variant, suffix in (("normal", "n"), ("anomalous", "ab"))
    }
    noise_arrays: dict[str, np.ndarray] = {}
    for variant, path in noise_paths.items():
        if path.is_file():
            values = load_npy(path)
            if values.shape != normal.shape:
                raise ValueError(f"AERCA {dataset} noise shape mismatch: {path}")
            noise_arrays[variant] = values

    source_files = [*required.values(), *noise_paths.values()]
    source_files = [path for path in source_files if path.is_file()]
    writer = DatasetWriter(
        source_dir / dataset,
        source=SOURCE,
        dataset=dataset,
        task="anomaly_detection",
        project_root=project_root,
    )
    for index in range(len(normal)):
        split = "train" if index < training_size else "eval"
        _, source_labels = point_and_source_labels(
            np.asarray(raw_labels[index]), rows=rows, feature_count=feature_count
        )
        normal_columns = source_labels.copy()
        normal_columns.loc[:, :] = 0
        if "normal" in noise_arrays:
            for feature in range(feature_count):
                normal_columns[f"noise_{feature}"] = noise_arrays["normal"][
                    index, :, feature
                ]
        writer.add_series(
            series_id=f"normal-{index:04d}",
            splits={
                split: SplitData(
                    normal[index],
                    np.arange(rows),
                    np.zeros(rows, dtype=np.int8),
                    feature_names,
                    label_columns=normal_columns,
                )
            },
            source_files=source_files,
            source_metadata={"variant": "normal", "source_index": index},
        )
        if index < training_size:
            continue
        point_labels, anomaly_columns = point_and_source_labels(
            np.asarray(raw_labels[index]), rows=rows, feature_count=feature_count
        )
        if "anomalous" in noise_arrays:
            for feature in range(feature_count):
                anomaly_columns[f"noise_{feature}"] = noise_arrays["anomalous"][
                    index, :, feature
                ]
        writer.add_series(
            series_id=f"anomalous-{index:04d}",
            splits={
                "eval": SplitData(
                    anomalous[index],
                    np.arange(rows),
                    point_labels,
                    feature_names,
                    label_columns=anomaly_columns,
                )
            },
            source_files=source_files,
            source_metadata={"variant": "anomalous", "source_index": index},
        )

    structural_metadata: dict[str, Any] = {
        "training_size": training_size,
        "normal_series": len(normal),
        "evaluated_anomalous_series": len(normal) - training_size,
    }
    for name in ("causal_struct", "signed_causal_struct", "causal_struct_value", "a"):
        path = dataset_dir / f"{name}.npy"
        if path.is_file():
            structural_metadata[name] = jsonable(load_npy(path, mmap=False))
    writer.finalize(source_metadata=structural_metadata)
    print(
        f"{SOURCE}/{dataset}: {len(normal)} normal + "
        f"{len(normal) - training_size} anomalous series"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/AERCA"))
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("data/raw/AERCA/msds/labels.csv"),
    )
    parser.add_argument("--synthetic-training-size", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    input_root = args.input_dir.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    labels_path = args.labels_path.resolve()
    available_synthetic: dict[str, Path] = {}
    for dataset, directory_names in SYNTHETIC_DIRECTORIES.items():
        matches = [
            input_root / name
            for name in directory_names
            if (input_root / name).is_dir()
        ]
        if len(matches) > 1:
            raise ValueError(
                f"multiple AERCA directories found for {dataset}: {matches}"
            )
        if matches:
            available_synthetic[dataset] = matches[0]
    msds_dir = input_root / "msds"
    has_msds = msds_dir.is_dir() and labels_path.is_file()
    if not has_msds and not available_synthetic:
        raise FileNotFoundError(f"no AERCA datasets found under {input_root}")

    with staged_source_directory(args.output_dir.resolve(), SOURCE) as source_dir:
        if has_msds:
            process_msds(msds_dir, labels_path, source_dir, project_root)
        for dataset, dataset_dir in available_synthetic.items():
            process_synthetic(
                dataset,
                dataset_dir,
                source_dir,
                project_root,
                args.synthetic_training_size,
            )


if __name__ == "__main__":
    sys.exit(main())
