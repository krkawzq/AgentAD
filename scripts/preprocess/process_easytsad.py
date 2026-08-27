"""Convert all EasyTSAD MTS/UTS arrays with per-series metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__:
    from .common import DatasetWriter, SplitData, staged_source_directory
else:
    from common import DatasetWriter, SplitData, staged_source_directory


SOURCE = "easytsad"
REQUIRED = ("train.npy", "test.npy", "train_label.npy", "test_label.npy")


def series_id_for(dataset_dir: Path, array_dir: Path) -> str:
    relative = array_dir.relative_to(dataset_dir)
    if relative.parts in ((), ("AllInOne",)):
        return "default"
    return relative.as_posix()


def load_source_info(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"EasyTSAD info.json is not an object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/EasyTSAD"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--only", choices=["MTS", "UTS"])
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    input_root = args.input_dir.resolve()
    with staged_source_directory(args.output_dir.resolve(), SOURCE) as source_dir:
        seen_datasets: set[str] = set()

        families = [args.only] if args.only else ["MTS", "UTS"]
        for family in families:
            family_dir = input_root / family
            if not family_dir.is_dir():
                raise FileNotFoundError(family_dir)
            for raw_dataset_dir in sorted(
                path for path in family_dir.iterdir() if path.is_dir()
            ):
                dataset = raw_dataset_dir.name
                if dataset in seen_datasets:
                    raise ValueError(
                        f"duplicate EasyTSAD dataset across families: {dataset}"
                    )
                seen_datasets.add(dataset)
                array_dirs = sorted(
                    path.parent for path in raw_dataset_dir.rglob("train.npy")
                )
                if not array_dirs:
                    raise FileNotFoundError(f"no train.npy under {raw_dataset_dir}")
                dataset_dir = source_dir / dataset
                num_series = 0
                writer = DatasetWriter(
                    dataset_dir,
                    source=SOURCE,
                    dataset=dataset,
                    task="anomaly_detection",
                    project_root=project_root,
                )
                for array_dir in array_dirs:
                    missing = [
                        name for name in REQUIRED if not (array_dir / name).is_file()
                    ]
                    if missing:
                        raise FileNotFoundError(f"missing {missing} under {array_dir}")
                    train = np.load(
                        array_dir / "train.npy", mmap_mode="r", allow_pickle=False
                    )
                    test = np.load(
                        array_dir / "test.npy", mmap_mode="r", allow_pickle=False
                    )
                    train_label = np.load(
                        array_dir / "train_label.npy", mmap_mode="r", allow_pickle=False
                    )
                    test_label = np.load(
                        array_dir / "test_label.npy", mmap_mode="r", allow_pickle=False
                    )
                    if train.ndim not in (1, 2) or test.ndim not in (1, 2):
                        raise ValueError(
                            f"EasyTSAD arrays must be 1D or 2D under {array_dir}"
                        )
                    train_rows = len(train)
                    test_rows = len(test)
                    feature_count = 1 if train.ndim == 1 else train.shape[1]
                    if (1 if test.ndim == 1 else test.shape[1]) != feature_count:
                        raise ValueError(
                            f"train/test feature mismatch under {array_dir}"
                        )
                    info_path = array_dir / "info.json"
                    source_files = [array_dir / name for name in REQUIRED]
                    if info_path.exists():
                        source_files.append(info_path)
                    feature_names = [f"feature_{i}" for i in range(feature_count)]
                    writer.add_series(
                        series_id=series_id_for(raw_dataset_dir, array_dir),
                        splits={
                            "train": SplitData(
                                train, np.arange(train_rows), train_label, feature_names
                            ),
                            "test": SplitData(
                                test,
                                np.arange(train_rows, train_rows + test_rows),
                                test_label,
                                feature_names,
                            ),
                        },
                        source_files=source_files,
                        source_metadata={
                            "family": family,
                            "source_info": load_source_info(info_path),
                            "source_array_dtypes": {
                                "train": str(train.dtype),
                                "test": str(test.dtype),
                                "train_label": str(train_label.dtype),
                                "test_label": str(test_label.dtype),
                            },
                        },
                    )
                    num_series += 1
                writer.finalize(source_metadata={"family": family})
                print(f"{SOURCE}/{dataset}: {num_series} series")


if __name__ == "__main__":
    sys.exit(main())
