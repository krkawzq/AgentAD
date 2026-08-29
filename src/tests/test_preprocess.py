from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from agentad.series import read
from scripts.preprocess.common import DatasetWriter, SplitData
from scripts.preprocess.process_aerca import MSDS_METRIC_FILES, load_msds_frames
from scripts.preprocess.process_gift_eval import gift_record, process_dataset
from scripts.preprocess.process_granite_tsfm import zafnoo_series
from scripts.preprocess.process_time_rcd import time_rcd_sample
from scripts.preprocess.process_wadi import load_wadi_csv


def test_dataset_writer_packs_source_label_columns(tmp_path: Path) -> None:
    writer = DatasetWriter(
        tmp_path / "source" / "toy",
        source="source",
        dataset="toy",
        task="anomaly_detection",
        project_root=tmp_path,
    )
    writer.add_series(
        series_id="default",
        splits={
            "test": SplitData(
                values=np.arange(6, dtype=np.float32).reshape(3, 2),
                timestamps=np.arange(3),
                labels=np.array([0, 1, 0]),
                feature_names=["a", "b"],
                label_columns=pd.DataFrame(
                    {"root_cause_a": [0, 1, 0], "root_cause_b": [0, 0, 0]}
                ),
            )
        },
    )
    writer.finalize()

    packed = read(tmp_path / "source" / "toy" / "test.zarr.zip")
    assert list(packed.labels.columns) == [
        "label",
        "root_cause_a",
        "root_cause_b",
    ]
    assert packed.labels.to_numpy().tolist() == [[0, 0, 0], [1, 1, 0], [0, 0, 0]]


def test_load_msds_frames_preserves_point_and_root_cause_labels(
    tmp_path: Path,
) -> None:
    timestamps = [f"2020-01-01 00:{index:02d}:00+0000" for index in range(20)]
    for file_index, name in enumerate(MSDS_METRIC_FILES):
        pd.DataFrame(
            {
                "now": timestamps,
                "cpu": np.arange(20) + file_index,
                "memory": np.arange(20) * 2 + file_index,
                "load.cpucore": 1,
                "load.min1": 1,
                "load.min5": 1,
                "load.min15": 1,
            }
        ).to_csv(tmp_path / name, index=False)
    labels_path = tmp_path / "labels.csv"
    labels = pd.DataFrame(np.zeros((9, 10), dtype=np.int8))
    labels.iloc[3, 4] = 1
    labels.to_csv(labels_path)

    train, test, root_causes, source_files, metadata = load_msds_frames(
        tmp_path, labels_path
    )

    assert train.shape == (9, 10)
    assert test.shape == (9, 10)
    assert root_causes.shape == (9, 10)
    assert root_causes.max(axis=1).tolist() == [0, 0, 0, 1, 0, 0, 0, 0, 0]
    assert len(source_files) == 6
    assert metadata["burn_in_rows"] == 2


def test_load_wadi_csv_handles_official_minus_one_attack_labels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "WADI_attackdataLABLE.csv"
    pd.DataFrame(
        {
            "Row": [1, 2, 3],
            "Date": ["10/9/2017"] * 3,
            "Time": ["18:00:00", "18:00:01", "18:00:02"],
            "sensor_a": [1.0, 2.0, 3.0],
            "sensor_b": [4.0, 5.0, 6.0],
            "Attack": [1, -1, 1],
        }
    ).to_csv(path, index=False)

    features, timestamps, labels, source_labels, metadata = load_wadi_csv(
        path, require_labels=True
    )

    assert list(features.columns) == ["sensor_a", "sensor_b"]
    assert timestamps.tolist()[1] == "10/9/2017 18:00:01"
    assert labels.tolist() == [0, 1, 0]
    assert source_labels["source_attack_label"].tolist() == [1, -1, 1]
    assert metadata["timestamp_source"] == "date+time"


def test_zafnoo_series_accepts_long_and_wide_layouts(tmp_path: Path) -> None:
    long_path = tmp_path / "long.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"] * 2,
            "data": [1.0, 2.0, 3.0, 4.0],
            "cols": ["north", "north", "south", "south"],
        }
    ).to_csv(long_path, index=False)
    long_records = list(zafnoo_series(long_path))
    assert [record[0] for record in long_records] == ["north", "south"]
    assert long_records[0][1].values.shape == (2, 1)

    wide_path = tmp_path / "wide.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "north": [1.0, 2.0],
            "south": [3.0, 4.0],
        }
    ).to_csv(wide_path, index=False)
    wide_records = list(zafnoo_series(wide_path))
    assert len(wide_records) == 1
    assert wide_records[0][1].values.shape == (2, 2)


def test_gift_record_preserves_targets_and_dynamic_covariates() -> None:
    series_id, split, metadata = gift_record(
        {
            "item_id": "sample",
            "start": pd.Timestamp("2020-01-01"),
            "freq": "D",
            "target": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "past_feat_dynamic_real": [[7.0, 8.0, 9.0]],
            "static_category": 2,
        },
        0,
    )

    assert series_id == "sample"
    assert split.values.shape == (3, 3)
    assert split.feature_names == [
        "target_0",
        "target_1",
        "past_feat_dynamic_real_0",
    ]
    assert metadata["target_features"] == 2
    assert metadata["dynamic_covariate_features"] == 1
    assert metadata["static_category"] == 2


def test_process_gift_eval_arrow_payload(tmp_path: Path) -> None:
    input_root = tmp_path / "raw"
    dataset_dir = input_root / "m4_yearly"
    dataset_dir.mkdir(parents=True)
    arrow_path = dataset_dir / "data-00000-of-00001.arrow"
    table = pa.Table.from_pylist(
        [
            {
                "item_id": "Y1",
                "start": pd.Timestamp("2000-01-01"),
                "freq": "YS",
                "target": [1.0, 2.0, 3.0],
            }
        ]
    )
    with pa.OSFile(str(arrow_path), "wb") as sink:
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    (dataset_dir / "dataset_info.json").write_text(
        json.dumps({"features": {"target": "float32"}}), encoding="utf-8"
    )
    (dataset_dir / "state.json").write_text("{}", encoding="utf-8")

    source_dir = tmp_path / "processed" / "gift-eval"
    process_dataset(input_root, arrow_path, source_dir, tmp_path)

    packed = read(source_dir / "m4_yearly" / "data.zarr.zip")
    assert packed.ids == ("Y1",)
    assert packed.data[:, 0].tolist() == [1.0, 2.0, 3.0]


def test_time_rcd_sample_keeps_normal_reference_in_label_columns() -> None:
    series_id, split, metadata = time_rcd_sample(
        {
            "normal_time_series": [[1.0, 2.0], [3.0, 4.0]],
            "time_series": [[1.0, 2.0], [30.0, 4.0]],
            "labels": [0, 1],
            "attribute": {"pattern": "spike"},
        },
        7,
    )

    assert series_id == "sample-00000007"
    assert split.values.shape == (2, 2)
    assert split.label_columns.columns.tolist() == [
        "normal_value_0",
        "normal_value_1",
    ]
    assert split.labels.tolist() == [0, 1]
    assert metadata == {"attribute": {"pattern": "spike"}}
