import gzip
import json
import stat
import zipfile

import numpy as np
import pandas as pd
import pytest
import zarr

import agentad
from agentad import SeriesData
from agentad.series import read, write
from agentad.series import _write as series_write


def make_packed() -> SeriesData:
    data = np.arange(30, dtype=np.float64).reshape(15, 2)
    timestamps = np.arange(15, dtype=np.int64)
    labels = pd.DataFrame({"is_anomaly": np.zeros(15, dtype=np.int64)})
    offsets = np.array([0, 5, 8, 15], dtype=np.int64)
    features = pd.DataFrame({"unit": ["m/s", "degC"]}, index=["speed", "temp"])
    return SeriesData(
        features=features,
        labels=labels,
        data=data,
        timestamps=timestamps,
        offsets=offsets,
        ids=["s0", "s1", "s2"],
        metas=[{"group": 0}, {"group": 1}, {"group": 2}],
        manifest={"name": "demo"},
    )


@pytest.fixture
def sdata() -> SeriesData:
    return make_packed()


class TestWriteRead:
    def test_top_level_io_exports(self):
        assert agentad.read is read
        assert agentad.write is write

    def test_write_validates_boundary_arguments(self, sdata, tmp_path):
        with pytest.raises(TypeError, match="SeriesData"):
            write(object(), tmp_path / "x.zip")
        with pytest.raises(TypeError, match="zipped"):
            write(sdata, tmp_path / "x.zip", zipped="yes")
        with pytest.raises(TypeError, match="create_parents"):
            write(sdata, tmp_path / "x.zip", create_parents=1)

    def test_write_can_create_parent_directories(self, sdata, tmp_path):
        target = tmp_path / "nested" / "path" / "x.zip"

        with pytest.raises(FileNotFoundError, match="parent directory"):
            write(sdata, target)

        write(sdata, target, create_parents=True)
        assert read(target).ids == sdata.ids

    def test_round_trip_zip(self, sdata, tmp_path):
        target = tmp_path / "zarr.zip"
        write(sdata, target)
        loaded = read(target)

        assert loaded.ids == sdata.ids
        assert loaded.total_points == sdata.total_points
        assert loaded.n_features == sdata.n_features
        assert loaded.data.dtype == sdata.data.dtype
        assert np.array_equal(loaded.data, sdata.data)
        assert np.array_equal(loaded.timestamps, sdata.timestamps)
        assert loaded.offsets.tolist() == sdata.offsets.tolist()
        pd.testing.assert_frame_equal(loaded.features, sdata.features)
        pd.testing.assert_frame_equal(loaded.labels, sdata.labels)
        assert [dict(meta) for meta in loaded.metas] == [
            dict(meta) for meta in sdata.metas
        ]
        assert loaded.manifest == dict(sdata.manifest)
        assert loaded["s1"].data.shape == (3, 2)
        assert loaded["s1"].sdata is loaded

    def test_round_trip_directory(self, sdata, tmp_path):
        target = tmp_path / "unpacked"
        write(sdata, target, zipped=False)
        loaded = read(target)
        assert loaded.ids == sdata.ids
        assert np.array_equal(loaded.data, sdata.data)
        pd.testing.assert_frame_equal(loaded.labels, sdata.labels)

    def test_dirty_labels_are_folded_on_write(self, sdata, tmp_path):
        item = sdata["s1"]
        item.labels = pd.DataFrame({"is_anomaly": [1, 0, 1]})
        target = tmp_path / "zarr.zip"
        write(sdata, target)
        loaded = read(target)
        assert loaded.labels["is_anomaly"].tolist() == ([0] * 5 + [1, 0, 1] + [0] * 7)

    def test_write_through_data_is_persisted(self, sdata, tmp_path):
        sdata[0].data[0, 0] = 42.0
        sdata[2].timestamps[0] = 777
        target = tmp_path / "zarr.zip"
        write(sdata, target)
        loaded = read(target)
        assert loaded.data[0, 0] == 42.0
        assert loaded.timestamps[8] == 777

    @pytest.mark.parametrize("mutation", ["length", "dtype", "offsets"])
    def test_structural_array_mutation_rejected_before_publish(
        self, sdata, tmp_path, mutation
    ):
        if mutation == "length":
            sdata.timestamps.resize(len(sdata.timestamps) - 1, refcheck=False)
        elif mutation == "dtype":
            with pytest.warns(DeprecationWarning):
                sdata.timestamps.dtype = np.uint64
        else:
            sdata.offsets.flags.writeable = True
            sdata.offsets[1] = sdata.offsets[2] + 1
        target = tmp_path / "zarr.zip"

        with pytest.raises((TypeError, ValueError)):
            write(sdata, target)

        assert not target.exists()

    def test_float32_dtype_preserved(self, tmp_path):
        sdata = SeriesData(
            features=pd.DataFrame(index=["f0"]),
            labels=pd.DataFrame({"x": [0, 1]}),
            data=np.zeros((2, 1), dtype=np.float32),
            timestamps=[0, 1],
            offsets=[0, 2],
            ids=["a"],
        )
        target = tmp_path / "zarr.zip"
        write(sdata, target)
        loaded = read(target)
        assert loaded.data.dtype == np.float32

    def test_unsupported_extended_data_dtype_rejected_before_publish(self, tmp_path):
        if np.dtype(np.longdouble).itemsize <= np.dtype(np.float64).itemsize:
            pytest.skip("platform longdouble is represented as float64")
        sdata = SeriesData(
            features=pd.DataFrame(index=["f0"]),
            labels=pd.DataFrame({"x": [0, 1]}),
            data=np.ones((2, 1), dtype=np.longdouble),
            timestamps=[0, 1],
            offsets=[0, 2],
            ids=["a"],
        )
        target = tmp_path / "zarr.zip"

        with pytest.raises(TypeError, match="represented losslessly in Zarr"):
            write(sdata, target)

        assert not target.exists()

    def test_empty_collection(self, tmp_path):
        sdata = SeriesData(
            features=pd.DataFrame(index=["f0"]),
            labels=pd.DataFrame(index=pd.RangeIndex(0)),
            data=np.empty((0, 1)),
            timestamps=np.empty(0, dtype=np.int64),
            offsets=np.array([0], dtype=np.int64),
            ids=[],
            manifest={"name": "empty"},
        )
        target = tmp_path / "zarr.zip"
        write(sdata, target)
        loaded = read(target)
        assert len(loaded) == 0
        assert loaded.data.shape == (0, 1)
        assert loaded.manifest == {"name": "empty"}

    def test_empty_categorical_schemas_round_trip(self, tmp_path):
        features = pd.DataFrame(
            {"kind": pd.Series(pd.Categorical([], categories=["sensor", "derived"]))}
        )
        labels = pd.DataFrame(
            {
                "state": pd.Series(
                    pd.Categorical([], categories=["normal", "alert"], ordered=True)
                )
            }
        )
        sdata = SeriesData.empty(features=features, labels=labels)
        target = tmp_path / "zarr.zip"

        write(sdata, target)
        loaded = read(target)

        assert loaded.features["kind"].cat.categories.tolist() == [
            "sensor",
            "derived",
        ]
        assert loaded.labels["state"].cat.categories.tolist() == [
            "normal",
            "alert",
        ]
        assert loaded.labels["state"].cat.ordered

    @pytest.mark.parametrize("zipped", [False, True])
    def test_zero_column_labels_round_trip(self, tmp_path, zipped):
        labels = pd.DataFrame(index=pd.RangeIndex(2))
        labels.attrs = {"kind": "empty"}
        sdata = SeriesData(
            features=pd.DataFrame(index=["f0"]),
            labels=labels,
            data=np.zeros((2, 1)),
            timestamps=[0, 1],
            offsets=[0, 2],
            ids=["a"],
        )
        target = tmp_path / ("zarr.zip" if zipped else "unpacked")

        write(sdata, target, zipped=zipped)
        loaded = read(target)

        assert loaded.labels.shape == (2, 0)
        assert loaded.labels.attrs == {"kind": "empty"}

    @pytest.mark.parametrize("zipped", [False, True])
    @pytest.mark.parametrize(
        "columns",
        [
            pd.Index([], dtype=object, name="label_axis"),
            pd.MultiIndex.from_arrays([[], []], names=["group", "kind"]),
        ],
        ids=["index", "multiindex"],
    )
    def test_zero_column_labels_preserve_column_axis(
        self, tmp_path, zipped, columns
    ):
        labels = pd.DataFrame(index=pd.RangeIndex(2))
        labels.columns = columns.copy()
        sdata = SeriesData(
            features=pd.DataFrame(index=["f0"]),
            labels=labels,
            data=np.zeros((2, 1)),
            timestamps=[0, 1],
            offsets=[0, 2],
            ids=["a"],
        )
        target = tmp_path / ("zarr.zip" if zipped else "unpacked")

        write(sdata, target, zipped=zipped)
        loaded = read(target)

        assert loaded.labels.columns.identical(columns)

    def test_zero_column_labels_do_not_materialize_range_index(self, tmp_path):
        rows = 100_000
        sdata = SeriesData(
            features=pd.DataFrame(index=["f0"]),
            labels=pd.DataFrame(index=pd.RangeIndex(rows)),
            data=np.zeros((rows, 1)),
            timestamps=np.arange(rows, dtype=np.int64),
            offsets=np.array([0, rows], dtype=np.int64),
            ids=["a"],
        )
        target = tmp_path / "zarr.zip"

        write(sdata, target)

        with zipfile.ZipFile(target) as archive:
            assert archive.getinfo(series_write.FILE_LABELS).file_size < 10_000
        assert read(target).labels.shape == (rows, 0)

    @pytest.mark.parametrize("zipped", [False, True])
    def test_non_string_label_columns_round_trip(self, tmp_path, zipped):
        sdata = SeriesData(
            features=pd.DataFrame(index=["f0"]),
            labels=pd.DataFrame({0: [0, 1]}),
            data=np.zeros((2, 1)),
            timestamps=[0, 1],
            offsets=[0, 2],
            ids=["a"],
        )
        target = tmp_path / ("zarr.zip" if zipped else "unpacked")

        write(sdata, target, zipped=zipped)
        loaded = read(target)

        assert loaded.labels.columns.tolist() == [0]

    def test_lossy_parquet_column_schema_rejected_before_publish(self, tmp_path):
        columns = pd.Index([("kind", 1)], tupleize_cols=False)
        sdata = SeriesData(
            features=pd.DataFrame(index=["f0"]),
            labels=pd.DataFrame([[0], [1]], columns=columns),
            data=np.zeros((2, 1)),
            timestamps=[0, 1],
            offsets=[0, 2],
            ids=["a"],
        )
        target = tmp_path / "zarr.zip"

        with pytest.raises(TypeError, match="represented losslessly"):
            write(sdata, target)

        assert not target.exists()

    @pytest.mark.parametrize("table_name", ["features", "labels"])
    def test_lossy_parquet_object_values_rejected_before_publish(
        self, tmp_path, table_name
    ):
        features = pd.DataFrame({"unit": ["u"]}, index=["f0"])
        labels = pd.DataFrame({"x": [0, 1]})
        if table_name == "features":
            features["metadata"] = pd.Series(
                [(1, 2)], index=features.index, dtype=object
            )
        else:
            labels["metadata"] = pd.Series([(1, 2), (3, 4)], dtype=object)
        sdata = SeriesData(
            features=features,
            labels=labels,
            data=np.zeros((2, 1)),
            timestamps=[0, 1],
            offsets=[0, 2],
            ids=["a"],
        )
        target = tmp_path / "zarr.zip"

        with pytest.raises(TypeError, match="object values.*losslessly"):
            write(sdata, target)

        assert not target.exists()

    @pytest.mark.parametrize("table_name", ["features", "labels"])
    def test_duplicate_columns_rejected_before_publish(
        self, tmp_path, table_name
    ):
        sdata = SeriesData(
            features=pd.DataFrame(
                [["a", "b"]], columns=["left", "right"], index=["f0"]
            ),
            labels=pd.DataFrame([[0, 1], [1, 0]], columns=["left", "right"]),
            data=np.zeros((2, 1)),
            timestamps=[0, 1],
            offsets=[0, 2],
            ids=["a"],
        )
        getattr(sdata, table_name).columns = ["duplicate", "duplicate"]
        target = tmp_path / "zarr.zip"

        with pytest.raises(ValueError, match="columns must be unique"):
            write(sdata, target)

        assert not target.exists()

    def test_lossy_feature_index_rejected_before_publish(self, tmp_path):
        feature_index = pd.Index([("sensor", 1)], tupleize_cols=False)
        sdata = SeriesData(
            features=pd.DataFrame({"unit": ["u"]}, index=feature_index),
            labels=pd.DataFrame({"x": [0, 1]}),
            data=np.zeros((2, 1)),
            timestamps=[0, 1],
            offsets=[0, 2],
            ids=["a"],
        )
        target = tmp_path / "zarr.zip"

        with pytest.raises(TypeError, match="features index"):
            write(sdata, target)

        assert not target.exists()

    def test_multiindex_feature_index_round_trip(self, tmp_path):
        feature_index = pd.MultiIndex.from_tuples(
            [("sensor", 1)], names=["kind", "channel"]
        )
        sdata = SeriesData(
            features=pd.DataFrame({"unit": ["u"]}, index=feature_index),
            labels=pd.DataFrame({"x": [0, 1]}),
            data=np.zeros((2, 1)),
            timestamps=[0, 1],
            offsets=[0, 2],
            ids=["a"],
        )
        target = tmp_path / "zarr.zip"

        write(sdata, target)
        loaded = read(target)

        assert loaded.features.index.identical(feature_index)

    def test_multiindex_label_columns_round_trip(self, tmp_path):
        columns = pd.MultiIndex.from_tuples(
            [("label", "x"), ("label", "y")], names=["group", "kind"]
        )
        sdata = SeriesData(
            features=pd.DataFrame(index=["f0"]),
            labels=pd.DataFrame([[0, 1], [1, 0]], columns=columns),
            data=np.zeros((2, 1)),
            timestamps=[0, 1],
            offsets=[0, 2],
            ids=["a"],
        )
        target = tmp_path / "zarr.zip"

        write(sdata, target)
        loaded = read(target)

        assert loaded.labels.columns.identical(columns)

    def test_unpacked_directory_honors_process_umask(self, sdata, tmp_path):
        reference = tmp_path / "reference"
        reference.mkdir()
        expected_mode = stat.S_IMODE(reference.stat().st_mode)
        target = tmp_path / "unpacked"

        write(sdata, target, zipped=False)

        assert stat.S_IMODE(target.stat().st_mode) == expected_mode

    def test_package_identity_is_checked_before_opening_members(self, tmp_path):
        with gzip.open(
            tmp_path / series_write.FILE_META, "wt", encoding="utf-8"
        ) as handle:
            json.dump({"format": "different.series"}, handle)

        with pytest.raises(ValueError, match="unsupported series package format"):
            read(tmp_path)

    @pytest.mark.parametrize("array_name", ["offsets", "data", "timestamps"])
    def test_corrupt_packed_arrays_report_structure_error(
        self, sdata, tmp_path, array_name
    ):
        target = tmp_path / "unpacked"
        write(sdata, target, zipped=False)
        group = zarr.open_group(store=str(target), mode="r+")
        group[array_name].resize((0,))

        with pytest.raises(ValueError, match=f"stored {array_name}"):
            read(target)

    def test_nested_json_metas_round_trip(self, tmp_path):
        sdata = SeriesData(
            features=pd.DataFrame(index=["f0"]),
            labels=pd.DataFrame({"x": [0, 1]}),
            data=np.zeros((2, 1)),
            timestamps=[0, 1],
            offsets=[0, 2],
            ids=["a"],
            metas=[{"nested": {"a": [1, 2.5, None, True]}, "tag": "x"}],
            manifest={"purpose": "round-trip"},
        )
        target = tmp_path / "zarr.zip"
        write(sdata, target)
        loaded = read(target)
        assert loaded.metas[0]["nested"]["a"] == [1, 2.5, None, True]
        assert loaded.metas[0]["tag"] == "x"
        assert loaded.manifest == {"purpose": "round-trip"}

    def test_json_dataframe_attrs_round_trip(self, tmp_path):
        features = pd.DataFrame({"unit": ["m"]}, index=["f0"])
        features.attrs = {"nested": {"values": [1, True, None]}}
        labels = pd.DataFrame({"x": [0, 1]})
        labels.attrs = {"kind": "point"}
        sdata = SeriesData(
            features=features,
            labels=labels,
            data=np.zeros((2, 1)),
            timestamps=[0, 1],
            offsets=[0, 2],
            ids=["a"],
        )
        target = tmp_path / "zarr.zip"

        write(sdata, target)
        loaded = read(target)

        assert loaded.features.attrs == features.attrs
        assert loaded.labels.attrs == labels.attrs

    def test_non_json_dataframe_attrs_rejected_before_publish(self, tmp_path):
        labels = pd.DataFrame({"x": [0, 1]})
        labels.attrs = {"opaque": object()}
        sdata = SeriesData(
            features=pd.DataFrame(index=["f0"]),
            labels=labels,
            data=np.zeros((2, 1)),
            timestamps=[0, 1],
            offsets=[0, 2],
            ids=["a"],
        )
        target = tmp_path / "zarr.zip"

        with pytest.raises(TypeError, match="labels attrs"):
            write(sdata, target)

        assert not target.exists()

    def test_non_json_meta_values_rejected(self):
        with pytest.raises(TypeError, match="JSON basic"):
            SeriesData(
                features=pd.DataFrame(index=["f0"]),
                labels=pd.DataFrame({"x": [0, 1]}),
                data=np.zeros((2, 1)),
                timestamps=[0, 1],
                offsets=[0, 2],
                ids=["a"],
                metas=[{"k": np.int64(7)}],
            )

    def test_failed_zip_overwrite_preserves_existing(
        self, sdata, tmp_path, monkeypatch
    ):
        target = tmp_path / "zarr.zip"
        write(sdata, target)
        original_contents = target.read_bytes()
        original_write = series_write.zipfile.ZipFile.write
        calls = 0

        def fail_during_archive(archive, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise RuntimeError("injected archive failure")
            return original_write(archive, *args, **kwargs)

        monkeypatch.setattr(series_write.zipfile.ZipFile, "write", fail_during_archive)
        with pytest.raises(RuntimeError, match="injected"):
            write(sdata, target)

        assert target.read_bytes() == original_contents
