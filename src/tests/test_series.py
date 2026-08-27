import copy
import gc

import numpy as np
import pandas as pd
import pytest

from agentad import SeriesData, SeriesItem


def make_packed(*, manifest=None) -> SeriesData:
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
        manifest={"name": "demo"} if manifest is None else manifest,
    )


@pytest.fixture
def sdata() -> SeriesData:
    return make_packed()


def make_standalone_items() -> list[SeriesItem]:
    return [
        SeriesItem(
            "a",
            np.arange(6, dtype=np.float64).reshape(3, 2),
            [0, 1, 2],
            pd.DataFrame({"x": [0, 1, 0]}),
            meta={"g": 0},
        ),
        SeriesItem(
            "b",
            np.ones((2, 2), dtype=np.float64),
            [5, 6],
            pd.DataFrame({"x": [1, 1]}),
            meta={"g": 1},
        ),
    ]


def make_features() -> pd.DataFrame:
    return pd.DataFrame({"unit": ["u", "v"]}, index=["f0", "f1"])


class TestCollection:
    def test_len_ids_iter(self, sdata):
        assert len(sdata) == 3
        assert sdata.ids == ("s0", "s1", "s2")
        assert [item.id for item in sdata] == ["s0", "s1", "s2"]

    def test_getitem(self, sdata):
        assert sdata[1].id == "s1"
        assert sdata[-1].id == "s2"
        assert sdata["s1"].id == "s1"
        with pytest.raises(KeyError):
            sdata["nope"]
        with pytest.raises(IndexError):
            sdata[3]
        with pytest.raises(TypeError):
            sdata[1.5]

    def test_manifest_readonly(self, sdata):
        assert sdata.manifest["name"] == "demo"
        with pytest.raises(TypeError):
            sdata.manifest["extra"] = 1

    def test_manifest_is_deeply_isolated(self):
        source = {"nested": {"values": [1, 2]}}
        sdata = make_packed(manifest=source)

        source["nested"]["values"].append(3)
        view = sdata.manifest
        view["nested"]["values"].append(4)

        assert sdata.manifest == {"nested": {"values": [1, 2]}}

    def test_features_table(self, sdata):
        assert list(sdata.features.index) == ["speed", "temp"]
        assert sdata.features.loc["temp", "unit"] == "degC"

    def test_metas_are_live_dicts(self, sdata):
        assert sdata.metas[1]["group"] == 1
        sdata.metas[1]["x"] = 1
        assert sdata["s1"].meta["x"] == 1

    def test_shape_properties(self, sdata):
        assert sdata.total_points == 15
        assert sdata.n_features == 2
        assert sdata.offsets.tolist() == [0, 5, 8, 15]

    def test_mapping_interfaces(self, sdata):
        assert list(sdata.keys()) == ["s0", "s1", "s2"]
        assert [item.id for item in sdata.values()] == ["s0", "s1", "s2"]
        assert [(key, item.id) for key, item in sdata.items()] == [
            ("s0", "s0"),
            ("s1", "s1"),
            ("s2", "s2"),
        ]
        assert sdata.get("s1") is sdata["s1"]
        sentinel = object()
        assert sdata.get("missing", sentinel) is sentinel
        assert sdata.get(99, sentinel) is sentinel
        assert sdata.index("s2") == 2
        assert sdata.index(-1) == 2

    def test_empty_constructor_preserves_schema(self):
        features = pd.DataFrame({"unit": ["u"]}, index=["f0"])
        labels = pd.DataFrame(
            {"kind": pd.Series(pd.Categorical([], categories=["a", "b"], ordered=True))}
        )
        labels.attrs = {"scope": "point"}

        sdata = SeriesData.empty(
            features=features,
            labels=labels,
            data_dtype=np.float32,
            manifest={"name": "empty"},
        )

        assert len(sdata) == 0
        assert sdata.data.shape == (0, 1)
        assert sdata.data.dtype == np.float32
        assert sdata.labels.dtypes.astype(str).tolist() == ["category"]
        assert sdata.labels["kind"].cat.ordered
        assert sdata.labels.attrs == {"scope": "point"}
        assert sdata.manifest == {"name": "empty"}

    def test_empty_constructor_rejects_nonempty_label_schema(self):
        with pytest.raises(ValueError, match="zero rows"):
            SeriesData.empty(
                features=pd.DataFrame(index=["f0"]),
                labels=pd.DataFrame({"x": [1]}),
            )


class TestItemViews:
    def test_sdata_backref(self, sdata):
        assert sdata[0].sdata is sdata

    def test_standalone_item_has_no_sdata(self):
        item = SeriesItem(
            "a",
            np.zeros((2, 1)),
            [0, 1],
            pd.DataFrame({"x": [0, 1]}),
        )
        assert item.sdata is None
        assert len(item) == 2
        assert item.n_features == 1

    def test_data_is_write_through_slice(self, sdata):
        item = sdata[1]
        assert np.shares_memory(item.data, sdata.data)
        item.data[0, 0] = 999.0
        assert sdata.data[5, 0] == 999.0

    def test_timestamps_are_write_through_slice(self, sdata):
        item = sdata[2]
        assert np.shares_memory(item.timestamps, sdata.timestamps)
        item.timestamps[0] = 4242
        assert sdata.timestamps[8] == 4242

    def test_item_shape_fields(self, sdata):
        item = sdata[1]
        assert len(item) == 3
        assert item.n_features == 2

    def test_item_labels_are_initialized_lazily(self, sdata):
        item = sdata[1]

        assert sdata.tracked_label_frames == 0
        assert item.data.shape == (3, 2)
        assert item.meta["group"] == 1
        assert sdata.tracked_label_frames == 0

        assert item.labels.shape == (3, 1)
        assert sdata.tracked_label_frames == 1

    def test_meta_write_through(self, sdata):
        item = sdata[2]
        item.meta = {"group": 9, "extra": True}
        assert item.meta["group"] == 9
        assert sdata.metas[2] == {"group": 9, "extra": True}

    def test_copy_returns_independent_standalone_item(self, sdata):
        item = sdata["s1"]
        copied = item.copy()

        assert copied.sdata is None
        assert copied.index is None
        assert copied.id == item.id
        assert not np.shares_memory(copied.data, item.data)
        assert not np.shares_memory(copied.timestamps, item.timestamps)
        copied.data[0, 0] = -1
        copied.timestamps[0] = -1
        copied.labels.iloc[0, 0] = 9
        copied.meta["group"] = 99

        assert item.data[0, 0] != -1
        assert item.timestamps[0] != -1
        assert item.labels.iloc[0, 0] != 9
        assert item.meta["group"] != 99


class TestLabels:
    def test_slice_index_is_global_row_number(self, sdata):
        assert list(sdata["s1"].labels.index) == [5, 6, 7]

    def test_clean_item_reads_through(self, sdata):
        sdata.labels.iloc[5, 0] = 42
        assert sdata["s1"].labels.iloc[0, 0] == 42

    def test_cached_clean_item_refreshes_after_parent_replaces_column(self, sdata):
        item = sdata["s1"]
        sdata.labels["is_anomaly"] = np.arange(15)

        assert item.labels["is_anomaly"].tolist() == [5, 6, 7]
        assert sdata.materialize() is sdata

    def test_cached_clean_item_refreshes_with_copy_on_write(self, sdata):
        with pd.option_context("mode.copy_on_write", True):
            item = sdata["s1"]
            sdata.labels.iloc[5, 0] = 42
            assert item.labels.iloc[0, 0] == 42

    def test_dirty_item_returns_assigned_frame(self, sdata):
        item = sdata["s1"]
        frame = pd.DataFrame({"is_anomaly": [5, 6, 7]})
        item.labels = frame
        assert item.labels is frame

    def test_explicit_override_survives_temporary_item(self, sdata):
        sdata["s1"].labels = pd.DataFrame({"is_anomaly": [5, 6, 7]})
        gc.collect()

        assert sdata.materialize().labels["is_anomaly"].tolist() == (
            [0] * 5 + [5, 6, 7] + [0] * 7
        )

    def test_explicit_overrides_survive_iteration(self, sdata):
        for item, value in zip(sdata, [1, 2, 3]):
            item.labels = pd.DataFrame({"is_anomaly": [value] * len(item)})
        del item
        gc.collect()

        assert sdata.materialize().labels["is_anomaly"].tolist() == (
            [1] * 5 + [2] * 3 + [3] * 7
        )

    def test_copied_item_rebinds_to_canonical_override(self, sdata):
        first = sdata["s1"]
        second = copy.copy(first)
        frame = pd.DataFrame({"is_anomaly": [3, 4, 5]})

        first.labels = frame

        assert second.labels is frame
        second.labels.iloc[0, 0] = 9
        assert sdata.materialize().labels["is_anomaly"].tolist() == (
            [0] * 5 + [9, 4, 5] + [0] * 7
        )

    def test_label_tracking_does_not_mutate_dataframe_attrs(self, tmp_path):
        labels = pd.DataFrame({"x": [0, 1]})
        labels.attrs["series_data_labels"] = "user value"
        sdata = SeriesData(
            features=pd.DataFrame(index=["f0"]),
            labels=labels,
            data=np.zeros((2, 1)),
            timestamps=[0, 1],
            offsets=[0, 2],
            ids=["a"],
        )

        item_labels = sdata["a"].labels

        assert sdata.labels.attrs == {"series_data_labels": "user value"}
        assert item_labels.attrs == {"series_data_labels": "user value"}
        item_labels.to_parquet(tmp_path / "labels.parquet")

    def test_local_label_attrs_mutation_is_rejected(self, sdata):
        item = sdata["s1"]
        item.labels.attrs["local"] = "value"

        with pytest.raises(ValueError, match="attrs do not match"):
            sdata.materialize()

    def test_unfingerprintable_values_allow_explicit_override(self):
        class Unfingerprintable:
            def __hash__(self):
                raise RuntimeError("not hashable")

            def __reduce__(self):
                raise RuntimeError("not pickleable")

        labels = pd.DataFrame({"x": [Unfingerprintable(), Unfingerprintable()]})
        sdata = SeriesData(
            features=pd.DataFrame(index=["f0"]),
            labels=labels,
            data=np.zeros((2, 1)),
            timestamps=[0, 1],
            offsets=[0, 2],
            ids=["a"],
        )
        item = sdata["a"]
        replacement = pd.DataFrame({"x": [Unfingerprintable(), Unfingerprintable()]})

        item.labels = replacement

        assert sdata.materialize()["a"].labels is not labels

    def test_setter_rejects_length_mismatch(self, sdata):
        with pytest.raises(ValueError, match="rows"):
            sdata["s1"].labels = pd.DataFrame({"is_anomaly": [1, 0]})

    def test_setter_rejects_column_mismatch(self, sdata):
        with pytest.raises(ValueError, match="columns"):
            sdata["s1"].labels = pd.DataFrame({"x": [1, 2, 3]})


class TestMaterialize:
    def test_identity_clean_returns_self(self, sdata):
        assert sdata.materialize() is sdata

    def test_dirty_labels_packed_into_snapshot(self, sdata):
        item = sdata["s1"]
        item.labels = pd.DataFrame({"is_anomaly": [1, 0, 1]})
        assert item._labels_dirty
        snapshot = sdata.materialize()
        assert snapshot is not sdata
        assert snapshot.labels["is_anomaly"].tolist() == ([0] * 5 + [1, 0, 1] + [0] * 7)
        assert sdata.labels["is_anomaly"].tolist() == [0] * 15
        assert snapshot["s1"].labels["is_anomaly"].tolist() == [1, 0, 1]
        assert snapshot.materialize() is snapshot

    def test_in_place_mutation_detected_by_fingerprint(self, sdata):
        with pd.option_context("mode.copy_on_write", True):
            item = sdata["s1"]
            item.labels.iloc[0, 0] = 5
            snapshot = sdata.materialize()
        assert snapshot.labels["is_anomaly"].tolist() == ([0] * 5 + [5, 0, 0] + [0] * 7)
        assert sdata.labels["is_anomaly"].tolist() == [0] * 15

    def test_object_scalar_type_mutation_detected_by_fingerprint(self):
        labels = pd.DataFrame({"x": pd.Series([b"a"], dtype=object)})
        sdata = SeriesData(
            features=pd.DataFrame(index=["f0"]),
            labels=labels,
            data=np.zeros((1, 1)),
            timestamps=[0],
            offsets=[0, 1],
            ids=["s"],
        )

        with pd.option_context("mode.copy_on_write", True):
            item = sdata["s"]
            item.labels.iloc[0, 0] = "a"
            snapshot = sdata.materialize()

        assert snapshot is not sdata
        assert snapshot.labels.iloc[0, 0] == "a"
        assert type(snapshot.labels.iloc[0, 0]) is str
        assert sdata.labels.iloc[0, 0] == b"a"

    @pytest.mark.parametrize("ordered", [False, True])
    def test_categorical_schema_mutation_detected_by_fingerprint(self, ordered):
        labels = pd.DataFrame(
            {"kind": pd.Categorical(["a", "b"], categories=["a", "b"], ordered=False)}
        )
        sdata = SeriesData(
            features=pd.DataFrame(index=["f0"]),
            labels=labels,
            data=np.zeros((2, 1)),
            timestamps=[0, 1],
            offsets=[0, 2],
            ids=["s"],
        )
        item = sdata["s"]
        if ordered:
            item.labels["kind"] = item.labels["kind"].cat.as_ordered()
        else:
            item.labels["kind"] = item.labels["kind"].cat.reorder_categories(["b", "a"])

        snapshot = sdata.materialize()
        assert snapshot is not sdata
        assert snapshot.labels["kind"].cat.ordered is ordered
        expected_categories = ["a", "b"] if ordered else ["b", "a"]
        assert snapshot.labels["kind"].cat.categories.tolist() == expected_categories

    def test_column_axis_schema_mutation_is_rejected(self, sdata):
        item = sdata["s1"]
        item.labels.columns = pd.Index(["is_anomaly"], name="per_item_axis")

        with pytest.raises(ValueError, match="columns do not match"):
            sdata.materialize()

    def test_wide_column_schema_mutation_is_detected(self):
        columns = [f"c{index}" for index in range(200)]
        sdata = SeriesData(
            features=pd.DataFrame(index=["f0"]),
            labels=pd.DataFrame(np.zeros((2, 200)), columns=columns),
            data=np.zeros((2, 1)),
            timestamps=[0, 1],
            offsets=[0, 2],
            ids=["s"],
        )
        item = sdata["s"]
        changed = columns.copy()
        changed[100] = "changed"
        item.labels.columns = changed

        with pytest.raises(ValueError, match="columns do not match"):
            sdata.materialize()

    def test_original_items_keep_dirty_mark(self, sdata):
        item = sdata["s0"]
        item.labels = pd.DataFrame({"is_anomaly": [1] * 5})
        assert sdata.materialize() is not sdata
        assert sdata.materialize() is not sdata


class TestReorder:
    def test_permutation(self, sdata):
        snapshot = sdata.reorder(["s2", "s0", "s1"])
        assert snapshot.ids == ("s2", "s0", "s1")
        assert snapshot.offsets.tolist() == [0, 7, 12, 15]
        assert np.array_equal(snapshot.data[:7], sdata.data[8:15])
        assert snapshot.metas[0]["group"] == 2

    def test_dirty_labels_folded_in(self, sdata):
        item = sdata["s0"]
        item.labels = pd.DataFrame({"is_anomaly": [7] * 5})
        snapshot = sdata.reorder([2, 1, 0])
        assert snapshot.labels["is_anomaly"].tolist() == [0] * 10 + [7] * 5

    def test_result_does_not_share_numpy_arrays(self, sdata):
        snapshot = sdata.reorder([0, 1, 2])
        assert not np.shares_memory(snapshot.data, sdata.data)
        snapshot.data[0, 0] = -1.0
        assert sdata.data[0, 0] != -1.0

    def test_validates_input(self, sdata):
        with pytest.raises(ValueError, match="exactly once"):
            sdata.reorder(["s0", "s1"])
        with pytest.raises(TypeError, match="iterable"):
            sdata.reorder("s0")
        with pytest.raises(KeyError):
            sdata.reorder(["s2", "s1", "nope"])
        with pytest.raises(IndexError):
            sdata.reorder([2, 1, 9])

    def test_empty_collection(self):
        sdata = SeriesData(
            features=pd.DataFrame(index=[]),
            labels=pd.DataFrame(index=pd.RangeIndex(0)),
            data=np.empty((0, 0)),
            timestamps=np.empty(0, dtype=np.int64),
            offsets=np.array([0], dtype=np.int64),
            ids=[],
        )
        assert sdata.reorder([]) is sdata


class TestSelect:
    def test_subset_and_order(self, sdata):
        selected = sdata.select(["s2", "s0"])

        assert selected.ids == ("s2", "s0")
        assert selected.offsets.tolist() == [0, 7, 12]
        assert np.array_equal(selected.data[:7], sdata.data[8:15])
        assert np.array_equal(selected.data[7:], sdata.data[:5])
        assert not np.shares_memory(selected.data, sdata.data)
        selected.features.iloc[0, 0] = "changed"
        assert sdata.features.iloc[0, 0] != "changed"

    def test_dirty_labels_are_folded_in(self, sdata):
        sdata["s1"].labels = pd.DataFrame({"is_anomaly": [3, 4, 5]})

        selected = sdata.select(["s1"])

        assert selected.labels["is_anomaly"].tolist() == [3, 4, 5]

    def test_empty_selection_preserves_schema(self, sdata):
        sdata.labels.attrs = {"scope": "point"}

        selected = sdata.select([])

        assert len(selected) == 0
        assert selected.data.shape == (0, sdata.n_features)
        assert selected.data.dtype == sdata.data.dtype
        assert selected.labels.columns.identical(sdata.labels.columns)
        assert selected.labels.dtypes.equals(sdata.labels.dtypes)
        assert selected.labels.attrs == sdata.labels.attrs
        assert selected.manifest == sdata.manifest

    def test_validates_keys(self, sdata):
        with pytest.raises(ValueError, match="once"):
            sdata.select(["s0", "s0"])
        with pytest.raises(TypeError, match="iterable"):
            sdata.select("s0")
        with pytest.raises(KeyError):
            sdata.select(["missing"])

    def test_selects_empty_series(self):
        sdata = SeriesData(
            features=pd.DataFrame(index=["f0"]),
            labels=pd.DataFrame({"x": [0, 1]}),
            data=np.zeros((2, 1)),
            timestamps=[0, 1],
            offsets=[0, 0, 2, 2],
            ids=["empty-a", "full", "empty-b"],
        )

        selected = sdata.select(["empty-b", "empty-a"])

        assert selected.ids == ("empty-b", "empty-a")
        assert selected.offsets.tolist() == [0, 0, 0]
        assert selected.data.shape == (0, 1)
        assert selected.labels.shape == (0, 1)


class TestFromItems:
    def test_packs_standalone_items(self):
        items = make_standalone_items()
        sdata = SeriesData.from_items(items, features=make_features())
        assert sdata.ids == ("a", "b")
        assert sdata.offsets.tolist() == [0, 3, 5]
        assert sdata.total_points == 5
        assert sdata.n_features == 2
        assert np.array_equal(sdata.data[:3], items[0].data)
        assert np.array_equal(sdata.data[3:], items[1].data)
        assert sdata.labels["x"].tolist() == [0, 1, 0, 1, 1]
        assert sdata.metas[1]["g"] == 1
        assert list(sdata.features.index) == ["f0", "f1"]

    def test_packs_existing_views(self):
        parent = make_packed()
        packed = SeriesData.from_items(
            [parent["s1"], parent["s2"]], features=parent.features
        )
        assert packed.ids == ("s1", "s2")
        assert np.array_equal(packed.data, parent.data[5:15])
        assert packed.metas[0] == {"group": 1}

    def test_copy_semantics(self):
        items = make_standalone_items()
        features = make_features()
        sdata = SeriesData.from_items(items, features=features)
        data_before = sdata.data.copy()

        items[0].data[0, 0] = 999.0
        items[1].labels = pd.DataFrame({"x": [7, 7]})
        features.loc["f0", "unit"] = "changed"

        assert np.array_equal(sdata.data, data_before)
        assert sdata.labels["x"].tolist() == [0, 1, 0, 1, 1]
        assert sdata.features.loc["f0", "unit"] == "u"

    def test_nested_meta_does_not_share_with_input_item(self):
        item = SeriesItem(
            "a",
            np.zeros((2, 2)),
            [0, 1],
            pd.DataFrame({"x": [0, 1]}),
            meta={"nested": {"value": 1}},
        )
        sdata = SeriesData.from_items([item], features=make_features())

        item.meta["nested"]["value"] = 99

        assert sdata.metas[0]["nested"]["value"] == 1

    def test_feature_count_mismatch_rejected(self):
        items = make_standalone_items()
        items[1] = SeriesItem(
            "b",
            np.ones((2, 3), dtype=np.float64),
            [5, 6],
            pd.DataFrame({"x": [1, 1]}),
        )
        with pytest.raises(ValueError, match="feature count"):
            SeriesData.from_items(items, features=make_features())

    def test_features_table_mismatch_rejected(self):
        with pytest.raises(ValueError, match="features has"):
            SeriesData.from_items(
                make_standalone_items(),
                features=pd.DataFrame(index=["f0"]),
            )

    def test_duplicate_item_ids_rejected(self):
        items = make_standalone_items()
        items[1] = SeriesItem(
            "a",
            items[1].data,
            items[1].timestamps,
            items[1].labels,
            meta=items[1].meta,
        )

        with pytest.raises(ValueError, match="unique"):
            SeriesData.from_items(items, features=make_features())

    def test_dtype_mismatch_rejected(self):
        items = make_standalone_items()
        items[1] = SeriesItem(
            "b",
            np.ones((2, 2), dtype=np.float32),
            [5, 6],
            pd.DataFrame({"x": [1, 1]}),
        )
        with pytest.raises(ValueError, match="dtype"):
            SeriesData.from_items(items, features=make_features())

    def test_label_columns_mismatch_rejected(self):
        items = make_standalone_items()
        items[1] = SeriesItem(
            "b",
            np.ones((2, 2), dtype=np.float64),
            [5, 6],
            pd.DataFrame({"y": [1, 1]}),
        )
        with pytest.raises(ValueError, match="label columns"):
            SeriesData.from_items(items, features=make_features())

    def test_empty_items(self):
        sdata = SeriesData.from_items((), features=make_features())
        assert len(sdata) == 0
        assert sdata.total_points == 0
        assert sdata.data.shape == (0, 2)

    def test_rejects_non_item(self):
        with pytest.raises(TypeError, match="SeriesItem"):
            SeriesData.from_items(["a"], features=make_features())


class TestStandaloneItem:
    def test_1d_data_becomes_single_feature(self):
        item = SeriesItem(
            "a",
            np.array([1.0, 2.0, 3.0]),
            [0, 1, 2],
            pd.DataFrame({"x": [0, 1, 0]}),
        )
        assert item.data.shape == (3, 1)
        assert item.n_features == 1

    def test_alignment_rejected(self):
        with pytest.raises(ValueError, match="timestamps"):
            SeriesItem("a", np.zeros((3, 1)), [0, 1], pd.DataFrame({"x": [0, 1, 0]}))
        with pytest.raises(ValueError, match="labels"):
            SeriesItem("a", np.zeros((3, 1)), [0, 1, 2], pd.DataFrame({"x": [0, 1]}))

    def test_string_timestamps_rejected(self):
        with pytest.raises(TypeError, match="int64"):
            SeriesItem("a", np.zeros((2, 1)), ["0", "1"], pd.DataFrame({"x": [0, 1]}))


class TestInvariants:
    def test_1d_data_becomes_single_feature(self):
        sdata = SeriesData(
            features=pd.DataFrame(index=["f0"]),
            labels=pd.DataFrame({"x": [0, 1]}),
            data=np.array([1.0, 2.0]),
            timestamps=[0, 1],
            offsets=[0, 2],
            ids=["a"],
        )
        assert sdata["a"].data.shape == (2, 1)
        assert sdata.n_features == 1

    def test_non_native_data_is_normalized(self):
        sdata = SeriesData(
            features=pd.DataFrame(index=["f0"]),
            labels=pd.DataFrame({"x": [0, 1]}),
            data=np.array([[1], [2]], dtype=">i4"),
            timestamps=[0, 1],
            offsets=[0, 2],
            ids=["a"],
        )

        assert sdata.data.dtype == np.dtype(np.int32)
        assert sdata.data[:, 0].tolist() == [1, 2]

    def test_3d_data_rejected(self):
        with pytest.raises(ValueError, match="1-D or 2-D"):
            SeriesData(
                features=pd.DataFrame(index=["f0"]),
                labels=pd.DataFrame({"x": [0, 1]}),
                data=np.zeros((2, 1, 1)),
                timestamps=[0, 1],
                offsets=[0, 2],
                ids=["a"],
            )

    def test_float_timestamps_rejected(self):
        with pytest.raises(TypeError, match="int64"):
            SeriesData(
                features=pd.DataFrame(index=["f0"]),
                labels=pd.DataFrame({"x": [0, 1]}),
                data=np.zeros((2, 1)),
                timestamps=[0.5, 1.5],
                offsets=[0, 2],
                ids=["a"],
            )

    def test_object_data_rejected(self):
        with pytest.raises(TypeError, match="numeric or boolean"):
            SeriesData(
                features=pd.DataFrame(index=["f0"]),
                labels=pd.DataFrame({"x": [0, 1]}),
                data=np.array([["a"], ["b"]], dtype=object),
                timestamps=[0, 1],
                offsets=[0, 2],
                ids=["a"],
            )

    def test_bad_offsets_rejected(self):
        kwargs = dict(
            features=pd.DataFrame(index=["f0"]),
            labels=pd.DataFrame({"x": [0, 1, 2]}),
            data=np.zeros((3, 1)),
            timestamps=[0, 1, 2],
            ids=["a", "b"],
        )
        with pytest.raises(ValueError, match="start at 0"):
            SeriesData(offsets=[1, 2, 3], **kwargs)
        with pytest.raises(ValueError, match="non-decreasing"):
            SeriesData(offsets=[0, 2, 1], **kwargs)
        with pytest.raises(ValueError, match="point count"):
            SeriesData(offsets=[0, 1, 2], **kwargs)
        with pytest.raises(ValueError, match="expected 3"):
            SeriesData(offsets=[0, 1, 2, 3], **kwargs)

    def test_labels_row_mismatch_rejected(self):
        with pytest.raises(ValueError, match="labels has"):
            SeriesData(
                features=pd.DataFrame(index=["f0"]),
                labels=pd.DataFrame({"x": [0, 1, 2]}),
                data=np.zeros((2, 1)),
                timestamps=[0, 1],
                offsets=[0, 2],
                ids=["a"],
            )

    def test_features_row_mismatch_rejected(self):
        with pytest.raises(ValueError, match="features has"):
            SeriesData(
                features=pd.DataFrame(index=["f0", "f1"]),
                labels=pd.DataFrame({"x": [0, 1]}),
                data=np.zeros((2, 1)),
                timestamps=[0, 1],
                offsets=[0, 2],
                ids=["a"],
            )

    def test_duplicate_table_columns_rejected(self):
        kwargs = dict(
            data=np.zeros((2, 1)),
            timestamps=[0, 1],
            offsets=[0, 2],
            ids=["a"],
        )
        with pytest.raises(ValueError, match="features columns must be unique"):
            SeriesData(
                features=pd.DataFrame([[1, 2]], columns=["x", "x"]),
                labels=pd.DataFrame({"label": [0, 1]}),
                **kwargs,
            )
        with pytest.raises(ValueError, match="labels columns must be unique"):
            SeriesData(
                features=pd.DataFrame(index=["f0"]),
                labels=pd.DataFrame([[0, 1], [1, 0]], columns=["x", "x"]),
                **kwargs,
            )

    def test_ids_validation(self):
        kwargs = dict(
            features=pd.DataFrame(index=["f0"]),
            labels=pd.DataFrame({"x": [0, 1]}),
            data=np.zeros((2, 1)),
            timestamps=[0, 1],
            offsets=[0, 2],
        )
        with pytest.raises(TypeError, match="not one string"):
            SeriesData(ids="ab", **kwargs)
        with pytest.raises(TypeError, match="must be a string"):
            SeriesData(ids=["a", 1], **kwargs)
        with pytest.raises(ValueError, match="unique"):
            SeriesData(ids=["a", "a"], **kwargs)

    def test_metas_validation(self):
        kwargs = dict(
            features=pd.DataFrame(index=["f0"]),
            labels=pd.DataFrame({"x": [0, 1]}),
            data=np.zeros((2, 1)),
            timestamps=[0, 1],
            offsets=[0, 2],
            ids=["a"],
        )
        with pytest.raises(TypeError, match="not one mapping"):
            SeriesData(metas={"g": 1}, **kwargs)
        with pytest.raises(ValueError, match="entries"):
            SeriesData(metas=[{"g": 1}, {"g": 2}], **kwargs)

    def test_meta_containers_reject_non_json_values(self):
        kwargs = dict(
            features=pd.DataFrame(index=["f0"]),
            labels=pd.DataFrame({"x": [0, 1]}),
            data=np.zeros((2, 1)),
            timestamps=[0, 1],
            offsets=[0, 2],
            ids=["a"],
        )
        with pytest.raises(TypeError, match="JSON basic"):
            SeriesData(metas=[{"k": np.int64(1)}], **kwargs)
        with pytest.raises(TypeError, match="JSON basic"):
            SeriesData(manifest={"k": (1, 2)}, **kwargs)
        with pytest.raises(TypeError, match="keys must be strings"):
            SeriesData(metas=[{1: "a"}], **kwargs)
        with pytest.raises(ValueError, match="non-finite"):
            SeriesData(manifest={"bad": float("nan")}, **kwargs)
        sdata = SeriesData(**kwargs)
        item = sdata["a"]
        with pytest.raises(TypeError, match="JSON basic"):
            item.meta = {"arr": np.zeros(2)}
        with pytest.raises(TypeError, match="JSON basic"):
            SeriesItem(
                "b",
                np.zeros((2, 1)),
                [0, 1],
                pd.DataFrame({"x": [0, 1]}),
                meta={"k": object()},
            )
        item.meta = {"nested": {"a": [1, 2.5, None, True]}}
        assert item.meta["nested"]["a"] == [1, 2.5, None, True]

    def test_empty_series_supported(self):
        sdata = SeriesData(
            features=pd.DataFrame(index=["f0"]),
            labels=pd.DataFrame({"x": [0, 1, 2]}),
            data=np.zeros((3, 1)),
            timestamps=[0, 1, 2],
            offsets=[0, 0, 3],
            ids=["a", "b"],
        )
        assert len(sdata["a"]) == 0
        assert sdata["a"].labels.shape == (0, 1)
        assert sdata["b"].labels.index[0] == 0

    def test_empty_collection(self):
        sdata = SeriesData(
            features=pd.DataFrame(index=[]),
            labels=pd.DataFrame(index=pd.RangeIndex(0)),
            data=np.empty((0, 0)),
            timestamps=np.empty(0, dtype=np.int64),
            offsets=np.array([0], dtype=np.int64),
            ids=[],
        )
        assert len(sdata) == 0
        assert sdata.data.shape == (0, 0)
        assert sdata.metas == ()
