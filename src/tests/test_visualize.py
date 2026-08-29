"""Tests for the AgentAD visualization API and view layer."""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import pytest

import agentad.visualize._tree as tree_module
import agentad.visualize._view as view_module
from agentad.series import SeriesData, write
from agentad.visualize import CSV_FORMAT, NORMALIZATIONS, WebUI, normalize, read_csv
from agentad.visualize.__main__ import _relative_to_root
from agentad.visualize._tree import PackageTree
from agentad.visualize._view import (
    DEFAULT_MAX_POINTS,
    HARD_MAX_POINTS,
    SeriesDataView,
    _downsample_indices,
)


def build_collection(
    series_lengths: tuple[int, ...] = (6, 4),
    *,
    n_features: int = 2,
) -> SeriesData:
    total = sum(series_lengths)
    rng = np.random.default_rng(7)
    data = rng.normal(size=(total, n_features))
    timestamps = np.arange(total, dtype=np.int64) * 1_000
    offsets = np.cumsum([0, *series_lengths], dtype=np.int64)
    labels = pd.DataFrame(
        {
            "is_anomaly": np.zeros(total, dtype=np.int64),
            "not_binary": np.linspace(0.0, 1.0, total),
        }
    )
    if total >= 4:
        labels.loc[2:3, "is_anomaly"] = 1
    feature_names = ["temp", "pres", *(f"feature-{i}" for i in range(2, n_features))]
    return SeriesData(
        features=pd.DataFrame(
            {"unit": [f"u-{i}" for i in range(n_features)]},
            index=feature_names[:n_features],
        ),
        labels=labels,
        data=data,
        timestamps=timestamps,
        offsets=offsets,
        ids=[f"s-{index}" for index in range(len(series_lengths))],
        metas=[{"split": "test"} for _ in series_lengths],
        manifest={"source": "synthetic"},
    )


def build_view() -> SeriesDataView:
    return SeriesDataView(build_collection())


@contextmanager
def running_server(app: WebUI) -> Iterator[tuple[str, int]]:
    server = app.create_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield str(host), int(port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_overview_reports_collection_facts() -> None:
    overview = build_view().overview("Academic workspace")
    assert overview["title"] == "Academic workspace"
    assert overview["series_count"] == 2
    assert overview["point_count"] == 10
    assert overview["feature_count"] == 2
    assert overview["label_count"] == 2
    assert overview["min_length"] == 4
    assert overview["max_length"] == 6
    assert [feature["name"] for feature in overview["features"]] == [
        "temp",
        "pres",
    ]
    assert [label["name"] for label in overview["binary_labels"]] == ["is_anomaly"]
    assert {item["id"] for item in overview["normalizations"]} == {
        item["id"] for item in NORMALIZATIONS
    }
    assert overview["manifest"] == {"source": "synthetic"}


def test_feature_metadata_handles_non_scalar_values_and_display_collisions() -> None:
    collection = SeriesData(
        features=pd.DataFrame(
            [["plain", ["nested"]]],
            index=[b"feature"],
            columns=[b"kind", "kind"],
        ),
        labels=pd.DataFrame(index=pd.RangeIndex(1)),
        data=np.array([[1.0]]),
        timestamps=np.array([0], dtype=np.int64),
        offsets=np.array([0, 1], dtype=np.int64),
        ids=["one"],
    )
    feature = SeriesDataView(collection).features[0]
    assert feature["name"] == "feature"
    assert feature["metadata"] == {
        "kind": "plain",
        "kind [1]": "['nested']",
    }


def test_items_support_search_paging_and_large_offsets() -> None:
    view = build_view()
    page = view.items(offset=0, limit=1)
    assert [item["id"] for item in page["items"]] == ["s-0"]
    assert page["total"] == 2 and page["has_more"]
    matched = view.items(query="S-1")
    assert [item["id"] for item in matched["items"]] == ["s-1"]
    assert not matched["has_more"]
    assert view.items(offset=99)["items"] == []
    with pytest.raises(TypeError, match="query"):
        view.items(query=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="between 1 and 200"):
        view.items(limit=0)


def test_data_returns_absolute_indices_timestamps_and_metadata() -> None:
    payload = build_view().data(series_index=0)
    assert payload["series"] == {
        "index": 0,
        "id": "s-0",
        "points": 6,
        "meta": {"split": "test"},
    }
    assert payload["window"] == {"start": 0, "stop": 6, "points": 6}
    assert payload["indices"] == [0, 1, 2, 3, 4, 5]
    assert payload["sampled_points"] == 6
    assert payload["timestamps"] == [str(1_000 * index) for index in range(6)]
    assert [feature["name"] for feature in payload["features"]] == [
        "temp",
        "pres",
    ]


def test_view_payloads_do_not_expose_mutable_internal_metadata() -> None:
    collection = build_collection()
    collection[0].meta = {"nested": {"value": 1}}
    view = SeriesDataView(collection)
    overview = view.overview("copy")
    overview["features"][0]["metadata"]["unit"] = "changed"
    overview["normalizations"][0]["name"] = "changed"
    payload = view.data(series_index=0)
    payload["series"]["meta"]["nested"]["value"] = 2
    assert view.overview("copy")["features"][0]["metadata"]["unit"] == "u-0"
    assert view.overview("copy")["normalizations"][0]["name"] == "Raw values"
    assert collection[0].meta["nested"]["value"] == 1


@pytest.mark.parametrize("max_points", [2, 3, 4, 5, 17, 50])
def test_downsampling_respects_exact_budget_and_window_bounds(max_points: int) -> None:
    rows = 3_000
    collection = SeriesData(
        features=pd.DataFrame({"unit": ["u"]}, index=["value"]),
        labels=pd.DataFrame({"flag": np.zeros(rows, dtype=np.int64)}),
        data=np.arange(rows, dtype=np.float64).reshape(-1, 1),
        timestamps=np.arange(rows, dtype=np.int64),
        offsets=np.array([0, rows], dtype=np.int64),
        ids=["big"],
    )
    payload = SeriesDataView(collection).data(
        series_index=0,
        start=100,
        stop=1_100,
        max_points=max_points,
    )
    indices = payload["indices"]
    assert len(indices) <= max_points
    assert indices == sorted(set(indices))
    assert indices[0] == 100 and indices[-1] == 1_099


def test_downsampling_handles_fully_missing_buckets() -> None:
    values = np.full((100, 2), np.nan)
    indices = _downsample_indices(values, 9)
    assert len(indices) <= 9
    assert indices[0] == 0 and indices[-1] == 99
    assert np.all(indices[1:] > indices[:-1])


def test_downsampling_budget_invariant_across_short_random_windows() -> None:
    rng = np.random.default_rng(11)
    for rows in range(3, 48):
        values = rng.normal(size=(rows, 3))
        for max_points in range(2, rows):
            indices = _downsample_indices(values, max_points)
            assert 2 <= len(indices) <= max_points
            assert indices[0] == 0 and indices[-1] == rows - 1
            assert np.all(indices[1:] > indices[:-1])


def test_data_label_runs_carry_absolute_indices_and_timestamps() -> None:
    view = build_view()
    payload = view.data(series_index=0, label_index=0)
    assert payload["label_runs"] == [
        {
            "start": "2000",
            "stop": "3000",
            "start_index": 2,
            "stop_index": 3,
        }
    ]
    with pytest.raises(ValueError, match="not binary"):
        view.data(series_index=0, label_index=1)


def test_data_supports_empty_series_and_zero_feature_collections() -> None:
    collection = SeriesData(
        features=pd.DataFrame(index=pd.RangeIndex(0)),
        labels=pd.DataFrame(index=pd.RangeIndex(3)),
        data=np.empty((3, 0), dtype=np.float64),
        timestamps=np.arange(3, dtype=np.int64),
        offsets=np.array([0, 0, 3], dtype=np.int64),
        ids=["empty", "timestamps-only"],
    )
    view = SeriesDataView(collection)
    empty = view.data(series_index=0)
    assert empty["indices"] == [] and empty["features"] == []
    populated = view.data(series_index=1, max_points=2)
    assert populated["indices"] == [0, 2] and populated["features"] == []


def test_data_validates_inputs_before_work() -> None:
    view = build_view()
    with pytest.raises(ValueError, match="series index"):
        view.data(series_index=9)
    with pytest.raises(ValueError, match="row window"):
        view.data(series_index=0, start=0, stop=99)
    with pytest.raises(ValueError, match="max_points"):
        view.data(series_index=0, max_points=1)
    with pytest.raises(ValueError, match="max_points"):
        view.data(series_index=0, max_points=HARD_MAX_POINTS + 1)
    with pytest.raises(ValueError, match="feature index"):
        view.data(series_index=0, feature_indices=[99])
    with pytest.raises(ValueError, match="unique"):
        view.data(series_index=0, feature_indices=[0, 0])
    with pytest.raises(ValueError, match="at least one feature"):
        view.data(series_index=0, feature_indices=[])


def test_data_bounds_total_response_values(monkeypatch: pytest.MonkeyPatch) -> None:
    view = build_view()
    monkeypatch.setattr(view_module, "MAX_RESPONSE_VALUES", 8)
    with pytest.raises(ValueError, match="too many values"):
        view.data(series_index=0, feature_indices=[0, 1], max_points=6)


@pytest.mark.parametrize(
    "method",
    ["zscore", "minmax", "robust", "maxabs", "l1", "l2"],
)
def test_normalize_is_finite_for_extreme_finite_values(method: str) -> None:
    limit = np.finfo(np.float64).max
    scaled = normalize(np.array([-limit, limit]), method)  # type: ignore[arg-type]
    assert np.isfinite(scaled).all()
    assert not np.array_equal(scaled, np.zeros(2))


def test_normalize_preserves_nonfinite_values_and_contract() -> None:
    values = np.array([1.0, 2.0, 3.0, np.nan, np.inf])
    scaled = normalize(values, "minmax")
    np.testing.assert_array_equal(scaled[:3], [0.0, 0.5, 1.0])
    assert np.isnan(scaled[-2]) and np.isposinf(scaled[-1])
    matrix = normalize(np.array([[0.0, 10.0], [5.0, 20.0]]), "minmax")
    np.testing.assert_array_equal(matrix, [[0.0, 0.0], [1.0, 1.0]])
    global_matrix = normalize(
        np.array([[0.0, 10.0], [5.0, 20.0]]),
        "minmax",
        scope="global",
    )
    np.testing.assert_array_equal(global_matrix, [[0.0, 0.5], [0.25, 1.0]])
    with pytest.raises(ValueError, match="unknown normalization"):
        normalize(values, "nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scope"):
        normalize(values, scope="row")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="1-D or 2-D"):
        normalize(np.zeros((1, 1, 1)))
    with pytest.raises(TypeError, match="complex"):
        normalize(np.array([1 + 2j]))
    assert DEFAULT_MAX_POINTS > 2


def test_view_rejects_complex_series_instead_of_dropping_imaginary_values() -> None:
    collection = SeriesData(
        features=pd.DataFrame(index=["complex"]),
        labels=pd.DataFrame(index=pd.RangeIndex(2)),
        data=np.array([[1 + 2j], [3 + 4j]]),
        timestamps=np.arange(2, dtype=np.int64),
        offsets=np.array([0, 2], dtype=np.int64),
        ids=["complex"],
    )
    with pytest.raises(TypeError, match="complex"):
        SeriesDataView(collection).data(series_index=0)


def test_webui_api_preserves_invalid_zero_parameters() -> None:
    app = WebUI(build_collection())
    with pytest.raises(ValueError, match="between 1 and 200"):
        app.api("/api/items", {"limit": ["0"]})
    with pytest.raises(ValueError, match="max_points"):
        app.api("/api/data", {"series": ["0"], "max_points": ["0"]})
    with pytest.raises(ValueError, match="comma-separated"):
        app.api("/api/data", {"series": ["0"], "features": ["0,"]})


def test_http_server_serves_assets_api_errors_and_security_headers() -> None:
    app = WebUI(build_collection(), title="HTTP test")
    with running_server(app) as (host, port):
        connection = http.client.HTTPConnection(host, port, timeout=3)
        connection.request("GET", "/api/overview")
        response = connection.getresponse()
        overview = json.loads(response.read())
        assert response.status == 200
        assert overview["title"] == "HTTP test"
        assert response.getheader("X-Content-Type-Options") == "nosniff"
        assert response.getheader("Content-Security-Policy")

        connection.request("GET", "/")
        response = connection.getresponse()
        page = response.read()
        etag = response.getheader("ETag")
        assert response.status == 200
        assert b'<html lang="en">' in page
        assert b'src="/app.js"' in page
        assert response.getheader("Cache-Control") == "no-cache"
        assert etag

        connection.request("GET", "/", headers={"If-None-Match": etag})
        response = connection.getresponse()
        assert response.status == 304
        assert response.read() == b""

        connection.request("GET", "/core.js")
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Cache-Control") == "no-cache"
        assert b"clampWindow" in response.read()

        connection.request("GET", "/api/missing")
        response = connection.getresponse()
        assert response.status == 404
        assert json.loads(response.read()) == {"error": "API endpoint not found"}

        connection.request("HEAD", "/api/data?series=0")
        response = connection.getresponse()
        assert response.status == 405
        assert response.getheader("Allow") == "GET"
        assert response.read() == b""

        connection.request("POST", "/api/data?series=0", body=b"")
        response = connection.getresponse()
        assert response.status == 405
        assert json.loads(response.read()) == {"error": "method not allowed"}

        connection.request("GET", "/api/items?query=" + "x" * 9_000)
        response = connection.getresponse()
        assert response.status == 414
        assert json.loads(response.read()) == {"error": "request target is too long"}
        connection.close()


def test_webui_validates_server_configuration() -> None:
    app = WebUI(build_collection())
    with pytest.raises(ValueError, match="title"):
        WebUI(build_collection(), title=" ")
    with pytest.raises(TypeError, match="max_csv_bytes"):
        WebUI(build_collection(), max_csv_bytes=True)
    with pytest.raises(ValueError, match="max_csv_bytes"):
        WebUI(build_collection(), max_csv_bytes=0)
    with pytest.raises(ValueError, match="path requires"):
        WebUI(path="sample.zarr.zip")
    with pytest.raises(TypeError, match="port"):
        app.create_server(port=True)
    with pytest.raises(ValueError, match="between 0 and 65535"):
        app.create_server(port=70_000)


def test_package_tree_is_rooted_bounded_and_marks_both_package_forms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    unpacked = root / "unpacked"
    unpacked.mkdir()
    (unpacked / tree_module.FILE_META).write_bytes(b"marker")
    (root / "packed.zarr.zip").write_bytes(b"marker")
    (root / "table.csv").write_text("marker", encoding="utf-8")
    (root / "ordinary.txt").write_text("plain", encoding="utf-8")
    (root / ".hidden.zarr.zip").write_bytes(b"marker")
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)

    tree = PackageTree(root)
    listing = tree.listing()
    records = {record["name"]: record for record in listing["entries"]}
    assert ".hidden.zarr.zip" not in records
    assert records["unpacked"]["dir"] and records["unpacked"]["loadable"]
    assert not records["packed.zarr.zip"]["dir"]
    assert records["packed.zarr.zip"]["loadable"]
    assert records["table.csv"]["loadable"]
    assert records["table.csv"]["format"] == "csv"
    assert not records["ordinary.txt"]["loadable"]
    assert tree.is_loadable("unpacked")
    assert tree.is_loadable("packed.zarr.zip")
    assert tree.is_loadable("table.csv")
    with pytest.raises(ValueError, match="must not contain"):
        tree.resolve("../outside")
    with pytest.raises(ValueError, match="escapes"):
        tree.resolve("escape")

    monkeypatch.setattr(tree_module, "MAX_ENTRIES", 2)
    limited = tree.listing()
    assert len(limited["entries"]) == 2
    assert limited["truncated"]


def test_relative_startup_path_uses_real_containment_not_string_prefix(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    dotted = root / "..data" / "sample.zarr.zip"
    dotted.parent.mkdir(parents=True)
    dotted.touch()
    outside = tmp_path / "outside.zarr.zip"
    outside.touch()
    assert _relative_to_root(str(dotted), str(root)) == "..data/sample.zarr.zip"
    assert _relative_to_root(str(outside), str(root)) is None


def test_browser_mode_lists_and_opens_a_package_over_http(tmp_path: Path) -> None:
    package = tmp_path / "sample.zarr.zip"
    write(build_collection(), package)
    app = WebUI(root=tmp_path, title="Browser mode")
    with running_server(app) as (host, port):
        connection = http.client.HTTPConnection(host, port, timeout=5)
        connection.request("GET", "/api/overview")
        response = connection.getresponse()
        assert response.status == 409
        assert json.loads(response.read()) == {"error": "no collection loaded"}

        connection.request("GET", "/api/tree")
        response = connection.getresponse()
        tree = json.loads(response.read())
        assert response.status == 200
        assert any(
            entry["name"] == package.name and entry["loadable"]
            for entry in tree["entries"]
        )

        connection.request("GET", f"/api/open?path={quote(package.name)}")
        response = connection.getresponse()
        overview = json.loads(response.read())
        assert response.status == 200
        assert overview["path"] == package.name
        assert overview["series_count"] == 2

        connection.request("GET", "/api/items?limit=1")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["items"][0]["id"] == "s-0"
        connection.close()


def test_csv_contract_loads_series_features_labels_metadata_and_datetimes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "samples.csv"
    source.write_text(
        "series_id,timestamp,feature.temperature,feature.pressure,"
        "label.is_anomaly,meta.split\n"
        "alpha,2026-01-01T00:00:00Z,1.0,10.0,0,train\n"
        "alpha,2026-01-01T00:01:00Z,2.0,11.0,1,train\n"
        "beta,2026-01-02T00:00:00Z,3.0,12.0,0,test\n",
        encoding="utf-8",
    )

    collection = read_csv(source)

    assert collection.ids == ("alpha", "beta")
    np.testing.assert_array_equal(collection.offsets, [0, 2, 3])
    np.testing.assert_array_equal(collection.data, [[1.0, 10.0], [2.0, 11.0], [3.0, 12.0]])
    assert collection.features.index.tolist() == ["temperature", "pressure"]
    assert collection.labels.columns.tolist() == ["is_anomaly"]
    assert collection[0].meta["split"] == "train"
    assert collection[0].meta["timestamp_encoding"] == {
        "original_values": "datetime",
        "source_timezone": "UTC",
    }
    assert collection.manifest["format"] == CSV_FORMAT
    assert SeriesDataView(collection).data(series_index=0)["timestamp_labels"] == [
        "2026-01-01 00:00:00+00:00",
        "2026-01-01 00:01:00+00:00",
    ]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("series_id,timestamp,value\na,0,1\n", "outside the contract"),
        (
            "series_id,timestamp,feature.value\na,0,1\nb,1,2\na,2,3\n",
            "contiguous block",
        ),
        (
            "series_id,timestamp,feature.value,meta.split\na,0,1,train\na,1,2,test\n",
            "not constant",
        ),
        (
            "series_id,timestamp,feature.value\na,9223372036854775808,1\n",
            "signed int64 range",
        ),
        (
            "series_id,timestamp,feature.value\na,0,1\na,2026-01-01T00:00:00Z,2\n",
            "one timestamp encoding",
        ),
        (
            "series_id,timestamp,feature.value\na,1.5,1\n",
            "exact signed int64",
        ),
    ],
)
def test_csv_contract_rejects_ambiguous_or_unsafe_inputs(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    source = tmp_path / "invalid.csv"
    source.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_csv(source)


def test_csv_size_limit_and_http_open(tmp_path: Path) -> None:
    source = tmp_path / "samples.csv"
    source.write_text(
        "series_id,timestamp,feature.value\na,0,1\na,1,2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="configured limit"):
        read_csv(source, max_bytes=8)

    app = WebUI(root=tmp_path)
    with running_server(app) as (host, port):
        connection = http.client.HTTPConnection(host, port, timeout=5)
        connection.request("GET", f"/api/open?path={quote(source.name)}")
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        assert payload["series_count"] == 1
        assert payload["manifest"]["format"] == CSV_FORMAT
        connection.close()


def test_csv_preserves_na_like_series_ids(tmp_path: Path) -> None:
    source = tmp_path / "na-id.csv"
    source.write_text(
        "series_id,timestamp,feature.value\nNA,0,1\nNA,1,2\n",
        encoding="utf-8",
    )
    assert read_csv(source).ids == ("NA",)
