"""Validate semantic package names and lossless SeriesData metadata."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from agentad.series import read


REQUIRED_ITEM_FIELDS = {
    "source",
    "dataset",
    "artifact",
    "series_id",
    "task",
    "split",
    "rows",
    "source_data_dtype",
    "data_dtype",
    "timestamp_kind",
    "source_timestamp_kind",
    "timestamp_encoding",
    "source_files",
    "source_metadata",
    "annotations",
}


def validate_package(path: Path) -> tuple[dict[str, Any], tuple[str, ...], int]:
    if "__schema" in path.name:
        raise ValueError(f"opaque schema package is forbidden: {path}")
    packed = read(path)
    manifest = dict(packed.manifest)
    required_manifest = {"source", "dataset", "artifact", "task", "split"}
    missing = required_manifest - set(manifest)
    if missing:
        raise ValueError(f"{path}: manifest is missing {sorted(missing)}")
    artifact = str(manifest["artifact"])
    split = str(manifest["split"])
    expected = artifact if split == "data" else f"{artifact}.{split}"
    if path.name != f"{expected}.zarr.zip":
        raise ValueError(
            f"{path}: filename does not match artifact/split {expected!r}"
        )

    for item in packed:
        meta = item.meta
        missing = REQUIRED_ITEM_FIELDS - set(meta)
        if missing:
            raise ValueError(f"{path}/{item.id}: metadata is missing {sorted(missing)}")
        for name in ("source", "dataset", "artifact", "task", "split"):
            if meta[name] != manifest[name]:
                raise ValueError(f"{path}/{item.id}: {name} differs from manifest")
        if int(meta["rows"]) != len(item):
            raise ValueError(f"{path}/{item.id}: row metadata is inconsistent")
        if str(meta["data_dtype"]) != str(item.data.dtype):
            raise ValueError(f"{path}/{item.id}: data dtype metadata is inconsistent")
        if not isinstance(meta["source_files"], list):
            raise TypeError(f"{path}/{item.id}: source_files must be a list")
        if not isinstance(meta["source_metadata"], dict):
            raise TypeError(f"{path}/{item.id}: source_metadata must be an object")
        annotations = meta["annotations"]
        if not isinstance(annotations, dict) or set(annotations) != {
            "source_events",
            "derived_label_runs",
        }:
            raise ValueError(f"{path}/{item.id}: annotations are incomplete")
    return manifest, packed.ids, packed.total_points


def validate(root: Path) -> dict[str, Any]:
    packages = sorted(root.rglob("*.zarr.zip"))
    if not packages:
        raise FileNotFoundError(f"no SeriesData packages under {root}")
    sources: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    unique_series: set[tuple[str, str, str, str, str]] = set()
    total_items = 0
    total_points = 0
    for path in packages:
        manifest, ids, points = validate_package(path)
        source = str(manifest["source"])
        collection = str(manifest.get("collection", ""))
        dataset = str(manifest["dataset"])
        artifact = str(manifest["artifact"])
        unique_series.update(
            (source, collection, dataset, artifact, series_id)
            for series_id in ids
        )
        sources[source] += 1
        tasks[str(manifest["task"])] += 1
        total_items += len(ids)
        total_points += points
    return {
        "root": str(root),
        "packages": len(packages),
        "split_items": total_items,
        "unique_series": len(unique_series),
        "points": total_points,
        "sources": dict(sorted(sources.items())),
        "tasks": dict(sorted(tasks.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.root.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
