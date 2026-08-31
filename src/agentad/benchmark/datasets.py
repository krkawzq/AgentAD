"""Discovery and loading of processed dataset archives.

A dataset directory is ``<root>/<source>/<dataset>`` and may hold several
artifact pairs ``<artifact>.train.zarr.zip`` / ``<artifact>.test.zarr.zip``
when one source dataset splits into distinct feature schemas (for example
``Exathlon`` has ``Exathlon-01`` .. ``Exathlon-27``). The evaluation unit is
one artifact pair. For TSB-AD the train split is the normal prefix encoded
in each source filename, so train and test are two segments of the same
series and evaluate together on the concatenated full length.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from ..series import SeriesData, read

DEFAULT_ROOT = Path("data/processed")
TSB_AD_ROOT = DEFAULT_ROOT / "tsb-ad"

__all__ = [
    "DatasetSplit",
    "discover_tsb_ad",
    "iter_tsb_ad",
    "load_split",
]


@dataclass(frozen=True)
class DatasetSplit:
    """One artifact pair of a dataset directory.

    ``source`` is the collection name such as ``"TSB-AD-M"``, ``dataset``
    the directory name such as ``"Exathlon"``, and ``name`` the artifact
    name such as ``"Exathlon-01"`` (equal to ``dataset`` for directories
    with a single pair). ``train`` is ``None`` when no train archive
    exists.
    """

    root: Path
    source: str
    dataset: str
    name: str
    train: SeriesData | None
    test: SeriesData

    @property
    def dataset_dir(self) -> Path:
        return self.root / self.source / self.dataset

    @property
    def unit_name(self) -> str:
        return f"{self.source}/{self.name}"


def _test_archives(dataset_dir: Path) -> list[Path]:
    return sorted(dataset_dir.glob("*.test.zarr.zip"))


def _resolve_pair(dataset_dir: Path, artifact: str | None) -> tuple[str, Path, Path]:
    """Return ``(artifact, train_archive_or_none, test_archive)``."""
    tests = _test_archives(dataset_dir)
    if not tests:
        raise FileNotFoundError(f"no test archive in {dataset_dir}")
    if artifact is None:
        if len(tests) != 1:
            names = [path.name.removesuffix(".test.zarr.zip") for path in tests]
            raise ValueError(
                f"{dataset_dir} holds {len(tests)} artifact pairs; pass one of {names}"
            )
        test_archive = tests[0]
    else:
        test_archive = dataset_dir / f"{artifact}.test.zarr.zip"
        if test_archive not in tests:
            raise FileNotFoundError(f"no test archive for artifact {artifact!r}")
    name = test_archive.name.removesuffix(".test.zarr.zip")
    train_archive = dataset_dir / f"{name}.train.zarr.zip"
    return name, train_archive if train_archive.is_file() else None, test_archive


def _partition_filter(collection: SeriesData, partition: str) -> SeriesData:
    kept = []
    for series_id, meta in zip(collection.ids, collection.metas):
        source_metadata = meta.get("source_metadata", {})
        if partition in source_metadata.get("benchmark_partitions", ()):
            kept.append(series_id)
    return collection.select(kept)


def load_split(
    dataset_dir: str | Path,
    artifact: str | None = None,
    *,
    partition: str | None = None,
) -> DatasetSplit:
    """Load one artifact pair of a dataset directory.

    ``artifact`` is required when the directory holds several pairs.
    ``partition`` keeps only series whose ``benchmark_partitions`` metadata
    contains the name (for example ``"Eva"``); ``None`` keeps every series.
    """
    directory = Path(dataset_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {directory}")
    name, train_archive, test_archive = _resolve_pair(directory, artifact)

    test = read(test_archive)
    train = read(train_archive) if train_archive is not None else None
    if partition is not None:
        test = _partition_filter(test, partition)
        if train is not None:
            kept = [sid for sid in test.ids if sid in train.ids]
            train = train.select(kept)
    return DatasetSplit(
        root=directory.parent.parent,
        source=directory.parent.name,
        dataset=directory.name,
        name=name,
        train=train,
        test=test,
    )


def discover_tsb_ad(
    *,
    source: str | None = None,
    datasets: Iterable[str] | None = None,
    artifacts: Iterable[str] | None = None,
    root: str | Path = TSB_AD_ROOT,
) -> list[tuple[str, str, str]]:
    """List ``(source, dataset, artifact)`` triples of available pairs."""
    wanted_datasets = set(datasets) if datasets is not None else None
    wanted_artifacts = set(artifacts) if artifacts is not None else None
    triples: list[tuple[str, str, str]] = []
    for source_dir in sorted(path for path in Path(root).iterdir() if path.is_dir()):
        if source is not None and source_dir.name != source:
            continue
        for dataset_dir in sorted(
            path for path in source_dir.iterdir() if path.is_dir()
        ):
            if wanted_datasets is not None and dataset_dir.name not in wanted_datasets:
                continue
            for test_archive in _test_archives(dataset_dir):
                artifact = test_archive.name.removesuffix(".test.zarr.zip")
                if wanted_artifacts is not None and artifact not in wanted_artifacts:
                    continue
                triples.append((source_dir.name, dataset_dir.name, artifact))
    if not triples:
        raise FileNotFoundError(f"no TSB-AD artifacts discovered under {root}")
    return triples


def iter_tsb_ad(
    *,
    source: str | None = None,
    datasets: Iterable[str] | None = None,
    artifacts: Iterable[str] | None = None,
    partition: str | None = None,
    root: str | Path = TSB_AD_ROOT,
) -> Iterator[DatasetSplit]:
    """Yield loaded TSB-AD artifact splits, skipping empty filtered ones."""
    for pair_source, dataset, artifact in discover_tsb_ad(
        source=source, datasets=datasets, artifacts=artifacts, root=root
    ):
        split = load_split(
            Path(root) / pair_source / dataset,
            artifact,
            partition=partition,
        )
        if len(split.test) == 0:
            continue
        yield split
