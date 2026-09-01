"""Transactional benchmark score, run-manifest and completion layout."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any

import numpy as np

from .datasets import DatasetSplit

RUN_SCHEMA = "agentad-benchmark-run-v1"
COMPLETION_SCHEMA = "agentad-benchmark-completion-v1"


class RunMismatchError(RuntimeError):
    """Raised when resume would combine outputs from different run contracts."""


__all__ = [
    "COMPLETION_SCHEMA",
    "RUN_SCHEMA",
    "RunMismatchError",
    "artifact_state",
    "atomic_write_file",
    "atomic_write_json",
    "checkpoints_dir",
    "completion_path",
    "has_score",
    "is_complete",
    "load_run_manifest",
    "load_score",
    "method_root",
    "metrics_path",
    "prepare_run",
    "run_manifest_path",
    "save_score",
    "score_path",
    "scores_dir",
    "unit_dir",
]


def method_root(output_root: str | Path, method: str) -> Path:
    """Directory of one method, e.g. ``outputs/benchmark/xLSTMAD``."""
    return Path(output_root) / method


@lru_cache(maxsize=None)
def artifact_state(path: str | Path) -> dict[str, Any]:
    """Return a stable file inventory for a pretrained model asset."""
    target = Path(path).resolve()
    if target.is_file():
        root = target.parent
        files = [target]
    elif target.is_dir():
        root = target
        files = sorted(
            candidate for candidate in target.rglob("*") if candidate.is_file()
        )
    else:
        return {"path": str(target), "missing": True, "files": []}
    return {
        "path": str(target),
        "missing": False,
        "files": [
            {
                "name": str(file.relative_to(root)),
                "size": (stat := file.stat()).st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
            for file in files
        ],
    }


def unit_dir(output_root: str | Path, method: str, split: DatasetSplit) -> Path:
    """Directory of one method on one dataset artifact."""
    return method_root(output_root, method) / split.source / split.name


def scores_dir(unit: str | Path) -> Path:
    return Path(unit) / "scores"


def run_manifest_path(unit: str | Path) -> Path:
    return Path(unit) / "run.json"


def completion_path(results_unit: str | Path) -> Path:
    return Path(results_unit) / "completion.json"


def metrics_path(results_unit: str | Path) -> Path:
    return Path(results_unit) / "metrics.csv"


def checkpoints_dir(unit: str | Path) -> Path:
    """Directory for method-owned checkpoints of one dataset unit."""
    return Path(unit) / "checkpoints"


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Publish one canonical JSON object through a same-directory replacement."""
    data = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    _atomic_write(Path(path), lambda handle: handle.write(data))


def atomic_write_file(path: str | Path, writer) -> None:
    """Publish a binary artifact produced by ``writer(handle)`` atomically."""
    _atomic_write(Path(path), writer)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"run metadata contains unsupported value {type(value).__name__}")


def _hash_files(paths: Sequence[Path], root: Path) -> tuple[str, list[str]]:
    digest = hashlib.sha256()
    names: list[str] = []
    candidates = sorted(
        {candidate.resolve() for candidate in paths if candidate.is_file()}
    )
    for path in candidates:
        try:
            name = str(path.relative_to(root))
        except ValueError:
            name = str(path)
        names.append(name)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest(), names


@lru_cache(maxsize=None)
def _implementation_fingerprint(
    implementation_file: str,
) -> tuple[str, tuple[str, ...], tuple[tuple[str, int, int], ...]]:
    implementation = Path(implementation_file).resolve()
    project = Path(__file__).resolve().parents[3]
    agentad = project / "src" / "agentad"
    paths: list[Path] = []
    for package in (agentad / "benchmark", agentad / "evaluation", agentad / "series"):
        paths.extend(package.rglob("*.py"))
    if implementation.parent.name == "baseline":
        # Baselines share helpers and selected implementations call sibling
        # modules, so fingerprint the package as one dependency boundary.
        paths.extend(implementation.parent.glob("*.py"))
        paths.append(implementation.parent.parent / "_utils.py")
    else:
        paths.extend(implementation.parent.glob("*.py"))
        paths.append(implementation.parent.parent / "_utils.py")
    paths.extend((project / "pyproject.toml", project / "uv.lock"))
    fingerprint, names = _hash_files(paths, project)
    state = []
    for name in names:
        path = Path(name) if Path(name).is_absolute() else project / name
        stat = path.stat()
        state.append((name, stat.st_size, stat.st_mtime_ns))
    return fingerprint, tuple(names), tuple(state)


def _collection_signature(collection) -> dict[str, Any] | None:
    if collection is None:
        return None
    return {
        "ids": list(collection.ids),
        "lengths": [len(collection[series_id]) for series_id in collection.ids],
        "features": list(map(str, collection.features.index)),
        "dtype": str(collection.data.dtype),
    }


def _dataset_signature(split: DatasetSplit) -> dict[str, Any]:
    files = []
    for partition in ("train", "test"):
        path = split.dataset_dir / f"{split.name}.{partition}.zarr.zip"
        if path.is_file():
            stat = path.stat()
            files.append(
                {
                    "name": path.name,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return {
        "root": str(split.root.resolve()),
        "source": split.source,
        "dataset": split.dataset,
        "artifact": split.name,
        "files": files,
        "train": _collection_signature(split.train),
        "test": _collection_signature(split.test),
    }


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def load_run_manifest(unit: str | Path) -> dict[str, Any]:
    """Read and validate the current run manifest of an output unit."""
    path = run_manifest_path(unit)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RunMismatchError(f"benchmark run manifest is missing: {path}") from None
    except (OSError, json.JSONDecodeError) as error:
        raise RunMismatchError(
            f"benchmark run manifest is unreadable: {path}"
        ) from error
    if payload.get("schema") != RUN_SCHEMA or not isinstance(
        payload.get("fingerprint"), str
    ):
        raise RunMismatchError(f"invalid benchmark run manifest: {path}")
    contract = dict(payload)
    fingerprint = contract.pop("fingerprint")
    if _canonical_digest(contract) != fingerprint:
        raise RunMismatchError(f"benchmark run manifest fingerprint mismatch: {path}")
    return payload


def prepare_run(
    unit: str | Path,
    split: DatasetSplit,
    *,
    method: str,
    partition: str | None,
    seed: int,
    hp: Mapping[str, Any] | None,
    resume: bool,
    implementation_file: str | Path,
    device: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind an output unit to one reproducible scoring contract.

    Existing legacy scores or a different contract may only be superseded by
    ``resume=False``. Score files live under a fingerprinted directory, so an
    interrupted replacement can never mix generations.
    """
    target = Path(unit)
    source_fingerprint, source_files, source_state = _implementation_fingerprint(
        str(Path(implementation_file).resolve())
    )
    contract: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "method": method,
        "partition": partition,
        "seed": int(seed),
        "hp": _jsonable(hp or {}),
        "device": device,
        "extra": _jsonable(extra or {}),
        "dataset": _dataset_signature(split),
        "source": {
            "sha256": source_fingerprint,
            "files": list(source_files),
            "state": [
                {"name": name, "size": size, "mtime_ns": mtime_ns}
                for name, size, mtime_ns in source_state
            ],
        },
        "python": platform.python_version(),
    }
    payload = {**contract, "fingerprint": _canonical_digest(contract)}
    path = run_manifest_path(target)
    existing: dict[str, Any] | None = None
    if path.exists():
        try:
            existing = load_run_manifest(target)
        except RunMismatchError:
            if resume:
                raise
    legacy_scores = scores_dir(target)
    has_legacy = legacy_scores.is_dir() and any(legacy_scores.glob("*.npy"))
    checkpoint_root = checkpoints_dir(target)
    has_legacy_checkpoint = checkpoint_root.is_dir() and any(
        candidate.is_file() for candidate in checkpoint_root.rglob("*")
    )
    if existing is None and (has_legacy or has_legacy_checkpoint) and resume:
        # In-flight Eva runs stored flat score files before run manifests
        # existed. Bind the current contract so remaining series can finish
        # under the fingerprinted layout instead of aborting the unit.
        atomic_write_json(path, payload)
        return payload
    if existing is not None and existing["fingerprint"] != payload["fingerprint"]:
        if resume:
            raise RunMismatchError(
                f"run contract changed for {target}: "
                f"{existing['fingerprint'][:12]} != {payload['fingerprint'][:12]}; "
                "rerun with resume=False or choose a new output root"
            )
    if existing is None or existing["fingerprint"] != payload["fingerprint"]:
        atomic_write_json(path, payload)
    return payload


def _active_score_path(unit: str | Path, series_id: str) -> Path:
    manifest = load_run_manifest(unit)
    return scores_dir(unit) / manifest["fingerprint"] / f"{series_id}.npy"


def score_path(unit: str | Path, series_id: str) -> Path:
    """Path of a score in the active run, or the legacy flat path if unbound."""
    try:
        return _active_score_path(unit, series_id)
    except RunMismatchError:
        return scores_dir(unit) / f"{series_id}.npy"


def has_score(unit: str | Path, series_id: str) -> bool:
    """Whether the active fingerprint owns an atomically published score."""
    try:
        return _active_score_path(unit, series_id).is_file()
    except RunMismatchError:
        return False


def save_score(unit: str | Path, series_id: str, scores: np.ndarray) -> None:
    """Atomically publish one finite, flat float64 score array."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("scores must not be empty")
    if not np.isfinite(values).all():
        raise ValueError("scores contain non-finite values")
    path = _active_score_path(unit, series_id)
    _atomic_write(path, lambda handle: np.save(handle, values, allow_pickle=False))


def load_score(unit: str | Path, series_id: str) -> np.ndarray:
    """Load one score from the active run and validate its storage contract."""
    path = _active_score_path(unit, series_id)
    values = np.load(path, allow_pickle=False)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"invalid score shape in {path}: {values.shape}")
    if values.dtype != np.float64 or not np.isfinite(values).all():
        raise ValueError(f"invalid score values in {path}")
    return values


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_is_current(manifest: Mapping[str, Any]) -> bool:
    project = Path(__file__).resolve().parents[3]
    try:
        expected = manifest["source"]["state"]
        for item in expected:
            name = item["name"]
            path = Path(name) if Path(name).is_absolute() else project / name
            stat = path.stat()
            if stat.st_size != item["size"] or stat.st_mtime_ns != item["mtime_ns"]:
                return False
        return bool(expected)
    except (KeyError, OSError, TypeError):
        return False


def _dataset_is_current(manifest: Mapping[str, Any]) -> bool:
    try:
        dataset = manifest["dataset"]
        directory = Path(dataset["root"]) / dataset["source"] / dataset["dataset"]
        for item in dataset["files"]:
            stat = (directory / item["name"]).stat()
            if stat.st_size != item["size"] or stat.st_mtime_ns != item["mtime_ns"]:
                return False
        return bool(dataset["files"])
    except (KeyError, OSError, TypeError):
        return False


def _assets_are_current(value: Any) -> bool:
    if isinstance(value, Mapping):
        if {"path", "missing", "files"}.issubset(value):
            return artifact_state(value["path"]) == value
        return all(_assets_are_current(item) for item in value.values())
    if isinstance(value, list):
        return all(_assets_are_current(item) for item in value)
    return True


def is_complete(unit: str | Path, results_unit: str | Path) -> bool:
    """Validate that metrics atomically complete the active run generation."""
    try:
        manifest = load_run_manifest(unit)
        completion = json.loads(
            completion_path(results_unit).read_text(encoding="utf-8")
        )
        metric_file = metrics_path(results_unit)
        return bool(
            completion.get("schema") == COMPLETION_SCHEMA
            and completion.get("run_fingerprint") == manifest["fingerprint"]
            and _source_is_current(manifest)
            and _dataset_is_current(manifest)
            and _assets_are_current(manifest.get("extra", {}))
            and metric_file.is_file()
            and completion.get("metrics_sha256") == _file_sha256(metric_file)
            and int(completion.get("rows", 0)) > 0
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError, RunMismatchError):
        return False
