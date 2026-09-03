"""Strict adapter for reproducing the published TSB-AD leaderboard.

The detector executes in a separate Python environment and calls the reference
``TSB_AD.model_wrapper`` dispatchers.  This is deliberate: TSB-AD currently
pins NumPy 1.x while AgentAD uses NumPy 2.x.  Scores are generated in the
official Eva-file order so the reference process owns one continuous RNG state.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..evaluation.period import find_period
from .datasets import discover_tsb_ad, load_split
from .layout import artifact_state, atomic_write_json, prepare_run, save_score, unit_dir
from .metrics import write_metrics

TSB_AD_LEADERBOARD_COMMIT = "e0975a5f7d3e65ab77e9fab24d1b5b51acda8f48"
REFERENCE_MARKER = ".agentad-reference.json"


@dataclass(frozen=True, slots=True)
class ProtocolProfile:
    requested_name: str
    official_name: str
    leaderboard_name: str
    provenance: Literal["tsb-ad-2024", "community-submission"]
    score_scope: Literal["full", "test"]
    hp_source: Literal["official-hp-list", "submitted-profile"]
    period_scope: Literal["full"] = "full"


_COMMUNITY_METHODS = frozenset(
    {
        "CHARM",
        "MMPAD",
        "StreamVAE",
        "Time_RCD",
        "TimeRCD_MAFT",
        "TSPulse_FT",
        "TSPulse_ZS",
        "xLSTMAD",
    }
)

_DISPLAY_NAMES = {
    "Sub_PCA": "PCA",
    "PCA": "PCA",
    "Sub_MCD": "MCD",
    "MCD": "MCD",
    "Sub_OCSVM": "OCSVM",
    "OCSVM": "OCSVM",
    "Sub_HBOS": "HBOS",
    "HBOS": "HBOS",
    "Sub_KNN": "KNN",
    "KNN": "KNN",
    "KMeansAD_U": "KMeansAD",
    "TSPulse_ZS": "TSPulse (ZS)",
    "TSPulse_FT": "TSPulse (FT)",
    "Time_RCD": "Time-RCD",
    "TimeRCD_MAFT": "Time-RCD+MAFT (FT)",
}

_SOURCE_ALIASES = {
    "PCA": {"U": "Sub_PCA", "M": "PCA"},
    "MCD": {"U": "Sub_MCD", "M": "MCD"},
    "OCSVM": {"U": "Sub_OCSVM", "M": "OCSVM"},
    "HBOS": {"U": "Sub_HBOS", "M": "HBOS"},
    "KNN": {"U": "Sub_KNN", "M": "KNN"},
    "KMEANSAD": {"U": "KMeansAD_U", "M": "KMeansAD"},
    "TSPULSEZS": {"U": "TSPulse_ZS", "M": "TSPulse_ZS"},
    "TSPULSEFT": {"U": "TSPulse_FT", "M": "TSPulse_FT"},
    "TIMERCD": {"U": "Time_RCD", "M": "Time_RCD"},
    "TIMERCDMAFTFT": {"U": "TimeRCD_MAFT", "M": "TimeRCD_MAFT"},
}

_COMMUNITY_RESULTS = {
    "CHARM": ("{prefix}_CHARM.csv", "CHARM"),
    "MMPAD": ("{prefix}_MMPAD.csv", "VUS-PR"),
    "StreamVAE": ("{prefix}_StreamVAE.csv", "StreamVAE"),
    "Time_RCD": ("{prefix}_Time_RCD.csv", "VUS-PR"),
    "TimeRCD_MAFT": ("Uni_TimeRCD_MAFT.csv", "VUS-PR"),
    "TSPulse_FT": ("{prefix}_TSPulse.csv", "TSPulse (FT)"),
    "TSPulse_ZS": ("{prefix}_TSPulse.csv", "TSPulse (ZS)"),
    "xLSTMAD": ("{prefix}_xLSTMAD.csv", "VUS-PR"),
}

_SUBMITTED_HP = {
    ("M", "TSPulse_ZS"): {"win_size": 96, "prediction_mode": "time"},
    ("M", "TSPulse_FT"): {
        "win_size": 96,
        "prediction_mode": "time",
        "lr": 1e-4,
    },
    # The submitted Time-RCD results use a feature-dependent multivariate
    # window. The worker resolves this marker after reading each CSV.
    ("M", "Time_RCD"): {
        "win_size": "min(10000, floor(400000 / n_features))",
        "batch_size": 64,
    },
}

_LEGACY_COLUMNS = {
    "Sub_IForest": "Sub-IForest",
    "Sub_LOF": "Sub-LOF",
    "Sub_PCA": "Sub-PCA",
    "Sub_HBOS": "Sub-HBOS",
    "Sub_OCSVM": "Sub-OCSVM",
    "Sub_MCD": "Sub-MCD",
    "Sub_KNN": "Sub-KNN",
    "KMeansAD_U": "KMeansAD",
    "Lag_Llama": "Lag-Llama",
    "MOMENT_ZS": "MOMENT (ZS)",
    "MOMENT_FT": "MOMENT (FT)",
}


def _normalized_name(value: str) -> str:
    return "".join(character for character in value.upper() if character.isalnum())


def resolve_protocol(method: str, source: Literal["U", "M"]) -> ProtocolProfile:
    """Resolve a leaderboard label to the source-specific official runner."""
    if source not in ("U", "M"):
        raise ValueError(f"source must be 'U' or 'M', got {source!r}")
    normalized = _normalized_name(method)
    aliases = {_normalized_name(key): key for key in _DISPLAY_NAMES}
    official = _SOURCE_ALIASES.get(normalized, {}).get(source)
    if official is None:
        official = aliases.get(normalized, method)
    provenance = (
        "community-submission" if official in _COMMUNITY_METHODS else "tsb-ad-2024"
    )
    # The archived Time-RCD integration evaluates only the suffix following
    # the filename-encoded normal prefix. Other official and submitted methods
    # use the full CSV as the scoring/evaluation sequence.
    scope: Literal["full", "test"] = "test" if official == "Time_RCD" else "full"
    hp_source = (
        "submitted-profile"
        if (source, official) in _SUBMITTED_HP
        else "official-hp-list"
    )
    return ProtocolProfile(
        requested_name=method,
        official_name=official,
        leaderboard_name=_DISPLAY_NAMES.get(official, official.replace("_", "-")),
        provenance=provenance,
        score_scope=scope,
        hp_source=hp_source,
    )


def _literal_assignment(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} is missing from {path}")


def official_hp(
    reference_root: str | Path,
    source: Literal["U", "M"],
    method: str,
) -> dict[str, Any]:
    dictionary = (
        "Optimal_Uni_algo_HP_dict" if source == "U" else "Optimal_Multi_algo_HP_dict"
    )
    values = _literal_assignment(
        Path(reference_root) / "TSB_AD" / "HP_list.py", dictionary
    )
    try:
        hp = values[method]
    except KeyError:
        raise KeyError(
            f"official method {method!r} has no {source} leaderboard HP"
        ) from None
    if not isinstance(hp, dict):
        raise TypeError(f"official HP for {method!r} is not a dictionary")
    return dict(hp)


def leaderboard_hp(
    reference_root: str | Path, source: Literal["U", "M"], method: str
) -> dict[str, Any]:
    """Return the HP contract used for the displayed leaderboard row.

    Most values come from the official selected-HP dictionary. A small number
    of later submissions omitted their multivariate configuration there; those
    contracts are preserved explicitly above from the submitted runner/results.
    """
    submitted = _SUBMITTED_HP.get((source, method))
    return (
        dict(submitted)
        if submitted is not None
        else official_hp(reference_root, source, method)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_reference(
    reference_root: str | Path,
    *,
    expected_commit: str = TSB_AD_LEADERBOARD_COMMIT,
) -> dict[str, Any]:
    """Require a clean checkout or pinned archive of the leaderboard source."""
    root = Path(reference_root).resolve()
    required = [
        root / "TSB_AD" / "model_wrapper.py",
        root / "TSB_AD" / "HP_list.py",
        root / "benchmark_exp" / "Run_Detector_U.py",
        root / "benchmark_exp" / "Run_Detector_M.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"invalid TSB-AD reference; missing: {missing}")

    commit: str | None = None
    kind: str
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if dirty:
            raise RuntimeError("TSB-AD reference checkout is dirty")
        kind = "git"
    else:
        marker_path = root / REFERENCE_MARKER
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            commit = marker["commit"]
        except (FileNotFoundError, KeyError, json.JSONDecodeError, TypeError):
            raise RuntimeError(
                f"TSB-AD archive has no valid {REFERENCE_MARKER} provenance marker"
            ) from None
        kind = "pinned-archive"
    if commit != expected_commit:
        raise RuntimeError(
            f"TSB-AD reference commit {commit!r} != leaderboard commit {expected_commit!r}"
        )
    return {
        "kind": kind,
        "commit": commit,
        "expected_commit": expected_commit,
        "root": str(root),
        "files": {
            str(path.relative_to(root)): {
                "sha256": _sha256(path),
                "artifact": artifact_state(path),
            }
            for path in required
        },
    }


def _reference_vus_pr(
    reference_root: Path, source: Literal["U", "M"], method: str
) -> pd.Series:
    prefix = "Uni" if source == "U" else "Multi"
    if method in _COMMUNITY_RESULTS:
        pattern, column = _COMMUNITY_RESULTS[method]
        path = (
            reference_root
            / "benchmark_exp"
            / "leaderboard_results"
            / pattern.format(prefix=prefix)
        )
    else:
        filename = (
            "uni_mergedTable_VUS-PR.csv"
            if source == "U"
            else "multi_mergedTable_VUS-PR.csv"
        )
        path = reference_root / "benchmark_exp" / "benchmark_eval_results" / filename
        column = _LEGACY_COLUMNS.get(method, method)
    if not path.is_file():
        raise FileNotFoundError(
            f"official per-series leaderboard result missing: {path}"
        )
    frame = pd.read_csv(path)
    id_column = "filename" if "filename" in frame else "file"
    if column not in frame:
        raise KeyError(f"official result column {column!r} missing from {path}")
    ids = frame[id_column].astype(str).str.removesuffix(".csv")
    values = pd.to_numeric(frame[column], errors="raise")
    return pd.Series(
        values.to_numpy(dtype=np.float64), index=ids, name="reference_vus_pr"
    )


def _parity_payload(
    frame: pd.DataFrame,
    expected: pd.Series,
    *,
    tolerance: float,
) -> dict[str, Any]:
    ids = frame["id"].astype(str)
    missing = sorted(set(ids) - set(expected.index))
    if missing:
        raise KeyError(
            f"{len(missing)} local series lack official results, e.g. {missing[:3]}"
        )
    reference = expected.loc[ids].to_numpy(dtype=np.float64)
    actual = frame["VUS-PR"].to_numpy(dtype=np.float64)
    differences = np.abs(actual - reference)
    maximum = float(differences.max(initial=0.0))
    return {
        "metric": "VUS-PR",
        "rows": len(frame),
        "tolerance": tolerance,
        "matches": bool(maximum <= tolerance),
        "max_abs_error": maximum,
        "mean_abs_error": float(differences.mean()) if len(differences) else 0.0,
        "actual_mean": float(actual.mean()),
        "reference_mean": float(reference.mean()),
    }


def _reference_environment(executable: str | Path) -> dict[str, Any]:
    code = """
import importlib.metadata as metadata
import json
import sys
import numpy
import pandas
import torch

def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

print(json.dumps({
    "python": sys.version.split()[0],
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "torch": torch.__version__,
    "granite-tsfm": version("granite-tsfm"),
    "huggingface-hub": version("huggingface-hub"),
    "lightning": version("lightning"),
    "xlstm": version("xlstm"),
}, sort_keys=True))
"""
    result = subprocess.run(
        [str(executable), "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    environment = json.loads(result.stdout.strip())
    if int(environment["numpy"].split(".", 1)[0]) >= 2:
        raise RuntimeError(
            "official TSB-AD environment must use NumPy 1.x as pinned by setup.py"
        )
    return environment


def _validate_method_environment(
    profile: ProtocolProfile, environment: dict[str, Any]
) -> None:
    if profile.official_name not in {"TSPulse_ZS", "TSPulse_FT"}:
        return
    if environment["python"] != "3.12.9" or environment["granite-tsfm"] != "0.3.2":
        raise RuntimeError(
            "TSPulse leaderboard reproduction requires Python 3.12.9 and "
            "granite-tsfm 0.3.2, as specified by the official model README; "
            f"got python={environment['python']!r}, "
            f"granite-tsfm={environment['granite-tsfm']!r}"
        )


def reproduce_leaderboard(
    method: str,
    source: Literal["U", "M"],
    *,
    reference_python: str | Path,
    reference_root: str | Path = "forks/TSB-AD",
    raw_root: str | Path = "data/raw/TSB-AD",
    processed_root: str | Path = "data/processed/tsb-ad",
    output_root: str | Path = "outputs/benchmark",
    results_root: str | Path = "results/benchmark",
    raw_scores_root: str | Path = "outputs/tsb-ad-reference-scores",
    seed: int = 2024,
    run_reference: bool = True,
    tolerance: float = 5.1e-6,
) -> pd.DataFrame:
    """Run one method across the complete official Eva split and verify it."""
    profile = resolve_protocol(method, source)
    reference = validate_reference(reference_root)
    reference_path = Path(reference_root).resolve()
    hp = leaderboard_hp(reference_path, source, profile.official_name)
    collection = f"TSB-AD-{source}"
    raw_dir = Path(raw_root).resolve() / collection
    file_list = reference_path / "Datasets" / "File_List" / f"{collection}-Eva.csv"
    raw_score_dir = (
        Path(raw_scores_root).resolve()
        / reference["commit"]
        / collection
        / profile.official_name
    )
    worker = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "benchmark"
        / "_tsbad_official_worker.py"
    )
    reference_environment = _reference_environment(reference_python)
    _validate_method_environment(profile, reference_environment)
    raw_contract = {
        "schema": "agentad-tsb-ad-reference-scores-v1",
        "complete": False,
        "reference_commit": reference["commit"],
        "source": source,
        "method": profile.official_name,
        "protocol": asdict(profile),
        "hp": hp,
        "seed": seed,
        "file_list_sha256": _sha256(file_list),
        "worker_sha256": _sha256(worker),
        "reference_environment": reference_environment,
    }
    raw_manifest = raw_score_dir / "run.json"
    if run_reference:
        raw_score_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(raw_manifest, raw_contract)
        subprocess.run(
            [
                str(reference_python),
                str(worker),
                "--reference-root",
                str(reference_path),
                "--source",
                source,
                "--method",
                profile.official_name,
                "--raw-dir",
                str(raw_dir),
                "--file-list",
                str(file_list),
                "--score-dir",
                str(raw_score_dir),
                "--score-scope",
                profile.score_scope,
                "--contract-file",
                str(raw_manifest),
                "--seed",
                str(seed),
            ],
            check=True,
        )
        atomic_write_json(raw_manifest, {**raw_contract, "complete": True})
    else:
        try:
            existing_contract = json.loads(raw_manifest.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            raise RuntimeError(
                f"completed official raw-score manifest missing: {raw_manifest}"
            ) from None
        if existing_contract != {**raw_contract, "complete": True}:
            raise RuntimeError(
                f"official raw-score contract mismatch or incomplete: {raw_manifest}"
            )

    expected_ids = (
        pd.read_csv(file_list)["file_name"]
        .astype(str)
        .str.removesuffix(".csv")
        .tolist()
    )
    observed_ids: list[str] = []
    oracle = _reference_vus_pr(reference_path, source, profile.official_name)
    summaries: list[dict[str, Any]] = []
    triples = discover_tsb_ad(source=collection, root=processed_root)
    output_method = f"TSB-AD-official/{profile.leaderboard_name}"
    for _, dataset, artifact in triples:
        split = load_split(
            Path(processed_root) / collection / dataset,
            artifact,
            partition="Eva",
        )
        if not split.test.ids:
            continue
        unit = unit_dir(output_root, output_method, split)
        results = unit_dir(results_root, output_method, split)
        prepare_run(
            unit,
            split,
            method=output_method,
            partition="Eva",
            seed=seed,
            hp=hp,
            resume=False,
            implementation_file=__file__,
            device="reference-environment",
            extra={"tsb_ad_reference": reference, "protocol": asdict(profile)},
        )
        windows: dict[str, int] = {}
        for series_id in split.test.ids:
            observed_ids.append(series_id)
            score_path = raw_score_dir / f"{series_id}.npy"
            if not score_path.is_file():
                raise FileNotFoundError(f"official score missing: {score_path}")
            score = np.load(score_path, allow_pickle=False).reshape(-1)
            train_item = split.train[series_id] if split.train is not None else None
            test_item = split.test[series_id]
            full = (
                np.concatenate((train_item.data, test_item.data), axis=0)
                if train_item is not None
                else np.asarray(test_item.data)
            )
            expected_length = (
                len(test_item) if profile.score_scope == "test" else len(full)
            )
            if score.shape != (expected_length,) or not np.isfinite(score).all():
                raise ValueError(
                    f"{series_id}: official score shape/value mismatch: {score.shape}, "
                    f"expected {(expected_length,)}"
                )
            windows[series_id] = find_period(np.asarray(full[:, 0]))
            save_score(unit, series_id, score)
        frame = write_metrics(
            split,
            unit,
            results,
            sliding_window=windows,
            score_scope=profile.score_scope,
        )
        parity = _parity_payload(frame, oracle, tolerance=tolerance)
        parity.update(
            {
                "reference_commit": reference["commit"],
                "official_method": profile.official_name,
                "provenance": profile.provenance,
                "score_scope": profile.score_scope,
            }
        )
        atomic_write_json(results / "parity.json", parity)
        summaries.append({"dataset": dataset, "artifact": artifact, **parity})

    if set(observed_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(observed_ids))
        extra = sorted(set(observed_ids) - set(expected_ids))
        raise RuntimeError(
            f"processed Eva membership differs from official list: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    failed = [row for row in summaries if not row["matches"]]
    if failed:
        worst = max(failed, key=lambda row: row["max_abs_error"])
        raise RuntimeError(
            f"official VUS-PR parity failed for {len(failed)} artifacts; "
            f"worst={worst['artifact']} max_abs_error={worst['max_abs_error']:.6g}"
        )
    return pd.DataFrame(summaries)


__all__ = [
    "ProtocolProfile",
    "REFERENCE_MARKER",
    "TSB_AD_LEADERBOARD_COMMIT",
    "official_hp",
    "leaderboard_hp",
    "reproduce_leaderboard",
    "resolve_protocol",
    "validate_reference",
]
