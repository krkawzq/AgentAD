"""Shared data IO and result-layout helpers for per-method scripts.

This package carries no method logic and defines no abstractions: it only
discovers dataset archives, resolves output paths, and assembles metric
tables from stored per-series scores. Every algorithm owns its training
and scoring code inside its own method directory.
"""

from .datasets import DatasetSplit, discover_tsb_ad, iter_tsb_ad, load_split
from .layout import (
    RunMismatchError,
    artifact_state,
    atomic_write_file,
    atomic_write_json,
    checkpoints_dir,
    completion_path,
    has_score,
    is_complete,
    load_run_manifest,
    load_score,
    method_root,
    metrics_path,
    prepare_run,
    run_manifest_path,
    save_score,
    score_path,
    scores_dir,
    unit_dir,
)
from .metrics import write_metrics
from .tsbad_reference import (
    ProtocolProfile,
    TSB_AD_LEADERBOARD_COMMIT,
    leaderboard_hp,
    official_hp,
    reproduce_leaderboard,
    resolve_protocol,
    validate_reference,
)

__all__ = [
    "DatasetSplit",
    "ProtocolProfile",
    "RunMismatchError",
    "TSB_AD_LEADERBOARD_COMMIT",
    "artifact_state",
    "atomic_write_file",
    "atomic_write_json",
    "checkpoints_dir",
    "completion_path",
    "discover_tsb_ad",
    "has_score",
    "is_complete",
    "iter_tsb_ad",
    "load_score",
    "load_run_manifest",
    "leaderboard_hp",
    "load_split",
    "method_root",
    "metrics_path",
    "official_hp",
    "prepare_run",
    "reproduce_leaderboard",
    "resolve_protocol",
    "run_manifest_path",
    "save_score",
    "score_path",
    "scores_dir",
    "unit_dir",
    "validate_reference",
    "write_metrics",
]
