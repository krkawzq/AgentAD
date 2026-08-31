#!/usr/bin/env python
"""Evaluate ClusterLocalOutlierFactor on one TSB-AD dataset unit and print its metric table."""

from __future__ import annotations

import argparse
import ast
import sys
import traceback
from datetime import datetime
from pathlib import Path

from agentad.methods.baseline.cluster_local_outlier_factor import evaluate


def _parse_hp(values: list[str]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for item in values:
        key, _, raw = item.partition("=")
        try:
            overrides[key] = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            overrides[key] = raw
    return overrides


class _Tee:
    """Forward every write to both the original stream and the log file."""

    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle

    def write(self, text):
        self._stream.write(text)
        self._handle.write(text)

    def flush(self):
        self._stream.flush()
        self._handle.flush()


def _open_log(args, method_name):
    """Open <log-root>/benchmark/<method>/<source>/<artifact>.log for appending."""
    artifact = args.artifact or args.dataset_dir.name
    path = (
        args.log_root
        / "benchmark"
        / method_name
        / args.dataset_dir.parent.name
        / f"{artifact}.log"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", encoding="utf-8")
    handle.write(
        f"\n===== {datetime.now().isoformat(timespec='seconds')} "
        f"{' '.join(sys.argv)} =====\n"
    )
    return handle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="dataset directory, e.g. data/processed/tsb-ad/TSB-AD-M/CATSv2",
    )
    parser.add_argument(
        "--artifact",
        default=None,
        help="artifact pair name when the directory holds several",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/benchmark"),
    )
    parser.add_argument(
        "--partition",
        default="Eva",
        help="benchmark partition filter; 'none' keeps every series",
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--hp",
        nargs="*",
        default=[],
        help="config overrides as key=value pairs",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="recompute even if stored results exist",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("logs"),
        help="run logs land under <root>/benchmark/<Method>/<source>/<artifact>.log",
    )
    args = parser.parse_args()
    handle = _open_log(args, "baseline/ClusterLocalOutlierFactor")
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(stdout, handle)
    sys.stderr = _Tee(stderr, handle)
    try:
        frame = evaluate(
            dataset_dir=args.dataset_dir,
            output_root=args.output_root,
            artifact=args.artifact,
            partition=None if args.partition.lower() == 'none' else args.partition,
            seed=args.seed,
            hp=_parse_hp(args.hp) or None,
            resume=not args.no_resume,
        )
        print(frame.to_string(index=False))
        handle.write(
            f"===== done "
            f"{datetime.now().isoformat(timespec='seconds')} =====\n"
        )
    except BaseException:
        handle.write(traceback.format_exc())
        handle.write(
            f"===== failed "
            f"{datetime.now().isoformat(timespec='seconds')} =====\n"
        )
        handle.flush()
        raise
    finally:
        sys.stdout, sys.stderr = stdout, stderr
        handle.close()


if __name__ == "__main__":
    main()
