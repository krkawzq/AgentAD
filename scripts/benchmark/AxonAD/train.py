#!/usr/bin/env python
"""Train AxonAD on one TSB-AD dataset unit and store its checkpoints."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

from agentad.methods.AxonAD import train


def _parse_hp(values: list[str]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for item in values:
        key, _, raw = item.partition("=")
        try:
            overrides[key] = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            overrides[key] = raw
    return overrides


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
        default=Path("benchmarks"),
    )
    parser.add_argument("--device", default="cuda")
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
    args = parser.parse_args()
    train(
        dataset_dir=args.dataset_dir,
        output_root=args.output_root,
        artifact=args.artifact,
        partition=None if args.partition.lower() == "none" else args.partition,
        device=args.device,
        seed=args.seed,
        hp=_parse_hp(args.hp) or None,
        resume=not args.no_resume,
    )
    print(
        f"checkpoints written under {args.output_root}/<source>/<artifact>/checkpoints/"
    )


if __name__ == "__main__":
    main()
