#!/usr/bin/env python
"""Reproduce and verify one method on the complete official TSB-AD Eva split."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentad.benchmark import reproduce_leaderboard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, help="leaderboard or wrapper name")
    parser.add_argument("--source", choices=["U", "M"], required=True)
    parser.add_argument(
        "--reference-python",
        required=True,
        help="Python executable from an environment containing official TSB-AD deps",
    )
    parser.add_argument("--reference-root", type=Path, default=Path("forks/TSB-AD"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/TSB-AD"))
    parser.add_argument(
        "--processed-root", type=Path, default=Path("data/processed/tsb-ad")
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--results-root", type=Path, default=Path("results/benchmark"))
    parser.add_argument(
        "--raw-scores-root",
        type=Path,
        default=Path("outputs/tsb-ad-reference-scores"),
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--import-only",
        action="store_true",
        help="verify and import an already completed raw-score run",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5.1e-6,
        help="maximum per-series VUS-PR error (TSPulse CSV is rounded to 5 decimals)",
    )
    args = parser.parse_args()
    summary = reproduce_leaderboard(
        args.method,
        args.source,
        reference_python=args.reference_python,
        reference_root=args.reference_root,
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        output_root=args.output_root,
        results_root=args.results_root,
        raw_scores_root=args.raw_scores_root,
        seed=args.seed,
        run_reference=not args.import_only,
        tolerance=args.tolerance,
    )
    columns = [
        "dataset",
        "artifact",
        "rows",
        "actual_mean",
        "reference_mean",
        "max_abs_error",
        "matches",
    ]
    print(summary[columns].to_string(index=False))
    print(
        f"verified {summary['rows'].sum()} series across {len(summary)} artifacts; "
        f"max |delta|={summary['max_abs_error'].max():.6g}"
    )


if __name__ == "__main__":
    main()
