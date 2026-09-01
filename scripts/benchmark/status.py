#!/usr/bin/env python
"""Validate transactional benchmark completions and print a compact status."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentad.benchmark import is_complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="", help="path below results/benchmark")
    parser.add_argument(
        "--count", action="store_true", help="print only the valid count"
    )
    parser.add_argument("--results-root", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/benchmark"))
    args = parser.parse_args()

    search_root = args.results_root / args.prefix
    valid: list[Path] = []
    invalid: list[Path] = []
    for marker in (
        sorted(search_root.rglob("completion.json")) if search_root.is_dir() else []
    ):
        results_unit = marker.parent
        relative = results_unit.relative_to(args.results_root)
        if is_complete(args.output_root / relative, results_unit):
            valid.append(relative)
        else:
            invalid.append(relative)
    if args.count:
        print(len(valid))
        return
    print(f"valid={len(valid)} invalid={len(invalid)}")
    for relative in invalid:
        print(f"INVALID {relative}")


if __name__ == "__main__":
    main()
