"""Download the GIFT-Eval datasets used by Granite TTM benchmarks.

This is a multi-gigabyte Hugging Face transfer and must run on the current
project pod. Cache Arrow files are excluded because each dataset already ships
its canonical ``data-*.arrow`` payload.

Usage:
    python scripts/download/download_gift_eval.py
    python scripts/download/download_gift_eval.py \
        --dataset m4_yearly electricity/15T
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ID = "Salesforce/GiftEval"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/GiftEval"))
    parser.add_argument(
        "--dataset",
        nargs="*",
        help="optional dataset paths such as m4_yearly or electricity/15T",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    hf = shutil.which("hf")
    if hf is None:
        sys.exit("the Hugging Face `hf` CLI is required")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = [
        hf,
        "download",
        REPO_ID,
        "--repo-type",
        "dataset",
        "--revision",
        args.revision,
        "--local-dir",
        str(output),
        "--max-workers",
        str(args.max_workers),
        "--exclude",
        "*/cache-*.arrow",
        "--exclude",
        "*/tmp*",
    ]
    if args.dataset:
        for dataset in args.dataset:
            clean = dataset.strip("/")
            if not clean or clean in {".", ".."} or ".." in Path(clean).parts:
                parser.error(f"invalid dataset path: {dataset!r}")
            command.extend(("--include", f"{clean}/*"))
        command.extend(("--include", "README.md"))

    subprocess.run(command, check=True)
    arrows = list(output.rglob("data-*.arrow"))
    if not arrows:
        raise FileNotFoundError(f"no canonical GIFT-Eval Arrow files under {output}")
    print(f"done: {output} ({len(arrows)} dataset payloads)")


if __name__ == "__main__":
    main()
