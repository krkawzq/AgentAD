"""Download EasyTSAD datasets (CSTCloudOps/datasets repo) via git clone.

Usage:
    python download_easytsad.py [--output-dir DIR]
"""

import argparse
import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/CSTCloudOps/datasets.git"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download EasyTSAD datasets")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/EasyTSAD"),
                        help="output directory (default: data/raw/EasyTSAD)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if (args.output_dir / ".git").exists():
        print("skip (already cloned)")
    else:
        print(f"cloning {REPO_URL} -> {args.output_dir}")
        subprocess.run(["git", "clone", REPO_URL, str(args.output_dir)],
                       check=True)

    # 清理 git 元数据，仅保留数据
    shutil.rmtree(args.output_dir / ".git", ignore_errors=True)
    print(f"done: {args.output_dir}")
    for sub in ["UTS", "MTS"]:
        d = args.output_dir / sub
        if d.exists():
            print(f"  {sub}: {', '.join(x.name for x in d.iterdir() if x.is_dir())}")


if __name__ == "__main__":
    main()
