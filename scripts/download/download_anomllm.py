"""Download anomllm synthetic anomaly detection data from Google Drive.

Usage:
    python download_anomllm.py [--output-dir DIR]
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

try:
    import gdown
except ImportError:
    sys.exit("gdown is required: pip install gdown")

FILE_ID = "19KNCiOm3UI_JXkzBAWOdqXwM0VH3xOwi"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download anomllm dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/anomllm"),
                        help="output directory (default: data/raw/anomllm)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if (args.output_dir / "data").exists():
        print("skip (already extracted)")
        return

    zip_path = args.output_dir / "anomllm_data.zip"
    if not zip_path.exists():
        print("downloading anomllm_data.zip (~233MB) from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}",
                       str(zip_path), quiet=False)

    print(f"extracting: {zip_path.name} -> {args.output_dir}")
    tmp = args.output_dir / "_tmp"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    inner = tmp
    while len(list(inner.iterdir())) == 1 and list(inner.iterdir())[0].is_dir():
        inner = list(inner.iterdir())[0]
    for item in inner.iterdir():
        target = args.output_dir / item.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        item.rename(target)
    shutil.rmtree(tmp)
    zip_path.unlink()

    synth = args.output_dir / "data" / "synthetic"
    print(f"done: {synth} ({', '.join(x.name for x in synth.iterdir())})")


if __name__ == "__main__":
    sys.exit(main())
