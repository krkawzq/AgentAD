"""Download CrossAD preprocessed datasets from Google Drive.

Usage:
    python download_crossad.py [--output-dir DIR]
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

FILE_ID = "1YU_d9kIaP2EubyUhGWOwSKAhJxNntB8W"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CrossAD dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/CrossAD"),
                        help="output directory (default: data/raw/CrossAD)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.output_dir / "CrossAD_data.zip"

    if (args.output_dir / "dataset" / "data").exists():
        print("skip (already extracted)")
        return

    if not zip_path.exists():
        print("downloading CrossAD_data.zip (~605MB) from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}",
                       str(zip_path), quiet=False)

    print(f"extracting: {zip_path} -> {args.output_dir}")
    tmp = args.output_dir / "_tmp"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    for item in tmp.iterdir():
        target = args.output_dir / item.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        item.rename(target)
    shutil.rmtree(tmp)
    zip_path.unlink()

    data_dir = args.output_dir / "dataset" / "data"
    print(f"done: {data_dir} ({len(list(data_dir.glob('*.csv')))} datasets)")


if __name__ == "__main__":
    sys.exit(main())
