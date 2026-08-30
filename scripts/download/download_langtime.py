"""Download LangTime forecasting datasets from Google Drive.

Usage:
    python download_langtime.py [--output-dir DIR]
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

FILE_ID = "1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LangTime dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/LangTime"),
                        help="output directory (default: data/raw/LangTime)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if (args.output_dir / "dataset").exists():
        print("skip (already extracted)")
        return

    zip_path = args.output_dir / "LangTime_data.zip"
    if not zip_path.exists():
        print("downloading LangTime_data.zip (~355MB) from Google Drive...")
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
        if item.name == "__MACOSX":
            continue
        target = args.output_dir / item.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        item.rename(target)
    shutil.rmtree(tmp)
    zip_path.unlink()

    ds = args.output_dir / "dataset"
    print(f"done: {ds} ({', '.join(x.name for x in ds.iterdir() if x.is_dir())})")


if __name__ == "__main__":
    main()
