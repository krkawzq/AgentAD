"""Download TSB-AD dataset (TSB-AD-U and TSB-AD-M) from thedatum.org.

Usage:
    python download_tsbad.py [--output-dir DIR] [--only U|M]
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://www.thedatum.org/datasets"
FILES = {
    "U": ("TSB-AD-U.zip", "TSB-AD-U"),
    "M": ("TSB-AD-M.zip", "TSB-AD-M"),
}


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"skip (exists): {dest}")
        return
    print(f"downloading: {url}")
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def extract(zip_path: Path, out_dir: Path, subdir: str) -> None:
    target = out_dir / subdir
    if target.exists():
        print(f"skip (extracted): {target}")
        return
    print(f"extracting: {zip_path} -> {out_dir}")
    tmp = out_dir / f"_{subdir}_tmp"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    (tmp / subdir).rename(target)
    shutil.rmtree(tmp)
    zip_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TSB-AD dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/TSB-AD"),
        help="output directory (default: data/raw/TSB-AD)",
    )
    parser.add_argument(
        "--only",
        choices=["U", "M"],
        default=None,
        help="download only U (univariate) or M (multivariate)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for key, (zip_name, subdir) in FILES.items():
        if args.only and key != args.only:
            continue
        zip_path = args.output_dir / zip_name
        download(f"{BASE_URL}/{zip_name}", zip_path)
        extract(zip_path, args.output_dir, subdir)

    print(f"done: {args.output_dir}")
    for d in sorted(args.output_dir.iterdir()):
        if d.is_dir():
            print(f"  {d.name}: {len(list(d.glob('*.csv')))} files")


if __name__ == "__main__":
    main()
