"""Download DCDetector/ScatterAD datasets from Google Drive (file-by-file).

gdown --folder mode fails with SSL errors on this folder; individual file
downloads work reliably.

Usage:
    python download_dcdetector.py [--output-dir DIR]
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

# (file_id, zip_name) — the 9 bundled files from the DCDetector Google Drive folder
FILES = [
    ("1gA6tdxtZaJac-DpovJJIYf3Mw6Lj50t9", "ECG.zip"),
    ("1CKns8asHMcVCUjFyxoupcnYT5eZHZ_s8", "IOPS_KPI.zip"),
    ("13ZgXufg-2s9HDrDJM1usP2fzslgU5233", "KDD21_UCR.zip"),
    ("1XWcVR88aR1DYwcScSOOMcC_1iLFQJ4dh", "MGAB.zip"),
    ("1IS-pLRF1_H1CnGfo8px1Qay7N4UyPbmK", "NAB.zip"),
    ("1v-t_phtuBY3s_UeWuNbEknqx9WoNp59y", "NASA-MSL.zip"),
    ("14OddigGk_VDgXJgcqYZra6BlK-cXsYG-", "NASA-SMAP.zip"),
    ("1kTMw64xS7wyTmBcnN_tM-v8zALGFSIgM", "SensorScope.zip"),
    ("1_c01r9zJtUwW2-ZFLwaoAU4MkG3hoXmi", "YAHOO.zip"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DCDetector datasets")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/DCDetector"),
        help="output directory (default: data/raw/DCDetector)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bm = args.output_dir / "benchmark"
    bm.mkdir(parents=True, exist_ok=True)

    for file_id, zip_name in FILES:
        stem = zip_name[:-4]
        target = bm / stem
        if target.exists() and any(target.iterdir()):
            print(f"skip (exists): {stem}")
            continue
        zip_path = bm / zip_name
        if not zip_path.exists():
            print(f"downloading {zip_name}...")
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}", str(zip_path), quiet=False
            )
        print(f"extracting: {zip_name}")
        tmp = bm / f"_{stem}_tmp"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        inner = tmp
        while len(list(inner.iterdir())) == 1 and list(inner.iterdir())[0].is_dir():
            inner = list(inner.iterdir())[0]
        inner.rename(target)
        shutil.rmtree(tmp)
        zip_path.unlink()

    print(f"done: {bm}")
    for d in sorted(bm.iterdir()):
        if d.is_dir():
            n = sum(1 for _ in d.rglob("*") if _.is_file() and "__MACOSX" not in str(_))
            print(f"  {d.name}: {n} files")


if __name__ == "__main__":
    main()
