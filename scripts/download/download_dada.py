"""Download DADA evaluation and Monash pretraining datasets from Google Drive.

Usage:
    python download_dada.py [--output-dir DIR] [--only eval|monash]
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

FILES = {
    "eval": ("1QumS8bSRsLZT7u5TWLaWctDWvGnSyeRB", "DADA_eval.zip", "eval"),
    "monash": ("1r2l2lO713HMAzeJUJtBPOHqiV26WwX4U", "DADA_monash.zip", "monash"),
}


def download(file_id: str, zip_path: Path) -> None:
    if zip_path.exists():
        print(f"skip (exists): {zip_path}")
        return
    print(f"downloading {zip_path.name} from Google Drive...")
    # --continue 支持断点续传（大文件下载易中断）
    gdown.download(f"https://drive.google.com/uc?id={file_id}",
                   str(zip_path), quiet=False, continue_ok=True)


def extract(zip_path: Path, out_dir: Path, subdir: str) -> None:
    target = out_dir / subdir
    if target.exists():
        print(f"skip (extracted): {target}")
        return
    print(f"extracting: {zip_path.name} -> {target}")
    tmp = out_dir / f"_{subdir}_tmp"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    # zip 内可能套一层目录（eval/dataset/...），找到实际内容根
    inner = tmp
    while len(list(inner.iterdir())) == 1 and list(inner.iterdir())[0].is_dir():
        inner = list(inner.iterdir())[0]
    inner.rename(target)
    shutil.rmtree(tmp)
    zip_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DADA dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/DADA"),
                        help="output directory (default: data/raw/DADA)")
    parser.add_argument("--only", choices=["eval", "monash"], default=None,
                        help="download only eval or monash")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for key, (file_id, zip_name, subdir) in FILES.items():
        if args.only and key != args.only:
            continue
        zip_path = args.output_dir / zip_name
        download(file_id, zip_path)
        extract(zip_path, args.output_dir, subdir)

    print(f"done: {args.output_dir}")
    for d in sorted(args.output_dir.iterdir()):
        if d.is_dir():
            n = sum(1 for _ in d.rglob("*.csv"))
            print(f"  {d.name}: {n} csv files")


if __name__ == "__main__":
    sys.exit(main())
