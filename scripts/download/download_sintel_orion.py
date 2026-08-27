"""Download Sintel-Orion (VLM4TS) datasets from AWS S3 bucket.

Downloads datasets.csv index + anomalies.csv, then each signal CSV.
Signals that 404 on S3 (some UCR/YAHOO) are skipped with a warning.

Usage:
    python download_sintel_orion.py [--output-dir DIR] [--dataset NAME ...]
"""

import argparse
import ast
import sys
from pathlib import Path

import requests

BASE_URL = "https://sintel-orion.s3.us-east-2.amazonaws.com"


def download_file(session: requests.Session, url: str, save_path: Path) -> bool:
    if save_path.exists():
        return True
    try:
        r = session.get(url, stream=True, timeout=60)
        r.raise_for_status()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return True
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            print(f"  skip (404): {save_path.name}")
        else:
            print(f"  fail: {save_path.name}: {e}")
        return False
    except requests.RequestException as e:
        print(f"  fail: {save_path.name}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Sintel-Orion datasets")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/Sintel-Orion"),
                        help="output directory (default: data/raw/Sintel-Orion)")
    parser.add_argument("--dataset", nargs="*", default=None,
                        help="specific dataset names; default: all")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    datasets_csv = args.output_dir / "datasets.csv"
    anomalies_csv = args.output_dir / "anomalies.csv"
    download_file(session, f"{BASE_URL}/datasets.csv", datasets_csv)
    download_file(session, f"{BASE_URL}/anomalies.csv", anomalies_csv)

    with open(datasets_csv) as f:
        lines = f.readlines()

    total = downloaded = 0
    for line in lines:
        parts = line.strip().split(",", 1)
        if len(parts) != 2:
            continue
        ds_name = parts[0]
        if args.dataset and ds_name not in args.dataset:
            continue
        signals = ast.literal_eval(parts[1].strip('"'))
        for sig in signals:
            total += 1
            fname = f"{sig}.csv"
            save_path = args.output_dir / ds_name / fname
            if download_file(session, f"{BASE_URL}/{fname}", save_path):
                downloaded += 1
        print(f"{ds_name}: {downloaded} signal files (of {total} attempted in this run)")

    print(f"done: {args.output_dir} ({downloaded} files)")


if __name__ == "__main__":
    sys.exit(main())
