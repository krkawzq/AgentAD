"""Download the MSDS concurrent-system metrics used by AERCA.

The archive is about 335 MB. Run this script on the current project pod, not
on the development box. Network access must be configured before invocation.

Usage:
    uv run --extra download python scripts/download/download_msds.py
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


URL = (
    "https://zenodo.org/api/records/3549604/files/"
    "concurrent%20data.zip/content"
)
MD5 = "be2ba2f17f774e7f3f045dcf5f406ff0"


def file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, partial.open(
            "wb"
        ) as handle:
            shutil.copyfileobj(response, handle, length=16 * 1024 * 1024)
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"unsafe ZIP member: {info.filename!r}")
            mode = info.external_attr >> 16
            if mode & 0o170000 == 0o120000:
                raise ValueError(f"ZIP symlink is not allowed: {info.filename!r}")
        handle.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/raw/AERCA/msds")
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("forks/AERCA/datasets/msds/labels.csv"),
        help="AERCA's released point/root-cause labels",
    )
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    labels_path = args.labels_path.resolve()
    if not labels_path.is_file():
        raise FileNotFoundError(labels_path)
    output.mkdir(parents=True, exist_ok=True)
    labels_target = output / "labels.csv"
    metric_files = list(output.rglob("*_metrics_concurrent.csv"))
    if metric_files and labels_target.is_file():
        print(f"skip (already extracted): {output}")
        return

    archive = output / "concurrent-data.zip"
    print(f"downloading MSDS concurrent data from Zenodo: {archive}")
    download(URL, archive)
    actual_md5 = file_md5(archive)
    if actual_md5 != MD5:
        raise ValueError(
            f"MSDS archive checksum mismatch: expected {MD5}, got {actual_md5}"
        )

    staging = output / "_extracting"
    if staging.exists():
        shutil.rmtree(staging)
    try:
        safe_extract(archive, staging)
        extracted_root = output / "MSDS"
        extracted_root.mkdir(exist_ok=True)
        for item in staging.iterdir():
            target = extracted_root / item.name
            if target.exists():
                raise FileExistsError(target)
            item.replace(target)
        shutil.copy2(labels_path, labels_target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    metric_files = list(output.rglob("*_metrics_concurrent.csv"))
    if not metric_files:
        raise FileNotFoundError(f"no MSDS metric CSVs extracted under {output}")
    if not args.keep_archive:
        archive.unlink()
    print(f"done: {output} ({len(metric_files)} metric files + labels.csv)")


if __name__ == "__main__":
    main()
