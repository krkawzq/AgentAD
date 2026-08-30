"""Download and extract ZafNoo from the TFB forecasting benchmark archive.

The upstream archive contains many datasets. Run this downloader on the
current project pod; only ``ZafNoo.csv`` is retained after extraction.

Usage:
    uv run --extra download python scripts/download/download_zafnoo.py
    uv run --extra download python scripts/download/download_zafnoo.py \
        --archive /path/to/tfb-datasets.zip
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

try:
    import gdown
except ImportError:
    sys.exit("gdown is required: uv run --extra download ...")


FILE_ID = "1vgpOmAygokoUt235piWKUjfwao6KwLv7"


def zafnoo_member(archive: zipfile.ZipFile) -> zipfile.ZipInfo:
    matches = [
        info
        for info in archive.infolist()
        if not info.is_dir()
        and PurePosixPath(info.filename).name.casefold() == "zafnoo.csv".casefold()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one ZafNoo.csv in TFB archive, found {len(matches)}"
        )
    member = PurePosixPath(matches[0].filename)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError(f"unsafe ZIP member: {matches[0].filename!r}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/raw/GraniteTSFM")
    )
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output / "ZafNoo.csv"
    if target.is_file():
        print(f"skip (already extracted): {target}")
        return

    supplied_archive = args.archive is not None
    archive_path = (
        args.archive.resolve() if supplied_archive else output / "tfb-datasets.zip"
    )
    if supplied_archive and not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if not archive_path.exists():
        print(f"downloading TFB dataset archive: {archive_path}")
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            str(archive_path),
            quiet=False,
            resume=True,
        )
    if not zipfile.is_zipfile(archive_path):
        raise ValueError(f"TFB dataset download is not a ZIP archive: {archive_path}")

    partial = target.with_suffix(".csv.part")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            info = zafnoo_member(archive)
            with archive.open(info) as source, partial.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    if not supplied_archive and not args.keep_archive:
        archive_path.unlink()
    print(f"done: {target} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
