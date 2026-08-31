"""Install an authorized WADI download into ``data/raw/WADI``.

WADI is not publicly redistributable. Request access from the official iTrust
page, then pass either the received archive/URL or the two authorized CSVs.

Usage:
    python scripts/download/download_wadi.py --archive /path/to/WADI.zip
    python scripts/download/download_wadi.py --url 'AUTHORIZED_SIGNED_URL'
    python scripts/download/download_wadi.py \
        --train-file /path/to/WADI_14days.csv \
        --test-file /path/to/WADI_attackdataLABLE.csv
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


OFFICIAL_URL = "https://www.sutd.edu.sg/itrust/itrust-labs/datasets/"


def download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with (
            urllib.request.urlopen(url, timeout=120) as response,
            partial.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle, length=16 * 1024 * 1024)
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def safe_extract(archive: Path, destination: Path) -> None:
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
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/WADI"))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--archive", type=Path)
    source.add_argument("--url")
    source.add_argument("--train-file", type=Path)
    parser.add_argument("--test-file", type=Path)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = list(output.rglob("*.csv"))
    if len(existing) >= 2 and not any((args.archive, args.url, args.train_file)):
        print(f"skip (already installed): {output}")
        return

    if args.train_file is not None:
        if args.test_file is None:
            parser.error("--train-file requires --test-file")
        for path in (args.train_file.resolve(), args.test_file.resolve()):
            if not path.is_file():
                raise FileNotFoundError(path)
            target = output / path.name
            if target.exists() and not target.samefile(path):
                raise FileExistsError(target)
            if not target.exists():
                shutil.copy2(path, target)
    elif args.test_file is not None:
        parser.error("--test-file requires --train-file")
    elif args.archive is not None or args.url is not None:
        archive = (
            args.archive.resolve()
            if args.archive is not None
            else output / "WADI-authorized.zip"
        )
        if args.archive is not None and not archive.is_file():
            raise FileNotFoundError(archive)
        if args.url is not None:
            print(f"downloading authorized WADI archive: {archive}")
            download(args.url, archive)
        if not zipfile.is_zipfile(archive):
            raise ValueError(f"authorized WADI input is not a ZIP archive: {archive}")
        staging = output / "_extracting"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        try:
            safe_extract(archive, staging)
            for item in staging.iterdir():
                target = output / item.name
                if target.exists():
                    raise FileExistsError(target)
                item.replace(target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    else:
        parser.error(
            "WADI requires an authorized download. Request it at "
            f"{OFFICIAL_URL}, then provide --archive, --url, or both CSV files"
        )

    csv_files = list(output.rglob("*.csv"))
    if len(csv_files) < 2:
        raise FileNotFoundError(
            f"WADI installation has fewer than two CSV files under {output}"
        )
    print(f"done: {output} ({len(csv_files)} CSV files)")


if __name__ == "__main__":
    main()
