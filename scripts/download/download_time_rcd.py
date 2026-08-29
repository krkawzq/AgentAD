"""Download the official Time-RCD synthetic dataset generator.

The released artifact is generation code, not a prebuilt corpus. Run the
generator on the current project pod and place trusted ``*.pkl`` outputs under
``data/raw/TimeRCD/datasets`` for ``process_time_rcd.py``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


COMMIT = "73eb733f3026e32bf9edfde1d4c3b787358908e7"
URL = f"https://github.com/thu-sail-lab/Datasets-RCD/archive/{COMMIT}.zip"


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
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/raw/TimeRCD/generator")
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if (output / "src" / "generate_dataset.py").is_file():
        print(f"skip (already downloaded): {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    archive = output.parent / f"Datasets-RCD-{COMMIT}.zip"
    partial = archive.with_suffix(".zip.part")
    staging = output.parent / f".generator-{COMMIT}.staging"
    try:
        if not archive.exists():
            with urllib.request.urlopen(URL, timeout=120) as response, partial.open(
                "wb"
            ) as handle:
                shutil.copyfileobj(response, handle, length=16 * 1024 * 1024)
            partial.replace(archive)
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        safe_extract(archive, staging)
        roots = [path for path in staging.iterdir() if path.is_dir()]
        if len(roots) != 1 or not (roots[0] / "src" / "generate_dataset.py").is_file():
            raise ValueError("unexpected Time-RCD generator archive layout")
        if output.exists():
            raise FileExistsError(output)
        roots[0].replace(output)
    finally:
        partial.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)
    archive.unlink()
    print(f"done: {output}")
    print(
        "Generate trusted pickle corpora on the project pod and save them under "
        f"{output.parent / 'datasets'}"
    )


if __name__ == "__main__":
    sys.exit(main())
