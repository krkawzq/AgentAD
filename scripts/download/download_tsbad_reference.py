"""Download the TSB-AD source snapshot used by the published 2026 leaderboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

COMMIT = "e0975a5f7d3e65ab77e9fab24d1b5b51acda8f48"
URL = f"https://github.com/TheDatumOrg/TSB-AD/archive/{COMMIT}.zip"


def _safe_extract(archive: zipfile.ZipFile, target: Path) -> None:
    root = target.resolve()
    for member in archive.infolist():
        destination = (target / member.filename).resolve()
        if destination != root and root not in destination.parents:
            raise ValueError(f"unsafe archive member: {member.filename}")
    archive.extractall(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("forks/TSB-AD"))
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination.exists():
        raise FileExistsError(
            f"refusing to replace existing reference: {destination}; move it first"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="tsbad-reference-", dir=destination.parent))
    archive_path = staging / "reference.zip"
    extracted = staging / "extracted"
    try:
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, extracted)
        roots = [path for path in extracted.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError("unexpected TSB-AD archive layout")
        marker = {
            "repository": "https://github.com/TheDatumOrg/TSB-AD",
            "commit": COMMIT,
            "archive_url": URL,
        }
        (roots[0] / ".agentad-reference.json").write_text(
            json.dumps(marker, indent=2) + "\n", encoding="utf-8"
        )
        roots[0].replace(destination)
        print(f"TSB-AD {COMMIT} -> {destination}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
