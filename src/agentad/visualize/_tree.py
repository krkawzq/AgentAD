"""Filesystem browsing rooted at one directory, for the local WebUI."""

from __future__ import annotations

import os
from typing import Any

from ..series._write import FILE_META

PACKAGE_SUFFIX = ".zarr.zip"
MAX_ENTRIES = 5_000


def _is_loadable(path: str, *, is_dir: bool, name: str) -> bool:
    if is_dir:
        return os.path.isfile(os.path.join(path, FILE_META))
    return os.path.isfile(path) and name.endswith(PACKAGE_SUFFIX)


class PackageTree:
    """List directories below ``root`` and locate SeriesData packages.

    Requested paths are resolved with ``os.path.realpath`` and must stay
    inside the root, so symlinks cannot make the server escape it. Entries
    whose name starts with ``.`` are hidden.
    """

    def __init__(self, root: str | os.PathLike) -> None:
        root = os.fspath(root)
        if not os.path.isdir(root):
            raise ValueError(f"tree root is not a directory: {root}")
        self.root = os.path.realpath(root)

    def _parts(self, relative: str) -> list[str]:
        if not isinstance(relative, str):
            raise TypeError("path must be a string")
        parts = [
            part
            for part in relative.replace("\\", "/").split("/")
            if part not in ("", ".")
        ]
        if any(part == ".." for part in parts):
            raise ValueError("path must not contain '..'")
        return parts

    def clean(self, relative: str) -> str:
        """Return the canonical slash-separated path relative to the root."""
        return "/".join(self._parts(relative))

    def resolve(self, relative: str) -> str:
        """Return the absolute path of ``relative``, constrained to the root."""
        parts = self._parts(relative)
        target = os.path.realpath(os.path.join(self.root, *parts))
        try:
            contained = os.path.commonpath((self.root, target)) == self.root
        except ValueError:
            contained = False
        if not contained:
            raise ValueError("path escapes the tree root")
        return target

    def is_loadable(self, relative: str) -> bool:
        """Return whether ``relative`` names a supported package."""
        target = self.resolve(relative)
        is_dir = os.path.isdir(target)
        return _is_loadable(
            target,
            is_dir=is_dir,
            name=os.path.basename(target),
        )

    def listing(self, relative: str = "") -> dict[str, Any]:
        """Return the entries of one directory as JSON-ready records."""
        parts = self._parts(relative)
        directory = self.resolve(relative)
        if not os.path.isdir(directory):
            raise ValueError(f"not a directory: {self.clean(relative) or '.'}")

        found: list[dict[str, Any]] = []
        truncated = False
        with os.scandir(directory) as scan:
            for entry in scan:
                if entry.name.startswith("."):
                    continue
                record = self._record(entry, parts)
                if record is None:
                    continue
                if len(found) >= MAX_ENTRIES:
                    truncated = True
                    break
                found.append(record)
        found.sort(
            key=lambda record: (
                not record["dir"],
                record["name"].casefold(),
                record["name"],
            )
        )
        return {
            "root": self.root,
            "path": "/".join(parts),
            "entries": found,
            "truncated": truncated,
        }

    def _record(
        self,
        entry: os.DirEntry,
        parent_parts: list[str],
    ) -> dict[str, Any] | None:
        try:
            is_dir = entry.is_dir()
        except OSError:
            return None
        try:
            size: int | None = (
                None if is_dir else entry.stat(follow_symlinks=False).st_size
            )
        except OSError:
            size = None
        return {
            "name": entry.name,
            "path": "/".join([*parent_parts, entry.name]),
            "dir": is_dir,
            "loadable": _is_loadable(entry.path, is_dir=is_dir, name=entry.name),
            "size": size,
        }
