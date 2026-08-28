"""Command-line entry point for ``python -m agentad.visualize``."""

from __future__ import annotations

import argparse
import os

from ..series import read
from ._server import serve


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("port must be an integer") from error
    if port < 0 or port > 65_535:
        raise argparse.ArgumentTypeError("port must be between 0 and 65535")
    return port


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agentad.visualize",
        description=(
            "Explore AgentAD SeriesData packages in a local WebUI; "
            "without a path the file tree picks a package after startup"
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="optional .zarr.zip package or unpacked package directory to open at startup",
    )
    parser.add_argument(
        "--root",
        default=None,
        help=(
            "directory shown in the file tree "
            "(default: the package's parent directory, else the current directory)"
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address (default: loopback only)",
    )
    parser.add_argument("--port", type=_port, default=8765, help="bind port")
    parser.add_argument(
        "--title",
        default="AgentAD Visualize",
        help="title shown in the WebUI",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="open the WebUI in the default browser after startup",
    )
    parser.add_argument(
        "--log-requests",
        action="store_true",
        help="write HTTP request logs to stderr",
    )
    return parser


def _relative_to_root(path: str, root: str) -> str | None:
    resolved_path = os.path.realpath(path)
    resolved_root = os.path.realpath(root)
    try:
        if os.path.commonpath((resolved_root, resolved_path)) != resolved_root:
            return None
    except ValueError:
        return None
    relative = os.path.relpath(resolved_path, resolved_root)
    return "" if relative == os.curdir else relative.replace(os.sep, "/")


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.path is not None:
        root = args.root
        if root is None:
            root = os.path.dirname(os.path.abspath(args.path)) or os.curdir
        if not os.path.isdir(root):
            parser.error(f"tree root is not a directory: {root}")
        sdata = read(args.path)
        path = _relative_to_root(args.path, root)
    else:
        sdata = None
        root = args.root if args.root is not None else os.getcwd()
        if not os.path.isdir(root):
            parser.error(f"tree root is not a directory: {root}")
        path = None
    serve(
        sdata,
        path=path,
        root=root,
        host=args.host,
        port=args.port,
        title=args.title,
        open_browser=args.open_browser,
        log_requests=args.log_requests,
    )


if __name__ == "__main__":
    main()
