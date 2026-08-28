"""Command-line entry point for ``python -m agentad.visualize``."""

from __future__ import annotations

import argparse

from ..series import read
from ._server import serve


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a SeriesData package")
    parser.add_argument("path", help=".zarr.zip package or unpacked package directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--title", default="AgentAD Visualize")
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--log-requests", action="store_true")
    args = parser.parse_args()
    serve(
        read(args.path),
        host=args.host,
        port=args.port,
        title=args.title,
        open_browser=args.open_browser,
        log_requests=args.log_requests,
    )


if __name__ == "__main__":
    main()
