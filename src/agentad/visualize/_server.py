"""Dependency-free local WebUI server for :class:`SeriesData`."""

from __future__ import annotations

import json
import os
import socket
import threading
import webbrowser
from collections.abc import Mapping, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any
from urllib.parse import parse_qs, urlsplit

from ..series import SeriesData, read
from ._tree import PackageTree
from ._view import DEFAULT_MAX_POINTS, SeriesDataView

_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/core.js": ("core.js", "text/javascript; charset=utf-8"),
}


class _RouteNotFound(LookupError):
    pass


class _NoCollection(LookupError):
    pass


class _Query:
    """Small typed accessor around ``parse_qs`` output."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, Sequence[str]]) -> None:
        self._values = values

    def text(self, name: str, default: str = "") -> str:
        values = self._values.get(name)
        return values[-1] if values else default

    def integer(self, name: str, default: int | None = None) -> int | None:
        raw = self.text(name)
        if raw == "":
            return default
        try:
            return int(raw)
        except ValueError as error:
            raise ValueError(f"{name} must be an integer") from error

    def integers(self, name: str) -> tuple[int, ...] | None:
        raw = self.text(name)
        if raw == "":
            return None
        parts = raw.split(",")
        if any(part == "" for part in parts):
            raise ValueError(f"{name} must be comma-separated integers")
        try:
            return tuple(int(part) for part in parts)
        except ValueError as error:
            raise ValueError(f"{name} must be comma-separated integers") from error


class WebUI:
    """A small local web application over one ``SeriesData`` collection.

    The collection may be provided up front or picked later from the file
    tree exposed by ``/api/tree`` and loaded through ``/api/open``; without
    a collection the chart endpoints answer ``409`` until one is opened.
    ``root`` is the directory the file tree is allowed to browse.
    """

    def __init__(
        self,
        sdata: SeriesData | None = None,
        *,
        path: str | None = None,
        root: str | os.PathLike | None = None,
        title: str = "AgentAD Visualize",
        log_requests: bool = False,
    ) -> None:
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-empty string")
        self.title = title.strip()
        self.tree = PackageTree(os.getcwd() if root is None else root)
        self._state_lock = threading.RLock()
        self._open_lock = threading.Lock()
        self.view: SeriesDataView | None = None
        self.collection_path: str | None = None
        if sdata is None and path is not None:
            raise ValueError("path requires an initial SeriesData collection")
        if sdata is not None:
            self.set_collection(sdata, path=path)
        self.log_requests = log_requests
        asset_root = files(__package__).joinpath("static")
        self._assets = {
            path: (asset_root.joinpath(name).read_bytes(), content_type)
            for path, (name, content_type) in _ASSETS.items()
        }

    def set_collection(self, sdata: SeriesData, *, path: str | None = None) -> None:
        """Install the collection served by the chart endpoints."""
        view = SeriesDataView(sdata)
        clean_path = None if path is None else self.tree.clean(path)
        with self._state_lock:
            self.view = view
            self.collection_path = clean_path

    def _require_collection(self) -> SeriesDataView:
        with self._state_lock:
            if self.view is None:
                raise _NoCollection
            return self.view

    def _collection_snapshot(self) -> tuple[SeriesDataView, str | None]:
        with self._state_lock:
            if self.view is None:
                raise _NoCollection
            return self.view, self.collection_path

    def _open_package(self, relative: str) -> dict[str, Any]:
        if not relative:
            raise ValueError("path is required")
        with self._open_lock:
            target = self.tree.resolve(relative)
            if not self.tree.is_loadable(relative):
                raise ValueError("path is not a SeriesData package")
            try:
                sdata = read(target)
                view = SeriesDataView(sdata)
            except MemoryError:
                raise
            except Exception as error:
                raise ValueError(f"failed to read package: {error}") from error
            clean_path = self.tree.clean(relative)
            with self._state_lock:
                self.view = view
                self.collection_path = clean_path
            payload = view.overview(self.title)
            payload["path"] = clean_path
            return payload

    def api(
        self,
        path: str,
        query: Mapping[str, Sequence[str]],
    ) -> dict[str, Any]:
        """Resolve an API request; public primarily for lightweight integrations."""
        params = _Query(query)
        if path == "/api/tree":
            return self.tree.listing(params.text("path"))
        if path == "/api/open":
            return self._open_package(params.text("path"))
        if path == "/api/overview":
            view, collection_path = self._collection_snapshot()
            payload = view.overview(self.title)
            if collection_path is not None:
                payload["path"] = collection_path
            return payload
        if path == "/api/items":
            return self._require_collection().items(
                query=params.text("query"),
                offset=params.integer("offset", 0),
                limit=params.integer("limit", 50),
            )
        if path == "/api/data":
            series_index = params.integer("series")
            if series_index is None:
                raise ValueError("series is required")
            return self._require_collection().data(
                series_index=series_index,
                feature_indices=params.integers("features"),
                normalization=params.text("normalization", "none"),
                scope=params.text("scope", "feature"),
                start=params.integer("start", 0),
                stop=params.integer("stop"),
                max_points=params.integer("max_points", DEFAULT_MAX_POINTS),
                label_index=params.integer("label"),
            )
        raise _RouteNotFound(path)

    def create_server(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> ThreadingHTTPServer:
        """Create, but do not start, the underlying HTTP server."""
        if not isinstance(host, str):
            raise TypeError("host must be a string")
        if isinstance(port, bool) or not isinstance(port, int):
            raise TypeError("port must be an integer")
        if port < 0 or port > 65_535:
            raise ValueError("port must be between 0 and 65535")
        app = self

        class Handler(BaseHTTPRequestHandler):
            def _headers(
                self,
                status: HTTPStatus,
                content_type: str,
                size: int,
                *,
                allow: str | None = None,
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(size))
                if allow is not None:
                    self.send_header("Allow", allow)
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Cross-Origin-Resource-Policy", "same-origin")
                self.send_header("Cross-Origin-Opener-Policy", "same-origin")
                self.send_header(
                    "Permissions-Policy",
                    "camera=(), geolocation=(), microphone=()",
                )
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; img-src 'self' data:; "
                    "style-src 'self'; script-src 'self'; connect-src 'self'; "
                    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
                )
                self.end_headers()

            def _send_json(
                self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK
            ) -> None:
                body = json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                self._headers(status, "application/json; charset=utf-8", len(body))
                if self.command != "HEAD":
                    self.wfile.write(body)

            def _send_asset(self, path: str) -> None:
                asset = app._assets.get(path)
                if asset is None:
                    body = b"Not found"
                    self._headers(
                        HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", len(body)
                    )
                    if self.command != "HEAD":
                        self.wfile.write(body)
                    return
                body, content_type = asset
                self._headers(HTTPStatus.OK, content_type, len(body))
                if self.command != "HEAD":
                    self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                request = urlsplit(self.path)
                if not request.path.startswith("/api/"):
                    self._send_asset(request.path)
                    return
                try:
                    query = parse_qs(
                        request.query,
                        keep_blank_values=True,
                        max_num_fields=32,
                    )
                    payload = app.api(request.path, query)
                except _RouteNotFound:
                    self._send_json(
                        {"error": "API endpoint not found"}, HTTPStatus.NOT_FOUND
                    )
                except _NoCollection:
                    self._send_json(
                        {"error": "no collection loaded"}, HTTPStatus.CONFLICT
                    )
                except (TypeError, ValueError) as error:
                    self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                except Exception:
                    self._send_json(
                        {"error": "internal server error"},
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                else:
                    self._send_json(payload)

            def do_HEAD(self) -> None:  # noqa: N802
                request = urlsplit(self.path)
                if request.path.startswith("/api/"):
                    body = b'{"error":"method not allowed"}'
                    self._headers(
                        HTTPStatus.METHOD_NOT_ALLOWED,
                        "application/json; charset=utf-8",
                        len(body),
                        allow="GET",
                    )
                    return
                self._send_asset(request.path)

            def log_message(self, format: str, *args: object) -> None:
                if app.log_requests:
                    super().log_message(format, *args)

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        if ":" in host:
            Server.address_family = socket.AF_INET6
        return Server((host, port), Handler)

    def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        *,
        open_browser: bool = False,
    ) -> None:
        """Run the WebUI until interrupted."""
        server = self.create_server(host, port)
        actual_host, actual_port = server.server_address[:2]
        browser_host = "127.0.0.1" if actual_host in {"0.0.0.0", "::"} else actual_host
        if ":" in browser_host:
            browser_host = f"[{browser_host}]"
        url = f"http://{browser_host}:{actual_port}"
        print(f"AgentAD Visualize: {url}")
        if actual_host not in {"127.0.0.1", "::1", "localhost"}:
            print(
                "Warning: the package browser is reachable from non-loopback "
                f"interfaces under {self.tree.root}"
            )
        if open_browser:
            timer = threading.Timer(0.2, webbrowser.open, args=(url,))
            timer.daemon = True
            timer.start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


def serve(
    sdata: SeriesData | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    path: str | None = None,
    root: str | os.PathLike | None = None,
    title: str = "AgentAD Visualize",
    open_browser: bool = False,
    log_requests: bool = False,
) -> None:
    """Convenience entry point for a blocking local WebUI server."""
    WebUI(
        sdata,
        path=path,
        root=root,
        title=title,
        log_requests=log_requests,
    ).serve(host, port, open_browser=open_browser)
