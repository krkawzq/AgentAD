"""Dependency-free local WebUI server for :class:`SeriesData`."""

from __future__ import annotations

import hashlib
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

from ..series import SeriesData
from ._csv import DEFAULT_MAX_CSV_BYTES, read_source
from ._tree import PackageTree
from ._view import DEFAULT_MAX_POINTS, SeriesDataView

_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/core.js": ("core.js", "text/javascript; charset=utf-8"),
}
MAX_REQUEST_TARGET_BYTES = 8_192
MAX_REQUEST_WORKERS = 16
CLIENT_TIMEOUT_SECONDS = 30


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
    """A small local web application over one or more ``SeriesData`` collections.

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
        title: str = "AgentAD Visualizer",
        log_requests: bool = False,
        max_csv_bytes: int = DEFAULT_MAX_CSV_BYTES,
    ) -> None:
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-empty string")
        if isinstance(max_csv_bytes, bool) or not isinstance(max_csv_bytes, int):
            raise TypeError("max_csv_bytes must be an integer")
        if max_csv_bytes < 1:
            raise ValueError("max_csv_bytes must be positive")
        self.title = title.strip()
        self.max_csv_bytes = max_csv_bytes
        self.tree = PackageTree(os.getcwd() if root is None else root)
        self._state_lock = threading.RLock()
        self._open_lock = threading.Lock()
        self.view: SeriesDataView | None = None
        self.collection_path: str | None = None
        self._collections: dict[str, tuple[SeriesDataView, str | None]] = {}
        self._current_source: str | None = None
        if sdata is None and path is not None:
            raise ValueError("path requires an initial SeriesData collection")
        if sdata is not None:
            self.set_collection(sdata, path=path)
        self.log_requests = log_requests
        asset_root = files(__package__).joinpath("static")
        self._assets: dict[str, tuple[bytes, str, str]] = {}
        for asset_path, (name, content_type) in _ASSETS.items():
            body = asset_root.joinpath(name).read_bytes()
            etag = f'"{hashlib.blake2s(body, digest_size=12).hexdigest()}"'
            self._assets[asset_path] = (body, content_type, etag)

    @staticmethod
    def _source_id(path: str | None) -> str:
        if path is None:
            return "initial"
        digest = hashlib.blake2s(path.encode("utf-8"), digest_size=12).hexdigest()
        return f"source-{digest}"

    def _install_view(self, view: SeriesDataView, path: str | None) -> str:
        source = self._source_id(path)
        with self._state_lock:
            self._collections[source] = (view, path)
            self._current_source = source
            self.view = view
            self.collection_path = path
        return source

    def set_collection(self, sdata: SeriesData, *, path: str | None = None) -> None:
        """Install a collection and make it the legacy default source."""
        view = SeriesDataView(sdata)
        clean_path = None if path is None else self.tree.clean(path)
        self._install_view(view, clean_path)

    def _require_collection(self, source: str = "") -> SeriesDataView:
        return self._collection_snapshot(source)[0]

    def _collection_snapshot(
        self, source: str = ""
    ) -> tuple[SeriesDataView, str | None, str]:
        with self._state_lock:
            selected = source or self._current_source
            collection = None if selected is None else self._collections.get(selected)
            if collection is None or selected is None:
                raise _NoCollection
            view, collection_path = collection
            return view, collection_path, selected

    def _overview_payload(
        self, view: SeriesDataView, collection_path: str | None, source: str
    ) -> dict[str, Any]:
        payload = view.overview(self.title)
        payload["source"] = source
        if collection_path is not None:
            payload["path"] = collection_path
        return payload

    def _open_package(self, relative: str) -> dict[str, Any]:
        if not relative:
            raise ValueError("path is required")
        with self._open_lock:
            clean_path = self.tree.clean(relative)
            source = self._source_id(clean_path)
            with self._state_lock:
                cached = self._collections.get(source)
                if cached is not None:
                    view, collection_path = cached
                    self._current_source = source
                    self.view = view
                    self.collection_path = collection_path
                    return self._overview_payload(view, collection_path, source)
            target = self.tree.resolve(relative)
            if not self.tree.is_loadable(relative):
                raise ValueError("path is not a supported data source")
            try:
                sdata = read_source(target, max_csv_bytes=self.max_csv_bytes)
                view = SeriesDataView(sdata)
            except MemoryError:
                raise
            except Exception as error:
                raise ValueError(f"failed to read package: {error}") from error
            source = self._install_view(view, clean_path)
            return self._overview_payload(view, clean_path, source)

    def _close_source(self, source: str) -> dict[str, Any]:
        if not source:
            raise ValueError("source is required")
        with self._state_lock:
            if self._collections.pop(source, None) is None:
                raise _NoCollection
            if self._current_source == source:
                self._current_source = next(reversed(self._collections), None)
                if self._current_source is None:
                    self.view = None
                    self.collection_path = None
                else:
                    self.view, self.collection_path = self._collections[
                        self._current_source
                    ]
            return {"closed": source, "active": self._current_source}

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
        if path == "/api/close":
            return self._close_source(params.text("source"))
        if path == "/api/overview":
            view, collection_path, source = self._collection_snapshot(
                params.text("source")
            )
            return self._overview_payload(view, collection_path, source)
        if path == "/api/items":
            return self._require_collection(params.text("source")).items(
                query=params.text("query"),
                offset=params.integer("offset", 0),
                limit=params.integer("limit", 50),
            )
        if path == "/api/data":
            series_index = params.integer("series")
            if series_index is None:
                raise ValueError("series is required")
            return self._require_collection(params.text("source")).data(
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
            server_version = "AgentAD"
            sys_version = ""

            def version_string(self) -> str:
                return self.server_version

            def _headers(
                self,
                status: HTTPStatus,
                content_type: str,
                size: int,
                *,
                allow: str | None = None,
                cache_control: str = "no-store",
                etag: str | None = None,
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(size))
                if allow is not None:
                    self.send_header("Allow", allow)
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Cache-Control", cache_control)
                if etag is not None:
                    self.send_header("ETag", etag)
                self.send_header("Cross-Origin-Resource-Policy", "same-origin")
                self.send_header("Cross-Origin-Opener-Policy", "same-origin")
                self.send_header(
                    "Permissions-Policy",
                    "camera=(), geolocation=(), microphone=()",
                )
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; img-src 'self' data:; "
                    "style-src 'self' 'unsafe-inline'; script-src 'self'; "
                    "connect-src 'self'; object-src 'none'; worker-src 'self'; "
                    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
                )
                self.end_headers()

            def _write_body(self, body: bytes) -> None:
                if self.command == "HEAD":
                    return
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

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
                self._write_body(body)

            def _send_asset(self, path: str) -> None:
                asset = app._assets.get(path)
                if asset is None:
                    body = b"Not found"
                    self._headers(
                        HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", len(body)
                    )
                    self._write_body(body)
                    return
                body, content_type, etag = asset
                # Asset names are stable across frontend builds. Revalidate on
                # every navigation so a restarted service cannot serve a new
                # index with an hour-old JavaScript or stylesheet from cache.
                cache_control = "no-cache"
                if self.headers.get("If-None-Match") == etag:
                    self._headers(
                        HTTPStatus.NOT_MODIFIED,
                        content_type,
                        0,
                        cache_control=cache_control,
                        etag=etag,
                    )
                    return
                self._headers(
                    HTTPStatus.OK,
                    content_type,
                    len(body),
                    cache_control=cache_control,
                    etag=etag,
                )
                self._write_body(body)

            def do_GET(self) -> None:  # noqa: N802
                if len(self.path.encode("utf-8", errors="surrogatepass")) > (
                    MAX_REQUEST_TARGET_BYTES
                ):
                    self._send_json(
                        {"error": "request target is too long"},
                        HTTPStatus.REQUEST_URI_TOO_LONG,
                    )
                    return
                request = urlsplit(self.path)
                if not request.path.startswith("/api/"):
                    self._send_asset(request.path)
                    return
                if request.path in {"/api/open", "/api/close"} and self.headers.get(
                    "Sec-Fetch-Site"
                ) not in {None, "none", "same-origin"}:
                    self._send_json(
                        {"error": "cross-origin collection changes are not allowed"},
                        HTTPStatus.FORBIDDEN,
                    )
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

            def _method_not_allowed(self) -> None:
                body = b'{"error":"method not allowed"}'
                self._headers(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    "application/json; charset=utf-8",
                    len(body),
                    allow="GET, HEAD",
                )
                self._write_body(body)

            do_DELETE = _method_not_allowed
            do_OPTIONS = _method_not_allowed
            do_PATCH = _method_not_allowed
            do_POST = _method_not_allowed
            do_PUT = _method_not_allowed

            def setup(self) -> None:
                super().setup()
                self.connection.settimeout(CLIENT_TIMEOUT_SECONDS)

            def log_message(self, format: str, *args: object) -> None:
                if app.log_requests:
                    super().log_message(format, *args)

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True
            request_queue_size = 64

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self._request_slots = threading.BoundedSemaphore(MAX_REQUEST_WORKERS)
                super().__init__(*args, **kwargs)

            def process_request(self, request: Any, client_address: Any) -> None:
                self._request_slots.acquire()
                try:
                    super().process_request(request, client_address)
                except BaseException:
                    self._request_slots.release()
                    raise

            def process_request_thread(
                self,
                request: Any,
                client_address: Any,
            ) -> None:
                try:
                    super().process_request_thread(request, client_address)
                finally:
                    self._request_slots.release()

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
        print(f"AgentAD Visualizer: {url}")
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
    title: str = "AgentAD Visualizer",
    open_browser: bool = False,
    log_requests: bool = False,
    max_csv_bytes: int = DEFAULT_MAX_CSV_BYTES,
) -> None:
    """Convenience entry point for a blocking local WebUI server."""
    WebUI(
        sdata,
        path=path,
        root=root,
        title=title,
        log_requests=log_requests,
        max_csv_bytes=max_csv_bytes,
    ).serve(host, port, open_browser=open_browser)
