"""Dependency-free local WebUI server for :class:`SeriesData`."""

from __future__ import annotations

import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any
from urllib.parse import parse_qs, urlsplit

from ..series import SeriesData
from ._view import DEFAULT_MAX_POINTS, SeriesDataView

_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


def _one(query: dict[str, list[str]], name: str, default: str = "") -> str:
    values = query.get(name)
    return values[-1] if values else default


def _integer(
    query: dict[str, list[str]],
    name: str,
    default: int | None = None,
) -> int | None:
    raw = _one(query, name)
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


class WebUI:
    """A small local web application backed by one ``SeriesData`` collection."""

    def __init__(
        self,
        sdata: SeriesData,
        *,
        title: str = "AgentAD Visualize",
        log_requests: bool = False,
    ) -> None:
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-empty string")
        self.title = title.strip()
        self.view = SeriesDataView(sdata)
        self.log_requests = log_requests
        asset_root = files(__package__).joinpath("static")
        self._assets = {
            path: (asset_root.joinpath(name).read_bytes(), content_type)
            for path, (name, content_type) in _ASSETS.items()
        }

    def api(self, path: str, query: dict[str, list[str]]) -> dict[str, Any]:
        """Resolve an API request; public primarily for lightweight integrations."""
        if path == "/api/overview":
            return self.view.overview(self.title)
        if path == "/api/items":
            return self.view.items(
                query=_one(query, "query"),
                offset=_integer(query, "offset", 0) or 0,
                limit=_integer(query, "limit", 50) or 50,
            )
        if path == "/api/data":
            series_index = _integer(query, "series")
            if series_index is None:
                raise ValueError("series is required")
            raw_features = _one(query, "features")
            feature_indices = None
            if raw_features:
                try:
                    feature_indices = tuple(
                        int(value) for value in raw_features.split(",") if value != ""
                    )
                except ValueError as error:
                    raise ValueError("features must be comma-separated integers") from error
            return self.view.data(
                series_index=series_index,
                feature_indices=feature_indices,
                normalization=_one(query, "normalization", "none"),
                scope=_one(query, "scope", "feature"),
                start=_integer(query, "start", 0) or 0,
                stop=_integer(query, "stop"),
                max_points=_integer(query, "max_points", DEFAULT_MAX_POINTS)
                or DEFAULT_MAX_POINTS,
                label_index=_integer(query, "label"),
            )
        raise KeyError(path)

    def create_server(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> ThreadingHTTPServer:
        """Create, but do not start, the underlying HTTP server."""
        app = self

        class Handler(BaseHTTPRequestHandler):
            def _headers(self, status: HTTPStatus, content_type: str, size: int) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(size))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Cache-Control", "no-store")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; img-src 'self' data:; "
                    "style-src 'self'; script-src 'self'; connect-src 'self'",
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

            def do_GET(self) -> None:  # noqa: N802
                request = urlsplit(self.path)
                if request.path.startswith("/api/"):
                    try:
                        payload = app.api(request.path, parse_qs(request.query))
                    except KeyError:
                        self._send_json(
                            {"error": "API endpoint not found"}, HTTPStatus.NOT_FOUND
                        )
                    except (TypeError, ValueError) as error:
                        self._send_json(
                            {"error": str(error)}, HTTPStatus.BAD_REQUEST
                        )
                    except Exception:
                        self._send_json(
                            {"error": "internal server error"},
                            HTTPStatus.INTERNAL_SERVER_ERROR,
                        )
                    else:
                        self._send_json(payload)
                    return

                asset = app._assets.get(request.path)
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

            def do_HEAD(self) -> None:  # noqa: N802
                self.do_GET()

            def log_message(self, format: str, *args: object) -> None:
                if app.log_requests:
                    super().log_message(format, *args)

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

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
        url = f"http://{browser_host}:{actual_port}"
        print(f"AgentAD Visualize: {url}")
        if open_browser:
            threading.Timer(0.2, webbrowser.open, args=(url,)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


def serve(
    sdata: SeriesData,
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    title: str = "AgentAD Visualize",
    open_browser: bool = False,
    log_requests: bool = False,
) -> None:
    """Convenience entry point for a blocking local WebUI server."""
    WebUI(sdata, title=title, log_requests=log_requests).serve(
        host, port, open_browser=open_browser
    )

