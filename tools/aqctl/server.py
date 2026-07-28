"""Read-only web console.

Serves the static console and a small JSON API over the collector's snapshot.
Nothing here mutates the cluster: changes go through cluster.yaml and
`aqctl apply`, so the deployment stays reproducible from git. A console that
could restart instances would quietly become a second source of truth.

Standard library only — no framework — so the tool's dependency stays PyYAML
alone. The load is a handful of internal viewers polling every few seconds.
"""

import json
import os
import posixpath
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import views

WEB_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}


def build_payload(path, cluster, specs, snapshot, gateway_url, links=None):
    """Map an API path to its JSON body. Pure — the tests drive it directly."""
    links = links or {}

    if path == "/api/cluster":
        return views.cluster_view(
            cluster, specs,
            snapshot["containers"], snapshot["health"], snapshot["metrics"],
            series=snapshot["series"],
            unreachable=snapshot["unreachable"],
            generated_at=snapshot["last_slow"],
        )
    if path == "/api/models":
        return views.models_view(
            cluster, specs, snapshot["health"], gateway_url,
            generated_at=snapshot["last_fast"],
        )
    if path == "/api/usage":
        return views.usage_view(
            snapshot["spend_rows"], generated_at=snapshot["last_slow"],
            error=snapshot["spend_error"], admin_url=links.get("gateway_admin"),
        )
    if path == "/api/links":
        # Where to go for the things this console deliberately does not do.
        return {"links": links}
    return None


class ConsoleHandler(SimpleHTTPRequestHandler):
    # Injected by serve().
    cluster = None
    specs = ()
    collector = None
    gateway_url = ""
    links = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_ROOT, **kwargs)

    def do_GET(self):
        path = self.path.split("?", 1)[0]

        if path.startswith("/api/"):
            return self._serve_api(path)
        if path == "/":
            path = "/index.html"

        # Everything else is a static asset from web/. SimpleHTTPRequestHandler
        # already refuses to escape its directory.
        self.path = path
        return super().do_GET()

    def _serve_api(self, path):
        try:
            payload = build_payload(
                path, self.cluster, self.specs, self.collector.read(),
                self.gateway_url, self.links,
            )
        except Exception as exc:
            return self._send_json({"error": str(exc)}, status=500)

        if payload is None:
            return self._send_json({"error": "no such endpoint"}, status=404)
        return self._send_json(payload)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # State changes every few seconds; a cached page would mislead.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def guess_type(self, path):
        ext = posixpath.splitext(path)[1]
        return CONTENT_TYPES.get(ext, super().guess_type(path))

    def log_message(self, fmt, *args):
        # One line per poll from every open tab is pure noise.
        pass


def serve(cluster, specs, collector, gateway_url, links=None,
          bind="127.0.0.1", port=8080):
    handler = type("BoundConsoleHandler", (ConsoleHandler,), {
        "cluster": cluster,
        "specs": specs,
        "collector": collector,
        "gateway_url": gateway_url,
        "links": links or {},
    })
    httpd = ThreadingHTTPServer((bind, port), handler)
    return httpd
