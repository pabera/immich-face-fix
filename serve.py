#!/usr/bin/env python3
"""Proxy server for Immich Face Fix tool. Serves static files and proxies API requests."""

import argparse
import http.server
import json
import os
import threading
import urllib.request
import urllib.error
import webbrowser


def main():
    parser = argparse.ArgumentParser(description="Immich Face Fix proxy server")
    parser.add_argument("--immich-url", default=os.environ.get("IMMICH_URL", ""),
                        help="Immich instance URL (e.g. https://photos.example.com)")
    parser.add_argument("--api-key", default=os.environ.get("IMMICH_API_KEY", ""),
                        help="Immich API key")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")),
                        help="Local port (default: 8080)")
    args = parser.parse_args()

    immich_url = args.immich_url.rstrip("/")
    api_key = args.api_key.strip()

    if not immich_url or not api_key:
        parser.error("--immich-url and --api-key are required")

    # Quick connectivity check
    test_url = immich_url + "/api/server/about"
    test_req = urllib.request.Request(test_url)
    test_req.add_header("x-api-key", api_key)
    try:
        with urllib.request.urlopen(test_req) as resp:
            print(f"Connected to Immich (status {resp.status})")
    except urllib.error.HTTPError as e:
        print(f"WARNING: Immich returned {e.code} for {test_url}")
        print(f"  Check your --api-key value (length={len(api_key)}, starts with '{api_key[:4]}...')")
    except Exception as e:
        print(f"WARNING: Cannot reach Immich at {immich_url}: {e}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dist_dir = os.path.join(base_dir, "dist")
    directory = dist_dir if os.path.isdir(dist_dir) else base_dir

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=directory, **kw)

        def do_GET(self):
            if self.path == "/api/_immich-url":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"url": immich_url}).encode())
            elif self.path.startswith("/api/"):
                self._proxy("GET")
            else:
                super().do_GET()

        def do_PUT(self):
            self._proxy("PUT")

        def do_POST(self):
            self._proxy("POST")

        def do_DELETE(self):
            self._proxy("DELETE")

        def _proxy(self, method):
            target = immich_url + self.path
            body = None
            if method in ("PUT", "POST"):
                length = int(self.headers.get("Content-Length", 0))
                if length:
                    body = self.rfile.read(length)

            req = urllib.request.Request(target, data=body, method=method)
            req.add_header("x-api-key", api_key)
            if self.headers.get("Content-Type"):
                req.add_header("Content-Type", self.headers["Content-Type"])
            req.add_header("Accept", self.headers.get("Accept", "application/json"))

            try:
                with urllib.request.urlopen(req) as resp:
                    resp_body = resp.read()
                    self.send_response(resp.status)
                    for key in ("Content-Type", "Content-Length"):
                        val = resp.headers.get(key)
                        if val:
                            self.send_header(key, val)
                    self.end_headers()
                    self.wfile.write(resp_body)
            except urllib.error.HTTPError as e:
                self.send_response(e.code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(e.read())
            except urllib.error.URLError as e:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        def log_message(self, fmt, *a):
            if a:
                print(f"  {' '.join(str(x) for x in a)}")

    server = http.server.HTTPServer(("", args.port), Handler)
    url = f"http://localhost:{args.port}/"
    print(f"Immich Face Fix running at {url}")
    print(f"Proxying to {immich_url}")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
