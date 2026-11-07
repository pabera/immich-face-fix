#!/usr/bin/env python3
"""Proxy server for Immich Face Fix tool. Serves static files and proxies API requests."""

import argparse
import array
import http.server
import json
import os
import re
import shlex
import struct
import subprocess
import sys
import threading
import urllib.parse
import urllib.request
import urllib.error
import webbrowser

UUID_RE = re.compile(r"^[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$")
PSQL_FLAGS = ["-X", "-q", "-t", "-A", "-v", "ON_ERROR_STOP=1"]
GALLERY_CAP = 1500  # faces per named person, enough to cover a lifetime of photos


def run_sql(psql_cmd, sql):
    """Run SQL through the user's psql command and return stdout as text."""
    result = subprocess.run(psql_cmd + PSQL_FLAGS, input=sql, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"psql exited with {result.returncode}")
    return result.stdout


def read_schema(psql_cmd):
    """Find the table and column names, which have changed between Immich releases."""
    found = set(run_sql(psql_cmd, """
        SELECT table_name || '.' || column_name FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name IN ('asset_face', 'asset_faces', 'asset', 'assets', 'face_search', 'person');
    """).split())

    def has(table, column):
        return f"{table}.{column}" in found

    faces = next((t for t in ("asset_face", "asset_faces") if has(t, "assetId")), None)
    assets = next((t for t in ("asset", "assets") if has(t, "localDateTime")), None)
    person = next((c for c in ("personGroupId", "personId") if faces and has(faces, c)), None)
    # Newer Immich keys person by (ownerId, personGroupId) and has no person.id.
    person_key = person if has("person", person or "") else "id"
    if not (faces and assets and person and has("person", person_key) and has("face_search", "embedding")):
        raise RuntimeError("Unrecognised Immich database schema (no face embeddings table found)")

    live = []
    if has(faces, "deletedAt"):
        live.append('f."deletedAt" IS NULL')
    if has(faces, "isVisible"):
        live.append('f."isVisible" IS TRUE')
    if has(assets, "deletedAt"):
        live.append('a."deletedAt" IS NULL')
    return {"faces": faces, "assets": assets, "person": person, "person_key": person_key, "live": live,
            "owned": has("person", "ownerId"), "hideable": has("person", "isHidden")}


def embeddings_query(schema, person_ids):
    person = schema["person"]
    where = [f'f."{person}" IN ({", ".join(chr(39) + p + chr(39) for p in person_ids)})'] + schema["live"]
    return f"""
        BEGIN TRANSACTION READ ONLY;
        COPY (
            SELECT f.id, f."assetId", f."{person}", extract(epoch from a."localDateTime")::bigint, s.embedding
            FROM "{schema['faces']}" f
            JOIN face_search s ON s."faceId" = f.id
            JOIN "{schema['assets']}" a ON a.id = f."assetId"
            WHERE {" AND ".join(where)}
            ORDER BY f.id
        ) TO STDOUT;
        COMMIT;
    """


def gallery_filter(schema, owner_of):
    """Faces worth matching: visible people, and only those of the API key's user.

    The database holds every user's people, so one of this user's people is passed
    in to find the owner. Nobody's faces may be merged across accounts.
    """
    where = list(schema["live"])
    if schema["hideable"]:
        where.append('p."isHidden" IS FALSE')
    if schema["owned"]:
        where.append(f"""p."ownerId" = (SELECT "ownerId" FROM person WHERE "{schema['person_key']}" = '{owner_of}')""")
    return where


def gallery_query(schema, owner_of, cap):
    """Every face of every unnamed person, and an even sample over time of each named one."""
    person, key, where = schema["person"], schema["person_key"], gallery_filter(schema, owner_of)
    return f"""
        BEGIN TRANSACTION READ ONLY;
        COPY (
            WITH faces AS (
                SELECT f.id, f."assetId", f."{person}" AS person, p.name <> '' AS named, a."localDateTime" AS taken,
                       row_number() OVER (PARTITION BY f."{person}" ORDER BY a."localDateTime", f.id) AS position,
                       count(*) OVER (PARTITION BY f."{person}") AS total
                FROM "{schema['faces']}" f
                JOIN person p ON p."{key}" = f."{person}"
                JOIN "{schema['assets']}" a ON a.id = f."assetId"
                WHERE {" AND ".join(where)}
            )
            SELECT v.id, v."assetId", v.person, v.named, extract(epoch from v.taken)::bigint, s.embedding
            FROM faces v
            JOIN face_search s ON s."faceId" = v.id
            WHERE NOT v.named OR v.total <= {cap} OR v.position % ceil(v.total::float / {cap})::bigint = 0
            ORDER BY v.person, v.taken, v.id
        ) TO STDOUT;
        COMMIT;
    """


def cooccurrence_query(schema, owner_of):
    """Unnamed and named people who appear in the same photo."""
    person, key, where = schema["person"], schema["person_key"], gallery_filter(schema, owner_of)
    same_owner = ' AND named."ownerId" = p."ownerId"' if schema["owned"] else ""
    return f"""
        BEGIN TRANSACTION READ ONLY;
        COPY (
            SELECT DISTINCT f."{person}", other."{person}"
            FROM "{schema['faces']}" f
            JOIN person p ON p."{key}" = f."{person}"
            JOIN "{schema['assets']}" a ON a.id = f."assetId"
            JOIN "{schema['faces']}" other ON other."assetId" = f."assetId" AND other.id <> f.id
            JOIN person named ON named."{key}" = other."{person}" AND named.name <> ''{same_owner}
            WHERE p.name = '' AND {" AND ".join(where)}
        ) TO STDOUT;
        COMMIT;
    """


def parse_vector(text, dim):
    """Return the vector and its length, or None if it doesn't match the others."""
    vector = array.array("f", map(float, text[1:-1].split(",")))
    return vector if not dim or len(vector) == dim else None


def pack_payload(header, vectors):
    """Binary payload: u32 header length, JSON header, then float32 embeddings."""
    if sys.byteorder == "big":
        vectors.byteswap()
    body = json.dumps(header).encode()
    body += b" " * (-len(body) % 4)  # keep the floats 4-byte aligned for the browser
    return struct.pack("<I", len(body)) + body + vectors.tobytes()


def seconds(ts):
    return int(ts) if ts.lstrip("-").isdigit() else 0


def fetch_embeddings(psql_cmd, wrong_id, correct_id):
    rows = run_sql(psql_cmd, embeddings_query(read_schema(psql_cmd), [wrong_id, correct_id]))
    faces, vectors, dim = [], array.array("f"), 0
    for line in rows.splitlines():
        parts = line.split("\t")
        if len(parts) != 5 or not parts[4].startswith("["):
            continue
        face_id, asset_id, person_id, ts, embedding = parts
        vector = parse_vector(embedding, dim)
        if vector is None:
            continue
        dim = dim or len(vector)
        faces.append([face_id, asset_id, 1 if person_id == correct_id else 0, seconds(ts)])
        vectors.extend(vector)
    return pack_payload({"dim": dim, "faces": faces}, vectors)


def fetch_gallery(psql_cmd, owner_of):
    """Faces of all this user's people, for matching unnamed people to named ones."""
    schema = read_schema(psql_cmd)
    rows = run_sql(psql_cmd, gallery_query(schema, owner_of, GALLERY_CAP))
    faces, people, index, vectors, dim = [], [], {}, array.array("f"), 0
    for line in rows.splitlines():
        parts = line.split("\t")
        if len(parts) != 6 or not parts[5].startswith("["):
            continue
        face_id, asset_id, person_id, named, ts, embedding = parts
        vector = parse_vector(embedding, dim)
        if vector is None:
            continue
        dim = dim or len(vector)
        if person_id not in index:
            index[person_id] = len(people)
            people.append({"id": person_id, "named": named == "t"})
        faces.append([face_id, asset_id, index[person_id], seconds(ts)])
        vectors.extend(vector)

    pairs = run_sql(psql_cmd, cooccurrence_query(schema, owner_of))
    cooccur = []
    for line in pairs.splitlines():
        unnamed, named = (line.split("\t") + [""])[:2]
        if unnamed in index and named in index:
            cooccur.append([index[unnamed], index[named]])
    return pack_payload({"dim": dim, "faces": faces, "people": people, "cooccur": cooccur}, vectors)


def main():
    parser = argparse.ArgumentParser(description="Immich Face Fix proxy server")
    parser.add_argument("--immich-url", default=os.environ.get("IMMICH_URL", ""),
                        help="Immich instance URL (e.g. https://photos.example.com)")
    parser.add_argument("--api-key", default=os.environ.get("IMMICH_API_KEY", ""),
                        help="Immich API key")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")),
                        help="Local port (default: 8080)")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"),
                        help="Address to bind (default: 127.0.0.1). The proxy has no login, so anyone who can "
                             "reach it can use your API key and read face embeddings.")
    parser.add_argument("--psql", default=os.environ.get("IMMICH_PSQL", ""),
                        help="Command that opens psql on the Immich database, enabling the pair model "
                             "(e.g. 'docker exec -i immich_postgres psql -U postgres -d immich')")
    args = parser.parse_args()

    immich_url = args.immich_url.rstrip("/")
    api_key = args.api_key.strip()
    psql_cmd = shlex.split(args.psql)

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
                self.wfile.write(json.dumps({"url": immich_url, "embeddings": bool(psql_cmd)}).encode())
            elif self.path.startswith("/api/_embeddings"):
                self._embeddings()
            elif self.path.startswith("/api/_gallery"):
                self._gallery()
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

        def _embeddings(self):
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            wrong_id = query.get("wrong", [""])[0]
            correct_id = query.get("correct", [""])[0]
            if not psql_cmd:
                return self._json_error(404, "Start serve.py with --psql to enable the pair model")
            if not (UUID_RE.match(wrong_id) and UUID_RE.match(correct_id)):
                return self._json_error(400, "wrong and correct must be person IDs")
            self._binary(lambda: fetch_embeddings(psql_cmd, wrong_id, correct_id))

        def _gallery(self):
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            owner_of = query.get("owner_of", [""])[0]
            if not psql_cmd:
                return self._json_error(404, "Start serve.py with --psql to enable automatic clean-up")
            if not UUID_RE.match(owner_of):
                return self._json_error(400, "owner_of must be a person ID")
            self._binary(lambda: fetch_gallery(psql_cmd, owner_of))

        def _binary(self, fetch):
            try:
                payload = fetch()
            except (RuntimeError, OSError, subprocess.SubprocessError) as e:
                return self._json_error(502, str(e))
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _json_error(self, status, message):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": message}).encode())

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

    server = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}/"
    print(f"Immich Face Fix running at {url}")
    print(f"Proxying to {immich_url}")
    print("Pair model: " + ("enabled" if psql_cmd else "off (add --psql to enable)"))
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
