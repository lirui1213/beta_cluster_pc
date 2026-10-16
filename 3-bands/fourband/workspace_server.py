"""Fixed, same-origin APIs and revision-scoped explorers on one loopback service."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
import threading
from urllib.parse import parse_qs, unquote, urlsplit


def make_workspace_server(session, *, token=None, assets_dir=None):
    token = token or secrets.token_urlsafe(32)
    assets = Path(assets_dir or Path(__file__).with_name("browser_assets")).resolve()
    prefix = f"/s/{token}/"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send(self, status, value, content_type="application/json"):
            body = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; frame-src 'self'; frame-ancestors 'self'; font-src 'self'; object-src 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def route(self):
            parsed = urlsplit(self.path)
            if not parsed.path.startswith(prefix):
                raise FileNotFoundError("Unknown session.")
            route = unquote(parsed.path[len(prefix):])
            if "\\" in route or "\x00" in route or ".." in route.split("/"):
                raise FileNotFoundError("Unknown route.")
            return route, parse_qs(parsed.query)

        def asset(self, route, *, explorer=False):
            target = route or ("index.html" if explorer else "workspace.html")
            path = (assets / target).resolve()
            if not path.is_relative_to(assets) or not path.is_file():
                raise FileNotFoundError("Unknown asset.")
            self.send(200, path.read_bytes(), mimetypes.guess_type(str(path))[0] or "application/octet-stream")

        def dispatch_get(self, route, query):
            if route == "api/workspace/catalog":
                self.send(200, dict(patients=session.catalog.patients()))
            elif route == "api/workspace/state":
                self.send(200, session.state())
            elif route == "api/workspace/history":
                self.send(200, dict(runs=session.catalog.history(query.get("subject", [None])[0], query.get("block", [None])[0])))
            elif route == "api/workspace/inputs":
                self.send(200, session.catalog.resolve(query.get("subject", [None])[0], query.get("block", [None])[0]))
            elif route.startswith("api/workspace/artifact/"):
                path = session.artifact(route.removeprefix("api/workspace/artifact/"))
                self.send(200, path.read_bytes(), mimetypes.guess_type(str(path))[0] or "application/octet-stream")
            elif route.startswith("explorer/"):
                parts = route.split("/", 2)
                if len(parts) != 3:
                    raise FileNotFoundError("Unknown explorer.")
                explorer, rest = session.explorer(parts[1]), parts[2]
                if rest == "api/meta":
                    self.send(200, explorer.metadata)
                elif rest == "api/surface":
                    self.send(200, explorer.surface())
                elif rest.startswith("api/job/"):
                    channel = query.get("channel", [explorer.metadata["channels"][0]])[0]
                    self.send(200, explorer.read_job(rest.removeprefix("api/job/"), channel))
                elif rest.startswith("api/image/"):
                    self.send(200, explorer.detail_image(rest.removeprefix("api/image/")), "image/png")
                elif rest.startswith("api/"):
                    raise FileNotFoundError("Unknown explorer API.")
                else:
                    self.asset(rest, explorer=True)
            elif route.startswith("api/"):
                raise FileNotFoundError("Unknown API.")
            else:
                self.asset(route)

        def do_GET(self):
            try:
                route, query = self.route()
                self.dispatch_get(route, query)
            except FileNotFoundError as error:
                self.send(404, dict(error=str(error)))
            except (ValueError, KeyError, TypeError) as error:
                self.send(400, dict(error=str(error)))
            except Exception as error:
                self.send(500, dict(error=str(error)))

        def do_POST(self):
            try:
                route, _ = self.route()
                origin = f"http://127.0.0.1:{self.server.server_port}"
                if self.headers.get("Origin") != origin or self.headers.get("Host") != origin[7:]:
                    self.send(403, dict(error="Same-origin session requests are required."))
                    return
                maximum = 24 * 1024**2 if route.endswith("api/export") else 1024**2
                length = int(self.headers.get("Content-Length", "-1"))
                if length < 0 or length > maximum or self.headers.get_content_type() != "application/json":
                    self.send(413 if length > maximum else 400, dict(error="Invalid JSON size or content type."))
                    return
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("JSON object required.")
                workspace = "api/workspace/"
                if route == workspace + "start":
                    result = session.start(body)
                elif route == workspace + "action":
                    result = session.action(body)
                elif route == workspace + "cancel":
                    result = session.cancel(body)
                elif route == workspace + "open":
                    result = session.open(body)
                elif route == workspace + "select_round":
                    result = session.select_round(body)
                elif route == workspace + "close":
                    session.closing = True
                    self.send(200, dict(state="closed"))
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                    return
                elif route.startswith("explorer/"):
                    parts = route.split("/", 2)
                    if len(parts) != 3:
                        raise FileNotFoundError("Unknown explorer.")
                    explorer, rest = session.explorer(parts[1]), parts[2]
                    if rest == "api/compute":
                        result = explorer.compute(body)
                    elif rest == "api/clear":
                        result = explorer.clear()
                    elif rest == "api/export":
                        result = explorer.export(body)
                    elif rest == "api/close":
                        explorer.closing = True
                        explorer.computation.abort()
                        result = dict(state="closed")
                    else:
                        raise FileNotFoundError("Unknown explorer API.")
                else:
                    raise FileNotFoundError("Unknown API.")
                self.send(200, result)
            except FileNotFoundError as error:
                self.send(404, dict(error=str(error)))
            except (ValueError, KeyError, TypeError) as error:
                self.send(400, dict(error=str(error)))
            except Exception as error:
                self.send(500, dict(error=str(error)))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server.session_url = f"http://127.0.0.1:{server.server_port}{prefix}"
    return server
