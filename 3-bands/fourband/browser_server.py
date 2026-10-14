"""Token-scoped loopback HTTP transport with no arbitrary filesystem API."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
import threading
from urllib.parse import unquote, urlsplit, parse_qs


def make_server(session, *, token=None, assets_dir=None):
    token = token or secrets.token_urlsafe(32)
    assets = Path(assets_dir or Path(__file__).with_name("browser_assets")).resolve()
    prefix = f"/s/{token}/"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def _send(self, status, body, content_type="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _route(self):
            parsed = urlsplit(self.path)
            if not parsed.path.startswith(prefix):
                return None, None
            route = unquote(parsed.path[len(prefix):])
            if "\\" in route or ".." in route.split("/") or "\x00" in route:
                return None, None
            return route, parse_qs(parsed.query)

        def do_GET(self):
            route, query = self._route()
            if route is None:
                self._send(404, dict(error="Not found"))
                return
            try:
                if route == "api/meta":
                    self._send(200, session.metadata)
                elif route == "api/surface":
                    self._send(200, session.surface())
                elif route.startswith("api/job/"):
                    channel = query.get("channel", [session.metadata["channels"][0]])[0]
                    self._send(200, session.read_job(route[len("api/job/"):], channel))
                elif route.startswith("api/image/"):
                    self._send(200, session.detail_image(route[len("api/image/"):]), "image/png")
                elif route.startswith("api/"):
                    self._send(404, dict(error="Not found"))
                else:
                    path = (assets / (route or "index.html")).resolve()
                    if not path.is_relative_to(assets) or not path.is_file():
                        self._send(404, dict(error="Not found"))
                    else:
                        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
                        self._send(200, path.read_bytes(), mime)
            except (ValueError, KeyError, TypeError) as error:
                self._send(400, dict(error=str(error)))
            except Exception as error:
                self._send(500, dict(error=str(error)))

        def do_POST(self):
            route, _ = self._route()
            if route is None:
                self._send(404, dict(error="Not found"))
                return
            origin = f"http://127.0.0.1:{self.server.server_port}"
            if self.headers.get("Origin") != origin or self.headers.get("Host") != origin[7:]:
                self._send(403, dict(error="Same-origin session requests are required."))
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                maximum = 24 * 1024**2 if route == "api/export" else 64 * 1024
                if length < 0 or length > maximum or self.headers.get_content_type() != "application/json":
                    self._send(413 if length > maximum else 400, dict(error="Invalid JSON request size or content type."))
                    return
                body = json.loads(self.rfile.read(length))
                if route == "api/compute":
                    response = session.compute(body)
                elif route == "api/clear":
                    response = session.clear()
                elif route == "api/export":
                    response = session.export(body)
                elif route == "api/close":
                    session.closing = True
                    self._send(200, dict(state="closed"))
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                    return
                else:
                    self._send(404, dict(error="Not found"))
                    return
                self._send(200, response)
            except (ValueError, KeyError, TypeError) as error:
                self._send(400, dict(error=str(error)))
            except Exception as error:
                self._send(500, dict(error=str(error)))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server.token, server.session = token, session
    server.session_url = f"http://127.0.0.1:{server.server_port}{prefix}"
    return server
