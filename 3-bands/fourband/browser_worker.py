"""Fresh owned local browser service; this entry point creates no native GUI."""
import os
os.environ["MPLBACKEND"] = "Agg"
import json
from pathlib import Path
import sys
import traceback

if __package__ in (None, ""):
    project = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(project / "3-bands"), str(project)]

if __name__ == "__mp_main__":
    from fourband.display_logging import configure_numerical_logging
    configure_numerical_logging()


def main():
    import faulthandler
    import joblib
    import webbrowser
    from fourband.browser_session import BrowserSession
    from fourband.browser_server import make_server
    faulthandler.enable()
    request_path = Path(sys.argv[1]).resolve()
    directory = request_path.parent
    session, server = None, None

    def status(stage, **details):
        payload = dict(stage=stage, pid=os.getpid())
        payload.update(details)
        (directory / "worker_status.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
        print(stage, flush=True)

    try:
        if os.environ.get("FOURBAND_DISPLAY_GATE") == "1" and not sys.stdin.buffer.read(1):
            raise RuntimeError("Display launcher disconnected before ownership was established.")
        status("loading_snapshot")
        request = joblib.load(request_path)
        status("loading_metadata")
        session = BrowserSession(request, request_path)
        server = make_server(session)
        details = dict(url=server.session_url, pid=os.getpid(), backend="browser")
        (directory / "browser_session.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
        print(server.session_url, flush=True)
        if os.environ.get("FOURBAND_NO_BROWSER") != "1":
            webbrowser.open(server.session_url)
        status("browser_ready", **details)
        server.serve_forever(poll_interval=.1)
        if session.closing:
            session.computation.raise_if_failed()
            session.computation.abort()
        else:
            session.computation.close()
            session.computation.raise_if_failed()
        status("complete")
        return 0
    except Exception as error:
        status("failed", error=str(error))
        traceback.print_exc()
        return 1
    finally:
        if server is not None:
            server.server_close()
        if session is not None:
            session.computation.abort()


if __name__ == "__main__":
    raise SystemExit(main())
