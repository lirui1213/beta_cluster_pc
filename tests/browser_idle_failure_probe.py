"""Diagnostic Close Session after a real idle numerical-child failure."""
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "3-bands"), str(ROOT)]
if __name__ == "__mp_main__":
    from fourband.display_logging import configure_numerical_logging
    configure_numerical_logging()


def main():
    import fourband.browser_session as sessions
    import fourband.browser_server as servers
    from fourband.display_computation import DisplayComputation
    from fourband.browser_worker import main as run_worker
    def create_session(request, request_path):
        return SimpleNamespace(computation=DisplayComputation(request_path), closing=False)

    class DiagnosticServer:
        session_url = "http://127.0.0.1:1/diagnostic-only/"
        def __init__(self, session):
            self.session = session
        def serve_forever(self, **kwargs):
            computation = self.session.computation
            child_pid = computation.submit_task(os.getpid).result(timeout=30)
            if os.environ.get("FOURBAND_PROBE_BUSY_CLOSE") == "1":
                future = computation.submit_task(time.sleep, 30)
                deadline = time.monotonic() + 5
                while not future.running() and time.monotonic() < deadline:
                    time.sleep(.01)
                assert future.running()
                self.session.closing = True
                return
            owned = list(computation.pool._processes.values())
            assert len(owned) == 1 and owned[0].pid == child_pid
            owned[0].terminate()
            deadline = time.monotonic() + 10
            while not computation.pool._broken and time.monotonic() < deadline:
                time.sleep(.02)
            assert computation.pool._broken
            self.session.closing = True
        def server_close(self):
            pass

    sessions.BrowserSession = create_session
    servers.make_server = DiagnosticServer
    return run_worker()


if __name__ == "__main__":
    raise SystemExit(main())
