"""Launch the offline four-band analysis workspace without a Notebook kernel."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "3-bands"), str(ROOT)]
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["MPLBACKEND"] = "Agg"


def serve(args):
    if os.environ.get("FOURBAND_WORKSPACE_SERVICE_GATE") == "1" and not sys.stdin.buffer.read(1):
        return 1
    from fourband.workspace_catalog import WorkspaceCatalog
    from fourband.workspace_session import WorkspaceSession
    from fourband.workspace_server import make_workspace_server
    from fourband.workspace_task import atomic_json
    import webbrowser
    session = WorkspaceSession(WorkspaceCatalog(args.data_root, args.output_root))
    server = make_workspace_server(session)
    atomic_json(Path(args.session_dir) / "session.json", dict(url=server.session_url, pid=os.getpid()))
    print(f"Four-band analysis workspace: {server.session_url}", flush=True)
    if not args.no_browser:
        webbrowser.open(server.session_url)
    try:
        server.serve_forever(poll_interval=.1)
    finally:
        session.close()
        server.server_close()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=r"I:\ECoG_data")
    parser.add_argument("--output-root", default=str(ROOT / "3-bands" / "outputs"))
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--session-dir", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.serve:
        return serve(args)
    from fourband.display_lifecycle import DisplayProcessTree
    directory = Path(args.output_root).resolve() / "workspace_sessions" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), "--serve", "--data-root", args.data_root,
               "--output-root", str(Path(args.output_root).resolve()), "--session-dir", str(directory)]
    if args.no_browser:
        command.append("--no-browser")
    environment = os.environ.copy()
    environment.update(FOURBAND_WORKSPACE_SERVICE_GATE="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with (directory / "stdout.log").open("wb") as out, (directory / "stderr.log").open("wb") as err:
        child = subprocess.Popen(command, stdout=out, stderr=err, stdin=subprocess.PIPE,
            env=environment, creationflags=flags, start_new_session=os.name != "nt")
    tree = None
    try:
        tree = DisplayProcessTree(child)
        child.stdin.write(b"\n")
        child.stdin.close()
        announced = False
        while child.poll() is None:
            if not announced and (directory / "session.json").is_file():
                try:
                    details = json.loads((directory / "session.json").read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
                else:
                    print(details["url"], flush=True)
                    print(f"Session logs: {directory}", flush=True)
                    announced = True
            time.sleep(.2)
        return child.returncode
    finally:
        if tree is not None:
            tree.close()
        elif child.poll() is None:
            child.terminate()
        child.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
