"""Launch an owned local browser service or an explicit legacy native display."""
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import joblib
from .display_lifecycle import DisplayProcessTree


_WORKER_SCRIPT = Path(__file__).with_name("browser_worker.py")
_NATIVE_WORKER_SCRIPT = Path(__file__).with_name("display_worker.py")


@dataclass(frozen=True)
class DisplayRun:
    directory: Path
    pid: int
    returncode: int


class DisplayProcessError(RuntimeError):
    def __init__(self, directory, returncode):
        self.directory = Path(directory)
        self.returncode = returncode
        super().__init__(f"Display process exited with code {returncode}. Classification remains in the parent. "
                         f"See {self.directory / 'stderr.log'}, {self.directory / 'stdout.log'} "
                         f"and numerical_stderr.log if present.")


def _stop_child(child):
    try:
        child.terminate()
    except OSError:
        pass
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def run_display(extraction, result, *, output_dir, fif_path=None, onset_path=None,
                offset_path=None, wav_path=None, surface_options=None, show_surface=False,
                show_selectable=False, link_explorer=True, explorer_settings=None,
                cache_max_mb=512, explorer_only=False, backend="browser"):
    """Block until the owned display process exits; use trusted local snapshots.

    Only data/model snapshots cross the boundary, never raw GUI objects or a
    live cache. Close Session ends the browser service; interrupting cleans up
    only this launch's process tree. Abnormal exits are ordinary Python errors.
    """
    if not (show_surface or show_selectable or explorer_only):
        return None
    if backend not in {"browser", "native"}:
        raise ValueError("backend must be browser or native.")
    sources = dict(fif_path=fif_path, onset_path=onset_path, offset_path=offset_path, wav_path=wav_path)
    if link_explorer or explorer_only:
        for name, path in sources.items():
            if path is None:
                raise ValueError(f"{name} is required for curve display.")
            if not Path(path).is_file():
                raise FileNotFoundError(f"{name}: {path}")
    if cache_max_mb < 0:
        raise ValueError("cache_max_mb must be nonnegative.")
    directory = Path(output_dir).resolve() / "display_sessions" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    request = dict(extraction=extraction, result=result, output_dir=str(Path(output_dir).resolve()),
        sources={key: str(Path(value).resolve()) if value is not None else None for key, value in sources.items()},
        surface_options=dict(surface_options or {}), show_surface=bool(show_surface),
        show_selectable=bool(show_selectable), link_explorer=bool(link_explorer),
        explorer_only=bool(explorer_only), explorer_settings=dict(explorer_settings or {}), cache_max_mb=cache_max_mb,
        backend=backend)
    request_path = directory / "request.joblib"
    joblib.dump(request, request_path)
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith(("PYDEVD", "DEBUGPY")) or key in {"JPY_PARENT_PID", "JPY_SESSION_NAME"}:
            environment.pop(key)
    environment.update(MPLBACKEND="Agg", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", FOURBAND_DISPLAY_GATE="1")
    environment.setdefault("OMP_NUM_THREADS", "1")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
        script = _WORKER_SCRIPT if backend == "browser" else _NATIVE_WORKER_SCRIPT
        child = subprocess.Popen([sys.executable, str(script), str(request_path)],
            stdout=stdout, stderr=stderr, stdin=subprocess.PIPE, shell=False,
            env=environment, creationflags=flags, start_new_session=os.name != "nt")
        tree = None
        try:
            tree = DisplayProcessTree(child)
            print(f"Display process PID {child.pid}; logs: {directory}", flush=True)
            child.stdin.write(b"\n")
            child.stdin.close()
            announced_url = False
            if backend == "browser" and script.name == "browser_worker.py":
                while child.poll() is None:
                    url_file = directory / "browser_session.json"
                    if not announced_url and url_file.is_file():
                        try:
                            url = json.loads(url_file.read_text(encoding="utf-8"))["url"]
                        except (OSError, ValueError, KeyError):
                            pass
                        else:
                            print(f"Local four-band explorer: {url}", flush=True)
                            announced_url = True
                    import time
                    time.sleep(.1)
            returncode = child.wait()
        except BaseException:
            try:
                if tree is not None:
                    tree.close()
            finally:
                _stop_child(child)
            raise
        finally:
            if child.stdin is not None and not child.stdin.closed:
                child.stdin.close()
            if tree is not None:
                tree.close()
    (directory / "process_exit.json").write_text(json.dumps(dict(pid=child.pid, returncode=returncode)), encoding="utf-8")
    if returncode != 0:
        raise DisplayProcessError(directory, returncode)
    return DisplayRun(directory, child.pid, returncode)
