"""Owned, cancellable headless numerical subprocess for one analysis action."""
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import threading
import time
import secrets

from .display_lifecycle import DisplayProcessTree


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        # Windows readers/indexers can briefly hold the old file without delete sharing.
        for attempt in range(9):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 8:
                    raise
                time.sleep(min(.02 * 2**attempt, .2))
    finally:
        temporary.unlink(missing_ok=True)


def copy_action_state(source, target):
    """Copy only accepted numerical inputs; cancelled actions cannot mutate them."""
    source, target = Path(source).resolve(), Path(target).resolve()
    target.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.is_file() and path.suffix.lower() in {".json", ".joblib", ".csv"}:
            if not path.resolve().is_relative_to(source):
                raise ValueError("State inputs must remain inside the accepted state directory.")
            if path.name != "workspace_manifest.json" and not path.name.startswith("workspace_data_"):
                shutil.copy2(path, target / path.name)
    extraction = source / "extraction"
    if extraction.is_dir():
        destination = target / "extraction"
        destination.mkdir()
        for path in extraction.iterdir():
            if path.is_file() and path.suffix.lower() in {".json", ".joblib", ".csv"}:
                if not path.resolve().is_relative_to(source):
                    raise ValueError("Extraction snapshot escaped accepted state.")
                shutil.copy2(path, destination / path.name)


class WorkspaceTask:
    def __init__(self, directory, run_dir, action, payload):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.child = self.tree = None
        self._close_lock = threading.Lock()
        run_dir = Path(run_dir).resolve()
        manifest_path = run_dir / "workspace_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        source_state = Path(manifest.get("numerical_state_dir", run_dir)).resolve()
        if not source_state.is_relative_to(run_dir):
            raise ValueError("Accepted numerical state escaped the current run.")
        atomic_json(self.directory / "task.json", dict(run_dir=str(run_dir), source_state_dir=str(source_state),
                                                      action=action, payload=payload))
        environment = os.environ.copy()
        for key in list(environment):
            if key.startswith(("PYDEVD", "DEBUGPY")) or key in {"JPY_PARENT_PID", "JPY_SESSION_NAME"}:
                environment.pop(key)
        environment.update(MPLBACKEND="Agg", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                           OMP_NUM_THREADS="1", FOURBAND_WORKSPACE_GATE="1")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            with (self.directory / "stdout.log").open("wb") as out, (self.directory / "stderr.log").open("wb") as err:
                self.child = subprocess.Popen([sys.executable, str(Path(__file__).with_name("workspace_worker.py")),
                    str(self.directory / "task.json")], stdout=out, stderr=err, stdin=subprocess.PIPE,
                    env=environment, creationflags=flags, start_new_session=os.name != "nt")
            self.tree = DisplayProcessTree(self.child)
            self.child.stdin.write(b"\n")
            self.child.stdin.close()
        except BaseException:
            self.close()
            raise

    def poll(self):
        return self.child.poll()

    def read_result(self):
        path = self.directory / "result.json"
        if not path.is_file():
            raise RuntimeError(f"Numerical worker exited without a result (code {self.poll()}); see stderr.log.")
        return json.loads(path.read_text(encoding="utf-8"))

    def close(self):
        with self._close_lock:
            self._close_owned()

    def _close_owned(self):
        if self.tree is not None:
            tree, self.tree = self.tree, None
            tree.close()
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()
            try:
                self.child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.child.kill()
                self.child.wait(timeout=5)
