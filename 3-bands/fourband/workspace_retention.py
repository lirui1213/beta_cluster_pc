"""Contained, browser-only single-record retention and context leases."""
import json
import os
from pathlib import Path
import re
import shutil

from .workspace_catalog import _identifier


_RUN = re.compile(r"\d{8}_\d{6}_\d{6}_web_[0-9a-f]{6}\Z")
_READY = {"trial_review", "trial_confirmed", "first_ready", "second_ready"}


class ContextLease:
    """The OS releases this lock even if the service exits unexpectedly."""

    def __init__(self, path):
        self.path = path
        self.stream = path.open("a+b")
        try:
            if path.stat().st_size == 0:
                self.stream.write(b"\0")
                self.stream.flush()
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            raise ValueError("This patient/block is in use by another analysis service.") from error

    def close(self):
        if self.stream.closed:
            return
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.stream.close()


class WorkspaceRetention:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def _checked(self, path):
        path = Path(path).absolute()
        if not path.is_relative_to(self.root) or path.resolve() != path:
            raise ValueError("Retention path escaped the output root or contains a link/junction.")
        for candidate in (path, *path.parents):
            if candidate.exists() or candidate.is_symlink():
                stat = candidate.lstat()
                if candidate.is_symlink() or getattr(stat, "st_file_attributes", 0) & 0x400:
                    raise ValueError("Retention refuses a link, junction or reparse point.")
            if candidate == self.root:
                break
        return path

    def parent(self, subject, block):
        _identifier(subject, "subject")
        _identifier(block, "block")
        return self._checked(self.root / subject / block)

    def lease(self, subject, block):
        parent = self.parent(subject, block)
        parent.mkdir(parents=True, exist_ok=True)
        return ContextLease(self._checked(parent / ".workspace.lock"))

    def _manifest(self, run):
        run = self._checked(run)
        relative = run.relative_to(self.root)
        if len(relative.parts) != 3 or not _RUN.fullmatch(run.name):
            raise ValueError("Not a recognized browser-owned analysis directory.")
        path = self._checked(run / "workspace_manifest.json")
        if not path.is_file() or path.stat().st_size > 1024 * 1024:
            raise ValueError("Browser ownership manifest is missing or too large.")
        data = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
                or data["schema_version"] != 1 or data.get("run_id") != run.name
                or data.get("subject") != relative.parts[0] or data.get("block") != relative.parts[1]
                or (run / "run_config.json").exists()):
            raise ValueError("Not a compatible browser-only ownership manifest.")
        return data

    def runs(self, subject, block):
        parent = self.parent(subject, block)
        if not parent.is_dir():
            return []
        result = []
        for run in sorted(parent.iterdir(), reverse=True):
            if not _RUN.fullmatch(run.name):
                continue
            try:
                data = self._manifest(run)
            except (OSError, ValueError):
                continue
            result.append((run, data))
        return result

    def owns(self, run):
        try:
            self._manifest(run)
        except ValueError:
            return False
        return True

    def discard_pending(self, subject, block):
        for run, data in self.runs(subject, block):
            if data.get("retention_pending"):
                self.remove(run)

    def current(self, subject, block):
        for run, data in self.runs(subject, block):
            if not data.get("retention_pending") and data.get("stage") in _READY:
                return dict(id=run.name, run_id=run.name, subject=subject, block=block,
                            stage=data["stage"], revision=data.get("revision"),
                            created_at=data.get("created_at"), read_only=False)
        return None

    def _tree(self, root):
        """Preflight the entire target without traversing any reparse point."""
        pending, paths = [self._checked(root)], []
        while pending:
            parent = pending.pop()
            with os.scandir(parent) as entries:
                for entry in entries:
                    path = self._checked(Path(entry.path))
                    paths.append(path)
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
        return paths

    def remove(self, run):
        run = self._checked(run)
        self._manifest(run)
        self._tree(run)
        # Ownership, absolute containment and all reparse points were checked.
        shutil.rmtree(run)

    def consolidate(self, keep):
        keep = self._checked(keep)
        data = self._manifest(keep)
        if data.get("retention_pending") or data.get("stage") not in _READY:
            raise ValueError("Cannot replace the current analysis with unaccepted extraction.")
        for run, _ in self.runs(data["subject"], data["block"]):
            if run != keep:
                self.remove(run)

    def collect(self, run):
        """Keep only files reachable from the atomically committed manifest."""
        run = self._checked(run)
        data = self._manifest(run)
        roots, files = [], set()

        def contained(value):
            path = self._checked(value if Path(value).is_absolute() else run / value)
            if not path.is_relative_to(run):
                raise ValueError("A retained snapshot reference escaped its analysis.")
            return path

        numerical = contained(data.get("numerical_state_dir", str(run)))
        roots.append(numerical if numerical != run else run / "extraction")
        for record in data.get("rounds", []):
            folder = record.get("artifact_root") or record.get("data", {}).get("round_dir")
            if folder:
                roots.append(contained(folder))
            if record.get("display_request"):
                files.add(contained(record["display_request"]))
        files.update(contained(value) for value in data.get("artifact_paths", {}).values())
        state_file = data.get("state_data_file")
        if state_file:
            files.add(contained(state_file))
        tasks = self._checked(run / "workspace_tasks")
        paths = self._tree(tasks) if tasks.is_dir() else []
        # All references and the entire deletion subtree are validated first.
        for path in paths:
            if path.is_file() and path not in files and not any(path.is_relative_to(root) for root in roots):
                path.unlink()
        for path in sorted(paths, key=lambda item: len(item.parts), reverse=True):
            protected = any(path.is_relative_to(root) or root.is_relative_to(path) for root in roots)
            if path.is_dir() and not protected and not any(path.iterdir()):
                path.rmdir()
        for path in run.glob("workspace_data_*.json"):
            self._checked(path)
            if path not in files:
                path.unlink()
