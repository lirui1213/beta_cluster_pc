"""Server-authoritative run revisions, explicit review gates and durable history."""
from datetime import datetime
import json
import mimetypes
from pathlib import Path
import secrets
import threading
import time
from zoneinfo import ZoneInfo

from .workspace_task import WorkspaceTask, atomic_json


_READ_ONLY = {"waveform", "preview_review", "stability"}
_STAGES = {
    "detect_qc": {"trial_review", "trial_confirmed", "first_ready", "second_ready"},
    "confirm_trials": {"trial_review", "trial_confirmed", "first_ready", "second_ready"},
    "fit_first": {"trial_confirmed", "first_ready", "second_ready"},
    "preview_review": {"first_ready", "second_ready"},
    "fit_second": {"first_ready", "second_ready"},
    "waveform": {"trial_review", "trial_confirmed", "first_ready", "second_ready"},
    "stability": {"first_ready", "second_ready"},
}


def _timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")


def _identifier(value):
    if not isinstance(value, str) or not value or len(value) > 180 or any(c in value for c in ("/", "\\", "\x00")) or value in {".", ".."}:
        raise ValueError("Invalid saved-run identifier.")
    return value


class WorkspaceSession:
    def __init__(self, catalog, *, task_factory=WorkspaceTask, config_validator=None):
        if config_validator is None:
            from .workspace_catalog import validate_config
            config_validator = validate_config
        self.catalog = catalog
        self.task_factory, self.config_validator = task_factory, config_validator
        self.run_dir = None
        self._state = dict(schema_version=1, run_id=None, revision=0, stage="idle",
                           subject=None, block=None, config={}, task=None, data={}, rounds=[])
        self._task = None
        self._lock = threading.RLock()
        self._explorer = None
        self._explorer_id = None
        self._load_generation = 0
        self._loader_task = None
        self._loader_thread = None
        self._loading_cancel = threading.Event()
        self._retirements = set()
        self._artifacts = {}
        self.closing = False
        self._stop = threading.Event()
        self._monitor = threading.Thread(target=self._watch, daemon=True, name="analysis-task-monitor")
        self._monitor.start()

    def _watch(self):
        while not self._stop.wait(.25):
            try:
                self.refresh()
            except Exception as error:
                with self._lock:
                    if self._state.get("task"):
                        self._state["task"].update(status="failed", error=str(error))

    def _save(self):
        if self.run_dir is not None and not self._state.get("read_only"):
            filename = f"workspace_data_{self._state['revision']}.json"
            atomic_json(self.run_dir / filename, dict(data=self._state["data"],
                round_data={item["id"]: item.get("data", {}) for item in self._state["rounds"]}))
            manifest = dict(self._state, data={}, state_data_file=filename,
                rounds=[{key: value for key, value in item.items() if key != "data"} for item in self._state["rounds"]])
            atomic_json(self.run_dir / "workspace_manifest.json", manifest)

    def _busy(self, *, include_retirement=True):
        task = self._state.get("task") or {}
        return (self._task is not None or task.get("status") in {"opening", "cancelling"}
                or (task.get("status") == "cancelled" and self._loader_thread is not None and self._loader_thread.is_alive())
                or (include_retirement and bool(self._retirements)))

    def _check(self, body):
        if (not isinstance(body, dict) or type(body.get("revision")) is not int
                or body.get("run_id") != self._state["run_id"] or body.get("revision") != self._state["revision"]):
            raise ValueError("Stale run or revision; refresh the active analysis.")
        if self.closing:
            raise ValueError("Workspace is closing.")

    def state(self):
        with self._lock:
            response = json.loads(json.dumps(self._state))
            task = response.get("task")
            if task and task.get("status") == "completed" and self._retirements:
                task.update(status="cancelling", stage="Closing previous display worker")
            if task and self._task is not None:
                task["elapsed_seconds"] = round(time.monotonic() - self._started, 1)
                directory = self._task.directory
                try:
                    task["stage"] = json.loads((directory / "progress.json").read_text(encoding="utf-8")).get("stage", task["action"])
                except (OSError, ValueError):
                    pass
                tails = []
                for name in ("stdout.log", "stderr.log"):
                    try:
                        with (directory / name).open("rb") as stream:
                            stream.seek(0, 2)
                            stream.seek(max(0, stream.tell() - 5000))
                            tails.append(stream.read().decode("utf-8", errors="replace"))
                    except OSError:
                        pass
                task["log_tail"] = "\n".join(tails)[-10000:]
            if self._explorer_id is not None:
                response["explorer_url"] = f"explorer/{self._explorer_id}/"
            response["artifacts"] = [{"id": key, "label": path.name,
                "content_type": mimetypes.guess_type(str(path))[0] or "application/octet-stream"}
                for key, path in self._artifacts.items() if key in self._visible_artifact_ids()]
            return response

    def _launch(self, action, payload, *, allow_retirement=False):
        if self._busy(include_retirement=not allow_retirement):
            raise ValueError("A numerical task is running; cancel it before changing context.")
        identifier = _timestamp() + "_" + secrets.token_hex(3)
        directory = self.run_dir / "workspace_tasks" / identifier
        task = self.task_factory(directory, self.run_dir, action, payload)
        self._task = task
        self._started = time.monotonic()
        self._state["revision"] += 1
        self._state["task"] = dict(id=identifier, action=action, status="running", stage=action,
                                   elapsed_seconds=0, error=None, log_tail="")
        self._save()

    def start(self, body):
        with self._lock:
            if self.closing or self._busy():
                raise ValueError("Workspace is closing or a numerical task is running.")
            if not isinstance(body, dict) or set(body) - {"subject", "block", "config", "selections"}:
                raise ValueError("Invalid start request.")
            resolved = self.catalog.resolve(body.get("subject"), body.get("block"), body.get("selections"))
            config = self.config_validator(body.get("config", {}), resolved)
            self._retire_explorer()
            identifier = _timestamp() + "_web_" + secrets.token_hex(3)
            root = Path(self.catalog.output_root).resolve()
            self.run_dir = (root / resolved["subject"] / resolved["block"] / identifier).resolve()
            if not self.run_dir.is_relative_to(root):
                raise ValueError("Output must remain inside the configured output root.")
            self.run_dir.mkdir(parents=True, exist_ok=False)
            atomic_json(self.run_dir / "config.json", config)
            self._artifacts = {}
            self._state = dict(schema_version=1, run_id=identifier, revision=0, stage="configured",
                subject=resolved["subject"], block=resolved["block"], config=config, task=None,
                data={}, rounds=[], created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
            self._launch("extract", {}, allow_retirement=True)
            return self.state()

    def action(self, body):
        with self._lock:
            self._check(body)
            if self._state.get("read_only"):
                raise ValueError("Legacy analysis is read-only; create a new run to confirm reviews and fit.")
            action = body.get("action")
            if action not in _STAGES or self._state["stage"] not in _STAGES[action]:
                raise ValueError("Action requires current confirmed trial review / first-round result.")
            selected = self._state.get("selected_round")
            current = {self._state.get("first_round_id"), self._state.get("second_round_id")}
            if selected is not None and selected not in current and action in {"preview_review", "fit_second", "stability"}:
                raise ValueError("A historical round is displayed; select the current accepted round before reviewing or testing stability.")
            payload = body.get("payload", {})
            if not isinstance(payload, dict):
                raise ValueError("Action payload must be an object.")
            if action == "confirm_trials" and "pairs" not in payload:
                raise ValueError("Explicit trial exclusion confirmation is required, including an empty pairs list.")
            self._launch(action, payload)
            return self.state()

    def refresh(self):
        with self._lock:
            task = self._task
            if task is None or self._state["task"]["status"] == "cancelling" or task.poll() is None:
                return
            action = self._state["task"]["action"]
            before_state = json.loads(json.dumps(self._state))
            before_artifacts = dict(self._artifacts)
            try:
                result = task.read_result()
                if not isinstance(result, dict) or result.get("error"):
                    raise RuntimeError(result.get("error", "Invalid numerical task result"))
                if task.poll() != 0:
                    raise RuntimeError("Numerical worker failed; inspect task logs.")
                # Check all file references before changing the accepted revision.
                self._validate_artifacts(result.get("artifacts", []))
                self._state["task"].update(status="completed", elapsed_seconds=round(time.monotonic() - self._started, 1))
                if action not in _READ_ONLY:
                    self._state["stage"] = result["stage"]
                    if action in {"detect_qc", "confirm_trials", "fit_first"}:
                        self._retire_explorer()
                    if action in {"detect_qc", "confirm_trials"}:
                        self._state.update(first_round_id=None, second_round_id=None, selected_round=None)
                if action in {"detect_qc", "confirm_trials", "fit_first", "fit_second"}:
                    for key in ("pca", "cluster_sizes", "summary", "plot_paths", "round", "round_dir", "warnings", "preview", "stability"):
                        self._state["data"].pop(key, None)
                data = result.get("data", {})
                if result.get("numerical_state_dir"):
                    numerical = Path(result["numerical_state_dir"]).resolve()
                    if not numerical.is_relative_to(self.run_dir.resolve()):
                        raise ValueError("Worker returned a numerical state outside the current run.")
                    self._state["numerical_state_dir"] = str(numerical)
                self._state["data"].update(data)
                updates = result.get("config_update", {})
                if not isinstance(updates, dict) or set(updates) - {"first_k", "second_k", "band_weights", "weight_preset", "group_weights", "correlation_threshold", "random_state"}:
                    raise ValueError("Invalid fitted-setting update.")
                self._state["config"].update(updates)
                self._register_artifacts(result.get("artifacts", []))
                new_ids = [key for key in self._artifacts if key not in before_artifacts]
                if action in {"extract", "detect_qc", "confirm_trials"}:
                    shared = self._state.get("shared_artifacts", [])
                    obsolete = {"trial_quality.csv", "trial_confirmation.json", "settings.json"}
                    self._state["shared_artifacts"] = [key for key in shared
                        if key in self._artifacts and self._artifacts[key].name not in obsolete] + new_ids
                if action == "stability":
                    number = data.get("stability", {}).get("round")
                    current = self._state.get("first_round_id" if number == 1 else "second_round_id")
                    record = next((item for item in self._state["rounds"] if item["id"] == current), None)
                    if record is not None:
                        record.setdefault("extra_artifacts", []).extend(new_ids)
                        report = data["stability"]
                        record.setdefault("data", {})["stability"] = dict(round=number, summary=report.get("summary", {}))
                if action in {"fit_first", "fit_second"}:
                    round_number = 1 if action == "fit_first" else 2
                    request = result.get("display_request")
                    round_id = self._state["task"]["id"]
                    record = dict(id=round_id, round=round_number,
                                  label=f"Round {round_number}: {round_id}", display_request=request,
                                  data=data, artifacts=new_ids, artifact_root=data.get("round_dir"),
                                  shared_artifacts=list(self._state.get("shared_artifacts", [])))
                    self._state["rounds"].append(record)
                    self._state["selected_round"] = round_id
                    if round_number == 1:
                        self._state.update(first_round_id=round_id, second_round_id=None)
                    else:
                        self._state["second_round_id"] = round_id
                    if request:
                        self._queue_attach(request)
                self._state["revision"] += 1
                self._save()
            except Exception as error:
                self._loading_cancel.set()
                self._load_generation += 1
                self._state, self._artifacts = before_state, before_artifacts
                self._state["task"].update(status="failed", error=str(error),
                    elapsed_seconds=round(time.monotonic() - self._started, 1))
                self._state["revision"] += 1
                try:
                    self._save()
                except OSError:
                    pass
            finally:
                task.close()
                self._task = None

    def _validate_artifacts(self, artifacts):
        if isinstance(artifacts, dict):
            artifacts = list(artifacts.values())
        accepted = []
        for item in artifacts:
            value = item.get("path") if isinstance(item, dict) else item
            if value is None:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = self.run_dir / path
            path = path.resolve()
            if not path.is_relative_to(self.run_dir.resolve()) or path.suffix.lower() not in {".png", ".json", ".csv"}:
                raise ValueError("Artifact outside accepted run or unsupported file type.")
            if path.is_file():
                accepted.append(path)
        return accepted

    def _register_artifacts(self, artifacts):
        for path in self._validate_artifacts(artifacts):
            self._artifacts[secrets.token_hex(12)] = path
        self._state["artifact_paths"] = {key: str(path.relative_to(self.run_dir)) for key, path in self._artifacts.items()}

    def artifact(self, identifier):
        with self._lock:
            path = self._artifacts.get(identifier)
            if path is None or identifier not in self._visible_artifact_ids() or not path.resolve().is_relative_to(self.run_dir.resolve()):
                raise ValueError("Unknown artifact.")
            return path

    def _visible_artifact_ids(self):
        selected = self._state.get("selected_round")
        record = next((item for item in self._state["rounds"] if item["id"] == selected), None)
        if record is None:
            return set(self._state.get("shared_artifacts", self._artifacts))
        root = record.get("artifact_root") or record.get("data", {}).get("round_dir")
        if root is None and record.get("display_request"):
            root = str(Path(record["display_request"]).parent)
        if root:
            directory = Path(root).resolve()
            own = {key for key, path in self._artifacts.items() if path.is_relative_to(directory)}
        else:
            own = set(record.get("artifacts", []))
        return own | set(record.get("shared_artifacts", [])) | set(record.get("extra_artifacts", []))

    def _retire_explorer(self):
        self._load_generation += 1
        previous, self._explorer = self._explorer, None
        self._explorer_id = None
        if previous is not None:
            previous.closing = True
            done = threading.Event()
            self._retirements.add(done)

            def stop():
                try:
                    previous.computation.abort()
                finally:
                    done.set()
                    with self._lock:
                        self._retirements.discard(done)

            threading.Thread(target=stop, daemon=True, name="analysis-display-retirement").start()

    def _load_explorer(self, request_path):
        path = Path(request_path).resolve()
        if not path.is_relative_to(self.run_dir.resolve()) or path.name != "request.joblib" or not path.is_file():
            raise ValueError("Display snapshot is not a trusted run artifact.")
        directory = self.run_dir / "workspace_tasks" / (_timestamp() + "_display_" + secrets.token_hex(3))
        task = WorkspaceTask(directory, self.run_dir, "prepare_display", {"request_path": str(path)})
        with self._lock:
            self._loader_task = task
        try:
            while task.poll() is None:
                if self._loading_cancel.wait(.1):
                    raise ValueError("Display loading cancelled.")
            prepared = task.read_result()
            if prepared.get("error"):
                raise ValueError(prepared["error"])
        finally:
            task.close()
            with self._lock:
                if self._loader_task is task:
                    self._loader_task = None
        from types import SimpleNamespace
        import numpy as np
        from .browser_session import BrowserSession
        from .workspace_display import guarded_computation
        metadata = prepared["metadata"]
        request = dict(output_dir=prepared["output_dir"], result=SimpleNamespace(
            channels=tuple(metadata["channels"]), labels=np.asarray(metadata["labels"], dtype=int)))
        computation = guarded_computation(path, prepared.get("sources", {}), prepared.get("input_identity"))
        explorer = BrowserSession(request, path, metadata=metadata, computation=computation)
        explorer._surface = prepared["surface"]
        return explorer

    def _queue_attach(self, request_path):
        self._retire_explorer()
        self._loading_cancel = threading.Event()
        generation = self._load_generation
        self._state["task"] = dict(id=secrets.token_hex(12), action="load_display", status="opening",
            stage="Loading saved model and anatomy", elapsed_seconds=0, error=None, log_tail="")
        started = time.monotonic()

        def load():
            explorer, error = None, None
            try:
                explorer = self._load_explorer(request_path)
            except Exception as caught:
                error = str(caught)
            with self._lock:
                if generation != self._load_generation or self.closing:
                    if explorer is not None:
                        explorer.computation.abort()
                    return
                if explorer is not None:
                    self._explorer, self._explorer_id = explorer, secrets.token_hex(16)
                self._state["task"].update(status="failed" if error else "completed", error=error,
                    elapsed_seconds=round(time.monotonic() - started, 1))
                self._state["revision"] += 1
                self._save()

        self._loader_thread = threading.Thread(target=load, daemon=True, name="analysis-anatomy-load")
        self._loader_thread.start()

    def explorer(self, identifier):
        with self._lock:
            if identifier != self._explorer_id or self._explorer is None:
                raise ValueError("This explorer is no longer the active accepted result.")
            return self._explorer

    def cancel(self, body):
        with self._lock:
            self._check(body)
            task = self._task
            if task is not None:
                self._state["task"].update(status="cancelling")
                self._save()
        if task is not None:
            task.close()
            with self._lock:
                self._task = None
                self._state["task"].update(status="cancelled", elapsed_seconds=round(time.monotonic() - self._started, 1))
                self._state["revision"] += 1
                self._save()
        else:
            with self._lock:
                if (self._state.get("task") or {}).get("status") != "opening":
                    return self.state()
                self._load_generation += 1
                self._loading_cancel.set()
                loader, loader_thread = self._loader_task, self._loader_thread
                self._state["task"].update(status="cancelling")
            if loader is not None:
                loader.close()
            if loader_thread is not None:
                loader_thread.join(timeout=10)
            with self._lock:
                self._state["task"].update(status="cancelled")
                self._state["revision"] += 1
                self._save()
        return self.state()

    def open(self, body):
        with self._lock:
            if self._busy():
                raise ValueError("A numerical task is running; cancel before opening history.")
            resolved = (self.catalog.context(body.get("subject"), body.get("block"))
                        if hasattr(self.catalog, "context") else self.catalog.resolve(body.get("subject"), body.get("block")))
            root = Path(self.catalog.output_root).resolve()
            directory = (root / resolved["subject"] / resolved["block"] / _identifier(body.get("run_id"))).resolve()
            if not directory.is_relative_to(root):
                raise ValueError("Invalid history location.")
            manifest = directory / "workspace_manifest.json"
            if not manifest.is_file():
                return self._open_legacy(directory, resolved, body.get("round_id"))
            state = json.loads(manifest.read_text(encoding="utf-8"))
            if (state.get("schema_version") != 1 or state.get("subject") != resolved["subject"]
                    or state.get("block") != resolved["block"] or state.get("run_id") != directory.name):
                raise ValueError("Unsupported or mismatched saved-run manifest.")
            identifier = body.get("round_id") or state.get("selected_round")
            if identifier and not any(item.get("id") == identifier for item in state.get("rounds", [])):
                raise ValueError("Unknown saved round.")
            if state.get("state_data_file"):
                data_path = (directory / _identifier(state["state_data_file"])).resolve()
                if not data_path.is_relative_to(directory) or data_path.stat().st_size > 32 * 1024**2:
                    raise ValueError("Saved analysis metadata is outside the run or too large.")
                saved_data = json.loads(data_path.read_text(encoding="utf-8"))
                state["data"] = saved_data["data"]
                for record in state["rounds"]:
                    record["data"] = saved_data.get("round_data", {}).get(record["id"], {})
            self._retire_explorer()
            self.run_dir, self._state = directory, state
            self._state["revision"] += 1
            if self._state.get("task") and self._state["task"]["status"] in {"running", "opening"}:
                self._state["task"].update(status="interrupted", error="Service stopped before this action completed.")
            self._artifacts = {}
            for key, relative in state.get("artifact_paths", {}).items():
                path = (directory / relative).resolve()
                if path.is_relative_to(directory) and path.is_file():
                    self._artifacts[key] = path
            if identifier:
                self._select_round(identifier)
            self._save()
            return self.state()

    def _open_legacy(self, directory, resolved, round_id):
        config_path = directory / "run_config.json"
        if not config_path.is_file():
            raise ValueError("No compatible saved analysis is available.")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rounds, artifacts = [], []
        for folder in sorted(directory.iterdir())[:256]:
            if not folder.is_dir() or not folder.name.startswith(("round1", "round2")) or not folder.resolve().is_relative_to(directory):
                continue
            requests = sorted(folder.glob("display_sessions/*/request.joblib"))[-32:]
            record = dict(id=folder.name, round=1 if folder.name.startswith("round1") else 2,
                          label=f"Historical {folder.name}", data={}, artifact_root=str(folder.resolve()))
            if requests:
                record["display_request"] = str(requests[-1].resolve())
            rounds.append(record)
            artifacts.extend(str(path) for path in sorted(folder.glob("*.png"))[:32])
        if round_id and not any(item["id"] == round_id for item in rounds):
            raise ValueError("Unknown saved round.")
        self._retire_explorer()
        self.run_dir = directory
        self._state = dict(schema_version=1, run_id=directory.name, revision=1, stage="historical",
            read_only=True, subject=resolved["subject"], block=resolved["block"], config=config,
            task=None, data={}, rounds=rounds)
        self._artifacts = {}
        self._register_artifacts(artifacts)
        if round_id or rounds:
            self._select_round(round_id or rounds[-1]["id"])
        return self.state()

    def _select_round(self, identifier):
        record = next((item for item in self._state["rounds"] if item["id"] == identifier), None)
        if record is None:
            raise ValueError("Unknown saved round.")
        self._retire_explorer()
        self._state["selected_round"] = identifier
        for key in ("pca", "cluster_sizes", "summary", "plot_paths", "round", "round_dir", "warnings", "preview", "stability"):
            self._state["data"].pop(key, None)
        self._state["data"].update(record.get("data", {}))
        if record.get("display_request"):
            self._queue_attach(record["display_request"])

    def select_round(self, body):
        with self._lock:
            self._check(body)
            if self._busy():
                raise ValueError("A numerical task is running.")
            self._select_round(body.get("round_id"))
            self._state["revision"] += 1
            self._save()
            return self.state()

    def close(self):
        self.closing = True
        self._stop.set()
        self._loading_cancel.set()
        loader = self._loader_task
        if loader is not None:
            loader.close()
        with self._lock:
            if self._task is not None:
                self._task.close()
                self._task = None
                self._state["task"].update(status="cancelled")
            self._retire_explorer()
            self._save()
