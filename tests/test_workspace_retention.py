"""Single-record browser retention, never Notebook/raw-data deletion."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3-bands"))

from fourband.workspace_session import WorkspaceSession
from test_workspace_service import Catalog, Task


@pytest.fixture
def workspace(tmp_path):
    session = WorkspaceSession(Catalog(tmp_path), task_factory=Task,
        config_validator=lambda config, resolved: dict(resolved, **config))
    yield session
    session.close()


def start(session, **extra):
    return session.start(dict(subject="HS0015", block="B03", config={}, **extra))


def finish(session, stage="trial_review", data=None):
    session._task.finish(stage, data)
    session.refresh()
    return session.state()


def saved(root, identifier, *, subject="HS0015", block="B03", pending=False):
    run = root / subject / block / identifier
    run.mkdir(parents=True)
    data = dict(schema_version=1, run_id=identifier, subject=subject, block=block,
                revision=2, stage="trial_review", data={}, rounds=[],
                retention_pending=pending, config={})
    (run / "workspace_manifest.json").write_text(json.dumps(data), encoding="utf-8")
    (run / "config.json").write_text("{}")
    return run


def test_current_ignores_notebook_pending_and_foreign_manifests(tmp_path):
    from fourband.workspace_retention import WorkspaceRetention
    old = saved(tmp_path, "20261016_000000_000000_web_aaaaaa")
    latest = saved(tmp_path, "20261017_000000_000000_web_bbbbbb")
    saved(tmp_path, "20261019_000000_000000_web_cccccc", pending=True)
    notebook = tmp_path / "HS0015/B03/20261020_000000_000000"
    notebook.mkdir()
    (notebook / "run_config.json").write_text("{}")
    saved(tmp_path, "foreign")
    store = WorkspaceRetention(tmp_path)
    assert store.current("HS0015", "B03")["run_id"] == latest.name
    assert old.exists() and notebook.exists()


def test_consolidation_removes_only_owned_runs_in_selected_context(tmp_path):
    from fourband.workspace_retention import WorkspaceRetention
    old = saved(tmp_path, "20261016_000000_000000_web_aaaaaa")
    keep = saved(tmp_path, "20261017_000000_000000_web_bbbbbb")
    other = saved(tmp_path, "20261016_000000_000000_web_aaaaaa", block="B05")
    notebook = tmp_path / "HS0015/B03/20261018_notebook"
    notebook.mkdir()
    (notebook / "run_config.json").write_text("{}")
    store = WorkspaceRetention(tmp_path)
    store.consolidate(keep)
    assert not old.exists()
    assert keep.exists() and notebook.exists() and other.exists()


def test_destructive_operations_reject_foreign_or_linked_paths(tmp_path):
    from fourband.workspace_retention import WorkspaceRetention
    store = WorkspaceRetention(tmp_path / "outputs")
    foreign = saved(tmp_path, "20261016_000000_000000_web_aaaaaa")
    with pytest.raises(ValueError):
        store.remove(foreign)
    inside = saved(store.root, "20261017_000000_000000_web_bbbbbb")
    sentinel = tmp_path / "raw.fif"
    sentinel.write_bytes(b"raw")
    try:
        (inside / "linked").symlink_to(sentinel)
    except OSError:
        pytest.skip("Symlinks require Windows developer mode/privilege")
    with pytest.raises(ValueError, match="link|junction|reparse"):
        store.remove(inside)
    assert sentinel.read_bytes() == b"raw" and inside.exists()


def test_context_lease_is_exclusive_and_reusable(tmp_path):
    from fourband.workspace_retention import WorkspaceRetention
    store = WorkspaceRetention(tmp_path)
    first = store.lease("HS0015", "B03")
    try:
        with pytest.raises(ValueError, match="another|occupied|使用"):
            store.lease("HS0015", "B03")
        second = store.lease("HS0015", "B05")
        second.close()
    finally:
        first.close()
    store.lease("HS0015", "B03").close()


def test_collect_preserves_current_inputs_rounds_artifacts_and_only_latest_metadata(tmp_path):
    from fourband.workspace_retention import WorkspaceRetention
    run = saved(tmp_path, "20261019_000000_000000_web_aaaaaa")
    active = run / "workspace_tasks/new/state"
    old = run / "workspace_tasks/old/state"
    first = old / "round1_saved"
    for path in (active / "extraction/source.joblib", old / "extraction/source.joblib",
                 first / "request.joblib", first / "extraction.joblib", old / "qc.csv",
                 run / "workspace_tasks/abandoned/state/source.joblib"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")
    for i in (1, 2):
        (run / f"workspace_data_{i}.json").write_text("{}")
    manifest = json.loads((run / "workspace_manifest.json").read_text())
    manifest.update(numerical_state_dir=str(active), state_data_file="workspace_data_2.json",
                    rounds=[dict(id="first", round=1, artifact_root=str(first))],
                    artifact_paths={"qc": "workspace_tasks/old/state/qc.csv"})
    (run / "workspace_manifest.json").write_text(json.dumps(manifest))
    WorkspaceRetention(tmp_path).collect(run)
    assert (active / "extraction/source.joblib").exists()
    assert (first / "request.joblib").exists() and (first / "extraction.joblib").exists()
    assert (old / "qc.csv").exists()
    assert not (old / "extraction/source.joblib").exists()
    assert not (run / "workspace_tasks/abandoned").exists()
    assert not (run / "workspace_data_1.json").exists()
    assert (run / "workspace_data_2.json").exists()


def test_replacement_requires_acknowledgement_and_commits_only_on_success(workspace):
    from fourband.workspace_retention import WorkspaceRetention
    original = start(workspace)
    finish(workspace)
    old = workspace.run_dir
    notebook = old.parent / "20261018_notebook"
    notebook.mkdir()
    (notebook / "run_config.json").write_text("{}")
    for value in ({}, {"replace_run_id": "stale"}):
        with pytest.raises(ValueError, match="replace|current|refresh"):
            start(workspace, **value)
    start(workspace, replace_run_id=original["run_id"])
    candidate = workspace.run_dir
    store = WorkspaceRetention(workspace.catalog.output_root)
    assert old.exists() and candidate != old
    assert store.current("HS0015", "B03")["run_id"] == original["run_id"]
    finish(workspace)
    assert not old.exists() and candidate.exists() and notebook.exists()
    assert len(store.runs("HS0015", "B03")) == 1


@pytest.mark.parametrize("cancel", [False, True])
def test_failed_or_cancelled_replacement_keeps_previous_and_removes_staging(workspace, cancel):
    from fourband.workspace_retention import WorkspaceRetention
    original = start(workspace)
    finish(workspace)
    old = workspace.run_dir
    pending = start(workspace, replace_run_id=original["run_id"])
    staging = workspace.run_dir
    if cancel:
        workspace.cancel(dict(run_id=pending["run_id"], revision=pending["revision"]))
    else:
        workspace._task.directory.mkdir(parents=True, exist_ok=True)
        (workspace._task.directory / "stderr.log").write_text("Traceback: extraction failed", encoding="utf-8")
        workspace._task.done = True
        workspace._task.result = {"error": "extraction failed"}
        workspace.refresh()
    assert old.exists() and not staging.exists()
    if not cancel:
        assert "Traceback: extraction failed" in workspace.state()["task"]["log_tail"]
    assert WorkspaceRetention(workspace.catalog.output_root).current("HS0015", "B03")["run_id"] == old.name


def test_continue_consolidates_browser_history_not_notebook(workspace):
    old = saved(workspace.catalog.output_root, "20261016_000000_000000_web_aaaaaa")
    latest = saved(workspace.catalog.output_root, "20261017_000000_000000_web_bbbbbb")
    state = workspace.open(dict(subject="HS0015", block="B03", run_id=latest.name))
    assert state["run_id"] == latest.name
    assert latest.exists() and not old.exists()


def test_services_cannot_share_one_mutable_context(workspace):
    start(workspace)
    other = WorkspaceSession(workspace.catalog, task_factory=Task,
        config_validator=lambda config, resolved: dict(resolved, **config))
    try:
        with pytest.raises(ValueError, match="another|use|使用"):
            start(other)
    finally:
        other.close()


def test_refitting_keeps_only_current_rounds_and_referenced_artifacts(workspace):
    start(workspace)
    ready = finish(workspace)
    workspace.action(dict(run_id=ready["run_id"], revision=ready["revision"],
        action="confirm_trials", payload={"pairs": []}))
    finish(workspace, "trial_confirmed")
    prior = []
    for number in (1, 2, 1):
        state = workspace.state()
        workspace.action(dict(run_id=state["run_id"], revision=state["revision"],
            action="fit_first" if number == 1 else "fit_second", payload={}))
        folder = workspace._task.directory / "state" / f"round{number}_saved"
        folder.mkdir(parents=True)
        artifact = folder / "curve.png"
        artifact.write_bytes(b"png")
        workspace._task.finish("first_ready" if number == 1 else "second_ready",
                               dict(round=number, round_dir=str(folder)))
        workspace._task.result["artifacts"] = [str(artifact)]
        workspace.refresh()
        prior.append(artifact)
    state = workspace.state()
    assert len(state["rounds"]) == 1
    assert state["second_round_id"] is None
    assert prior[-1].exists() and not any(p.exists() for p in prior[:-1])
    assert len(workspace._artifacts) == 1
    assert len(list(workspace.run_dir.glob("workspace_data_*.json"))) == 1


def test_failed_worker_launch_does_not_leave_unresumable_cache(workspace):
    def broken(*args):
        raise OSError("could not launch")
    workspace.task_factory = broken
    with pytest.raises(OSError, match="launch"):
        start(workspace)
    assert not list(workspace.catalog.output_root.glob("HS0015/B03/*_web_*"))


def test_current_http_endpoint_is_read_only_and_notebook_is_not_a_current_record(workspace):
    from urllib.request import urlopen
    from fourband.workspace_server import make_workspace_server
    import threading
    old = saved(workspace.catalog.output_root, "20261016_000000_000000_web_aaaaaa")
    latest = saved(workspace.catalog.output_root, "20261017_000000_000000_web_bbbbbb")
    server = make_workspace_server(workspace)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        response = json.load(urlopen(server.session_url + "api/workspace/current?subject=HS0015&block=B03"))
        assert response["analysis"]["id"] == latest.name
        assert old.exists() and latest.exists(), "Discovery must not delete before the selection"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def test_collect_failure_keeps_committed_record_and_exposes_retry_warning(workspace, monkeypatch):
    start(workspace)
    def blocked(*args):
        raise PermissionError("file is in use")
    monkeypatch.setattr(workspace.retention, "collect", blocked)
    state = finish(workspace)
    assert state["stage"] == "trial_review" and state["task"]["status"] == "completed"
    assert "file is in use" in state["retention_warning"]
    assert (workspace.run_dir / "workspace_manifest.json").exists()


def test_new_analysis_removes_abandoned_staging_after_crash(workspace):
    stale = saved(workspace.catalog.output_root, "20261016_000000_000000_web_aaaaaa", pending=True)
    start(workspace)
    assert not stale.exists()


def test_state_clone_does_not_propagate_workspace_ui_metadata(tmp_path):
    from fourband.workspace_task import copy_action_state
    source = tmp_path / "accepted"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "workspace_data_999.json").write_text('{"large_qc_rows":[]}')
    copy_action_state(source, tmp_path / "clone")
    assert not list((tmp_path / "clone").glob("workspace_data_*.json"))


def test_continue_reads_mutable_state_only_after_acquiring_lease(workspace, monkeypatch):
    run = saved(workspace.catalog.output_root, "20261016_000000_000000_web_aaaaaa")
    newer = run / "workspace_tasks/newer/state"
    newer.mkdir(parents=True)
    actual = workspace.retention.lease
    def previous_service_commits_then_releases(*args):
        lease = actual(*args)
        path = run / "workspace_manifest.json"
        data = json.loads(path.read_text())
        data.update(stage="trial_confirmed", numerical_state_dir=str(newer))
        path.write_text(json.dumps(data))
        return lease
    monkeypatch.setattr(workspace.retention, "lease", previous_service_commits_then_releases)
    opened = workspace.open(dict(subject="HS0015", block="B03", run_id=run.name))
    assert opened["stage"] == "trial_confirmed"
    assert opened["numerical_state_dir"] == str(newer) and newer.exists()


def test_close_drains_loader_startup_before_releasing_context(workspace, monkeypatch):
    import threading
    import time
    import fourband.workspace_session as module
    start(workspace)
    finish(workspace)
    request = workspace.run_dir / "round1/request.joblib"
    request.parent.mkdir()
    request.write_bytes(b"test")
    entered, release = threading.Event(), threading.Event()
    class SlowTask(Task):
        def __init__(self, *args):
            entered.set()
            release.wait(timeout=5)
            super().__init__(*args)
    monkeypatch.setattr(module, "WorkspaceTask", SlowTask)
    workspace._queue_attach(str(request))
    assert entered.wait(timeout=1)
    closer = threading.Thread(target=workspace.close)
    closer.start()
    try:
        deadline = time.monotonic() + 1
        while not workspace.closing and time.monotonic() < deadline:
            time.sleep(.01)
        time.sleep(.03)
        with pytest.raises(ValueError, match="another|use|使用"):
            workspace.retention.lease("HS0015", "B03")
    finally:
        release.set()
        closer.join(timeout=5)
    assert not closer.is_alive() and not workspace._loader_thread.is_alive()


def test_manifest_probe_error_cannot_bypass_another_services_lease(workspace, monkeypatch):
    state = start(workspace)
    finish(workspace)
    state = workspace.state()
    workspace.action(dict(run_id=state["run_id"], revision=state["revision"],
        action="confirm_trials", payload={"pairs": []}))
    live = workspace._task.directory / "state/live.joblib"
    live.parent.mkdir(parents=True)
    live.write_bytes(b"running")
    other = WorkspaceSession(workspace.catalog, task_factory=Task,
        config_validator=lambda config, resolved: dict(resolved, **config))
    actual, attempts = other.retention._manifest, []
    def probe_fails_once(*args):
        attempts.append(True)
        if len(attempts) == 1:
            raise PermissionError("transient reader contention")
        return actual(*args)
    monkeypatch.setattr(other.retention, "_manifest", probe_fails_once)
    try:
        with pytest.raises((ValueError, PermissionError)):
            other.open(dict(subject="HS0015", block="B03", run_id=state["run_id"]))
        assert live.read_bytes() == b"running"
    finally:
        other.close()
