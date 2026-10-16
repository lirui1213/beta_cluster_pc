"""State and HTTP regression tests for the standalone analysis workspace."""
import json
from pathlib import Path
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3-bands"))

from fourband.workspace_session import WorkspaceSession
from fourband.workspace_server import make_workspace_server


class Catalog:
    def __init__(self, root):
        self.output_root = root

    def patients(self):
        return [{"id": "HS0015", "blocks": ["B03", "B05"]}]

    def resolve(self, subject, block, selections=None):
        if subject != "HS0015" or block not in {"B03", "B05"}:
            raise ValueError("Unknown subject or block")
        return dict(subject=subject, block=block, sources={}, surface_options={})

    def history(self, subject, block):
        return []


class Task:
    def __init__(self, directory, run_dir, action, payload):
        self.directory = directory
        self.action = action
        self.payload = payload
        self.done = False
        self.cancelled = False
        self.result = None

    def poll(self):
        return 0 if self.done else None

    def finish(self, stage, data=None):
        self.result = dict(stage=stage, data=data or {}, artifacts=[])
        self.done = True

    def read_result(self):
        return self.result

    def close(self):
        self.cancelled = True
        self.done = True


@pytest.fixture
def session(tmp_path):
    instance = WorkspaceSession(Catalog(tmp_path), task_factory=Task,
        config_validator=lambda config, resolved: dict(resolved, **config))
    yield instance
    instance.close()


def start(session):
    return session.start(dict(subject="HS0015", block="B03", config={}))


def finish(session, stage, data=None):
    session._task.finish(stage, data)
    session.refresh()
    return session.state()


def test_confirmation_and_stale_revision(session):
    first = start(session)
    with pytest.raises(ValueError, match="running|busy"):
        session.start(dict(subject="HS0015", block="B05", config={}))
    ready = finish(session, "trial_review", {"channels": ["ECoG_1"]})
    with pytest.raises(ValueError, match="revision|Stale"):
        session.action(dict(run_id=first["run_id"], revision=first["revision"],
                            action="confirm_trials", payload={"pairs": []}))
    with pytest.raises(ValueError, match="trial|confirm"):
        session.action(dict(run_id=ready["run_id"], revision=ready["revision"],
                            action="fit_first", payload={"first_k": 2}))
    session.action(dict(run_id=ready["run_id"], revision=ready["revision"],
                        action="confirm_trials", payload={"pairs": []}))
    assert session._task.payload == {"pairs": []}
    confirmed = finish(session, "trial_confirmed")
    session.action(dict(run_id=confirmed["run_id"], revision=confirmed["revision"],
                        action="fit_first", payload={"first_k": 2}))
    assert session._task.action == "fit_first"


def test_cancel_preserves_accepted_state_and_allows_new_run(session):
    running = start(session)
    task = session._task
    cancelled = session.cancel(dict(run_id=running["run_id"], revision=running["revision"]))
    assert task.cancelled
    assert cancelled["task"]["status"] == "cancelled"
    assert cancelled["stage"] == "configured"
    second = session.start(dict(subject="HS0015", block="B05", config={}))
    assert second["run_id"] != running["run_id"]


def test_failure_does_not_confirm_or_fit(session):
    start(session)
    session._task.done = True
    session._task.result = {"error": "broken input"}
    session.refresh()
    state = session.state()
    assert state["stage"] == "configured"
    assert state["task"]["status"] == "failed"
    assert "broken input" in state["task"]["error"]


def test_manifest_atomic_history_and_path_rejection(session):
    start(session)
    ready = finish(session, "trial_review", {"channels": ["ECoG_1"]})
    manifest = session.run_dir / "workspace_manifest.json"
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 1
    assert saved["stage"] == ready["stage"]
    with pytest.raises(ValueError):
        session.open(dict(subject="HS0015", block="B03", run_id="../../other"))


def test_http_token_origin_and_assets(session):
    server = make_workspace_server(session, token="test-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = server.session_url
    try:
        data = json.load(urlopen(url + "api/workspace/catalog"))
        assert data["patients"][0]["id"] == "HS0015"
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url + "api/workspace/start", data=b"{}",
                            headers={"Content-Type": "application/json"}))
        assert error.value.code == 403
        with pytest.raises(HTTPError):
            urlopen(url.replace("test-token", "wrong") + "api/workspace/state")
        with pytest.raises(HTTPError):
            urlopen(url + "../workspace_session.py")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_history_restore_rejects_foreign_identity(session):
    running = start(session)
    finish(session, "trial_review")
    path = session.run_dir / "workspace_manifest.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["block"] = "B05"
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatched"):
        session.open(dict(subject="HS0015", block="B03", run_id=running["run_id"]))


def test_history_restore_retains_explicit_review_gate(session):
    running = start(session)
    finish(session, "trial_review")
    opened = session.open(dict(subject="HS0015", block="B03", run_id=running["run_id"]))
    assert opened["stage"] == "trial_review"
    with pytest.raises(ValueError, match="confirm"):
        session.action(dict(run_id=opened["run_id"], revision=opened["revision"], action="fit_first", payload={}))


def test_artifact_cannot_serve_outside_file(session, tmp_path):
    start(session)
    outside = tmp_path / "private.json"
    outside.write_text("{}")
    session._artifacts["foreign"] = outside
    with pytest.raises(ValueError, match="Unknown artifact"):
        session.artifact("foreign")


def test_unknown_actions_and_implicit_confirmation_rejected(session):
    start(session)
    ready = finish(session, "trial_review")
    for action, payload in [("exec", {}), ("fit_second", {}), ("confirm_trials", {})]:
        with pytest.raises(ValueError):
            session.action(dict(run_id=ready["run_id"], revision=ready["revision"], action=action, payload=payload))


def test_surface_loading_never_blocks_state_polling(session, monkeypatch):
    start(session)
    finish(session, "trial_review")
    entered, release = threading.Event(), threading.Event()

    def slow_load(path):
        entered.set()
        release.wait(timeout=3)
        raise ValueError("test anatomy unavailable")

    monkeypatch.setattr(session, "_load_explorer", slow_load)
    session._queue_attach("request.joblib")
    assert entered.wait(timeout=1)
    # state() must stay responsive while the anatomy/snapshot is read.
    import time
    before = time.monotonic()
    state = session.state()
    assert time.monotonic() - before < .5
    assert state["task"]["status"] == "opening"
    release.set()


def test_historical_round_cannot_be_refitted_as_current_first(session):
    start(session)
    finish(session, "first_ready")
    session._state.update(first_round_id="new-first", second_round_id=None, selected_round="old-first")
    state = session.state()
    with pytest.raises(ValueError, match="historical|current"):
        session.action(dict(run_id=state["run_id"], revision=state["revision"], action="fit_second", payload={"second_k": 2}))


def test_legacy_history_is_read_only_and_not_overwritten(session):
    directory = session.catalog.output_root / "HS0015" / "B03" / "legacy-run"
    directory.mkdir(parents=True)
    config = directory / "run_config.json"
    config.write_text('{"subject":"HS0015","block":"B03"}')
    (directory / "round1_abc").mkdir()
    state = session.open(dict(subject="HS0015", block="B03", run_id="legacy-run"))
    assert state["read_only"] is True
    assert not (directory / "workspace_manifest.json").exists()
    with pytest.raises(ValueError, match="read-only|confirmed"):
        session.action(dict(run_id=state["run_id"], revision=state["revision"], action="fit_first", payload={}))


def test_task_state_is_private_until_success(tmp_path):
    from fourband.workspace_task import copy_action_state
    source, target = tmp_path / "accepted", tmp_path / "staging"
    (source / "extraction").mkdir(parents=True)
    (source / "config.json").write_text("{}")
    (source / "reviewed.joblib").write_bytes(b"accepted")
    (source / "extraction/source.joblib").write_bytes(b"source")
    (source / "workspace_tasks/previous").mkdir(parents=True)
    (source / "workspace_tasks/previous/result.json").write_text("{}")
    copy_action_state(source, target)
    (target / "reviewed.joblib").write_bytes(b"unconfirmed change")
    assert (source / "reviewed.joblib").read_bytes() == b"accepted"
    assert (target / "extraction/source.joblib").read_bytes() == b"source"
    assert not (target / "workspace_tasks").exists()


def test_failed_export_does_not_advance_accepted_stage(session, tmp_path):
    start(session)
    session._task.result = dict(stage="trial_review", data={"channels": ["not accepted"]},
                                artifacts=[str(tmp_path / "outside.csv")])
    session._task.done = True
    session.refresh()
    state = session.state()
    assert state["task"]["status"] == "failed"
    assert state["stage"] == "configured"
    assert "channels" not in state["data"]


def test_reconfirmation_clears_stale_fit_diagnostics(session):
    start(session)
    finish(session, "second_ready", dict(pca=["old"], preview={"old": True}, stability={"old": True}))
    state = session.state()
    session.action(dict(run_id=state["run_id"], revision=state["revision"], action="confirm_trials", payload={"pairs": []}))
    state = finish(session, "trial_confirmed", {"channels": ["ECoG_1"]})
    assert not {"pca", "preview", "stability"} & set(state["data"])


def test_rejected_round_open_preserves_active_context(session):
    start(session)
    active = finish(session, "trial_review")
    old = session.catalog.output_root / "HS0015/B05/saved"
    old.mkdir(parents=True)
    (old / "workspace_manifest.json").write_text(json.dumps(dict(schema_version=1,
        subject="HS0015", block="B05", run_id="saved", revision=1, rounds=[], stage="trial_review")))
    with pytest.raises(ValueError, match="Unknown saved round"):
        session.open(dict(subject="HS0015", block="B05", run_id="saved", round_id="missing"))
    assert session.state()["run_id"] == active["run_id"]
    assert session.state()["revision"] == active["revision"]


def test_cancel_teardown_keeps_state_polling_responsive(session):
    state = start(session)
    entered, release = threading.Event(), threading.Event()
    task = session._task

    def blocked_close():
        entered.set()
        release.wait(timeout=3)
        task.done = True

    task.close = blocked_close
    thread = threading.Thread(target=lambda: session.cancel(dict(run_id=state["run_id"], revision=state["revision"])))
    thread.start()
    assert entered.wait(timeout=1)
    assert session.state()["task"]["status"] == "cancelling"
    with pytest.raises(ValueError, match="running"):
        session.start(dict(subject="HS0015", block="B05", config={}))
    release.set()
    thread.join(timeout=3)
    assert not thread.is_alive()


def test_history_manifest_stays_small_with_large_trial_report(session):
    running = start(session)
    rows = [{"channel": "ECoG_1", "original_trial_id": i, "reasons": "x" * 600} for i in range(3000)]
    finish(session, "trial_review", {"qc_rows": rows})
    manifest = session.run_dir / "workspace_manifest.json"
    assert manifest.stat().st_size < 100_000
    opened = session.open(dict(subject="HS0015", block="B03", run_id=running["run_id"]))
    assert len(opened["data"]["qc_rows"]) == 3000


def test_display_source_identity_rejects_changed_audio(tmp_path):
    from fourband.workspace_display import validate_source_identity
    audio = tmp_path / "analog1.wav"
    audio.write_bytes(b"original")
    identity = {"wav_path": {"size": audio.stat().st_size, "mtime_ns": audio.stat().st_mtime_ns}}
    validate_source_identity({"wav_path": str(audio)}, identity)
    audio.write_bytes(b"changed audio content")
    with pytest.raises(ValueError, match="changed|Changed"):
        validate_source_identity({"wav_path": str(audio)}, identity)


def test_atomic_manifest_retries_windows_reader_contention(tmp_path, monkeypatch):
    from fourband.workspace_task import atomic_json
    real_replace = Path.replace
    attempts = []

    def temporarily_locked(path, target):
        attempts.append(path)
        if len(attempts) < 3:
            raise PermissionError("Windows reader has not closed the old manifest")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", temporarily_locked)
    target = tmp_path / "manifest.json"
    atomic_json(target, {"accepted": True})
    assert json.loads(target.read_text()) == {"accepted": True}
    assert len(attempts) == 3


def test_failed_manifest_commit_rolls_back_accepted_revision(session, monkeypatch):
    import fourband.workspace_session as module
    start(session)
    real_write = module.atomic_json

    def fail_new_manifest(path, value):
        if path.name == "workspace_manifest.json" and value.get("stage") == "trial_review":
            raise OSError("simulated disk failure")
        return real_write(path, value)

    monkeypatch.setattr(module, "atomic_json", fail_new_manifest)
    state = finish(session, "trial_review", {"channels": ["not accepted"]})
    assert state["stage"] == "configured"
    assert state["task"]["status"] == "failed"
    assert "channels" not in state["data"]


def test_new_analysis_can_retire_an_existing_explorer(session):
    from types import SimpleNamespace
    release = threading.Event()
    session._explorer = SimpleNamespace(closing=False, computation=SimpleNamespace(abort=lambda: release.wait(timeout=3)))
    session._explorer_id = "previous"
    try:
        state = start(session)
        assert state["task"]["status"] == "running"
        assert session._task.action == "extract"
    finally:
        release.set()


def test_snapshotless_round_clears_previous_explorer(session):
    from types import SimpleNamespace
    start(session)
    state = finish(session, "first_ready")
    session._explorer = SimpleNamespace(closing=False, computation=SimpleNamespace(abort=lambda: None))
    session._explorer_id = "old-brain"
    session._state["rounds"] = [dict(id="no-snapshot", round=1, data={})]
    selected = session.select_round(dict(run_id=state["run_id"], revision=state["revision"], round_id="no-snapshot"))
    assert "explorer_url" not in selected


def test_artifacts_are_bound_to_selected_round(session):
    start(session)
    finish(session, "second_ready")
    paths = []
    for number in (1, 2):
        path = session.run_dir / f"round{number}_saved" / "cluster_assignments.csv"
        path.parent.mkdir()
        path.write_text(f"round,{number}")
        paths.append(path)
    session._register_artifacts([str(path) for path in paths])
    ids = list(session._artifacts)
    session._state["rounds"] = [dict(id=f"r{n}", round=n, data={"round_dir": str(path.parent)}, artifacts=ids)
                                for n, path in enumerate(paths, 1)]
    session._state["selected_round"] = "r2"
    visible = session.state()["artifacts"]
    assert len(visible) == 1
    assert session.artifact(visible[0]["id"]) == paths[1]
    with pytest.raises(ValueError, match="Unknown artifact"):
        session.artifact(ids[0])


def test_stability_summary_survives_round_reselection(session):
    start(session)
    state = finish(session, "first_ready")
    session._state.update(first_round_id="r1", selected_round="r1")
    session._state["rounds"] = [dict(id="r1", round=1, data={})]
    session.action(dict(run_id=state["run_id"], revision=state["revision"], action="stability", payload={"round": 1, "n_resamples": 2}))
    summary = {"round": 1, "summary": {"supported": 8, "attempted": 8, "ari_full_mean": .9}}
    state = finish(session, "first_ready", {"stability": summary})
    selected = session.select_round(dict(run_id=state["run_id"], revision=state["revision"], round_id="r1"))
    assert selected["data"]["stability"] == summary
