"""The classification process never owns native display windows."""
import os
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import joblib
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3-bands"))


def test_display_snapshot_is_local_and_uses_same_interpreter(tmp_path, monkeypatch):
    from fourband import display_process as module
    captured = {}
    model = SimpleNamespace(channels=("E1",), labels=[0])

    class Child:
        pid = 123
        def __init__(self, command, **kwargs):
            self.stdin = io.BytesIO()
            captured.update(command=command, kwargs=kwargs)
            request = joblib.load(command[-1])
            assert request["result"].labels == [0]
            assert request["surface_options"]["brain_opacity"] == .5
        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", Child)
    monkeypatch.setattr(module, "DisplayProcessTree", lambda child: SimpleNamespace(close=lambda: None), raising=False)
    run = module.run_display({"epochs": "snapshot"}, model, output_dir=tmp_path,
        show_surface=True, link_explorer=False, surface_options={"brain_opacity": .5})
    assert run.pid == 123 and run.returncode == 0
    assert captured["command"][0] == sys.executable
    assert captured["kwargs"].get("shell", False) is False
    assert run.directory.is_relative_to(tmp_path)
    assert (run.directory / "request.joblib").is_file()
    assert model.labels == [0]


def test_child_crash_is_an_exception_not_parent_exit(tmp_path, monkeypatch):
    from fourband import display_process as module
    parent_pid = os.getpid()

    class Child:
        pid = 124
        def __init__(self, command, **kwargs):
            self.stdin = io.BytesIO()
            kwargs["stderr"].write(b"Fatal Python error: PyEval_RestoreThread\n")
        def wait(self, timeout=None):
            return 3221226505

    monkeypatch.setattr(module.subprocess, "Popen", Child)
    monkeypatch.setattr(module, "DisplayProcessTree", lambda child: SimpleNamespace(close=lambda: None), raising=False)
    with pytest.raises(module.DisplayProcessError, match="3221226505") as error:
        module.run_display({}, {}, output_dir=tmp_path, show_surface=True, link_explorer=False)
    assert "stderr.log" in str(error.value)
    assert "PyEval_RestoreThread" in (error.value.directory / "stderr.log").read_text()
    assert os.getpid() == parent_pid


def test_interrupt_terminates_only_owned_child(tmp_path, monkeypatch):
    from fourband import display_process as module
    events = []

    class Child:
        pid = 125
        def __init__(self, *args, **kwargs):
            self.stdin = io.BytesIO()
        def wait(self, timeout=None):
            if timeout is None:
                raise KeyboardInterrupt
            events.append(("wait", timeout))
            return -1
        def terminate(self):
            events.append("terminate")

    monkeypatch.setattr(module.subprocess, "Popen", Child)
    monkeypatch.setattr(module, "DisplayProcessTree", lambda child: SimpleNamespace(close=lambda: None), raising=False)
    with pytest.raises(KeyboardInterrupt):
        module.run_display({}, {}, output_dir=tmp_path, show_surface=True, link_explorer=False)
    assert events[0] == "terminate"
    assert events[1][0] == "wait"


def test_startup_announcement_interrupt_also_cleans_child(tmp_path, monkeypatch):
    from fourband import display_process as module
    events = []
    children = []
    class Child:
        pid = 126
        def __init__(self, *args, **kwargs):
            self.stdin = io.BytesIO()
            children.append(self)
        def terminate(self):
            events.append("terminate")
        def wait(self, timeout=None):
            events.append("wait")
            return 0
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(module.subprocess, "Popen", Child)
    monkeypatch.setattr(module, "print", interrupted, raising=False)
    monkeypatch.setattr(module, "DisplayProcessTree", lambda child: SimpleNamespace(close=lambda: None))
    with pytest.raises(KeyboardInterrupt):
        module.run_display({}, {}, output_dir=tmp_path, show_surface=True, link_explorer=False)
    assert events == ["terminate", "wait"]
    assert children[0].stdin.closed


def test_linked_display_validates_recording_paths_before_start(tmp_path, monkeypatch):
    from fourband import display_process as module
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="fif_path"):
        module.run_display({}, {}, output_dir=tmp_path, show_surface=True)
    assert not (tmp_path / "display_sessions").exists()


def test_disabled_display_does_not_spawn_or_dump(tmp_path, monkeypatch):
    from fourband import display_process as module
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    assert module.run_display({}, {}, output_dir=tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_real_child_exit_keeps_parent_and_fitted_labels_alive(tmp_path, monkeypatch):
    from fourband import display_process as module
    monkeypatch.setattr(module, "_WORKER_SCRIPT", Path(__file__).with_name("display_exit_worker.py"))
    result = SimpleNamespace(labels=[0, 1])
    pid = os.getpid()
    with pytest.raises(module.DisplayProcessError, match="17") as error:
        module.run_display({}, result, output_dir=tmp_path, show_surface=True, link_explorer=False)
    assert "Deliberate display" in (error.value.directory / "stderr.log").read_text()
    assert result.labels == [0, 1] and os.getpid() == pid


def test_worker_owns_linked_callbacks_and_closes_explorer(tmp_path, monkeypatch):
    from fourband import display_worker as worker
    calls = []
    instances = []

    class Explorer:
        def __init__(self, *args, **kwargs):
            self.settings = kwargs
            instances.append(self)
        def show(self, block=True):
            calls.append(("show", block))
        def select_channel(self, channel):
            calls.append(("pick", channel))
        def pump(self):
            calls.append("pump")
        def close(self):
            calls.append("close")

    monkeypatch.setattr(worker, "FourBandExplorer", Explorer)
    monkeypatch.setattr(worker, "load_ecog_fif", lambda *a, **k: "child raw")
    monkeypatch.setattr(worker, "load_speech_events", lambda *a: "child events")
    monkeypatch.setattr(worker, "extract_audio_envelope", lambda *a: "child audio")
    def render(*args, **kwargs):
        assert kwargs["brain_opacity"] == .5 and kwargs["channel_label_toggle"] is True
        kwargs["on_channel_pick"]("E1")
        kwargs["on_window_tick"]()
        calls.append("render")
    monkeypatch.setattr(worker, "render_electrode_clusters", render)
    monkeypatch.setattr(worker, "render_selectable_electrode_class", render)
    request = dict(extraction={}, result=SimpleNamespace(channels=("E1",), labels=[0]),
        output_dir=str(tmp_path), sources=dict(fif_path="raw", onset_path="on", offset_path="off", wav_path="wav"),
        show_surface=True, show_selectable=True, link_explorer=True, explorer_only=False,
        cache_max_mb=1, explorer_settings={"sigma_seconds": .15},
        surface_options={"electrode_csv": "electrodes", "brain_opacity": .5})
    worker.run_request(request, tmp_path)
    assert calls == [("show", False), ("pick", "E1"), "pump", "render", ("pick", "E1"), "pump", "render", "close"]
    assert instances[0].settings["sigma_seconds"] == .15
    assert callable(instances[0].settings["continuous_submitter"])
    assert callable(instances[0].settings["cache_clearer"])


def test_worker_does_not_load_recording_for_surface_only(tmp_path, monkeypatch):
    from fourband import display_worker as worker
    monkeypatch.setattr(worker, "load_ecog_fif", lambda *a, **k: pytest.fail("no curves requested"))
    monkeypatch.setattr(worker, "render_electrode_clusters", lambda *a, **k: None)
    worker.run_request(dict(extraction={}, result=SimpleNamespace(channels=("E1",), labels=[0]),
        output_dir=str(tmp_path), sources={}, show_surface=True, show_selectable=False,
        link_explorer=False, explorer_only=False, cache_max_mb=1, explorer_settings={},
        surface_options={"electrode_csv": "electrodes"}), tmp_path)


def test_worker_never_marks_numerical_crash_complete(tmp_path, monkeypatch):
    from fourband import display_worker as worker
    class Computation:
        def __init__(self, *args):
            pass
        def submit(self, *args):
            pass
        def clear(self):
            pass
        def close(self):
            pass
        def raise_if_failed(self):
            raise RuntimeError("Numerical child died")
    class Explorer:
        def __init__(self, *args, **kwargs):
            pass
        def show(self, block=True):
            pass
        def close(self):
            pass
    monkeypatch.setattr(worker, "DisplayComputation", Computation)
    monkeypatch.setattr(worker, "FourBandExplorer", Explorer)
    monkeypatch.setattr(worker, "load_ecog_fif", lambda *a, **k: None)
    monkeypatch.setattr(worker, "load_speech_events", lambda *a: None)
    monkeypatch.setattr(worker, "extract_audio_envelope", lambda *a: None)
    request = dict(extraction={}, result={}, output_dir=str(tmp_path),
        sources=dict(fif_path="raw", onset_path="on", offset_path="off", wav_path="wav"),
        explorer_only=True, link_explorer=False, explorer_settings={})
    with pytest.raises(RuntimeError, match="Numerical child died"):
        worker.run_request(request, tmp_path)
    assert json.loads((tmp_path / "worker_status.json").read_text())["stage"] != "complete"


@pytest.mark.skipif(os.name != "nt", reason="Windows process-tree regression")
def test_interrupt_waits_for_owned_descendant_exit(tmp_path, monkeypatch):
    import psutil
    from fourband import display_process as module
    monkeypatch.setattr(module, "_WORKER_SCRIPT", Path(__file__).with_name("display_tree_worker.py"))
    real_popen = module.subprocess.Popen
    descendants = []
    def launch(command, **kwargs):
        child = real_popen(command, **kwargs)
        original_wait = child.wait
        def interrupt_after_spawn(timeout=None):
            if timeout is not None:
                return original_wait(timeout=timeout)
            marker = Path(command[-1]).with_name("descendant.json")
            deadline = time.monotonic() + 10
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(.02)
            assert marker.exists(), "Diagnostic child failed to create its descendant"
            pid = json.loads(marker.read_text(encoding="utf-8"))["pid"]
            descendants.append(psutil.Process(pid))
            raise KeyboardInterrupt
        child.wait = interrupt_after_spawn
        return child
    monkeypatch.setattr(module.subprocess, "Popen", launch)
    try:
        with pytest.raises(KeyboardInterrupt):
            module.run_display({}, {}, output_dir=tmp_path, show_surface=True, link_explorer=False)
        assert descendants and descendants[0].wait(timeout=1) is not None
    finally:
        for process in descendants:
            try:
                process.wait(timeout=.1)
            except psutil.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
