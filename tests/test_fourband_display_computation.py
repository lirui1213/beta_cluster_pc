"""Numerical display jobs retain a private cache outside the GUI interpreter."""
from pathlib import Path
import sys
import os
import subprocess

import joblib
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3-bands"))
from test_fourband_continuous import recording


def test_compute_snapshot_retains_cache_and_never_changes_extraction(recording, tmp_path, monkeypatch):
    from fourband import display_computation as module
    raw, events, audio, extraction = recording
    request_path = tmp_path / "request.joblib"
    joblib.dump(dict(extraction=extraction, sources=dict(fif_path="raw", onset_path="on", offset_path="off", wav_path="wav"),
                     cache_max_mb=10), request_path)
    loads = []
    def load(*args, **kwargs):
        loads.append(True)
        return raw
    monkeypatch.setattr(module, "load_ecog_fif", load)
    monkeypatch.setattr(module, "load_speech_events", lambda *a: events)
    monkeypatch.setattr(module, "extract_audio_envelope", lambda *a: audio)
    monkeypatch.setattr(module, "_SESSION", None)
    settings = dict(trial_ids=(10, 11), pre_seconds=1., post_seconds=1., sigma_seconds=.10, display_sfreq=100.)
    first = module.prepare_snapshot(request_path, settings)
    second = module.prepare_snapshot(request_path, dict(settings, sigma_seconds=.15))
    assert len(loads) == 1
    assert first.metadata["power_cache"]["misses"] == 1
    assert second.metadata["power_cache"]["hits"] == 1
    module.clear_snapshot_cache()
    third = module.prepare_snapshot(request_path, settings)
    assert third.metadata["power_cache"]["hits"] == 0
    assert third.metadata["power_cache"]["misses"] == 1
    assert next(iter(extraction.epochs.values())).channel_names == ("E1", "E2")


def test_same_display_settings_reuse_full_curve_data(recording, tmp_path, monkeypatch):
    from fourband import display_computation as module
    raw, events, audio, extraction = recording
    request_path = tmp_path / "request.joblib"
    joblib.dump(dict(extraction=extraction, sources=dict(fif_path="raw", onset_path="on", offset_path="off", wav_path="wav"),
                     cache_max_mb=10), request_path)
    monkeypatch.setattr(module, "load_ecog_fif", lambda *a, **k: raw)
    monkeypatch.setattr(module, "load_speech_events", lambda *a: events)
    monkeypatch.setattr(module, "extract_audio_envelope", lambda *a: audio)
    monkeypatch.setattr(module, "_SESSION", None)
    calls = []
    real_prepare = module.prepare_continuous_trials
    def prepare(*args, **kwargs):
        calls.append(True)
        return real_prepare(*args, **kwargs)
    monkeypatch.setattr(module, "prepare_continuous_trials", prepare)
    settings = dict(trial_ids=(10, 11), pre_seconds=1., post_seconds=1., sigma_seconds=.10, display_sfreq=100.)
    module.prepare_snapshot(request_path, settings)
    module.prepare_snapshot(request_path, settings)
    assert len(calls) == 1, "Picking another electrode must not smooth the full recording again."
    module.clear_snapshot_cache()
    module.prepare_snapshot(request_path, settings)
    assert len(calls) == 2


def test_numerical_crash_is_retained_for_worker_exit(tmp_path):
    from concurrent.futures import Future
    from concurrent.futures.process import BrokenProcessPool
    from types import SimpleNamespace
    from fourband.display_computation import DisplayComputation
    failed = Future()
    failed.set_exception(BrokenProcessPool("numerical child exited"))
    session = DisplayComputation(tmp_path / "request.joblib")
    session.pool = SimpleNamespace(submit=lambda *args: failed)
    session.submit({})
    with pytest.raises(RuntimeError, match="numerical child exited"):
        session.raise_if_failed()


@pytest.mark.skipif(os.name != "nt", reason="Windows consoleless interpreter")
def test_numerical_interpreter_uses_sibling_not_path(tmp_path, monkeypatch):
    from fourband import display_computation as module
    executable = tmp_path / "python.exe"
    quiet = tmp_path / "pythonw.exe"
    quiet.touch()
    monkeypatch.setattr(sys, "executable", str(executable))
    assert module.numerical_executable() == str(quiet)
    quiet.unlink()
    with pytest.raises(FileNotFoundError, match="pythonw.exe"):
        module.numerical_executable()


def test_consoleless_bootstrap_retains_stdout_and_stderr(tmp_path):
    from fourband.display_computation import numerical_executable
    project = Path(__file__).resolve().parents[1]
    source = ("import sys; sys.path[:0]=" + repr([str(project / "3-bands"), str(project)]) + "; "
        "from fourband.display_logging import configure_numerical_logging; configure_numerical_logging(); "
        "print('NUMERICAL_STDOUT', flush=True); sys.stderr.write('NUMERICAL_STDERR\\n'); sys.stderr.flush()")
    environment = dict(os.environ, FOURBAND_NUMERICAL_LOG_DIR=str(tmp_path))
    process = subprocess.run([numerical_executable(), "-c", source], env=environment, timeout=20,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert process.returncode == 0
    assert "NUMERICAL_STDOUT" in (tmp_path / "numerical_stdout.log").read_text()
    assert "NUMERICAL_STDERR" in (tmp_path / "numerical_stderr.log").read_text()


def test_idle_broken_pool_is_not_lost_during_shutdown(tmp_path):
    from types import SimpleNamespace
    from fourband.display_computation import DisplayComputation
    session = DisplayComputation(tmp_path / "request.joblib")
    session.pool = SimpleNamespace(_broken="Idle numerical child exited", shutdown=lambda **kwargs: None)
    session.close()
    with pytest.raises(RuntimeError, match="Idle numerical child exited"):
        session.raise_if_failed()


def test_numerical_startup_error_is_retained(tmp_path, monkeypatch):
    from fourband import display_computation as module
    session = module.DisplayComputation(tmp_path / "request.joblib")
    def missing():
        raise FileNotFoundError("missing pythonw.exe")
    monkeypatch.setattr(module, "numerical_executable", missing)
    with pytest.raises(FileNotFoundError):
        session.submit({})
    with pytest.raises(RuntimeError, match="missing pythonw.exe"):
        session.raise_if_failed()
