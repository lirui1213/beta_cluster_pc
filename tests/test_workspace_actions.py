"""File-backed numerical actions keep explicit review and accepted snapshots."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3-bands"))

import joblib
import mne
import numpy as np
import pandas as pd
import pytest

from fourband.browser_data import session_metadata
from fourband.trial_qc import detect_trial_quality
from test_fourband_stability import fixture as extraction_fixture


def actions():
    assert importlib.util.find_spec("fourband.workspace_actions"), "Numerical actions are missing"
    return importlib.import_module("fourband.workspace_actions")


@pytest.fixture
def run(tmp_path, monkeypatch):
    extraction, _ = extraction_fixture()
    names = list(extraction.features.channel)
    times = np.arange(3400) / 100.
    values = np.array([1e-5 * np.sin(2 * np.pi * (i + 2) * times) for i in range(len(names))])
    raw = mne.io.RawArray(values, mne.create_info(names, 100., "ecog"), verbose=False)
    raw.info["bads"] = [names[-1]]
    raw_path = tmp_path / "source-raw.fif"
    raw.save(raw_path, fmt="double", overwrite=True, verbose=False)
    np.save(tmp_path / "onset.npy", np.arange(8) * 4.)
    np.save(tmp_path / "offset.npy", np.arange(8) * 4. + 2.)
    config = dict(subject="synthetic", block="test", sources=dict(fif_path=str(raw_path),
        onset_path=str(tmp_path / "onset.npy"), offset_path=str(tmp_path / "offset.npy"),
        wav_path=str(tmp_path / "audio.wav"), baseline_fif=None, baseline_segments_csv=None),
        surface_options={}, baseline_mode="pre_onset", baseline_window=[-1.3, -.3],
        baseline_edge_seconds=.3, baseline_relative_floor=.001, feature_sigma_seconds=.06,
        first_k=2, second_k=2, band_weights=None, weight_preset="legacy", group_weights=None,
        correlation_threshold=.95, random_state=50, duration_mad_threshold=2.5,
        min_trials_after_filter=3, progress_points=41, pre_post_window=.5,
        slope_method="savgol", max_baseline_invalid_fraction=.25,
        display_sigma_seconds=.1, display_sfreq=25.)
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "extraction").mkdir()
    joblib.dump(extraction, tmp_path / "extraction/source.joblib")
    joblib.dump(detect_trial_quality(raw, extraction), tmp_path / "extraction/trial_report.joblib")
    task = tmp_path / "task"
    task.mkdir()
    monkeypatch.setenv("FOURBAND_WORKSPACE_TASK_DIR", str(task))
    return tmp_path, extraction, raw


def execute(run, action, payload=None):
    result = actions().execute_action(run[0], action, payload or {})
    json.dumps(result, allow_nan=False)
    assert all(Path(item if isinstance(item, str) else item["path"]).is_absolute()
               for item in result["artifacts"])
    assert (run[0] / "task/progress.json").is_file()
    return result


def test_explicit_empty_confirmation_preserves_features_and_provenance(run):
    with pytest.raises(ValueError, match="pairs|explicit"):
        execute(run, "confirm_trials")
    result = execute(run, "confirm_trials", {"pairs": []})
    assert result["stage"] == "trial_confirmed"
    reviewed = joblib.load(run[0] / "extraction/reviewed.joblib")
    assert reviewed.trial_exclusions == ()
    assert reviewed.trial_exclusions.provenance["raw_fingerprint"]
    assert reviewed.trial_qc_provenance["qc_settings"]["spike_z"] == 10.
    np.testing.assert_allclose(reviewed.features.filter(regex="__").values,
                               run[1].features.filter(regex="__").values, equal_nan=True)


def test_confirmation_replaces_masks_across_all_bands_and_rejects_stale_report(run):
    execute(run, "confirm_trials", {"pairs": [["E1", 17]]})
    for epoch in joblib.load(run[0] / "extraction/reviewed.joblib").epochs.values():
        assert np.isnan(epoch.speech[0, 1]).all()
        assert np.isfinite(epoch.speech[1:, 1]).all()
    execute(run, "confirm_trials", {"pairs": []})
    assert np.isfinite(joblib.load(run[0] / "extraction/reviewed.joblib").epochs["theta"].speech[:, 1]).all()
    before = (run[0] / "extraction/reviewed.joblib").read_bytes()
    report_path = run[0] / "extraction/trial_report.joblib"
    report = joblib.load(report_path)
    report.loc[0, "spike_z"] = 12345.
    joblib.dump(report, report_path)
    with pytest.raises(ValueError, match="provenance"):
        execute(run, "confirm_trials", {"pairs": []})
    assert (run[0] / "extraction/reviewed.joblib").read_bytes() == before


def test_detect_recomputes_candidates_and_requires_new_confirmation(run):
    execute(run, "confirm_trials", {"pairs": []})
    result = execute(run, "detect_qc", {"settings": {"spike_z": 25.}})
    assert result["stage"] == "trial_review"
    assert len(result["data"]["qc_rows"]) == 56
    assert result["data"]["qc_settings"]["spike_z"] == 25.
    with pytest.raises(ValueError, match="(?i)confirm|review"):
        execute(run, "fit_first")


@pytest.mark.parametrize("action,payload,excluded_count", [
    ("detect_qc", {"settings": {}}, 0),
    ("confirm_trials", {"pairs": [["E1", 17]]}, 1),
])
def test_qc_rows_are_compact_and_full_diagnostics_remain_saved(run, action, payload, excluded_count):
    result = execute(run, action, payload)
    data = result["data"]
    expected_keys = {"channel", "original_trial_id", "candidate", "reasons", "excluded", "raw_bad",
                     "flag_spike", "flag_jump", "flag_flat", "flag_clipping_candidate", "flag_nonfinite"}
    assert all(set(row) == expected_keys for row in data["qc_rows"])
    flags = {"flag_spike", "flag_jump", "flag_flat", "flag_clipping_candidate", "flag_nonfinite"}
    assert all(isinstance(row[name], bool) for row in data["qc_rows"] for name in flags)
    assert data["qc_summary"] == dict(total_pairs=56, candidate_count=7, excluded_count=excluded_count)
    report = joblib.load(run[0] / "extraction/trial_report.joblib")
    for index, row in enumerate(data["qc_rows"]):
        assert all(row[name] == bool(report.iloc[index][name]) for name in flags)
    assert {"spike_z", "peak_abs_v", "raw_mad_scale_v", "detection_start_seconds"} <= set(report)
    if action == "detect_qc":
        csv_path = next(Path(path) for path in result["artifacts"] if path.endswith("trial_quality.csv"))
        assert set(pd.read_csv(csv_path)) == set(report)
    else:
        selected = [row for row in data["qc_rows"] if row["excluded"]]
        assert [(row["channel"], row["original_trial_id"]) for row in selected] == [("E1", 17)]


def test_waveform_is_original_bounded_voltage_and_validates_pair(run):
    result = execute(run, "waveform", {"channel": "E1", "trial_id": 20})
    wave = result["data"]["waveform"]
    assert wave["channel"] == "E1" and wave["trial_id"] == 20
    assert wave["units"] == "volts"
    assert 0 < len(wave["time"]) == len(wave["voltage"]) <= 4096
    assert min(wave["time"]) >= 3.5 and max(wave["time"]) <= 6.5
    indices = np.rint(np.asarray(wave["time"]) * 100).astype(int)
    np.testing.assert_allclose(wave["voltage"], run[2].get_data(picks=["E1"])[0, indices])
    for payload in ({"channel": "unknown", "trial_id": 20},
                    {"channel": "E1", "trial_id": 18}, {"channel": "E1", "trial_id": True}):
        with pytest.raises(ValueError):
            execute(run, "waveform", payload)


def test_waveform_preserves_narrow_positive_and_negative_spikes_within_response_bound(run):
    from dataclasses import replace
    root, extraction, _ = run
    for band, epoch in extraction.epochs.items():
        extraction.epochs[band] = replace(epoch, sfreq=2000.)
    times = np.arange(68000) / 2000.
    names = list(extraction.features.channel)
    values = np.array([1e-5 * np.sin(2 * np.pi * (i + 2) * times) for i in range(len(names))])
    values[1, 7003], values[1, 7006] = .01, -.02
    raw = mne.io.RawArray(values, mne.create_info(names, 2000., "ecog"), verbose=False)
    raw_path = json.loads((root / "config.json").read_text(encoding="utf-8"))["sources"]["fif_path"]
    raw.save(raw_path, fmt="double", overwrite=True, verbose=False)
    joblib.dump(extraction, root / "extraction/source.joblib")
    report = detect_trial_quality(raw, extraction)
    joblib.dump(report, root / "extraction/trial_report.joblib")
    assert report.loc[(report.channel == "E1") & (report.original_trial_id == 20), "flag_spike"].item()
    wave = execute(run, "waveform", {"channel": "E1", "trial_id": 20})["data"]["waveform"]
    assert len(wave["time"]) <= 4096
    assert np.all(np.diff(wave["time"]) > 0)
    assert wave["time"][0] == 3.5 and wave["time"][-1] == 6.5
    assert max(wave["voltage"]) == .01
    assert min(wave["voltage"]) == -.02
    indices = np.rint(np.asarray(wave["time"]) * 2000).astype(int)
    np.testing.assert_array_equal(wave["voltage"], values[1, indices])


def test_waveform_rejects_changed_raw_recording_against_saved_qc_report(run):
    raw = run[2].copy()
    raw._data *= 2
    config = json.loads((run[0] / "config.json").read_text(encoding="utf-8"))
    raw.save(config["sources"]["fif_path"], fmt="double", overwrite=True, verbose=False)
    with pytest.raises(ValueError, match="provenance"):
        execute(run, "waveform", {"channel": "E1", "trial_id": 20})


@pytest.mark.parametrize("source_name", ["onset_path", "wav_path"])
@pytest.mark.parametrize("change", ["size", "mtime", "missing"])
def test_action_rejects_stale_event_or_audio_identity_after_confirmation(run, source_name, change):
    root = run[0]
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    Path(config["sources"]["wav_path"]).write_bytes(b"synthetic audio identity")
    identity = {}
    for name, value in config["sources"].items():
        if value is not None:
            stat = Path(value).stat()
            identity[name] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    config["input_identity"] = identity
    config_path.write_text(json.dumps(config), encoding="utf-8")
    execute(run, "confirm_trials", {"pairs": []})
    before = (root / "extraction/reviewed.joblib").read_bytes()
    path = Path(config["sources"][source_name])
    if change == "size":
        with path.open("ab") as stream:
            stream.write(b"changed")
    elif change == "mtime":
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    else:
        path.unlink()
    with pytest.raises(ValueError, match="(?i)source|changed|missing|identity"):
        execute(run, "waveform", {"channel": "E1", "trial_id": 20})
    assert (root / "extraction/reviewed.joblib").read_bytes() == before


def test_display_request_carries_recorded_source_identity(run):
    root, extraction, _ = run
    from fourband.analysis import fit_joint
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    stat = Path(config["sources"]["onset_path"]).stat()
    config["input_identity"] = {"onset_path": dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)}
    model = fit_joint(extraction.features, n_clusters=2)
    folder = root / "display_test"
    folder.mkdir()
    path, _ = actions()._display_request(root, folder, extraction, model, config, 1)
    assert joblib.load(path)["input_identity"] == config["input_identity"]


def test_first_requires_reviewed_snapshot_and_rejects_invalid_manual_k(run):
    with pytest.raises(ValueError, match="(?i)confirm|review"):
        execute(run, "fit_first")
    execute(run, "confirm_trials", {"pairs": []})
    for k in (True, 1, 2.5, 100):
        with pytest.raises(ValueError):
            execute(run, "fit_first", {"first_k": k})
    assert not (run[0] / "first.joblib").exists()


@pytest.mark.parametrize("settings", [
    {"feature_sigma_seconds": .2}, {"random_state": True},
    {"correlation_threshold": True}, None,
])
def test_first_rejects_invalid_fit_setting_overrides_without_persisting_them(run, settings):
    execute(run, "confirm_trials", {"pairs": []})
    config_path = run[0] / "config.json"
    before = config_path.read_bytes()
    with pytest.raises(ValueError):
        execute(run, "fit_first", {"fit_settings": settings})
    assert config_path.read_bytes() == before
    assert not (run[0] / "first.joblib").exists()


def test_weight_only_first_refit_persists_policy_and_second_reuses_accepted_budgets(run):
    execute(run, "confirm_trials", {"pairs": [["E1", 17]]})
    root = run[0]
    reviewed_before = (root / "extraction/reviewed.joblib").read_bytes()
    execute(run, "fit_first", {"first_k": 2})
    old_state = json.loads((root / "action_state.json").read_text(encoding="utf-8"))
    old_request = Path(old_state["first"]) / "request.joblib"
    overrides = dict(band_weights=dict(theta=2., beta1=3., beta2=4., high_gamma=5.),
                     weight_preset="balanced", correlation_threshold=1., random_state=17)
    first = execute(run, "fit_first", {"first_k": 3, "fit_settings": overrides})
    fitted = joblib.load(root / "first.joblib")
    assert fitted.kmeans.n_clusters == 3
    for band, budget in overrides["band_weights"].items():
        assert sum(w for c, w in zip(fitted.feature_columns, fitted.feature_weights)
                   if c.startswith(band + "__")) == pytest.approx(budget)
    assert first["config_update"]["first_k"] == 3
    assert all(first["config_update"][key] == value for key, value in overrides.items())
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    assert config["first_k"] == 3
    assert all(config[key] == value for key, value in overrides.items())
    assert (root / "extraction/reviewed.joblib").read_bytes() == reviewed_before
    assert old_request.is_file()
    # A later draft policy must not change the population's accepted first-fit policy.
    config.update(band_weights=dict(theta=7., beta1=7., beta2=7., high_gamma=7.),
                  weight_preset="legacy", correlation_threshold=.95, random_state=50)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    second = execute(run, "fit_second", {"second_k": 2})
    fitted = joblib.load(root / "second.joblib")
    for band, budget in overrides["band_weights"].items():
        assert sum(w for c, w in zip(fitted.feature_columns, fitted.feature_weights)
                   if c.startswith(band + "__")) == pytest.approx(budget)
    state = json.loads((root / "action_state.json").read_text(encoding="utf-8"))
    assert state["second_fit_settings"] == state["first_fit_settings"]
    round_config = json.loads((Path(second["display_request"]).parent / "round_config.json").read_text(encoding="utf-8"))
    assert all(round_config[key] == value for key, value in overrides.items())


def test_two_round_fit_preview_stability_and_immutable_history(run):
    execute(run, "confirm_trials", {"pairs": [["E1", 17]]})
    first_response = execute(run, "fit_first", {"first_k": 2})
    assert first_response["stage"] == "first_ready"
    first = joblib.load(run[0] / "first.joblib")
    assert first.kmeans.n_clusters == 2
    assert not any("audio" in c or "same_sign" in c or "slope_product" in c for c in first.feature_columns)
    request_path = Path(first_response["display_request"])
    request = joblib.load(request_path)
    assert request["extraction"].trial_exclusions == (("E1", 17),)
    metadata = session_metadata(request)
    assert metadata["round"] == 1
    assert not metadata["curves_enabled"]  # Fixture IDs are deliberately non-adjacent.
    assert first_response["data"]["warnings"]
    assert len(first_response["data"]["pca"]) == 7
    assert Path(first_response["data"]["plot_paths"]["pca_clusters"]).is_file()
    assert sum(first_response["data"]["cluster_sizes"].values()) == 6
    before = (run[0] / "first.joblib").read_bytes()
    controls = dict(bad_clusters=[], manual_bad_channels=["E2"],
                    pca_distance_threshold=None, distance_reference="cluster", second_k=2)
    preview = execute(run, "preview_review", controls)["data"]["preview"]
    assert preview["remaining"] == 5
    second_response = execute(run, "fit_second", controls)
    assert second_response["stage"] == "second_ready"
    second = joblib.load(run[0] / "second.joblib")
    assert second.kmeans.n_clusters == 2
    assert second.labels[2] == second.labels[-1] == -1
    assert second.imputer is not first.imputer
    audit = pd.read_csv(Path(second_response["display_request"]).parent / "review_audit.csv")
    assert audit.excluded_final.tolist() == [row["excluded_final"] for row in preview["rows"]]
    assert (run[0] / "first.joblib").read_bytes() == before
    stability = execute(run, "stability", {"round": 2, "n_resamples": 0})["data"]["stability"]
    assert stability["summary"]["requested_k"] == 2
    assert stability["summary"]["included_channels"] == ["E0", "E1", "E3", "E4", "E5"]
    assert (run[0] / "first.joblib").read_bytes() == before
    with pytest.raises(ValueError):
        execute(run, "fit_second", dict(controls, second_k=6))
    assert request_path.is_file()
    assert len(list(run[0].glob("round1_*"))) == 1
    assert len(list(run[0].glob("round2_*"))) == 1
    # A new first fit retires the active second fit while retaining old exports.
    old_request = request_path.read_bytes()
    new_first = execute(run, "fit_first", {"first_k": 3})
    assert new_first["display_request"] != first_response["display_request"]
    assert joblib.load(run[0] / "first.joblib").kmeans.n_clusters == 3
    assert request_path.read_bytes() == old_request
    assert len(list(run[0].glob("round1_*"))) == 2
    with pytest.raises(ValueError, match="accepted round 2"):
        execute(run, "stability", {"round": 2, "n_resamples": 0})


def test_stability_uses_accepted_fit_policy_after_draft_config_changes(run):
    from fourband.analysis import fit_joint
    execute(run, "confirm_trials", {"pairs": []})
    extraction = joblib.load(run[0] / "extraction/reviewed.joblib")
    first = fit_joint(extraction.features, n_clusters=2, weight_preset="balanced", random_state=17)
    joblib.dump(first, run[0] / "first.joblib")
    joblib.dump(extraction, run[0] / "extraction/first.joblib")
    state_path = run[0] / "action_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(first=str(run[0] / "round1_previous"),
                 first_fit_settings=dict(weight_preset="balanced", random_state=17))
    state_path.write_text(json.dumps(state), encoding="utf-8")
    result = execute(run, "stability", {"round": 1, "n_resamples": 0})
    summary = result["data"]["stability"]["summary"]
    assert summary["fit_settings"]["weight_preset"] == "balanced"
    assert summary["fit_settings"]["random_state"] == 17
    assert summary["requested_k"] == 2


def test_failed_export_never_overwrites_accepted_model(run, monkeypatch):
    execute(run, "confirm_trials", {"pairs": []})
    accepted = b"accepted model sentinel"
    (run[0] / "first.joblib").write_bytes(accepted)

    def fail(*args, **kwargs):
        raise OSError("export failed")

    monkeypatch.setattr(actions(), "save_round", fail)
    with pytest.raises(OSError, match="export failed"):
        execute(run, "fit_first")
    assert (run[0] / "first.joblib").read_bytes() == accepted


def test_cloned_task_state_can_preview_and_refit_without_copying_round_directories(run):
    import shutil
    execute(run, "confirm_trials", {"pairs": []})
    extraction = joblib.load(run[0] / "extraction/reviewed.joblib")
    from fourband.analysis import fit_joint
    first = fit_joint(extraction.features, n_clusters=2)
    joblib.dump(first, run[0] / "first.joblib")
    joblib.dump(extraction, run[0] / "extraction/first.joblib")
    state = json.loads((run[0] / "action_state.json").read_text(encoding="utf-8"))
    state["first"] = str(run[0] / "round1_previous")
    (run[0] / "action_state.json").write_text(json.dumps(state), encoding="utf-8")
    cloned = run[0] / "workspace_tasks/new/state"
    cloned.mkdir(parents=True)
    for path in run[0].iterdir():
        if path.is_file() and path.suffix in {".json", ".joblib"}:
            shutil.copy2(path, cloned / path.name)
    shutil.copytree(run[0] / "extraction", cloned / "extraction")
    response = actions().execute_action(cloned, "preview_review", {"second_k": 2})
    assert response["data"]["preview"]["remaining"] == 6
    response = actions().execute_action(cloned, "fit_second", {"second_k": 2})
    assert response["stage"] == "second_ready"
    assert (cloned / "second.joblib").is_file()
    assert session_metadata(joblib.load(response["display_request"]))["subject"] == "synthetic"


def test_snapshot_serialization_failure_keeps_previous_review(run, monkeypatch):
    execute(run, "confirm_trials", {"pairs": []})
    path = run[0] / "extraction/reviewed.joblib"
    before = path.read_bytes()
    state_before = (run[0] / "action_state.json").read_bytes()
    real_dump = joblib.dump

    def partial_dump(value, destination, *args, **kwargs):
        destination = Path(destination)
        if destination.parent == path.parent and destination.name.startswith(".reviewed.joblib"):
            destination.write_bytes(b"partial serialization")
            raise OSError("snapshot write failed")
        return real_dump(value, destination, *args, **kwargs)

    monkeypatch.setattr(joblib, "dump", partial_dump)
    with pytest.raises(OSError, match="snapshot write failed"):
        execute(run, "confirm_trials", {"pairs": [["E1", 17]]})
    assert path.read_bytes() == before
    assert (run[0] / "action_state.json").read_bytes() == state_before
    assert not list(path.parent.glob(".*.tmp"))


def test_fresh_worker_loads_review_without_gui_imports(run):
    import subprocess
    execute(run, "confirm_trials", {"pairs": []})
    source_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(source_root / "3-bands"), str(source_root)]))
    script = """import json, sys
from fourband.workspace_actions import execute_action
result = execute_action(sys.argv[1], 'fit_first', {'first_k': 2})
assert not any(name in sys.modules for name in ('tkinter', 'PyQt5', 'PyQt6', 'pyvista'))
print(json.dumps(result, allow_nan=False))
"""
    child = subprocess.run([sys.executable, "-c", script, str(run[0])], env=environment,
                           text=True, capture_output=True, timeout=90)
    assert child.returncode == 0, child.stderr
    response = json.loads(child.stdout.splitlines()[-1])
    assert response["stage"] == "first_ready"
    assert (run[0] / "first.joblib").is_file()
    assert Path(response["display_request"]).is_file()


def test_real_short_fourband_extraction_matches_direct_functions_and_links_adjacent_trials(run):
    from scipy.io import wavfile
    from fourband.analysis import load_fourband
    root = run[0]
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    times = np.arange(400 * 20) / 400.
    names = list(run[1].features.channel)
    signal = np.array([sum((1 + .25 * np.sin(times * (i + 1))) *
        np.sin(2 * np.pi * freq * times + i / 5) for freq in (6, 18, 30, 90)) * 1e-5
        for i in range(len(names))])
    raw = mne.io.RawArray(signal, mne.create_info(names, 400., "ecog"), verbose=False)
    raw.save(config["sources"]["fif_path"], fmt="double", overwrite=True, verbose=False)
    np.save(config["sources"]["onset_path"], np.array([2., 6., 10., 14.]))
    np.save(config["sources"]["offset_path"], np.array([3., 7., 11., 15.]))
    wavfile.write(config["sources"]["wav_path"], 4000,
                  np.sin(2 * np.pi * 110 * np.arange(80000) / 4000).astype(np.float32))
    result = execute(run, "extract")
    assert result["stage"] == "trial_review"
    source = joblib.load(root / "extraction/source.joblib")
    direct = load_fourband(**{key: config["sources"][key] for key in
        ("fif_path", "onset_path", "offset_path", "wav_path")}, picks="ecog",
        baseline_mode="pre_onset", baseline_window=[-1.3, -.3], baseline_edge_seconds=.3,
        baseline_relative_floor=.001, smoothing_sigma_seconds=.06, duration_mad_threshold=2.5,
        min_trials_after_filter=3, progress_points=41, pre_post_window=.5,
        slope_method="savgol", max_baseline_invalid_fraction=.25)
    pd.testing.assert_frame_equal(source.features, direct.features)
    assert result["data"]["events"][0]["original_trial_id"] == 0
    execute(run, "confirm_trials", {"pairs": []})
    first = execute(run, "fit_first", {"first_k": 2})
    metadata = session_metadata(joblib.load(first["display_request"]))
    assert metadata["curves_enabled"]
    assert metadata["settings"]["trial_ids"] == [0, 1]


def test_extract_uses_effective_config_and_saves_source_and_report(tmp_path, monkeypatch):
    module = actions()
    extraction, _ = extraction_fixture()
    times = np.arange(3400) / 100.
    raw = mne.io.RawArray(np.array([1e-5 * np.sin(times * (i + 1)) for i in range(7)]),
                          mne.create_info(list(extraction.features.channel), 100., "ecog"), verbose=False)
    config = dict(sources=dict(fif_path="fif", onset_path="onset", offset_path="offset", wav_path="wav",
                  baseline_fif="baseline", baseline_segments_csv="segments"), baseline_mode="file",
                  baseline_window=[-1.3, -.3], baseline_edge_seconds=.4, baseline_relative_floor=.002,
                  feature_sigma_seconds=.08, progress_points=51, duration_mad_threshold=3.,
                  min_trials_after_filter=4, pre_post_window=.8, slope_method="gradient",
                  max_baseline_invalid_fraction=.2)
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    def load(fif, onset, offset, wav, **kwargs):
        assert (fif, onset, offset, wav) == ("fif", "onset", "offset", "wav")
        assert kwargs == dict(picks="ecog", baseline_mode="file", baseline_window=[-1.3, -.3],
            baseline_auto_discover_segments=False,
            baseline_edge_seconds=.4, baseline_relative_floor=.002, baseline_fif="baseline",
            baseline_segments_csv="segments", smoothing_sigma_seconds=.08, progress_points=51,
            duration_mad_threshold=3., min_trials_after_filter=4, pre_post_window=.8,
            slope_method="gradient", max_baseline_invalid_fraction=.2)
        return extraction

    monkeypatch.setattr(module, "load_fourband", load)
    monkeypatch.setattr(module, "load_ecog_fif", lambda *a, **kw: raw)
    result = module.execute_action(tmp_path, "extract", {})
    assert result["stage"] == "trial_review"
    assert len(result["data"]["qc_rows"]) == 56
    assert result["data"]["channels"] == list(extraction.features.channel)
    assert len(result["data"]["events"]) == 8
    assert "baseline_summary" in result["data"]
    assert (tmp_path / "extraction/source.joblib").is_file()
    assert (tmp_path / "extraction/trial_report.joblib").is_file()
    assert (tmp_path / "progress.json").is_file()


def test_weight_only_refit_reuses_review_and_second_round_settings(run):
    execute(run, "confirm_trials", {"pairs": []})
    accepted = (run[0] / "extraction/reviewed.joblib").read_bytes()
    settings = dict(band_weights=dict(theta=2., beta1=1., beta2=1., high_gamma=1.),
                    weight_preset="balanced", group_weights=None)
    first = execute(run, "fit_first", dict(first_k=2, fit_settings=settings))
    assert first["config_update"]["band_weights"]["theta"] == 2.
    assert (run[0] / "extraction/reviewed.joblib").read_bytes() == accepted
    saved = json.loads((run[0] / "config.json").read_text(encoding="utf-8"))
    assert saved["weight_preset"] == "balanced"
    assert saved["band_weights"]["theta"] == 2.
    second = execute(run, "fit_second", dict(second_k=2))
    second_config = json.loads((Path(second["data"]["round_dir"]) / "round_config.json").read_text(encoding="utf-8"))
    assert second_config["band_weights"] == settings["band_weights"]
    assert second_config["weight_preset"] == "balanced"


@pytest.mark.parametrize("settings", [{"unknown": 1}, {"random_state": True},
    {"correlation_threshold": float("nan")}, {"group_weights": {"slope": -1}}])
def test_invalid_fit_setting_override_never_replaces_accepted_config(run, settings):
    execute(run, "confirm_trials", {"pairs": []})
    before = (run[0] / "config.json").read_bytes()
    with pytest.raises((ValueError, TypeError)):
        execute(run, "fit_first", dict(first_k=2, fit_settings=settings))
    assert (run[0] / "config.json").read_bytes() == before
