"""Browser adapters preserve fitted identity, coordinates and display semantics."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import joblib
import matplotlib
matplotlib.use("Agg")
import nibabel as nib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "3-bands"))


def adapter():
    return importlib.import_module("fourband.browser_data")


@pytest.fixture
def snapshot(tmp_path):
    # Retained row positions must not replace the source recording's trial IDs.
    original = pd.DataFrame(dict(onset=10. + 3 * np.arange(13),
                                 offset=11. + 3 * np.arange(13)))
    event_path = tmp_path / "events.csv"
    original.to_csv(event_path, index=False)
    retained = original.iloc[[7, 8, 9]].copy()
    retained["original_trial_id"] = [7, 8, 9]
    retained["duration"] = 1.
    bands = {"theta": (4., 8.), "beta1": (13., 24.),
             "beta2": (24., 40.), "high_gamma": (70., 150.)}
    channels = ("E9", "E2", "E7")
    epochs = {name: SimpleNamespace(channel_names=channels, events=retained.copy(),
              progress_percent=np.linspace(0, 100, 5),
              speech=np.arange(45, dtype=float).reshape(3, 3, 5),
              baseline_mode="pre_onset") for name in bands}
    extraction = SimpleNamespace(epochs=epochs, bands=bands, trial_exclusions=(("E9", 8),))
    # Deliberately inconsistent clean_mask proves continuous membership follows explorer labels.
    result = SimpleNamespace(channels=channels, labels=np.array([2, 2, -1]),
                             clean_mask=np.array([True, False, False]))
    run_dir = tmp_path / "HS0015" / "B03" / "run"
    output = run_dir / "round2_test"
    output.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(json.dumps(dict(subject="HS0015", block="B03")))
    (output / "round_config.json").write_text(json.dumps(dict(round=2)))
    return dict(extraction=extraction, result=result, output_dir=str(output),
        sources=dict(onset_path=str(event_path), offset_path=None),
        surface_options=dict(subject="recon", brain_opacity=.25),
        show_surface=True, show_selectable=False, explorer_only=False, link_explorer=True,
        explorer_settings=dict(trial_ids=(8, 9), sigma_seconds=.15))


def test_json_safe_recursively_converts_missing_numpy_and_pandas_values():
    values = dict(array=np.array([1., np.nan, np.inf, -np.inf]),
                  nested=(np.int64(9), np.bool_(True), pd.NA, pd.NaT),
                  scalar=np.array(np.nan), label=np.float32(2.5))
    converted = adapter().json_safe(values)
    assert converted == dict(array=[1., None, None, None], nested=[9, True, None, None],
                             scalar=None, label=2.5)
    json.dumps(converted, allow_nan=False)
    assert np.isnan(values["array"][1])


def test_metadata_contract_loads_original_events_and_saved_run_identity(snapshot):
    before = joblib.hash(snapshot)
    meta = adapter().session_metadata(snapshot)
    assert set(meta) == {"subject", "block", "round", "channels", "labels", "bands", "pairs",
                         "settings", "surface_options", "surface_enabled", "curves_enabled", "warnings",
                         "summary_band_limits", "summary_speech_duration_seconds"}
    assert (meta["subject"], meta["block"], meta["round"]) == ("HS0015", "B03", 2)
    assert meta["channels"] == ["E9", "E2", "E7"]
    assert meta["labels"] == [2, 2, -1]
    assert meta["bands"] == ["theta", "beta1", "beta2", "high_gamma"]
    assert meta["summary_band_limits"] == {name:list(limits) for name,limits in snapshot["extraction"].bands.items()}
    assert meta["summary_speech_duration_seconds"] == 1.
    assert [(p["first_trial_id"], p["second_trial_id"]) for p in meta["pairs"]] == [(7, 8), (8, 9)]
    assert meta["pairs"][1] == dict(first_trial_id=8, second_trial_id=9, first_onset=34.,
                                    first_offset=35., second_onset=37., second_offset=38., gap_seconds=2.)
    assert meta["settings"] == dict(trial_ids=[8, 9], sigma_seconds=.15,
                                     pre_seconds=1., post_seconds=1., display_sfreq=100.)
    assert meta["surface_options"] == dict(brain_color="lightgray", brain_opacity=.25, point_size=14.)
    assert meta["surface_enabled"] is True and meta["curves_enabled"] is True
    assert joblib.hash(snapshot) == before
    json.dumps(meta, allow_nan=False)


def test_metadata_original_adjacency_does_not_bridge_a_dropped_trial(snapshot):
    for epoch in snapshot["extraction"].epochs.values():
        epoch.events = epoch.events.iloc[[0, 2]].copy()
    with pytest.raises(ValueError, match="adjacent"):
        adapter().session_metadata(snapshot)


def test_metadata_defaults_to_first_pair_and_preserves_explicit_identity(snapshot):
    snapshot["explorer_settings"] = {}
    snapshot.update(subject="HS9999", block="B99", round=1)
    meta = adapter().session_metadata(snapshot)
    assert meta["settings"]["trial_ids"] == [7, 8]
    assert (meta["subject"], meta["block"], meta["round"]) == ("HS9999", "B99", 1)


def test_metadata_emits_integer_labels_without_mutating_float_storage(snapshot):
    snapshot["result"].labels = np.array([2., 2., -1.])
    meta = adapter().session_metadata(snapshot)
    assert all(type(label) is int for label in meta["labels"])
    assert snapshot["result"].labels.dtype == np.dtype(float)


def test_brain_only_metadata_never_loads_signal_or_event_sources(snapshot):
    snapshot.update(link_explorer=False, sources={})
    meta = adapter().session_metadata(snapshot)
    assert meta["curves_enabled"] is False
    assert meta["pairs"] == []
    assert meta["settings"]["trial_ids"] == [8, 9]


def test_brain_only_metadata_has_json_list_when_no_pair_was_requested(snapshot):
    snapshot.update(link_explorer=False, sources={}, explorer_settings={})
    assert adapter().session_metadata(snapshot)["settings"]["trial_ids"] == []


def test_metadata_infers_patient_block_from_recording_path_without_saved_config(snapshot, tmp_path):
    snapshot["output_dir"] = str(tmp_path / "exports" / "round1_custom")
    snapshot["sources"]["fif_path"] = str(tmp_path / "HS0023" / "blocks" / "B07" /
                                            "standardized" / "ECoG" / "ecog_clean-raw.fif")
    meta = adapter().session_metadata(snapshot)
    assert (meta["subject"], meta["block"], meta["round"]) == ("HS0023", "B07", 1)
    assert meta["warnings"] == []


def test_explorer_only_disables_surface(snapshot):
    snapshot["explorer_only"] = True
    meta = adapter().session_metadata(snapshot)
    assert meta["surface_enabled"] is False and meta["curves_enabled"] is True
    assert adapter().surface_payload(snapshot) == dict(meshes=[], electrodes=[], warnings=[])


@pytest.mark.parametrize("change", ["order", "labels", "bands"])
def test_metadata_rejects_misaligned_snapshot(snapshot, change):
    if change == "order":
        snapshot["result"].channels = ("E2", "E9", "E7")
    elif change == "labels":
        snapshot["result"].labels = np.array([2., np.nan, -1.])
    else:
        snapshot["extraction"].epochs.pop("beta1")
    with pytest.raises(ValueError, match="align|label"):
        adapter().session_metadata(snapshot)


@pytest.mark.parametrize("settings", [dict(trial_ids=[7, 19]), dict(sigma_seconds=np.nan),
                                      dict(display_sfreq=0), dict(pre_seconds=-1)])
def test_metadata_rejects_invalid_initial_display_settings(snapshot, settings):
    snapshot["explorer_settings"].update(settings)
    with pytest.raises(ValueError):
        adapter().session_metadata(snapshot)


@pytest.fixture
def anatomy(snapshot, tmp_path):
    root = tmp_path / "anatomy" / "recon"
    (root / "surf").mkdir(parents=True)
    (root / "mri").mkdir()
    vertices = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    nib.freesurfer.write_geometry(str(root / "surf" / "lh.pial"), vertices, np.array([[0, 1, 2]]))
    affine = np.array([[0., -2., 0., 10.], [2., 0., 0., 20.], [0., 0., 3., 30.], [0., 0., 0., 1.]])
    image = nib.MGHImage(np.zeros((4, 6, 8), dtype=np.float32), affine)
    mri = root / "mri" / "orig.mgz"
    nib.save(image, str(mri))
    csv = tmp_path / "electrodes.csv"
    pd.DataFrame(dict(channel=["E2", "extra", "E9"], x=[11., 0., 12.],
                      y=[22., 0., 24.], z=[33., 0., 36.])).to_csv(csv, index=False)
    snapshot["surface_options"].update(electrode_csv=str(csv), subjects_dir=str(root.parent),
                                        hemispheres=("lh",))
    return snapshot, root, csv


def test_surface_uses_real_geometry_and_scanner_to_tkras_without_projection(anatomy):
    request, root, csv = anatomy
    before = (csv.read_bytes(), (root / "surf" / "lh.pial").read_bytes(), joblib.hash(request))
    payload = adapter().surface_payload(request)
    assert payload["meshes"] == [dict(name="lh", positions=[1., 2., 3., 4., 5., 6., 7., 8., 9.],
                                      indices=[0, 1, 2])]
    assert [e["channel"] for e in payload["electrodes"]] == ["E9", "E2"]
    assert [e["label"] for e in payload["electrodes"]] == [2, 2]
    # Independent physical transform: scanner (12,24,36) -> voxel (2,-1,2),
    # FreeSurfer tkRAS (-2*x+4, 3*z-12, -2*y+6) -> (0,-6,8).
    np.testing.assert_allclose(payload["electrodes"][0]["position"], [0., -6., 8.])
    np.testing.assert_allclose(payload["electrodes"][1]["position"], [2., -9., 7.])
    assert any("E7" in w for w in payload["warnings"])
    assert any("extra" in w for w in payload["warnings"])
    assert before == (csv.read_bytes(), (root / "surf" / "lh.pial").read_bytes(), joblib.hash(request))
    json.dumps(payload, allow_nan=False)


def test_surface_missing_mri_reports_failure_without_mixing_coordinate_frames(anatomy):
    request, root, _ = anatomy
    request["surface_options"]["reference_mri"] = str(root / "missing.mgz")
    payload = adapter().surface_payload(request)
    assert payload["meshes"] and payload["electrodes"] == []
    assert any("MRI" in w or "transform" in w for w in payload["warnings"])


def test_surface_unreadable_mri_is_reported_without_scanner_points_on_tkras_mesh(anatomy, tmp_path):
    request, _, _ = anatomy
    corrupt = tmp_path / "corrupt.mgz"
    corrupt.write_bytes(b"not an MRI")
    request["surface_options"]["reference_mri"] = str(corrupt)
    payload = adapter().surface_payload(request)
    assert payload["meshes"] and payload["electrodes"] == []
    assert any("transform" in w for w in payload["warnings"])


def test_surface_missing_anatomy_and_nonfinite_coordinates_are_reported(snapshot, tmp_path):
    csv = tmp_path / "coords.csv"
    pd.DataFrame(dict(channel=["E9", "E2"], x=[np.nan, 2.], y=[1., 3.], z=[1., 4.])).to_csv(csv, index=False)
    snapshot["surface_options"] = dict(electrode_csv=str(csv))
    payload = adapter().surface_payload(snapshot)
    assert payload["meshes"] == []
    assert payload["electrodes"] == [dict(channel="E2", label=2, position=[2., 3., 4.])]
    assert any("anatom" in w.lower() for w in payload["warnings"])
    assert any("E9" in w for w in payload["warnings"])
    assert any("E7" in w for w in payload["warnings"])
    json.dumps(payload, allow_nan=False)


@pytest.fixture
def continuous(snapshot):
    return SimpleNamespace(channels=snapshot["result"].channels,
        times=np.array([-1., 0., 1., 2.]), audio=np.array([np.nan, .1, .4, .2]),
        bands=snapshot["extraction"].bands,
        power={name: np.array([[1., 2., np.nan, np.nan], [3., 4., 8., np.nan],
                               [100., 200., 300., np.nan]]) for name in snapshot["extraction"].bands},
        trial_ids=(7, 11), events=pd.DataFrame(dict(original_trial_id=[7, 11, 19],
            onset=[10., 13., 16.], offset=[11., 14., 17.])),
        metadata=dict(first_onset_seconds=10., baseline_mode="pre_onset",
            baseline_description="First selected trial baseline for entire display",
            time_axis="Real seconds from first onset; no duration normalization",
            trial_exclusions=[("E9", 11), ("E2", 19)]))


def test_continuous_preserves_fixed_membership_qc_relative_events_and_missing_values(snapshot, continuous):
    before = joblib.hash((continuous, snapshot["result"]))
    payload = adapter().continuous_payload(continuous, snapshot["result"], "E9")
    assert payload["times"] == [-1., 0., 1., 2.]
    assert payload["audio"] == [None, .1, .4, .2]
    assert payload["channel"] == "E9" and payload["label"] == 2
    assert payload["trial_ids"] == [7, 11]
    assert payload["bands"][0] == dict(name="theta", selected=[1., 2., None, None], mean=[2., 3., 8., None])
    assert [b["name"] for b in payload["bands"]] == ["theta", "beta1", "beta2", "high_gamma"]
    assert payload["events"] == [dict(original_trial_id=7, onset=0., offset=1., excluded=False),
                                 dict(original_trial_id=11, onset=3., offset=4., excluded=True),
                                 dict(original_trial_id=19, onset=6., offset=7., excluded=False)]
    assert payload["baseline_description"] == continuous.metadata["baseline_description"]
    assert payload["metadata"]["time_axis"] == continuous.metadata["time_axis"]
    assert joblib.hash((continuous, snapshot["result"])) == before
    json.dumps(payload, allow_nan=False)
    other = adapter().continuous_payload(continuous, snapshot["result"], "E2")
    assert [e["excluded"] for e in other["events"]] == [False, False, True]


def test_continuous_excluded_channel_keeps_original_negative_label(snapshot, continuous):
    payload = adapter().continuous_payload(continuous, snapshot["result"], "E7")
    assert payload["label"] == -1
    assert payload["bands"][0]["mean"] == [100., 200., 300., None]


@pytest.mark.parametrize("change", ["channel", "order", "shape", "bands"])
def test_continuous_rejects_misalignment(snapshot, continuous, change):
    channel = "E9"
    if change == "channel":
        channel = "missing"
    elif change == "order":
        continuous.channels = tuple(reversed(continuous.channels))
    elif change == "shape":
        continuous.power["theta"] = np.zeros((2, 4))
    else:
        continuous.power.pop("theta")
    with pytest.raises(ValueError, match="channel|align|band"):
        adapter().continuous_payload(continuous, snapshot["result"], channel)


def test_detail_returns_real_agg_png_and_preserves_snapshot(snapshot, tmp_path):
    from matplotlib import pyplot as plt

    request_path = tmp_path / "request.joblib"
    joblib.dump(snapshot, request_path)
    before = request_path.read_bytes()
    figures_before = set(plt.get_fignums())
    image = adapter().render_detail_snapshot(request_path, dict(channel="E9", band="beta1", sigma_seconds=0))
    assert image.startswith(b"\x89PNG\r\n\x1a\n") and len(image) > 1000
    assert matplotlib.get_backend().lower() == "agg"
    assert request_path.read_bytes() == before
    assert set(plt.get_fignums()) == figures_before


def test_import_and_detail_do_not_import_native_gui_renderers(snapshot, tmp_path):
    request_path = tmp_path / "request.joblib"
    joblib.dump(snapshot, request_path)
    script = """
import sys
from pathlib import Path
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from fourband.browser_data import render_detail_snapshot
render_detail_snapshot(Path(sys.argv[3]), {'channel': 'E9', 'band': 'theta', 'sigma_seconds': 0})
assert not any(name.split('.')[0] in {'tkinter', 'pyvista', 'vtk', 'vtkmodules', 'PyQt5', 'PySide6'} for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", script, str(ROOT / "3-bands"), str(ROOT),
                             str(request_path)], capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stderr
