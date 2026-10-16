"""Catalog contracts exercised against real temporary workspace layouts."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3-bands"))


def test_catalog_api_is_available_for_server_startup():
    assert importlib.util.find_spec("fourband.workspace_catalog") is not None, "Task1 catalog is not implemented"
    module = importlib.import_module("fourband.workspace_catalog")
    assert callable(module.WorkspaceCatalog) and callable(module.validate_config)


@pytest.fixture
def api():
    assert importlib.util.find_spec("fourband.workspace_catalog") is not None, "Task1 catalog is not implemented"
    return importlib.import_module("fourband.workspace_catalog")


def put(root, relative, content=""):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def layout(tmp_path):
    data, output = tmp_path / "data", tmp_path / "outputs"
    prefix = "HS0015/blocks/B03/standardized/"
    for relative in ("ECoG/ecog_TEST_clean_car-raw.fif", "events/onset.npy",
                     "events/offset.npy", "audio/analog1.wav"):
        put(data, prefix + relative)
    for relative in ("baseline/baseline.fif", "recon/electrodes/electrodes.csv",
                     "recon/mri/T1.mgz", "recon/surf/lh.pial"):
        put(data, "HS0015/" + relative)
    return data, output


def test_patients_lists_only_actual_standardized_block_directories(api, layout):
    data, output = layout
    (data / "HS0015/blocks/B05/standardized").mkdir(parents=True)
    (data / "HS0012/blocks/B01/standardized").mkdir(parents=True)
    (data / "HS0007/blocks/B02").mkdir(parents=True)
    put(data, "notes/blocks/B01/standardized")
    (data / "not valid/blocks/B01/standardized").mkdir(parents=True)
    assert api.WorkspaceCatalog(data, output).patients() == [
        {"id": "HS0012", "blocks": ["B01"]},
        {"id": "HS0015", "blocks": ["B03", "B05"]},
    ]


def test_missing_root_is_empty_and_does_not_create_directories(api, tmp_path):
    data, output = tmp_path / "absent", tmp_path / "outputs"
    assert api.WorkspaceCatalog(data, output).patients() == []
    assert not data.exists() and not output.exists()


def test_resolve_returns_documented_sources_surface_and_availability(api, layout):
    data, output = layout
    resolved = api.WorkspaceCatalog(data, output).resolve("HS0015", "B03")
    assert resolved["subject"] == "HS0015" and resolved["block"] == "B03"
    assert resolved["sources"] == {
        "fif_path": str(data / "HS0015/blocks/B03/standardized/ECoG/ecog_TEST_clean_car-raw.fif"),
        "onset_path": str(data / "HS0015/blocks/B03/standardized/events/onset.npy"),
        "offset_path": str(data / "HS0015/blocks/B03/standardized/events/offset.npy"),
        "wav_path": str(data / "HS0015/blocks/B03/standardized/audio/analog1.wav"),
        "baseline_fif": str(data / "HS0015/baseline/baseline.fif"),
        "baseline_segments_csv": None,
    }
    assert resolved["surface_options"] == {
        "electrode_csv": str(data / "HS0015/recon/electrodes/electrodes.csv"),
        "subjects_dir": str(data / "HS0015"), "subject": "recon",
        "reference_mri": str(data / "HS0015/recon/mri/T1.mgz"), "hemispheres": ["lh"],
    }
    available = {row["name"]: row for row in resolved["availability"]}
    assert available["fif_path"]["exists"] is True
    assert available["rh.pial"]["exists"] is False
    assert resolved["errors"] == []
    json.dumps(resolved, allow_nan=False)


def test_empty_standardized_folder_retains_missing_paths_and_errors(api, layout):
    data, output = layout
    (data / "HS0015/blocks/B05/standardized").mkdir(parents=True)
    result = api.WorkspaceCatalog(data, output).resolve("HS0015", "B05")
    for name in ("fif_path", "onset_path", "offset_path", "wav_path"):
        assert result["sources"][name]
        assert any(name in error for error in result["errors"])
    with pytest.raises(ValueError, match="fif_path|onset_path|offset_path|wav_path"):
        api.validate_config({}, result)


@pytest.mark.parametrize("subject,block", [
    ("../HS0015", "B03"), ("HS0015", "../B03"), ("HS0015", "B03/.."),
    ("HS0015", "B03\\x"), ("I:/ECoG_data", "B03"), ("HS0015", "B03:ads"),
    ("HS0015.", "B03"), ("HS0015 ", "B03"), (None, "B03"), (15, "B03"),
    ("HS0015", "B99"), ("hs0015", "B03"), ("HS0015", "b03"),
])
def test_resolve_rejects_invalid_or_non_catalog_ids(api, layout, subject, block):
    catalog = api.WorkspaceCatalog(*layout)
    with pytest.raises(ValueError):
        catalog.resolve(subject, block)
    with pytest.raises(ValueError):
        catalog.history(subject, block)


def link_directory(link, target):
    if os.name == "nt":
        # Junction creation works without the Windows symbolic-link privilege.
        environment = dict(os.environ, CATALOG_TEST_LINK=str(link), CATALOG_TEST_TARGET=str(target))
        result = subprocess.run(["powershell", "-NoProfile", "-Command",
            "New-Item -ItemType Junction -Path $env:CATALOG_TEST_LINK -Target $env:CATALOG_TEST_TARGET | Out-Null"],
            capture_output=True, text=True, env=environment)
        assert result.returncode == 0, result.stderr
    else:
        link.symlink_to(target, target_is_directory=True)


def test_discovery_and_resolve_reject_external_patient_junction(api, layout, tmp_path):
    data, output = layout
    outside = tmp_path / "outside"
    (outside / "blocks/B01/standardized").mkdir(parents=True)
    link_directory(data / "HS9999", outside)
    catalog = api.WorkspaceCatalog(data, output)
    assert [p["id"] for p in catalog.patients()] == ["HS0015"]
    with pytest.raises(ValueError):
        catalog.resolve("HS9999", "B01")


@pytest.mark.parametrize("relative", ["HS0015/blocks/B03/standardized/events", "HS0015/recon/surf"])
def test_resolve_rejects_source_and_anatomy_junction_escapes(api, layout, tmp_path, relative):
    data, output = layout
    target = data / relative
    for child in target.iterdir():
        child.unlink()
    target.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link_directory(target, outside)
    with pytest.raises(ValueError, match="root|contain|outside"):
        api.WorkspaceCatalog(data, output).resolve("HS0015", "B03")


def test_output_context_junction_cannot_escape_configured_root(api, layout, tmp_path):
    data, output = layout
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link_directory(output / "HS0015", outside)
    catalog = api.WorkspaceCatalog(data, output)
    with pytest.raises(ValueError):
        catalog.resolve("HS0015", "B03")
    with pytest.raises(ValueError):
        catalog.history("HS0015", "B03")


def test_baseline_csv_requires_explicit_discovered_id(api, layout):
    data, output = layout
    csv = put(data, "HS0015/baseline/baseline__segments.csv", "start,end\n0,1\n")
    put(data, "HS0015/baseline/unrelated.csv")
    catalog = api.WorkspaceCatalog(data, output)
    default = catalog.resolve("HS0015", "B03")
    assert default["sources"]["baseline_segments_csv"] is None
    assert default["selection_options"]["baseline_segments_csv"] == [
        {"id": "baseline__segments.csv", "label": "baseline__segments.csv"}]
    selected = catalog.resolve("HS0015", "B03", {"baseline_segments_csv": "baseline__segments.csv"})
    assert selected["sources"]["baseline_segments_csv"] == str(csv)
    assert api.validate_config({}, selected)["sources"]["baseline_segments_csv"] == str(csv)
    for selections in ({"baseline_segments_csv": str(csv)}, {"baseline_segments_csv": "../segments.csv"},
                       {"baseline_segments_csv": "unrelated.csv"}, {"fif_path": "alternate"}, [], "bad"):
        with pytest.raises(ValueError):
            catalog.resolve("HS0015", "B03", selections)


def test_config_defaults_are_json_safe_and_server_paths_are_authoritative(api, layout):
    resolved = api.WorkspaceCatalog(*layout).resolve("HS0015", "B03")
    settings = api.validate_config({}, resolved)
    assert {key: value for key, value in settings.items() if key != "input_identity"} == {
        "subject": "HS0015", "block": "B03", "sources": resolved["sources"],
        "surface_options": resolved["surface_options"],
        "baseline_mode": "file", "baseline_window": [-1.3, -.3],
        "baseline_edge_seconds": .3, "baseline_relative_floor": .001,
        "feature_sigma_seconds": .06, "first_k": 10, "second_k": 4,
        "band_weights": {"theta": 1., "beta1": 1., "beta2": 1., "high_gamma": 1.},
        "weight_preset": "legacy", "group_weights": None,
        "correlation_threshold": .95, "random_state": 50,
        "duration_mad_threshold": 2.5, "min_trials_after_filter": 3,
        "progress_points": 101, "pre_post_window": None, "slope_method": "savgol",
        "max_baseline_invalid_fraction": .25, "display_sigma_seconds": .1, "display_sfreq": 100.,
    }
    json.dumps(settings, allow_nan=False)
    settings["sources"]["fif_path"] = "mutated"
    settings["band_weights"]["theta"] = 2
    assert resolved["sources"]["fif_path"] != "mutated"
    assert api.validate_config({}, resolved)["band_weights"]["theta"] == 1.


def test_config_records_server_stat_identity_for_all_existing_sources(api, layout):
    data, output = layout
    put(data, "HS0015/baseline/baseline__segments.csv", "start,end\n0,1\n")
    resolved = api.WorkspaceCatalog(data, output).resolve("HS0015", "B03",
        {"baseline_segments_csv": "baseline__segments.csv"})
    content = {"fif_path": "fif01", "onset_path": "on1", "offset_path": "off1",
               "wav_path": "audio01", "baseline_fif": "baseline", "baseline_segments_csv": "start,end\n0,1\n"}
    for name, text in content.items():
        path = Path(resolved["sources"][name])
        path.write_bytes(text.encode("ascii"))
        os.utime(path, ns=(1700000000000000000, 1700000000000000000))
    settings = api.validate_config({}, resolved)
    assert settings["input_identity"] == {
        "fif_path": {"size": 5, "mtime_ns": 1700000000000000000},
        "onset_path": {"size": 3, "mtime_ns": 1700000000000000000},
        "offset_path": {"size": 4, "mtime_ns": 1700000000000000000},
        "wav_path": {"size": 7, "mtime_ns": 1700000000000000000},
        "baseline_fif": {"size": 8, "mtime_ns": 1700000000000000000},
        "baseline_segments_csv": {"size": 14, "mtime_ns": 1700000000000000000},
    }
    json.dumps(settings, allow_nan=False)
    Path(resolved["sources"]["wav_path"]).write_bytes(b"changed audio")
    updated = api.validate_config({}, resolved)
    assert updated["input_identity"]["wav_path"]["size"] == 13
    assert updated["input_identity"]["wav_path"] != settings["input_identity"]["wav_path"]


def test_config_identity_omits_missing_optional_baseline_and_unselected_csv(api, layout):
    data, output = layout
    put(data, "HS0015/baseline/baseline__segments.csv", "start,end\n0,1\n")
    (data / "HS0015/baseline/baseline.fif").unlink()
    resolved = api.WorkspaceCatalog(data, output).resolve("HS0015", "B03")
    settings = api.validate_config({"baseline_mode": "pre_onset"}, resolved)
    assert set(settings["input_identity"]) == {"fif_path", "onset_path", "offset_path", "wav_path"}


def test_missing_baseline_blocks_only_file_mode_and_anatomy_is_nonfatal(api, layout):
    data, output = layout
    (data / "HS0015/baseline/baseline.fif").unlink()
    (data / "HS0015/recon/electrodes/electrodes.csv").unlink()
    (data / "HS0015/recon/mri/T1.mgz").unlink()
    (data / "HS0015/recon/surf/lh.pial").unlink()
    resolved = api.WorkspaceCatalog(data, output).resolve("HS0015", "B03")
    assert any("baseline_fif" in error for error in resolved["errors"])
    assert any("anatom" in error.lower() for error in resolved["errors"])
    with pytest.raises(ValueError, match="baseline_fif"):
        api.validate_config({}, resolved)
    assert api.validate_config({"baseline_mode": "pre_onset"}, resolved)["baseline_mode"] == "pre_onset"


@pytest.mark.parametrize("key,value", [
    ("unknown", 1), ("sources", {}), ("surface_options", {}), ("subject", "HS9999"),
    ("block", "B05"), ("fif_path", "C:/private.fif"), ("baseline_segments_csv", "segments.csv"),
    ("input_identity", {"wav_path": {"size": 0, "mtime_ns": 0}}),
    ("first_k", True), ("first_k", 2.0), ("first_k", 1), ("second_k", 0),
    ("random_state", -1), ("random_state", 2**32), ("random_state", False),
    ("progress_points", 2), ("progress_points", 3.5), ("min_trials_after_filter", 0),
    ("baseline_mode", "auto"), ("baseline_mode", []), ("slope_method", "invalid"),
    ("baseline_window", [-.3, -1.3]), ("baseline_window", [-1., .1]),
    ("baseline_window", [-1., -1.]), ("baseline_window", [True, 0]),
    ("baseline_window", [-1.]), ("baseline_window", "-1,0"),
    ("baseline_edge_seconds", -.1), ("baseline_relative_floor", 0),
    ("feature_sigma_seconds", float("nan")), ("display_sigma_seconds", float("inf")),
    ("display_sfreq", 0), ("display_sfreq", "100"), ("feature_sigma_seconds", True),
    ("duration_mad_threshold", 0), ("correlation_threshold", 0), ("correlation_threshold", 1.1),
    ("max_baseline_invalid_fraction", -.1), ("max_baseline_invalid_fraction", 1.1),
    ("pre_post_window", 0), ("pre_post_window", [1, 2]),
    ("band_weights", [1, 1, 1, 1]), ("band_weights", {"theta": 1}),
    ("band_weights", {"theta": 1, "beta1": 1, "beta2": 1, "high_gamma": 0}),
    ("band_weights", {"theta": 1, "beta1": 1, "beta2": 1, "audio": 1}),
    ("weight_preset", "automatic"), ("group_weights", {"slope": 2}),
])
def test_config_rejects_invalid_values_and_browser_path_overrides(api, layout, key, value):
    resolved = api.WorkspaceCatalog(*layout).resolve("HS0015", "B03")
    with pytest.raises(ValueError, match=key):
        api.validate_config({key: value}, resolved)


@pytest.mark.parametrize("config", [None, [], "settings", True])
def test_config_requires_json_object(api, layout, config):
    with pytest.raises(ValueError):
        api.validate_config(config, api.WorkspaceCatalog(*layout).resolve("HS0015", "B03"))


def test_valid_custom_weights_and_numerical_overrides(api, layout):
    resolved = api.WorkspaceCatalog(*layout).resolve("HS0015", "B03")
    config = {"first_k": 2, "second_k": 3, "baseline_mode": "pre_onset",
        "baseline_window": [-2, -.5], "weight_preset": "custom",
        "group_weights": {"morphology": 0, "amplitude": 2, "slope": 0},
        "band_weights": {"theta": 2, "beta1": 1, "beta2": 3, "high_gamma": 4},
        "pre_post_window": .5, "slope_method": "gradient", "feature_sigma_seconds": 0}
    effective = api.validate_config(config, resolved)
    for key, value in config.items():
        assert effective[key] == value
    json.dumps(effective, allow_nan=False)


@pytest.mark.parametrize("groups", [[], {"audio": 1}, {"slope": -1}, {"slope": True},
    {"slope": float("nan")}, {"morphology": 0, "amplitude": 0, "slope": 0}])
def test_invalid_custom_group_weights(api, layout, groups):
    with pytest.raises(ValueError, match="group_weights"):
        api.validate_config({"weight_preset": "custom", "group_weights": groups},
                            api.WorkspaceCatalog(*layout).resolve("HS0015", "B03"))


def test_required_files_rechecked_after_resolution(api, layout):
    resolved = api.WorkspaceCatalog(*layout).resolve("HS0015", "B03")
    Path(resolved["sources"]["onset_path"]).unlink()
    with pytest.raises(ValueError, match="onset_path"):
        api.validate_config({}, resolved)


def test_pre_onset_window_cannot_end_at_onset(api, layout):
    with pytest.raises(ValueError, match="baseline_window"):
        api.validate_config({"baseline_mode": "pre_onset", "baseline_window": [-1, 0]},
                            api.WorkspaceCatalog(*layout).resolve("HS0015", "B03"))


def test_history_unsupported_schema_has_bounded_summary(api, layout):
    data, output = layout
    for number in range(30):
        identifier = f"run_{number:03d}"
        put(output, f"HS0015/B03/{identifier}/workspace_manifest.json", json.dumps({
            "schema_version": 10**4000, "subject": "HS0015", "block": "B03", "run_id": identifier,
            "revision": 0, "stage": "configured", "created_at": "x" * 10000}))
    history = api.WorkspaceCatalog(data, output).history("HS0015", "B03")
    assert len(history) == 30 and all(row["read_only"] for row in history)
    assert len(json.dumps(history)) < 30000


def test_swapped_input_junction_is_rechecked_at_validation(api, layout, tmp_path):
    data, output = layout
    resolved = api.WorkspaceCatalog(data, output).resolve("HS0015", "B03")
    events = data / "HS0015/blocks/B03/standardized/events"
    for path in events.iterdir():
        path.unlink()
    events.rmdir()
    outside = tmp_path / "outside"
    put(outside, "onset.npy")
    put(outside, "offset.npy")
    link_directory(events, outside)
    with pytest.raises(ValueError, match="root|outside"):
        api.validate_config({}, resolved)


def test_swapped_anatomy_junction_is_rechecked_at_validation(api, layout, tmp_path):
    data, output = layout
    resolved = api.WorkspaceCatalog(data, output).resolve("HS0015", "B03")
    surfaces = data / "HS0015/recon/surf"
    (surfaces / "lh.pial").unlink()
    surfaces.rmdir()
    outside = tmp_path / "outside"
    put(outside, "lh.pial")
    link_directory(surfaces, outside)
    with pytest.raises(ValueError, match="root|outside"):
        api.validate_config({}, resolved)


def test_history_summarizes_manifests_and_marks_legacy_read_only(api, layout):
    data, output = layout
    manifest = {"schema_version": 1, "run_id": "20261016_web", "subject": "HS0015",
        "block": "B03", "revision": 4, "stage": "trial_review", "created_at": "2026-10-16",
        "task": {"status": "completed"}, "data": {"private_path": "C:/private"}, "config": {}}
    put(output, "HS0015/B03/20261016_web/workspace_manifest.json", json.dumps(manifest))
    put(output, "HS0015/B03/20261015_old/run_config.json", "{}")
    put(output, "HS0015/B03/20261015_old/round1_001/summary.json", "{}")
    put(output, "HS0015/B03/random_notes/readme.txt")
    history = api.WorkspaceCatalog(data, output).history("HS0015", "B03")
    assert [row["id"] for row in history] == ["20261016_web", "20261015_old"]
    current, legacy = history
    assert current["kind"] == "workspace" and current["read_only"] is False
    assert current["stage"] == "trial_review" and current["revision"] == 4
    assert current["status"] == "completed"
    assert all(row["id"] == row["run_id"] for row in history)
    assert legacy["kind"] == "legacy" and legacy["read_only"] is True
    assert "private" not in json.dumps(history) and str(output) not in json.dumps(history)
    assert not (output / "HS0015/B03/20261015_old/workspace_manifest.json").exists()


@pytest.mark.parametrize("stage", ["trial_review", "first_ready", "second_ready"])
def test_history_accepts_lean_manifest_with_large_separate_workspace_data(api, layout, stage):
    data, output = layout
    identifier = "20261016_182943_780124_web_2c8872"
    prefix = f"HS0015/B03/{identifier}/"
    sidecar = put(output, prefix + "workspace_data.json", json.dumps({"qc": "x" * (2 * 1024 * 1024)}))
    manifest = put(output, prefix + "workspace_manifest.json", json.dumps({
        "schema_version": 1, "run_id": identifier, "subject": "HS0015", "block": "B03",
        "revision": 6, "stage": stage, "created_at": "2026-10-16T18:29:43+08:00",
        "task": {"action": "fit_second", "status": "completed"},
        "config": {}, "data": {}, "data_file": "workspace_data.json",
        "rounds": [{"id": "round1_001", "data": {"pca": [[1, 2], [3, 4]]}}],
    }))
    before = (manifest.stat().st_size, sidecar.stat().st_size)
    assert before[0] < 1024 * 1024 < before[1]
    history = api.WorkspaceCatalog(data, output).history("HS0015", "B03")
    assert len(history) == 1
    row = history[0]
    assert row["id"] == row["run_id"] == identifier
    assert row["schema_version"] == 1 and row["revision"] == 6
    assert row["kind"] == "workspace" and row["read_only"] is False
    assert row["stage"] == stage and row["status"] == "completed"
    assert "data_file" not in row and "data" not in row and "rounds" not in row
    assert len(json.dumps(history)) < 1000
    assert (manifest.stat().st_size, sidecar.stat().st_size) == before
    # History must not inspect or parse the separately persisted QC payload.
    sidecar.write_text("{", encoding="utf-8")
    assert api.WorkspaceCatalog(data, output).history("HS0015", "B03") == history


def test_history_bounds_count_json_size_and_rejects_foreign_identity(api, layout):
    data, output = layout
    for number in range(120):
        put(output, f"HS0015/B03/run_{number:03d}/run_config.json", "{}")
    put(output, "HS0015/B03/z_foreign/workspace_manifest.json", json.dumps({
        "schema_version": 1, "subject": "HS9999", "block": "B03", "run_id": "z_foreign"}))
    put(output, "HS0015/B03/z_huge/workspace_manifest.json", " " * (1024 * 1024 + 1))
    put(output, "HS0015/B03/z_broken/workspace_manifest.json", "{")
    history = api.WorkspaceCatalog(data, output).history("HS0015", "B03")
    assert len(history) == 100
    assert not any(row["id"] == "z_foreign" for row in history)
    for identifier in ("z_huge", "z_broken"):
        row = next(row for row in history if row["id"] == identifier)
        assert row["read_only"] is True and row["status"] == "incomplete"
    assert len(json.dumps(history)) < 100000


def test_history_ignores_external_run_junctions(api, layout, tmp_path):
    data, output = layout
    parent = output / "HS0015/B03"
    parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    put(outside, "run_config.json", "{}")
    link_directory(parent / "foreign", outside)
    assert api.WorkspaceCatalog(data, output).history("HS0015", "B03") == []


def saved_workspace(output, subject="HS0015", block="B03", identifier="saved_web"):
    return put(output, f"{subject}/{block}/{identifier}/workspace_manifest.json", json.dumps({
        "schema_version": 1, "subject": subject, "block": block, "run_id": identifier,
        "revision": 6, "stage": "second_ready", "task": {"status": "completed"},
        "state_data_file": "workspace_data.json"}))


@pytest.mark.parametrize("kind", ["workspace", "legacy_config", "legacy_round"])
def test_history_is_available_when_data_root_is_offline(api, tmp_path, kind):
    data, output = tmp_path / "unmounted", tmp_path / "outputs"
    if kind == "workspace":
        saved_workspace(output)
    elif kind == "legacy_config":
        put(output, "HS0015/B03/saved_web/run_config.json", "{}")
    else:
        put(output, "HS0015/B03/saved_web/round1_001/summary.json", "{}")
    history = api.WorkspaceCatalog(data, output).history("HS0015", "B03")
    assert len(history) == 1 and history[0]["id"] == history[0]["run_id"] == "saved_web"
    assert history[0]["read_only"] is (kind != "workspace")
    assert not data.exists()


def test_patients_unions_source_and_known_saved_output_contexts(api, layout):
    data, output = layout
    saved_workspace(output, subject="HS0012", block="B01")
    saved_workspace(output)
    put(output, "HS0015/B05/saved_old/run_config.json", "{}")
    put(output, "HS9999/B01/notes/readme.txt")
    put(output, "HS0012/B99/foreign/workspace_manifest.json", json.dumps({
        "subject": "HS9999", "block": "B99", "run_id": "foreign", "schema_version": 1}))
    (output / "HS0012/B98").mkdir()
    assert api.WorkspaceCatalog(data, output).patients() == [
        {"id": "HS0012", "blocks": ["B01"]},
        {"id": "HS0015", "blocks": ["B03", "B05"]}]


def test_resolve_saved_offline_context_reports_expected_paths_and_missing_inputs(api, tmp_path):
    data, output = tmp_path / "unmounted", tmp_path / "outputs"
    saved_workspace(output)
    catalog = api.WorkspaceCatalog(data, output)
    assert catalog.patients() == [{"id": "HS0015", "blocks": ["B03"]}]
    resolved = catalog.resolve("HS0015", "B03")
    assert resolved["sources"]["fif_path"] == str(data / "HS0015/blocks/B03/standardized/ECoG/ecog_TEST_clean_car-raw.fif")
    for name in ("fif_path", "onset_path", "offset_path", "wav_path", "baseline_fif"):
        assert any(name in error for error in resolved["errors"])
    assert all(not item["exists"] for item in resolved["availability"])
    with pytest.raises(ValueError, match="fif_path"):
        api.validate_config({}, resolved)
    for subject, block in (("HS9999", "B03"), ("HS0015", "B99"), ("hs0015", "B03")):
        with pytest.raises(ValueError):
            catalog.resolve(subject, block)
        with pytest.raises(ValueError):
            catalog.history(subject, block)
    assert not data.exists()


def test_saved_context_survives_loss_of_standardized_directory(api, layout):
    data, output = layout
    saved_workspace(output, block="B99")
    (data / "HS0015/blocks/B99").mkdir()
    catalog = api.WorkspaceCatalog(data, output)
    assert catalog.history("HS0015", "B99")[0]["stage"] == "second_ready"
    assert any("fif_path" in error for error in catalog.resolve("HS0015", "B99")["errors"])


def test_public_context_validates_known_ids_without_requiring_offline_sources(api, tmp_path):
    data, output = tmp_path / "unmounted", tmp_path / "outputs"
    saved_workspace(output)
    catalog = api.WorkspaceCatalog(data, output)
    assert catalog.context("HS0015", "B03") == {"subject": "HS0015", "block": "B03"}
    with pytest.raises(ValueError):
        catalog.context("HS0015", "B03", require_data=True)
    for subject, block in (("../HS0015", "B03"), ("HS0015", "B03:ads"), ("HS0015", "B99"),
                           ("hs0015", "B03"), ("HS0015", "b03")):
        with pytest.raises(ValueError):
            catalog.context(subject, block)


def test_offline_output_context_rejects_external_block_junction(api, tmp_path):
    data, output = tmp_path / "unmounted", tmp_path / "outputs"
    outside = tmp_path / "outside"
    saved_workspace(outside)
    (output / "HS0015").mkdir(parents=True)
    link_directory(output / "HS0015/B03", outside / "HS0015/B03")
    catalog = api.WorkspaceCatalog(data, output)
    assert catalog.patients() == []
    for method in (catalog.context, catalog.history, catalog.resolve):
        with pytest.raises(ValueError):
            method("HS0015", "B03")


def test_history_saved_context_does_not_follow_external_source_junction(api, tmp_path):
    data, output = tmp_path / "data", tmp_path / "outputs"
    saved_workspace(output)
    outside = tmp_path / "outside"
    outside.mkdir()
    data.mkdir()
    link_directory(data / "HS0015", outside)
    catalog = api.WorkspaceCatalog(data, output)
    assert catalog.history("HS0015", "B03")[0]["id"] == "saved_web"
    assert catalog.context("HS0015", "B03") == {"subject": "HS0015", "block": "B03"}
    with pytest.raises(ValueError, match="root|outside"):
        catalog.resolve("HS0015", "B03")
