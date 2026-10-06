"""Execute notebook display cells without allowing a native GUI in the kernel."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import nbformat
import numpy as np
import pandas as pd
import pytest

from test_fourband_continuous_notebook import ROOT, generator
sys.path.insert(0, str(ROOT / "3-bands"))
from fourband.session import ensure_fit_settings, ensure_trial_review
from fourband.workflow import ensure_extraction_config, ensure_first_round


@pytest.fixture(params=["current", "reliability", "legacy"])
def notebook(request):
    if request.param == "current":
        return nbformat.read(ROOT / "3-bands/ECoG_4band_clustering_pipeline.ipynb", as_version=4)
    builder = generator()
    return builder.build_reliability_notebook() if request.param == "reliability" else builder.build_notebook()


def cell_source(notebook, marker):
    return next(c.source for c in notebook.cells if c.cell_type == "code" and marker in c.source)


def forbidden(*args, **kwargs):
    pytest.fail("Notebook attempted to create a native window or load display data in the parent")


def display_namespace(tmp_path):
    extraction = object()
    first = SimpleNamespace(channels=["E1", "E2"], labels=[0, 1])
    second = SimpleNamespace(channels=["E1", "E2"], labels=[1, 0])
    calls = []
    display_run = SimpleNamespace(directory=tmp_path / "display_sessions/session", pid=17, returncode=0)

    def launch(extraction, result, *, output_dir, fif_path=None, onset_path=None,
               offset_path=None, wav_path=None, surface_options=None, show_surface=False,
               show_selectable=False, link_explorer=True, explorer_settings=None,
               cache_max_mb=512, explorer_only=False):
        calls.append(dict(extraction=extraction, result=result, output_dir=output_dir,
            fif_path=fif_path, onset_path=onset_path, offset_path=offset_path, wav_path=wav_path,
            surface_options=surface_options, show_surface=show_surface, show_selectable=show_selectable,
            link_explorer=link_explorer, explorer_settings=explorer_settings,
            cache_max_mb=cache_max_mb, explorer_only=explorer_only))
        return display_run

    namespace = dict(extraction=extraction, first_extraction=extraction,
        first_result=first, second_result=second,
        ROUND1_DIR=tmp_path / "round1", ROUND2_DIR=tmp_path / "round2",
        RUN_3D_RENDER=True, RUN_SELECTABLE_CLUSTER_RENDER=True,
        SECOND_ROUND_READY=True, TRIAL_QC_READY=True, OPEN_EXPLORER=True,
        LINK_SURFACE_EXPLORER=True, CACHE_MAX_MB=123,
        FIF_PATH=Path("recording.fif"), ONSET_PATH=Path("onset.npy"),
        OFFSET_PATH=Path("offset.npy"), WAV_PATH=Path("audio.wav"),
        ELECTRODE_CSV=Path("electrodes.csv"), SUBJECTS_DIR=Path("anatomy"),
        SURFACE_SUBJECT="patient-recon", REFERENCE_MRI=Path("T1.mgz"),
        HEMISPHERES=("lh", "rh"), SURFACE_COLOR="ivory", SURFACE_OPACITY=1,
        DISPLAY_SMOOTH_SIGMA_SECONDS=.23, DISPLAY_SFREQ=75.0,
        TRIAL_PRE_SECONDS=2.5, TRIAL_POST_SECONDS=3.5, TRIAL_PAIR_IDS=(10, 11),
        extraction_settings=lambda: {"subject": "patient"}, extracted_settings={"subject": "patient"},
        fit_settings=lambda: {"weight_preset": "amplitude"}, FIRST_N_CLUSTERS=7,
        first_fit_settings={"weight_preset": "amplitude", "first_k": 7},
        TRIAL_QC_SETTINGS={"spike_z": 12}, reviewed_trial_settings={"spike_z": 12},
        ensure_extraction_config=ensure_extraction_config, ensure_first_round=ensure_first_round,
        ensure_fit_settings=ensure_fit_settings, ensure_trial_review=ensure_trial_review,
        run_display=launch)
    for name in ("render_electrode_clusters", "render_selectable_electrode_class",
                 "get_explorer", "FourBandExplorer", "load_ecog_fif", "extract_audio_envelope",
                 "load_speech_events", "load_continuous_trials"):
        namespace[name] = forbidden
    return namespace, calls, display_run


def assert_recording_options(call, namespace, *, defaults=False):
    for name in ("fif_path", "onset_path", "offset_path", "wav_path"):
        assert call[name] == namespace[name.upper()]
    assert call["explorer_settings"] == dict(pre_seconds=1.0 if defaults else 2.5,
        post_seconds=1.0 if defaults else 3.5, trial_ids=None if defaults else (10, 11),
        sigma_seconds=.23, display_sfreq=75.0)
    assert call["cache_max_mb"] == 123


@pytest.mark.parametrize("surface,selectable,linked", [(True, True, True), (True, False, False), (False, True, True)])
def test_surface_exec_launches_once_with_paths_style_and_link_flags(notebook, tmp_path, surface, selectable, linked):
    namespace, calls, display_run = display_namespace(tmp_path)
    namespace.update(RUN_3D_RENDER=surface, RUN_SELECTABLE_CLUSTER_RENDER=selectable, LINK_SURFACE_EXPLORER=linked)
    exec(cell_source(notebook, "if RUN_3D_RENDER"), namespace)
    assert len(calls) == 1
    call = calls[0]
    assert call["extraction"] is namespace["extraction"]
    assert call["result"] is namespace["second_result"]
    assert call["output_dir"] == tmp_path / "round2"
    assert (call["show_surface"], call["show_selectable"], call["link_explorer"]) == (surface, selectable, linked)
    assert call["explorer_only"] is False
    assert call["surface_options"] == dict(electrode_csv=Path("electrodes.csv"), subjects_dir=Path("anatomy"),
        subject="patient-recon", reference_mri=Path("T1.mgz"), hemispheres=("lh", "rh"),
        brain_color="ivory", brain_opacity=1, channel_label_toggle=True)
    assert_recording_options(call, namespace)
    assert namespace["surface_display_run"] is display_run
    assert namespace["SECOND_ROUND_READY"] is True


def test_surface_defaults_before_continuous_cell_exec(notebook, tmp_path):
    namespace, calls, _ = display_namespace(tmp_path)
    for name in ("TRIAL_PRE_SECONDS", "TRIAL_POST_SECONDS", "TRIAL_PAIR_IDS"):
        namespace.pop(name)
    exec(cell_source(notebook, "if RUN_3D_RENDER"), namespace)
    assert len(calls) == 1
    assert_recording_options(calls[0], namespace, defaults=True)


def test_disabled_surface_needs_no_fit_or_gui_state(notebook):
    exec(cell_source(notebook, "if RUN_3D_RENDER"),
         dict(RUN_3D_RENDER=False, RUN_SELECTABLE_CLUSTER_RENDER=False, SECOND_ROUND_READY=False))


def test_cancelled_second_round_cannot_launch_stale_result(notebook, tmp_path):
    namespace, calls, _ = display_namespace(tmp_path)
    namespace["SECOND_ROUND_READY"] = False
    namespace.pop("second_result")
    namespace.pop("ROUND2_DIR")
    exec(cell_source(notebook, "if RUN_3D_RENDER"), namespace)
    assert calls == []


@pytest.fixture(params=["current", "reliability"])
def reliable_notebook(request):
    if request.param == "current":
        return nbformat.read(ROOT / "3-bands/ECoG_4band_clustering_pipeline.ipynb", as_version=4)
    return generator().build_reliability_notebook()


@pytest.mark.parametrize("marker", ["if RUN_3D_RENDER", "if OPEN_EXPLORER:"])
@pytest.mark.parametrize("changed,match", [
    ({"TRIAL_QC_READY": False}, "Confirm trial QC"),
    ({"extracted_settings": {}}, "Extraction settings changed"),
    ({"first_extraction": None}, "rerunning round one"),
    ({"reviewed_trial_settings": None}, "trial quality"),
    ({"first_fit_settings": {}}, "Classification settings changed"),
])
def test_display_guards_reject_stale_state_before_launch(reliable_notebook, tmp_path, marker, changed, match):
    namespace, calls, _ = display_namespace(tmp_path)
    namespace.update(changed)
    with pytest.raises(ValueError, match=match):
        exec(cell_source(reliable_notebook, marker), namespace)
    assert calls == []
    assert namespace["SECOND_ROUND_READY"] is True


@pytest.mark.parametrize("ready", [False, True])
def test_explorer_exec_uses_current_round_in_child(reliable_notebook, tmp_path, ready):
    namespace, calls, display_run = display_namespace(tmp_path)
    namespace["SECOND_ROUND_READY"] = ready
    if not ready:
        namespace.pop("second_result")
        namespace.pop("ROUND2_DIR")
    exec(cell_source(reliable_notebook, "if OPEN_EXPLORER:"), namespace)
    assert len(calls) == 1
    call = calls[0]
    assert call["extraction"] is namespace["extraction"]
    assert call["result"] is namespace["second_result" if ready else "first_result"]
    assert call["output_dir"] == tmp_path / ("round2" if ready else "round1")
    assert call["explorer_only"] is True
    assert call["show_surface"] is call["show_selectable"] is False
    assert call["surface_options"] is None
    assert_recording_options(call, namespace)
    assert namespace["explorer_display_run"] is display_run
    assert namespace["SECOND_ROUND_READY"] is ready


def test_disabled_explorer_does_not_launch(reliable_notebook, tmp_path):
    namespace, calls, _ = display_namespace(tmp_path)
    namespace["OPEN_EXPLORER"] = False
    exec(cell_source(reliable_notebook, "if OPEN_EXPLORER:"), namespace)
    assert calls == []


def test_notebook_imports_process_launcher_without_native_display_helpers(notebook):
    imports = [node for c in notebook.cells if c.cell_type == "code"
               for node in ast.walk(ast.parse(c.source)) if isinstance(node, ast.ImportFrom)]
    assert any(node.module == "fourband.display_process" and any(n.name == "run_display" for n in node.names)
               for node in imports)
    assert not any(node.module == "fourband.explorer" for node in imports)
    assert not any(node.module == "visualization" and any(n.name.startswith("render_") for n in node.names)
                   for node in imports)
    assert not any("def get_explorer(" in c.source for c in notebook.cells)


def test_initialization_preserves_static_continuous_event_loading(notebook, tmp_path):
    initialization = ast.parse(cell_source(notebook, "PROJECT_DIR ="))
    imports = [node for node in initialization.body if isinstance(node, ast.ImportFrom)
               and node.module in {"audio_cross_3band_utils", "fourband.continuous"}]
    namespace = {}
    # Execute the real data-helper imports without initializing any GUI backend.
    exec(compile(ast.Module(body=imports, type_ignores=[]), "initialization", "exec"), namespace)
    onset, offset = tmp_path / "onset.npy", tmp_path / "offset.npy"
    np.save(onset, [1.0, 3.0])
    np.save(offset, [2.0, 4.0])
    namespace.update(ONSET_PATH=onset, OFFSET_PATH=offset,
        extraction=SimpleNamespace(epochs={"theta": SimpleNamespace(
            events=pd.DataFrame({"original_trial_id": [0, 1]}))}))
    continuous = ast.parse(cell_source(notebook, "TRIAL_PAIR_IDS ="))
    assignment = next(node for node in continuous.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "trial_pairs"
                              for target in node.targets))
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "continuous", "exec"), namespace)
    assert namespace["trial_pairs"].to_dict("records") == [dict(
        first_trial_id=0, second_trial_id=1, first_onset=1.0, first_offset=2.0,
        second_onset=3.0, second_offset=4.0, gap_seconds=1.0)]


@pytest.mark.parametrize("migration", ["update_review_cells", "update_continuous_cells", "update_reliability_cells"])
@pytest.mark.parametrize("reliable", [False, True])
def test_migrations_replace_legacy_display_and_remain_repeatable(tmp_path, migration, reliable):
    builder = generator()
    notebook = builder.build_reliability_notebook() if reliable else builder.build_notebook()
    # Reproduce the previous direct-renderer layout accepted by all migration entry points.
    surface = next(c for c in notebook.cells if c.cell_type == "code" and "if RUN_3D_RENDER" in c.source)
    surface.source = "if RUN_3D_RENDER and SECOND_ROUND_READY:\n    render_electrode_clusters(ELECTRODE_CSV, {})"
    if reliable:
        explorer = next(c for c in notebook.cells if c.cell_type == "code" and "if OPEN_EXPLORER:" in c.source)
        explorer.source = 'if OPEN_EXPLORER:\n    get_explorer("second", second_result, ROUND2_DIR).show(block=True)'
    path = tmp_path / "legacy.ipynb"
    nbformat.write(notebook, path)
    update = getattr(builder, migration)
    update(path)
    updated = nbformat.read(path, as_version=4)
    namespace, calls, _ = display_namespace(tmp_path)
    exec(cell_source(updated, "if RUN_3D_RENDER"), namespace)
    assert len(calls) == 1 and calls[0]["result"] is namespace["second_result"]
    if reliable or migration == "update_reliability_cells":
        exec(cell_source(updated, "if OPEN_EXPLORER:"), namespace)
        assert len(calls) == 2 and calls[1]["explorer_only"] is True
    test_notebook_imports_process_launcher_without_native_display_helpers(updated)
    update(path)
    twice = nbformat.read(path, as_version=4)
    assert [c.source for c in twice.cells] == [c.source for c in updated.cells]
    nbformat.validate(twice)


@pytest.mark.parametrize("insert_at", [2, 12, 14, 16])
def test_review_migration_preserves_inserted_user_cell_and_routes_original_surface(tmp_path, insert_at):
    builder = generator()
    notebook = builder.build_notebook()
    surface = notebook.cells[16]
    surface.source = "if RUN_3D_RENDER and SECOND_ROUND_READY:\n    render_electrode_clusters(ELECTRODE_CSV, {})"
    surface_id = surface.id
    notebook.cells[2].source = notebook.cells[2].source.replace(
        "from fourband.display_process import run_display",
        "from visualization import render_electrode_clusters, render_selectable_electrode_class")
    parameters = cell_source(notebook, "def extraction_settings():")
    user_cell = nbformat.v4.new_code_cell("important_user_analysis = 42", execution_count=9,
        outputs=[nbformat.v4.new_output("stream", name="stdout", text="keep my output\n")],
        metadata={"user_note": "preserve"})
    notebook.cells.insert(insert_at, user_cell)
    path = tmp_path / "inserted_review.ipynb"
    nbformat.write(notebook, path)

    builder.update_review_cells(path)
    updated = nbformat.read(path, as_version=4)
    assert next(c for c in updated.cells if c.id == user_cell.id) == user_cell
    assert len(updated.cells) == len(notebook.cells)
    assert cell_source(updated, "def extraction_settings():") == parameters
    updated_surface = next(c for c in updated.cells if c.id == surface_id)
    namespace, calls, _ = display_namespace(tmp_path)
    exec(updated_surface.source, namespace)
    assert len(calls) == 1
    assert calls[0]["result"] is namespace["second_result"]
    assert calls[0]["show_surface"] is calls[0]["show_selectable"] is True
    test_notebook_imports_process_launcher_without_native_display_helpers(updated)
    nbformat.validate(updated)


@pytest.mark.parametrize("ambiguous", [False, True])
def test_review_migration_checks_all_owners_before_source_changes_or_write(tmp_path, monkeypatch, ambiguous):
    builder = generator()
    notebook = builder.build_notebook()
    notebook.cells[11].source += "\nLegacy review note"
    if ambiguous:
        notebook.cells.insert(16, nbformat.v4.new_code_cell(notebook.cells[16].source))
    else:
        notebook.cells[16].source = "unrecognized_surface_analysis = 42"
    path = tmp_path / "unowned_review.ipynb"
    nbformat.write(notebook, path)
    original_bytes = path.read_bytes()
    original_sources = [c.source for c in notebook.cells]
    monkeypatch.setattr(builder.nbf, "read", lambda *args, **kwargs: notebook)
    owned_cell = builder._owned_cell

    def checked_owner(existing, *markers):
        assert [c.source for c in existing.cells] == original_sources
        return owned_cell(existing, *markers)

    monkeypatch.setattr(builder, "_owned_cell", checked_owner)
    with pytest.raises(ValueError, match="Cannot uniquely identify owned notebook cell"):
        builder.update_review_cells(path)
    assert [c.source for c in notebook.cells] == original_sources
    assert path.read_bytes() == original_bytes
