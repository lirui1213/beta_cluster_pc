"""Targeted notebook migration and real-time display integration."""
import ast
import importlib.util
from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]


def generator():
    spec = importlib.util.spec_from_file_location("fourband_generator", ROOT / "3-bands/create_notebook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_targeted_update_preserves_existing_analysis_and_style_assignments(tmp_path):
    source = ROOT / "3-bands/ECoG_4band_clustering_pipeline.ipynb"
    notebook = nbformat.read(source, as_version=4)
    # User-added cells after the pipeline are not owned by the migration.
    notebook.cells.append(nbformat.v4.new_code_cell("user_notes = 'keep me'"))
    original_parameters = notebook.cells[4].source
    destination = tmp_path / "experiment.ipynb"
    nbformat.write(notebook, destination)
    generator().update_continuous_cells(destination)
    revised = nbformat.read(destination, as_version=4)
    assert revised.cells[4].source == original_parameters
    assert all(not cell.outputs and cell.execution_count is None for cell in revised.cells if cell.cell_type == "code")
    added = 0 if any(c.cell_type == "code" and "TRIAL_PAIR_IDS =" in c.source for c in notebook.cells) else 2
    assert len(revised.cells) == len(notebook.cells) + added
    assert revised.cells[-1].source == "user_notes = 'keep me'"
    continuous_cell = next(c for c in revised.cells if c.cell_type == "code" and "TRIAL_PAIR_IDS =" in c.source)
    assert "TRIAL_PAIR_IDS = None" in continuous_cell.source
    assert "TRIAL_PRE_SECONDS = 1.0" in continuous_cell.source
    assert "TRIAL_POST_SECONDS = 1.0" in continuous_cell.source
    assert 'TRIAL_DISPLAY_ROUND = "both"' in continuous_cell.source
    assert "ensure_extraction_config" in continuous_cell.source
    assert "SECOND_ROUND_READY" in continuous_cell.source
    assert "load_continuous_trials" in revised.cells[2].source
    assert any("channel_label_toggle=True" in c.source for c in revised.cells)
    assert any("predictor_policy" in c.source for c in revised.cells)
    nbformat.validate(revised)
    for cell in revised.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    generator().update_continuous_cells(destination)
    twice = nbformat.read(destination, as_version=4)
    assert [cell.source for cell in twice.cells] == [cell.source for cell in revised.cells]


def test_generated_notebook_has_same_continuous_workflow_and_label_controls():
    notebook = generator().build_notebook()
    assert len(notebook.cells) == 19
    assert "adjacent_trial_pairs" in notebook.cells[18].source
    assert "save_continuous_trial_plot" in notebook.cells[18].source
    assert "channel_label_toggle=True" in notebook.cells[16].source


def test_cancelled_second_round_cannot_display_old_second_labels():
    source = generator().build_notebook().cells[18].source
    # The selected-round list only admits round 2 after fresh review confirmation.
    assert 'if TRIAL_DISPLAY_ROUND in {"second", "both"} and SECOND_ROUND_READY:' in source
    assert "ensure_first_round" in source


def test_failed_first_round_invalidates_old_second_round_before_fit():
    source = generator().build_notebook().cells[10].source
    def fail(*args, **kwargs):
        raise ValueError("Invalid first K")
    namespace = dict(SECOND_ROUND_READY=True, fit_joint=fail,
        ensure_extraction_config=lambda *args: None, extraction_settings=lambda: {}, extracted_settings={},
        features=None, FIRST_N_CLUSTERS=999, BANDS={}, BAND_WEIGHTS={}, FEATURE_CORRELATION_THRESHOLD=.95, RANDOM_STATE=50)
    with pytest.raises(ValueError, match="Invalid first K"):
        exec(source, namespace)
    assert namespace["SECOND_ROUND_READY"] is False


def test_migration_preserves_cells_inserted_inside_the_pipeline(tmp_path):
    notebook = generator().build_notebook()
    notebook.cells[9:9] = [nbformat.v4.new_markdown_cell("## User analysis"),
        nbformat.v4.new_code_cell("important_user_analysis = 42")]
    destination = tmp_path / "inserted.ipynb"
    nbformat.write(notebook, destination)
    generator().update_continuous_cells(destination)
    updated = nbformat.read(destination, as_version=4)
    assert any(c.source == "important_user_analysis = 42" for c in updated.cells)
    assert any("pending_first = fit_joint(" in c.source for c in updated.cells)
    assert len(updated.cells) == len(notebook.cells)


def test_repeated_migration_preserves_selected_display_parameters(tmp_path):
    notebook = generator().build_notebook()
    notebook.cells[18].source = notebook.cells[18].source.replace("TRIAL_PAIR_IDS = None", "TRIAL_PAIR_IDS = (10, 11)").replace(
        "TRIAL_PRE_SECONDS = 1.0", "TRIAL_PRE_SECONDS = 2.5")
    destination = tmp_path / "selected.ipynb"
    nbformat.write(notebook, destination)
    generator().update_continuous_cells(destination)
    updated = nbformat.read(destination, as_version=4)
    assert "TRIAL_PAIR_IDS = (10, 11)" in updated.cells[18].source
    assert "TRIAL_PRE_SECONDS = 2.5" in updated.cells[18].source
