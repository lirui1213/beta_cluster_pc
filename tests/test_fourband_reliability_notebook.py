"""Reliability migration preserves user settings and guards downstream fits."""
import ast
from pathlib import Path
import sys

import nbformat
import pytest

from test_fourband_continuous_notebook import generator
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3-bands"))


def code_with(nb, marker):
    return next(c for c in nb.cells if c.cell_type == "code" and marker in c.source)


def test_reliability_migration_preserves_parameters_added_cells_and_new_options(tmp_path):
    nb = generator().build_notebook()
    params = nb.cells[4].source.replace('SUBJECT_ID = "HS0023"', 'SUBJECT_ID = "HS0015"')
    nb.cells[4].source = params
    nb.cells.insert(9, nbformat.v4.new_code_cell("user_value = 321"))
    nb.cells[-1].source = nb.cells[-1].source.replace("TRIAL_PRE_SECONDS = 1.0", "TRIAL_PRE_SECONDS = 3.0")
    path = tmp_path / "custom.ipynb"
    nbformat.write(nb, path)
    generator().update_reliability_cells(path)
    updated = nbformat.read(path, as_version=4)
    assert code_with(updated, "def extraction_settings").source == params
    assert any(c.source == "user_value = 321" for c in updated.cells)
    assert "TRIAL_PRE_SECONDS = 3.0" in code_with(updated, "TRIAL_PAIR_IDS =").source
    settings = code_with(updated, "WEIGHT_PRESET =")
    settings.source = settings.source.replace('WEIGHT_PRESET = "legacy"', 'WEIGHT_PRESET = "amplitude"')
    nbformat.write(updated, path)
    generator().update_reliability_cells(path)
    twice = nbformat.read(path, as_version=4)
    assert 'WEIGHT_PRESET = "amplitude"' in code_with(twice, "WEIGHT_PRESET =").source
    assert len(twice.cells) == len(updated.cells)
    for c in twice.cells:
        if c.cell_type == "code":
            ast.parse(c.source)
            assert c.outputs == [] and c.execution_count is None
    nbformat.validate(twice)


def test_generated_workflow_requires_confirmed_qc_and_matching_first_fit():
    nb = generator().build_reliability_notebook()
    qc = code_with(nb, "TrialQualityReviewWindow(").source
    assert "initial_pairs=()" in qc
    assert "if selected_trial_pairs is not None:" in qc
    first = code_with(nb, "pending_first = fit_joint(").source
    assert first.index("SECOND_ROUND_READY = False") < first.index("pending_first = fit_joint(")
    assert "TRIAL_QC_READY" in first and "weight_preset" in first
    second = code_with(nb, "selected_review = FourBandReviewWindow(").source
    assert "ensure_fit_settings" in second
    assert "weight_preset" in second
    diagnostics = code_with(nb, "run_stability(").source
    call = next(n for n in ast.walk(ast.parse(diagnostics)) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id == "run_stability")
    assert "n_clusters" not in {kw.arg for kw in call.keywords}  # K comes from fitted model
    assert "NoEligible" not in diagnostics


def test_changed_weight_settings_reject_stale_fit():
    from fourband.session import ensure_fit_settings
    with pytest.raises(ValueError, match="round one"):
        ensure_fit_settings({"weight_preset": "legacy"}, {"weight_preset": "balanced"})


def test_changed_trial_quality_settings_require_fresh_confirmation():
    from fourband.session import ensure_trial_review
    ensure_trial_review(None, None)
    with pytest.raises(ValueError, match="trial quality"):
        ensure_trial_review({"spike_z": 12}, {"spike_z": 10})


def test_legacy_migration_cannot_strip_reliability_controls(tmp_path):
    nb = generator().build_reliability_notebook()
    path = tmp_path / "reliability.ipynb"
    nbformat.write(nb, path)
    generator().update_continuous_cells(path)
    assert "TRIAL_QC_READY" in code_with(nbformat.read(path, as_version=4), "pending_first = fit_joint(").source


def test_disabled_surface_cell_needs_no_explorer_or_fitted_state():
    source = code_with(generator().build_reliability_notebook(), "if RUN_3D_RENDER").source
    exec(source, dict(RUN_3D_RENDER=False, RUN_SELECTABLE_CLUSTER_RENDER=False, SECOND_ROUND_READY=False))
