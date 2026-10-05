"""Four-band review settings, real Tk controls, and notebook cancellation."""
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3-bands"))


def module():
    return importlib.import_module("fourband.review_window")


def first_model():
    return SimpleNamespace(channels=tuple(f"E{i}" for i in range(6)),
        labels=np.array([0, 0, 1, 1, 2, -1]), clean_mask=np.array([True] * 5 + [False]),
        pca_scores=np.array([[0, 0, 0], [.1, .1, 0], [.2, 0, 0], [.5, 0, 0], [9, 0, 0], [np.nan] * 3]))


def test_review_preview_unions_exclusions_and_preserves_initial_bads():
    settings, audit = module().parse_review_settings(first_model(), bad_clusters=(2,), manual_channels="E0; E0",
        n_clusters="2", distance_enabled=True, distance_threshold="3", distance_reference="global")
    assert settings.manual_bad_channels == ("E0",)
    assert settings.pca_distance_threshold == 3
    assert settings.n_clusters == 2
    np.testing.assert_array_equal(audit.excluded_final, [True, False, False, False, True, True])
    assert audit.excluded_distance.iloc[4] and audit.excluded_cluster.iloc[4]


def test_disabled_distance_ignores_unfinished_threshold_text():
    settings, audit = module().parse_review_settings(first_model(), n_clusters="2",
        distance_enabled=False, distance_threshold="not-a-number")
    assert settings.pca_distance_threshold is None
    assert not audit.excluded_distance.any()


@pytest.mark.parametrize("options", [dict(n_clusters="1"), dict(n_clusters="2.5"), dict(n_clusters="6"),
    dict(n_clusters="nan"), dict(manual_channels="Unknown"), dict(bad_clusters=(99,)),
    dict(distance_reference="unknown"), dict(distance_enabled=True, distance_threshold=""),
    dict(distance_enabled=True, distance_threshold="nan"), dict(distance_enabled=True, distance_threshold="-1"),
    dict(distance_enabled=True, distance_threshold="inf"), dict(manual_channels="E0,E1,E2,E3", n_clusters="2")])
def test_review_rejects_invalid_settings_before_acceptance(options):
    with pytest.raises(ValueError):
        module().parse_review_settings(first_model(), **dict(dict(n_clusters="2"), **options))


def test_settings_map_to_existing_refit_contract_and_cluster_reference():
    settings, audit = module().parse_review_settings(first_model(), n_clusters="2",
        distance_enabled=True, distance_threshold="3", distance_reference="cluster")
    assert not audit.excluded_distance.any()  # singleton is its own centroid
    assert settings.refit_kwargs() == dict(n_clusters=2, bad_clusters=(), manual_bad_channels=(),
        pca_distance_threshold=3, distance_reference="cluster")


def test_native_popup_checks_in_isolated_process_without_fd_capture():
    import subprocess
    worker = Path(__file__).with_name("fourband_review_gui_checks.py")
    # Tcl's native IO must not share pytest's fd-capture lifecycle on Windows.
    result = subprocess.run([sys.executable, "-m", "pytest", str(worker), "-q", "--capture=sys"],
        cwd=worker.parents[1], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "5 passed" in result.stdout


def notebook_cells():
    path = Path(__file__).resolve().parents[1] / "3-bands/ECoG_4band_clustering_pipeline.ipynb"
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("same_model", [False, True])
def test_notebook_only_fits_and_exports_after_accepted_popup(tmp_path, accepted, same_model):
    from test_fourband import feature_table
    from fourband.analysis import fit_joint, DEFAULT_BANDS
    from fourband.review import refit_reviewed
    from fourband.workflow import ensure_extraction_config, ensure_first_round
    from fourband.session import ensure_fit_settings, ensure_trial_review, close_explorers
    features = feature_table()
    first = fit_joint(features, n_clusters=3)
    extraction = object()
    calls = []
    selection = module().ReviewSettings(n_clusters=2, manual_bad_channels=("E0",))
    class Popup:
        def __init__(self, model, *, initial):
            assert model is first
            assert initial == (selection if same_model else module().ReviewSettings(n_clusters=4))
        def show(self):
            calls.append("popup")
            return selection if accepted else None
    def fit(*args, **kwargs):
        calls.append("fit")
        assert kwargs["n_clusters"] == 2 and kwargs["manual_bad_channels"] == ("E0",)
        return refit_reviewed(*args, **kwargs)
    def save(extraction, result, path, **kwargs):
        calls.append("save")
        assert kwargs["round_config"]["manual_bad_channels"] == ("E0",)
        return result.to_frame()
    source = next("".join(c["source"]) for c in notebook_cells()
        if c["cell_type"] == "code" and "selected_bad_clusters =" in "".join(c["source"]))
    ns = dict(extraction_settings=lambda: {}, extracted_settings={}, ensure_extraction_config=ensure_extraction_config,
        ensure_first_round=ensure_first_round, first_extraction=extraction, extraction=extraction, features=features,
        first_result=first, OPEN_REVIEW_WINDOW=True, review_settings=selection, review_model=first if same_model else object(),
        FourBandReviewWindow=Popup, refit_reviewed=fit, save_round=save, ReviewSettings=module().ReviewSettings,
        RUN_DIR=tmp_path, ROUND1_DIR=tmp_path / "round1", timestamp=lambda: "second", BANDS=DEFAULT_BANDS,
        BAND_WEIGHTS={b: 1 for b in DEFAULT_BANDS}, FEATURE_CORRELATION_THRESHOLD=.95, RANDOM_STATE=50,
        DISPLAY_SMOOTH_SIGMA_SECONDS=.1, DISPLAY_SFREQ=100, FEATURE_SLOPE_METHOD="savgol", SECOND_N_CLUSTERS=4,
        display=lambda value: None, Image=lambda **kw: None, SECOND_ROUND_READY=True,
        close_explorers=close_explorers, EXPLORERS={}, TRIAL_QC_READY=True,
        ensure_fit_settings=ensure_fit_settings, fit_settings=lambda: {}, first_fit_settings={"first_k": 3}, FIRST_N_CLUSTERS=3,
        WEIGHT_PRESET="legacy", GROUP_WEIGHTS={}, ensure_trial_review=ensure_trial_review,
        TRIAL_QC_SETTINGS=None, reviewed_trial_settings=None)
    exec(compile(source, "round_two_cell", "exec"), ns)
    assert ns["SECOND_ROUND_READY"] is accepted
    assert calls == (["popup", "fit", "save"] if accepted else ["popup"])
    if accepted:
        assert ns["second_result"].labels[0] == -1
        assert ns["selected_second_k"] == 2
    else:
        assert "ROUND2_DIR" not in ns


def test_notebook_defaults_to_popup_and_anatomy_is_gated_on_completed_second_round():
    sources = ["".join(c["source"]) for c in notebook_cells() if c["cell_type"] == "code"]
    assert any("OPEN_REVIEW_WINDOW = True" in s for s in sources)
    assert any("RUN_3D_RENDER and SECOND_ROUND_READY" in s for s in sources)
    assert any("RUN_SELECTABLE_CLUSTER_RENDER and SECOND_ROUND_READY" in s for s in sources)


def test_notebook_guard_failure_invalidates_old_round_two_ready_flag():
    source = next("".join(c["source"]) for c in notebook_cells()
        if c["cell_type"] == "code" and "selected_bad_clusters =" in "".join(c["source"]))
    def fail(*args):
        raise ValueError("Extraction settings changed")
    ns = dict(SECOND_ROUND_READY=True, ensure_extraction_config=fail, extraction_settings=lambda: {}, extracted_settings={})
    with pytest.raises(ValueError, match="Extraction settings changed"):
        exec(source, ns)
    assert ns["SECOND_ROUND_READY"] is False
