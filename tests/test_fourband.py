"""Four actual passbands, one joint label, manual review and paired audio."""
import importlib
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3-bands"))


def module(name="analysis"):
    return importlib.import_module("fourband." + name)


def feature_table():
    rows = 16
    table = pd.DataFrame({"channel": [f"E{i}" for i in range(rows)],
                          "is_bad": [False] * (rows - 1) + [True]})
    x = np.r_[np.linspace(-2, -1, 7), np.linspace(1, 2, 7), 70, np.nan]
    for band in ("theta", "beta1", "beta2", "high_gamma"):
        table[f"{band}__early_mean"] = x
        table[f"{band}__audio_slope_corr"] = np.sin(np.arange(rows))
        table[f"qc_{band}_valid_trials"] = np.arange(rows) + 5
    return table


def test_joint_fit_uses_all_bands_and_no_quality_metadata_with_equal_budgets():
    result = module().fit_joint(feature_table(), n_clusters=3)
    assert len(set(result.labels[result.clean_mask])) == 3
    assert result.labels[-1] == -1
    assert not any("qc_" in name for name in result.feature_columns)
    for band in module().DEFAULT_BANDS:
        budget = sum(w for c, w in zip(result.feature_columns, result.feature_weights) if c.startswith(band + "__"))
        assert budget == pytest.approx(1)
    assert result.feature_columns == tuple(c for c in result.feature_columns if c.split("__")[0] in module().DEFAULT_BANDS)


def test_band_weight_budget_is_independent_of_number_of_selected_features():
    table = feature_table()
    table["theta__middle_mean"] = np.cos(np.arange(len(table)) * .7)
    weights = dict(theta=2, beta1=1, beta2=3, high_gamma=4)
    result = module().fit_joint(table, n_clusters=2, band_weights=weights)
    for band, expected in weights.items():
        assert sum(w for c, w in zip(result.feature_columns, result.feature_weights) if c.startswith(band + "__")) == pytest.approx(expected)


def test_constant_or_missing_band_cannot_silently_become_beta_only():
    table = feature_table()
    table["theta__early_mean"] = 1
    table["theta__audio_slope_corr"] = 1
    with pytest.raises(ValueError, match="theta"):
        module().fit_joint(table, n_clusters=2)


@pytest.mark.parametrize("kwargs", [{"band_weights": {"theta": 0}}, {"n_clusters": 1}, {"n_clusters": 99}])
def test_invalid_joint_fit_parameters_fail(kwargs):
    with pytest.raises(ValueError):
        module().fit_joint(feature_table(), **kwargs)


def test_pca_global_distance_detects_a_singleton_outlier():
    result = SimpleNamespace(labels=np.array([0, 0, 0, 1, -1]),
        pca_scores=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 2], [0, 0, 50], [np.nan] * 3]),
        channels=tuple(f"E{i}" for i in range(5)))
    table = module("review").pca_review_table(result)
    assert table.pca_global_distance.iloc[3] == pytest.approx(49)
    assert table.pca_cluster_distance.iloc[3] == 0
    assert np.isnan(table.pca_global_distance.iloc[-1])


def test_second_fit_uses_manual_k_and_refits_every_estimator_after_review():
    table = feature_table()
    first = module().fit_joint(table, n_clusters=3)
    audit = module("review").pca_review_table(first)
    threshold = (audit.pca_global_distance.iloc[14] + audit.pca_global_distance.iloc[:14].max()) / 2
    second, review = module("review").refit_reviewed(table, first, n_clusters=2,
        pca_distance_threshold=threshold, manual_bad_channels=("E0",))
    assert first.labels[0] >= 0
    assert second.labels[0] == second.labels[14] == second.labels[15] == -1
    assert len(set(second.labels[second.clean_mask])) == 2
    assert second.scaler is not first.scaler
    assert second.imputer is not first.imputer
    assert second.kmeans is not first.kmeans
    assert second.pca is not first.pca
    assert review.excluded_manual.iloc[0]
    assert review.excluded_distance.iloc[14]
    assert review.excluded_initial.iloc[15]


@pytest.mark.parametrize("kwargs", [{"manual_bad_channels": ("unknown",)}, {"bad_clusters": (999,)},
                                    {"pca_distance_threshold": -1}, {"distance_reference": "invalid"}])
def test_review_rejects_unknown_or_invalid_selections(kwargs):
    table = feature_table()
    first = module().fit_joint(table, n_clusters=3)
    with pytest.raises(ValueError):
        module("review").refit_reviewed(table, first, n_clusters=2, **kwargs)


def test_four_passbands_are_really_extracted_and_aligned(monkeypatch):
    from test_beta_optimization import raw_epochs
    analysis = module()
    reference = raw_epochs()
    calls = []
    def extract(raw, events, audio, **kwargs):
        calls.append(kwargs["beta_band"])
        return replace(reference, beta_band=kwargs["beta_band"])
    monkeypatch.setattr(analysis, "extract_beta_progress_epochs", extract)
    raw = SimpleNamespace(info={"sfreq": 400})
    result = analysis.extract_fourband(raw, reference.events, None)
    assert calls == [(4, 8), (13, 24), (24, 40), (70, 150)]
    assert len(result.epochs) == 4
    assert all(any(c.startswith(band + "__") for c in result.features) for band in analysis.DEFAULT_BANDS)
    assert all("beta_progress__" not in c for c in result.features)
    assert all("beta_slope" not in c for c in result.features)


def test_four_passbands_reject_nyquist_truncation_before_extraction():
    with pytest.raises(ValueError, match="Nyquist"):
        module().extract_fourband(SimpleNamespace(info={"sfreq": 250}), None, None)


def test_band_trial_mismatch_cannot_silently_mix_different_trials(monkeypatch):
    from test_beta_optimization import raw_epochs
    analysis = module()
    reference = raw_epochs()
    def extract(raw, events, audio, **kwargs):
        if kwargs["beta_band"] == (13, 24):
            reference.events.loc[0, "onset"] += 1
        return replace(reference, beta_band=kwargs["beta_band"], events=reference.events.copy())
    monkeypatch.setattr(analysis, "extract_beta_progress_epochs", extract)
    with pytest.raises(ValueError, match="align"):
        analysis.extract_fourband(SimpleNamespace(info={"sfreq": 400}), reference.events.copy(), None)


def test_any_band_quality_failure_excludes_electrode_in_joint_model(monkeypatch):
    from test_beta_optimization import raw_epochs
    analysis = module()
    reference = raw_epochs()
    def extract(raw, events, audio, **kwargs):
        epochs = replace(reference, beta_band=kwargs["beta_band"])
        if kwargs["beta_band"] == (24, 40):
            epochs = replace(epochs, speech=np.full_like(epochs.speech, np.nan))
        return epochs
    monkeypatch.setattr(analysis, "extract_beta_progress_epochs", extract)
    result = analysis.extract_fourband(SimpleNamespace(info={"sfreq": 400}), reference.events, None)
    assert result.features.is_bad.all()
    assert result.features.exclusion_reason.str.contains("beta2").all()


def test_paired_audio_curves_and_products_use_matching_trials():
    progress = np.linspace(0, 100, 5)
    epochs = SimpleNamespace(speech=np.array([[2 * progress], [np.full(5, np.nan)]]),
        audio_speech=np.array([progress / 100, 1 - progress / 100]),
        progress_percent=progress, events=pd.DataFrame({"duration": [1., 1.]}))
    result = module("plots").summarize_relationship(epochs, sigma_seconds=0, slope_method="gradient")
    np.testing.assert_allclose(result["power"][0], 2 * progress)
    np.testing.assert_allclose(result["audio"][0], progress / 100)
    np.testing.assert_allclose(result["power_slope"][0], 2)
    np.testing.assert_allclose(result["audio_slope"][0], .01)
    np.testing.assert_allclose(result["interaction"][0], .02)
    assert result["valid_trial_count"][0] == 1


def test_plotting_exports_all_four_bands_and_audio_relationships(tmp_path):
    from test_beta_optimization import curve_epochs
    reference = curve_epochs(np.tile(np.sin(np.linspace(0, 6, 101)), (6, 16, 1)))
    result = module().fit_joint(feature_table(), n_clusters=3)
    epochs = {band: replace(reference, channel_names=result.channels) for band in module().DEFAULT_BANDS}
    summary = module("plots").save_round_plots(epochs, result, tmp_path, sigma_seconds=.10)
    assert set(summary.band) == set(module().DEFAULT_BANDS)
    assert (tmp_path / "fourband_cluster_grid.png").is_file()
    assert (tmp_path / "pca_clusters.png").is_file()
    for band in module().DEFAULT_BANDS:
        assert (tmp_path / f"{band}_audio_dynamics.png").is_file()
        assert (tmp_path / f"{band}_audio_dynamics.csv").is_file()


def test_notebook_pyvista_keywords_match_existing_renderer_api():
    import ast
    import inspect
    import json
    import visualization
    path = Path(__file__).resolve().parents[1] / "3-bands/ECoG_4band_clustering_pipeline.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.walk(ast.parse("".join(cell["source"]))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.startswith("render_"):
                signature = inspect.signature(getattr(visualization, node.func.id))
                assert all(k.arg in signature.parameters for k in node.keywords)


def test_round_export_preserves_manual_exclusions_and_requested_k(tmp_path, monkeypatch):
    import json
    from test_beta_optimization import curve_epochs
    analysis, review, workflow = module(), module("review"), module("workflow")
    features = feature_table()
    first = analysis.fit_joint(features, n_clusters=3)
    second, audit = review.refit_reviewed(features, first, n_clusters=2, manual_bad_channels=("E0", "E14"))
    reference = replace(curve_epochs(np.zeros((6, 16, 101))), channel_names=first.channels)
    extraction = analysis.FourBandExtraction(features, {b: reference for b in analysis.DEFAULT_BANDS}, dict(analysis.DEFAULT_BANDS))
    monkeypatch.setattr(workflow, "save_round_plots", lambda *a, **k: None)
    assignments = workflow.save_round(extraction, second, tmp_path / "round2", n_clusters=2, audit=audit)
    assert assignments.label.iloc[0] == assignments.label.iloc[14] == -1
    assert "excluded_manual" in assignments.exclusion_reason.iloc[0]
    summary = json.loads((tmp_path / "round2/summary.json").read_text())
    assert summary["requested_k"] == 2
    assert summary["stability_evaluated"] is False
    assert len(summary["band_weight_budgets"]) == 4
    with pytest.raises(FileExistsError):
        workflow.save_round(extraction, second, tmp_path / "round2", n_clusters=2)


def test_changed_extraction_settings_require_reextracting_not_just_refitting():
    workflow = module("workflow")
    saved = {"bands": {"theta": [4, 8]}, "baseline_mode": "file", "feature_sigma_seconds": .06}
    workflow.ensure_extraction_config(dict(saved), saved)
    for key, value in (("bands", {"theta": [5, 9]}), ("baseline_mode", "pre_onset"), ("feature_sigma_seconds", .1)):
        changed = dict(saved, **{key: value})
        with pytest.raises(ValueError, match=key):
            workflow.ensure_extraction_config(changed, saved)


def test_failed_reextraction_cannot_publish_a_new_config_for_old_features():
    import json
    path = Path(__file__).resolve().parents[1] / "3-bands/ECoG_4band_clustering_pipeline.ipynb"
    source = next("".join(c["source"]) for c in json.loads(path.read_text(encoding="utf-8"))["cells"]
        if c["cell_type"] == "code" and "pending_settings = extraction_settings()" in "".join(c["source"]))
    old = {"bands": {"theta": (4, 8)}}
    def fail(*args, **kwargs):
        raise RuntimeError("Failed extraction")
    ns = dict(extraction_settings=lambda: {"bands": {"theta": (5, 9)}}, extracted_settings=old,
              load_fourband=fail, close_explorers=lambda value: None, EXPLORERS={})
    for name in ("FIF_PATH", "ONSET_PATH", "OFFSET_PATH", "WAV_PATH", "BANDS", "PROGRESS_POINTS",
                 "DURATION_MAD_THRESHOLD", "MIN_TRIALS_AFTER_FILTER", "PRE_POST_WINDOW_SECONDS",
                 "BASELINE_MODE", "BASELINE_WINDOW_SECONDS", "BASELINE_FIF", "BASELINE_SEGMENTS_CSV",
                 "BASELINE_EDGE_SECONDS", "BASELINE_RELATIVE_FLOOR", "FEATURE_SMOOTH_SIGMA_SECONDS",
                 "FEATURE_SLOPE_METHOD", "MAX_BASELINE_INVALID_FRACTION"):
        ns[name] = None
    with pytest.raises(RuntimeError, match="Failed extraction"):
        exec(source, ns)
    assert ns["extracted_settings"] is old
    with pytest.raises(ValueError, match="bands"):
        module("workflow").ensure_extraction_config(ns["extraction_settings"](), ns["extracted_settings"])


def test_round_two_requires_first_round_from_the_same_extraction():
    workflow = module("workflow")
    original, newer = object(), object()
    workflow.ensure_first_round(original, original)
    with pytest.raises(ValueError, match="round one"):
        workflow.ensure_first_round(original, newer)


def test_relationship_export_distinguishes_mean_curve_from_paired_trial_products(tmp_path, monkeypatch):
    from test_beta_optimization import curve_epochs
    analysis, workflow = module(), module("workflow")
    features = feature_table()
    wave = np.sin(np.linspace(0, 2 * np.pi, 101))
    polarity = np.array([1, -1, 1, -1])
    speech = np.broadcast_to(polarity[:, None, None] * wave, (4, 16, 101)).copy()
    reference = replace(curve_epochs(speech), audio_speech=2 + polarity[:, None] * wave)
    result = analysis.fit_joint(features, n_clusters=3)
    reference = replace(reference, channel_names=result.channels)
    for band in analysis.DEFAULT_BANDS:
        features[f"{band}__slope_product_mean"] = 0.0
    extraction = analysis.FourBandExtraction(features, {b: reference for b in analysis.DEFAULT_BANDS}, dict(analysis.DEFAULT_BANDS))
    monkeypatch.setattr(workflow, "save_round_plots", lambda *a, **k: None)
    workflow.save_round(extraction, result, tmp_path / "round", n_clusters=3, sigma_seconds=0, slope_method="gradient")
    table = pd.read_csv(tmp_path / "round/electrode_audio_relationship.csv")
    assert table.theta__feature_mean_curve__slope_product_mean.iloc[0] == 0
    assert table.theta__display_paired_trial_slope_product_mean.iloc[0] > 0
    cache = module("plots").summarize_relationship(reference, sigma_seconds=0, slope_method="gradient")
    assert table.theta__display_paired_trial_slope_product_mean.iloc[0] == pytest.approx(cache["interaction"][0].mean())
