"""Regression tests for descriptive, independently refitted stability metrics."""

from dataclasses import fields, replace
import importlib
import importlib.util

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import adjusted_rand_score, silhouette_score

from audio_cross_3band_utils import BetaProgressEpochs
import clustering_pipeline


FEATURES = ("beta_progress__energy_mean", "beta_progress__post_mean")


def test_independent_evaluation_module_is_available():
    assert importlib.util.find_spec("cluster_evaluation") is not None, (
        "Task 3 requires the independent cluster_evaluation module"
    )


@pytest.fixture
def evaluator():
    assert importlib.util.find_spec("cluster_evaluation") is not None, (
        "Task 3 requires the independent cluster_evaluation module"
    )
    module = importlib.import_module("cluster_evaluation")
    assert callable(getattr(module, "evaluate_beta_cluster_stability", None))
    return module


@pytest.fixture
def epochs():
    rng = np.random.default_rng(12)
    trials, channels = 8, 12
    progress = np.linspace(0, 100, 101)
    signs = np.repeat([1.0, -1.0], 6)
    response = np.exp(-0.5 * ((progress - 45) / 15) ** 2)
    amplitudes = signs * (3 + 0.03 * (np.arange(channels) % 6))
    speech = amplitudes[None, :, None] * response[None, None, :]
    speech = speech + rng.normal(0, 0.01, (trials, channels, len(progress)))
    pre = rng.normal(0, 0.01, (trials, channels, 21))
    post = np.broadcast_to(amplitudes[None, :, None], (trials, channels, 21)).copy()
    post += rng.normal(0, 0.01, post.shape)
    names = tuple(f"E{i}" for i in range(channels))
    trial_ids = np.arange(trials) + 100
    kwargs = {}
    if "baseline_diagnostics" in {field.name for field in fields(BetaProgressEpochs)}:
        kwargs["baseline_diagnostics"] = pd.DataFrame({
            "original_trial_id": np.repeat(trial_ids, channels),
            "channel": np.tile(names, trials),
            "baseline_mean": 0.0,
            "baseline_std": 1.0,
            "denominator_floor": 0.01,
            "is_low_variance": False,
            "valid": True,
        })
    return BetaProgressEpochs(
        speech=speech, pre=pre, post=post,
        audio_speech=np.tile(response, (trials, 1)),
        audio_pre=np.zeros((trials, 21)), audio_post=np.zeros((trials, 21)),
        progress_percent=progress, pre_times=np.linspace(-0.5, 0, 21),
        post_times=np.linspace(0, 0.5, 21),
        events=pd.DataFrame({"duration": np.linspace(1.5, 2.5, trials),
                             "original_trial_id": trial_ids}, index=np.arange(trials)[::-1]),
        dropped_events=pd.DataFrame({"original_trial_id": [99]}),
        channel_names=names, bad_channels=frozenset(), sfreq=100.0,
        beta_band=(13.0, 30.0), pre_post_window=0.5, **kwargs,
    )


def evaluate(evaluator, epochs, **kwargs):
    options = dict(k_values=(2, 3), n_resamples=3, feature_columns=FEATURES)
    options.update(kwargs)
    return evaluator.evaluate_beta_cluster_stability(epochs, **options)


def test_clear_groups_recommend_two_and_report_descriptive_metrics(evaluator, epochs):
    result = evaluate(evaluator, epochs)
    assert isinstance(result, evaluator.ClusterStabilityEvaluation)
    assert result.recommended_k == 2
    assert result.summary["k"].tolist() == [2, 3]
    row = result.summary.set_index("k").loc[2]
    assert row.silhouette > 0.95
    assert row.split_half_ari_mean == pytest.approx(1)
    assert row.bootstrap_ari_mean == pytest.approx(1)
    assert row.split_half_ari_std == pytest.approx(0)
    assert row.bootstrap_ari_std == pytest.approx(0)
    assert row.min_cluster_size == 6
    assert row.waveform_correlation_mean > 0.99
    assert row.split_half_valid_count == row.bootstrap_valid_count == 3
    assert row.selection_rank == 1
    assert len(result.resamples) == 6


def test_fixed_seed_reproduces_tables_and_recommendation(evaluator, epochs):
    first = evaluate(evaluator, epochs, random_state=24)
    second = evaluate(evaluator, epochs, random_state=24)
    pd.testing.assert_frame_equal(first.summary, second.summary)
    pd.testing.assert_frame_equal(first.resamples, second.resamples)
    assert first.recommended_k == second.recommended_k


def test_label_permutations_do_not_change_ari(evaluator, epochs, monkeypatch):
    real_fit = evaluator.cluster_electrodes
    calls = []

    def permuting_fit(*args, **kwargs):
        fit = real_fit(*args, **kwargs)
        labels = fit.labels.copy()
        if len(calls) % 2:
            labels[fit.clean_mask] = kwargs["n_clusters"] - 1 - labels[fit.clean_mask]
        calls.append(fit)
        return replace(fit, labels=labels)

    monkeypatch.setattr(evaluator, "cluster_electrodes", permuting_fit)
    result = evaluate(evaluator, epochs, k_values=(2,))
    np.testing.assert_allclose(result.resamples.split_half_ari, 1)
    np.testing.assert_allclose(result.resamples.bootstrap_ari, 1)


def test_refits_are_independent_and_silhouette_uses_weighted_features(evaluator, epochs, monkeypatch):
    real_fit = evaluator.cluster_electrodes
    fits = []

    def recording_fit(*args, **kwargs):
        fit = real_fit(*args, **kwargs)
        fits.append(fit)
        return fit

    monkeypatch.setattr(evaluator, "cluster_electrodes", recording_fit)
    weights = {FEATURES[0]: 4.0, FEATURES[1]: 0.2}
    result = evaluate(evaluator, epochs, k_values=(2,), feature_weights=weights)
    assert len(fits) == 10
    for attribute in ("imputer", "scaler", "kmeans"):
        assert len({id(getattr(fit, attribute)) for fit in fits}) == 10
    expected = silhouette_score(fits[0].weighted_clean_features, fits[0].labels[fits[0].clean_mask])
    assert result.summary.iloc[0].silhouette == pytest.approx(expected)


def test_trial_slices_cache_features_across_k_and_preserve_diagnostics(evaluator, epochs, monkeypatch):
    real_extract = evaluator.extract_beta_progress_features
    samples = []

    def recording_extract(sample, **kwargs):
        samples.append(sample)
        positions = sample.events.original_trial_id.to_numpy() - 100
        for name in ("speech", "pre", "post", "audio_speech", "audio_pre", "audio_post"):
            np.testing.assert_array_equal(getattr(sample, name), getattr(epochs, name)[positions])
        if getattr(sample, "baseline_diagnostics", None) is not None:
            counts = sample.baseline_diagnostics.groupby("original_trial_id").size()
            expected = sample.events.original_trial_id.value_counts().sort_index() * 12
            pd.testing.assert_series_equal(counts, expected, check_names=False)
        return real_extract(sample, **kwargs)

    monkeypatch.setattr(evaluator, "extract_beta_progress_features", recording_extract)
    before = epochs.speech.copy()
    events_before = epochs.events.copy(deep=True)
    result = evaluate(evaluator, epochs)
    assert len(samples) == 10  # Full data plus two halves and a bootstrap per replicate.
    for _, row in result.resamples.iterrows():
        left, right, bootstrap = row.half_a_trials, row.half_b_trials, row.bootstrap_trials
        assert len(left) == len(right) == 4
        assert set(left).isdisjoint(right)
        assert set(left) | set(right) == set(range(8))
        assert len(bootstrap) == 8
    for _, group in result.resamples.groupby("resample"):
        assert group.half_a_trials.nunique() == group.half_b_trials.nunique() == 1
        assert group.bootstrap_trials.nunique() == 1
    assert any(len(set(row.bootstrap_trials)) < 8 for _, row in result.resamples.iterrows())
    np.testing.assert_array_equal(epochs.speech, before)
    pd.testing.assert_frame_equal(epochs.events, events_before)


def test_common_mask_excludes_fif_supplied_and_all_invalid_channels(evaluator, epochs, monkeypatch):
    speech = epochs.speech.copy()
    speech[:, 2, :] = np.nan
    unreliable = replace(epochs, speech=speech, bad_channels=frozenset({"E0"}))
    supplied = np.zeros(12, dtype=bool)
    supplied[1] = True
    real_fit = evaluator.cluster_electrodes
    masks = []

    def recording_fit(*args, **kwargs):
        fit = real_fit(*args, **kwargs)
        masks.append(fit.clean_mask)
        return fit

    monkeypatch.setattr(evaluator, "cluster_electrodes", recording_fit)
    result = evaluate(evaluator, unreliable, k_values=(2,), bad_mask=supplied,
                      feature_kwargs={"mask_bad_features": False})
    assert result.summary.iloc[0].n_clean_electrodes == 9
    for mask in masks:
        np.testing.assert_array_equal(mask, [False] * 3 + [True] * 9)
    assert result.summary.iloc[0].min_cluster_size == 3


def test_original_unreliable_electrode_is_excluded_before_resampling(evaluator, epochs, monkeypatch):
    speech = epochs.speech.copy()
    speech[2:, 0, :] = np.nan
    real_fit = evaluator.cluster_electrodes
    masks = []

    def recording_fit(*args, **kwargs):
        fit = real_fit(*args, **kwargs)
        masks.append(fit.clean_mask)
        return fit

    monkeypatch.setattr(evaluator, "cluster_electrodes", recording_fit)
    result = evaluate(evaluator, replace(epochs, speech=speech), k_values=(2,),
                      n_resamples=12, random_state=6)
    assert result.summary.iloc[0].n_clean_electrodes == 11
    assert all(not mask[0] for mask in masks)
    assert all(np.array_equal(mask, masks[0]) for mask in masks)


def test_original_good_membership_does_not_change_with_resample_count(evaluator, epochs, monkeypatch):
    speech = epochs.speech.copy()
    speech[0, 0, :] = np.nan  # 7/8 valid trials: original QC passes the 25% threshold.
    sample = replace(epochs, speech=speech)
    real_fit = evaluator.cluster_electrodes
    masks = []

    def recording_fit(*args, **kwargs):
        fit = real_fit(*args, **kwargs)
        masks.append(fit.clean_mask)
        return fit

    monkeypatch.setattr(evaluator, "cluster_electrodes", recording_fit)
    once = evaluate(evaluator, sample, k_values=(2,), n_resamples=1, random_state=50)
    repeated = evaluate(evaluator, sample, k_values=(2,), n_resamples=20, random_state=50)
    np.testing.assert_array_equal(once.clean_mask, np.ones(12, dtype=bool))
    np.testing.assert_array_equal(repeated.clean_mask, once.clean_mask)
    assert all(np.array_equal(mask, once.clean_mask) for mask in masks)
    unsupported = repeated.resamples.loc[~repeated.resamples.bootstrap_supported]
    assert not unsupported.empty
    assert unsupported.bootstrap_ari.isna().all()
    assert unsupported.bootstrap_reason.str.contains("E0").all()
    assert repeated.summary.iloc[0].n_clean_electrodes == 12
    assert repeated.summary.iloc[0].bootstrap_valid_count == repeated.resamples.bootstrap_supported.sum()


@pytest.mark.parametrize("target", ["half_a", "half_b", "bootstrap"])
@pytest.mark.parametrize("problem", ["is_bad", "wholly_missing"])
def test_unsupported_sample_is_skipped_without_changing_other_fits(
    evaluator, epochs, monkeypatch, target, problem,
):
    real_extract = evaluator.extract_beta_progress_features
    real_fit = evaluator.cluster_electrodes
    extraction_count = 0
    fit_masks = []
    target_call = {"half_a": 1, "half_b": 2, "bootstrap": 3}[target]

    def unavailable_features(sample, **kwargs):
        nonlocal extraction_count
        table = real_extract(sample, **kwargs)
        if extraction_count == target_call:
            if problem == "is_bad":
                table.loc[table.channel == "E0", "is_bad"] = True
            else:
                table.loc[table.channel == "E0", list(FEATURES)] = np.nan
        extraction_count += 1
        return table

    def recording_fit(table, **kwargs):
        assert not table.loc[table.channel == "E0", "is_bad"].any()
        assert table.loc[table.channel == "E0", list(FEATURES)].notna().any(axis=1).all()
        fit = real_fit(table, **kwargs)
        fit_masks.append(fit.clean_mask)
        return fit

    monkeypatch.setattr(evaluator, "extract_beta_progress_features", unavailable_features)
    monkeypatch.setattr(evaluator, "cluster_electrodes", recording_fit)
    result = evaluate(evaluator, epochs, k_values=(2,), n_resamples=5)
    np.testing.assert_array_equal(result.clean_mask, np.ones(12, dtype=bool))
    assert len(fit_masks) == 15  # Reference plus 14 supported samples, not a reduced population fit.
    assert all(np.array_equal(mask, result.clean_mask) for mask in fit_masks)
    first = result.resamples.iloc[0]
    assert not first[f"{target}_supported"]
    reason = "sample_is_bad" if problem == "is_bad" else "wholly_missing_features"
    assert reason in first[f"{target}_reason"] and "E0" in first[f"{target}_reason"]
    metric = "bootstrap" if target == "bootstrap" else "split_half"
    assert pd.isna(first[f"{metric}_ari"])
    assert result.summary.iloc[0][f"{metric}_valid_count"] == 4
    for other in {"half_a", "half_b", "bootstrap"} - {target}:
        assert first[f"{other}_supported"]
    assert result.recommended_k == 2


def test_qc_diagnostics_invalid_fraction_uses_original_threshold(evaluator, epochs):
    quality = epochs.baseline_diagnostics.copy()
    quality.loc[(quality.channel == "E0") & (quality.original_trial_id < 103), "valid"] = False
    sample = replace(epochs, baseline_diagnostics=quality)
    strict = evaluate(evaluator, sample, k_values=(2,), n_resamples=1)
    np.testing.assert_array_equal(strict.clean_mask, [False] + [True] * 11)
    lenient = evaluate(evaluator, sample, k_values=(2,), n_resamples=5,
                       feature_kwargs={"max_baseline_invalid_fraction": 0.75})
    np.testing.assert_array_equal(lenient.clean_mask, np.ones(12, dtype=bool))


def test_baseline_quality_excludes_unreliable_finite_channels(evaluator, epochs):
    if getattr(epochs, "baseline_diagnostics", None) is None:
        pytest.skip("Parent baseline diagnostics API is not yet available")
    quality = epochs.baseline_diagnostics.copy()
    quality.loc[quality.channel == "E0", "valid"] = False
    result = evaluate(evaluator, replace(epochs, baseline_diagnostics=quality), k_values=(2,))
    assert result.summary.iloc[0].n_clean_electrodes == 11


def test_default_selection_uses_parent_selector_and_never_qc(evaluator, epochs, monkeypatch):
    real_extract = evaluator.extract_beta_progress_features
    real_fit = evaluator.cluster_electrodes
    columns_seen = []

    def with_quality(sample, **kwargs):
        table = real_extract(sample, **kwargs)
        table["baseline_invalid_fraction"] = np.arange(12) / 12
        return table

    def selector(table, *, bad_mask=None, correlation_threshold=0.95):
        return FEATURES

    def recording_fit(*args, **kwargs):
        columns_seen.append(tuple(kwargs["feature_columns"]))
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(evaluator, "extract_beta_progress_features", with_quality)
    monkeypatch.setattr(clustering_pipeline, "select_beta_progress_feature_columns", selector, raising=False)
    monkeypatch.setattr(evaluator, "cluster_electrodes", recording_fit)
    result = evaluate(evaluator, epochs, feature_columns=None, k_values=(2,))
    assert result.recommended_k == 2
    assert all(columns == FEATURES for columns in columns_seen)


def test_default_fallback_does_not_cluster_quality_metadata(evaluator, epochs, monkeypatch):
    real_extract = evaluator.extract_beta_progress_features
    real_fit = evaluator.cluster_electrodes

    def with_quality(sample, **kwargs):
        table = real_extract(sample, **kwargs)
        table["baseline_invalid_fraction"] = np.arange(12) / 12
        return table

    def checking_fit(*args, **kwargs):
        assert "baseline_invalid_fraction" not in kwargs["feature_columns"]
        return real_fit(*args, **kwargs)

    monkeypatch.delattr(clustering_pipeline, "select_beta_progress_feature_columns", raising=False)
    monkeypatch.setattr(evaluator, "extract_beta_progress_features", with_quality)
    monkeypatch.setattr(evaluator, "cluster_electrodes", checking_fit)
    evaluate(evaluator, epochs, feature_columns=None, k_values=(2,), n_resamples=1)


def test_singleton_k_is_flagged_and_cannot_win_recommendation(evaluator, epochs):
    result = evaluate(evaluator, epochs, k_values=(2, 11), n_resamples=1)
    row = result.summary.set_index("k").loc[11]
    assert row.min_cluster_size == 1
    assert row.has_tiny_cluster
    assert not row.eligible
    assert pd.isna(row.selection_rank)
    replicates = result.resamples.loc[result.resamples.k == 11]
    assert replicates.split_half_ari.notna().all()
    assert replicates.bootstrap_ari.notna().all()
    assert result.recommended_k == 2


def test_leave_one_out_waveform_correlation_avoids_self_inflation(evaluator, epochs):
    progress = epochs.progress_percent / 100
    curves = np.stack([np.sin(2 * np.pi * progress), np.cos(2 * np.pi * progress),
                       np.sin(4 * np.pi * progress), np.cos(4 * np.pi * progress)])
    speech = np.tile(curves[None], (8, 1, 1))
    pre = np.zeros((8, 4, 21))
    post = np.broadcast_to(np.array([0, 0.01, 4, 4.01])[None, :, None], (8, 4, 21)).copy()
    sample = replace(epochs, speech=speech, pre=pre, post=post,
                     channel_names=epochs.channel_names[:4], baseline_diagnostics=None)
    result = evaluate(evaluator, sample, k_values=(2,), n_resamples=1,
                      feature_columns=(FEATURES[1],))
    assert abs(result.summary.iloc[0].waveform_correlation_mean) < 0.03


@pytest.mark.parametrize("options,match", [
    ({"n_resamples": 0}, "n_resamples"),
    ({"n_resamples": 1.5}, "n_resamples"),
    ({"k_values": []}, "k_values"),
    ({"k_values": [1]}, "k_values"),
    ({"k_values": [12]}, "clean"),
    ({"k_values": [2.5]}, "k_values"),
    ({"bad_mask": [False]}, "bad_mask"),
    ({"feature_columns": []}, "feature_columns"),
    ({"feature_columns": ["missing_feature"]}, "Unknown feature"),
    ({"feature_columns": ["baseline_invalid_fraction"]}, "metadata"),
    ({"feature_columns": ["valid_trial_count"]}, "metadata"),
])
def test_invalid_arguments_fail_clearly(evaluator, epochs, options, match):
    with pytest.raises(ValueError, match=match):
        evaluate(evaluator, epochs, **options)


def test_fewer_than_four_trials_rejected_before_fitting(evaluator, epochs):
    sample = replace(epochs, speech=epochs.speech[:3], pre=epochs.pre[:3], post=epochs.post[:3],
                     audio_speech=epochs.audio_speech[:3], audio_pre=epochs.audio_pre[:3],
                     audio_post=epochs.audio_post[:3], events=epochs.events.iloc[:3])
    with pytest.raises(ValueError, match="at least 4 trials"):
        evaluate(evaluator, sample)


def test_four_trials_use_two_per_disjoint_half(evaluator, epochs):
    sample = replace(epochs, speech=epochs.speech[:4], pre=epochs.pre[:4], post=epochs.post[:4],
                     audio_speech=epochs.audio_speech[:4], audio_pre=epochs.audio_pre[:4],
                     audio_post=epochs.audio_post[:4], events=epochs.events.iloc[:4])
    result = evaluate(evaluator, sample, k_values=(2,), n_resamples=1)
    row = result.resamples.iloc[0]
    assert len(row.half_a_trials) == len(row.half_b_trials) == 2
    assert set(row.half_a_trials).isdisjoint(row.half_b_trials)


def test_all_candidates_with_singletons_cannot_produce_a_recommendation(evaluator, epochs):
    with pytest.raises(ValueError, match="No eligible") as caught:
        evaluate(evaluator, epochs, k_values=(11,), n_resamples=1)
    assert not caught.value.summary.eligible.any()
    assert caught.value.summary.has_tiny_cluster.all()
    assert len(caught.value.resamples) == 1
    assert caught.value.clean_mask.sum() == 12


def test_collapsed_clusters_are_invalid_replicates(evaluator, epochs):
    speech = np.tile(epochs.speech[:, :1], (1, 12, 1))
    pre = np.tile(epochs.pre[:, :1], (1, 12, 1))
    post = np.tile(epochs.post[:, :1], (1, 12, 1))
    with pytest.warns(Warning), pytest.raises(ValueError, match="No eligible"):
        evaluate(evaluator, replace(epochs, speech=speech, pre=pre, post=post),
                 k_values=(2,), n_resamples=1)


def test_instability_compares_halves_to_each_other_and_bootstrap_to_reference(evaluator, epochs, monkeypatch):
    rng = np.random.default_rng(324)
    levels = rng.normal(size=(8, 12))
    post = np.repeat(levels[:, :, None], 21, axis=2)
    sample = replace(epochs, post=post)
    real_fit = evaluator.cluster_electrodes
    fits = []

    def recording_fit(*args, **kwargs):
        fit = real_fit(*args, **kwargs)
        fits.append(fit)
        return fit

    monkeypatch.setattr(evaluator, "cluster_electrodes", recording_fit)
    result = evaluate(evaluator, sample, k_values=(2,), n_resamples=8,
                      feature_columns=(FEATURES[1],))
    for replicate, row in result.resamples.iterrows():
        left, right, boot = fits[1 + 3 * replicate:4 + 3 * replicate]
        if row.split_half_valid:
            assert row.split_half_ari == pytest.approx(adjusted_rand_score(left.labels, right.labels))
        if row.bootstrap_valid:
            assert row.bootstrap_ari == pytest.approx(adjusted_rand_score(fits[0].labels, boot.labels))
    assert result.resamples.split_half_ari.min() < 0.8
    assert result.summary.iloc[0].split_half_ari_std > 0
    assert result.summary.iloc[0].bootstrap_ari_std > 0


def test_default_api_selects_parent_morphology_features(evaluator, epochs):
    result = evaluator.evaluate_beta_cluster_stability(epochs, n_resamples=1)
    assert result.summary.k.tolist() == [2, 3, 4, 5, 6]
    assert result.resamples.k.tolist() == [2, 3, 4, 5, 6]
    assert "baseline_invalid_fraction" not in result.summary.attrs["feature_columns"]
    assert "valid_trial_count" not in result.summary.attrs["feature_columns"]
    assert "beta_progress__early_mean" in result.summary.attrs["feature_columns"]
    assert result.recommended_k in result.summary.loc[result.summary.eligible, "k"].tolist()


def test_feature_selection_excludes_electrodes_missing_only_selected_predictors(evaluator, epochs, monkeypatch):
    real_extract = evaluator.extract_beta_progress_features

    def missing_predictors(sample, **kwargs):
        table = real_extract(sample, **kwargs)
        table.loc[table.channel == "E0", list(FEATURES)] = np.nan
        return table

    monkeypatch.setattr(evaluator, "extract_beta_progress_features", missing_predictors)
    result = evaluate(evaluator, epochs, k_values=(2,), n_resamples=1)
    assert result.summary.iloc[0].n_clean_electrodes == 11
    assert result.summary.attrs["excluded_channels"] == ("E0",)


def test_exposed_clean_mask_aligns_notebook_final_fit_with_evaluation(evaluator, epochs):
    supplied_bad = np.zeros(12, dtype=bool)
    supplied_bad[1] = True
    sample = replace(epochs, bad_channels=frozenset({"E0"}))
    result = evaluate(evaluator, sample, bad_mask=supplied_bad, k_values=(2,), n_resamples=1)
    assert fields(result)[-1].name == "clean_mask"
    assert result.clean_mask.dtype == np.bool_
    np.testing.assert_array_equal(result.clean_mask, [False, False] + [True] * 10)
    table = evaluator.extract_beta_progress_features(sample)
    final_fit = clustering_pipeline.cluster_electrodes(
        table, n_clusters=result.recommended_k, bad_mask=~result.clean_mask,
        feature_columns=result.summary.attrs["feature_columns"],
    )
    np.testing.assert_array_equal(final_fit.clean_mask, result.clean_mask)
    assert (final_fit.labels[~result.clean_mask] == -1).all()


def test_tiny_complete_resamples_contribute_to_ari_means(evaluator, epochs, monkeypatch):
    levels = np.random.default_rng(324).normal(size=(8, 12))
    sample = replace(epochs, post=np.repeat(levels[:, :, None], 21, axis=2))
    real_fit = evaluator.cluster_electrodes
    fits = []

    def recording_fit(*args, **kwargs):
        fit = real_fit(*args, **kwargs)
        fits.append(fit)
        return fit

    monkeypatch.setattr(evaluator, "cluster_electrodes", recording_fit)
    result = evaluate(evaluator, sample, k_values=(2,), n_resamples=8,
                      feature_columns=(FEATURES[1],))
    assert (result.resamples.split_half_min_cluster_size == 1).any()
    assert (result.resamples.bootstrap_min_cluster_size == 1).any()
    expected_split, expected_bootstrap = [], []
    for replicate in range(8):
        left, right, boot = fits[1 + 3 * replicate:4 + 3 * replicate]
        expected_split.append(adjusted_rand_score(left.labels, right.labels))
        expected_bootstrap.append(adjusted_rand_score(fits[0].labels, boot.labels))
    np.testing.assert_allclose(result.resamples.split_half_ari, expected_split)
    np.testing.assert_allclose(result.resamples.bootstrap_ari, expected_bootstrap)
    summary = result.summary.iloc[0]
    assert summary.split_half_ari_mean == pytest.approx(np.mean(expected_split))
    assert summary.bootstrap_ari_mean == pytest.approx(np.mean(expected_bootstrap))
    assert summary.split_half_ari_std == pytest.approx(np.std(expected_split, ddof=1))
    assert summary.bootstrap_ari_std == pytest.approx(np.std(expected_bootstrap, ddof=1))
    assert summary.split_half_valid_count == summary.bootstrap_valid_count == 8
    np.testing.assert_array_equal(result.resamples.split_half_has_tiny_cluster,
                                  result.resamples.split_half_min_cluster_size < 2)
    np.testing.assert_array_equal(result.resamples.bootstrap_has_tiny_cluster,
                                  result.resamples.bootstrap_min_cluster_size < 2)


@pytest.mark.parametrize("kind", ["split_half", "bootstrap"])
@pytest.mark.parametrize("n_resamples,n_collapsed,eligible", [(5, 1, True), (5, 2, False), (3, 1, False)])
def test_eligibility_requires_eighty_percent_complete_replicates(
    evaluator, epochs, monkeypatch, kind, n_resamples, n_collapsed, eligible,
):
    levels = np.repeat([0.0, 5.0, 10.0], 4) + np.arange(12) * 0.01
    sample = replace(epochs, post=np.broadcast_to(levels[None, :, None], (8, 12, 21)).copy())
    real_fit = evaluator.cluster_electrodes
    calls_for_three = 0

    def occasionally_collapsed_fit(*args, **kwargs):
        nonlocal calls_for_three
        fit = real_fit(*args, **kwargs)
        if kwargs["n_clusters"] != 3:
            return fit
        phase = calls_for_three
        calls_for_three += 1
        if phase > 0:
            replicate, sample_kind = divmod(phase - 1, 3)
            targeted_kind = 0 if kind == "split_half" else 2
            if replicate < n_collapsed and sample_kind == targeted_kind:
                labels = fit.labels.copy()
                labels[fit.clean_mask] = 0
                return replace(fit, labels=labels)
        return fit

    monkeypatch.setattr(evaluator, "cluster_electrodes", occasionally_collapsed_fit)
    result = evaluate(evaluator, sample, k_values=(2, 3), n_resamples=n_resamples,
                      feature_columns=(FEATURES[1],))
    summary = result.summary.set_index("k").loc[3]
    assert bool(summary.eligible) is eligible
    assert summary[f"{kind}_valid_count"] == n_resamples - n_collapsed
    assert summary[f"{kind}_valid_fraction"] == pytest.approx((n_resamples - n_collapsed) / n_resamples)
    rows = result.resamples.loc[result.resamples.k == 3]
    assert rows[f"{kind}_ari"].isna().sum() == n_collapsed
    assert rows[f"{kind}_has_tiny_cluster"].iloc[:n_collapsed].eq(False).all()
    assert result.recommended_k == (3 if eligible else 2)
    if not eligible:
        assert pd.isna(summary.selection_rank)


@pytest.mark.parametrize("sigma_seconds", [None, 0.0, 0.15])
@pytest.mark.parametrize("slope_method", ["savgol", "gradient"])
def test_waveform_correlation_smooths_each_trial_on_its_physical_time_axis(
    evaluator, epochs, sigma_seconds, slope_method,
):
    progress = epochs.progress_percent / 100
    durations = np.array([0.4, 0.7, 1.0, 1.5, 2.2, 3.0, 4.0, 5.0])
    response = np.exp(-0.5 * ((progress - 0.45) / 0.15) ** 2)
    signs = np.repeat([1.0, -1.0], 6)
    alternating = np.tile([1.0, -1.0], 6)
    noise = 2 * np.sin(2 * np.pi * 13 * progress)
    speech = signs[None, :, None] * 3 * response[None, None, :]
    speech = speech + np.linspace(0.5, 1.5, 8)[:, None, None] * alternating[None, :, None] * noise
    events = epochs.events.copy()
    events["duration"] = durations
    sample = replace(epochs, speech=speech, events=events)
    feature_kwargs = {"slope_method": slope_method}
    if sigma_seconds is not None:
        feature_kwargs["smoothing_sigma_seconds"] = sigma_seconds
    result = evaluate(evaluator, sample, k_values=(2,), n_resamples=1, feature_kwargs=feature_kwargs,
                      feature_columns=(FEATURES[1],))
    sigma = 0.06 if sigma_seconds is None else sigma_seconds
    smoothed = np.stack([
        gaussian_filter1d(trial, sigma * 100 / duration, axis=-1, mode="reflect")
        if sigma > 0 else trial.copy()
        for trial, duration in zip(speech, durations)
    ])
    curves = smoothed.mean(axis=0)
    correlations = []
    for members in (np.arange(6), np.arange(6, 12)):
        for member in members:
            others = curves[members[members != member]].mean(axis=0)
            correlations.append(np.corrcoef(curves[member], others)[0, 1])
    assert result.summary.iloc[0].waveform_correlation_mean == pytest.approx(np.mean(correlations), abs=1e-12)
    assert result.summary.iloc[0].waveform_valid_count == 12
