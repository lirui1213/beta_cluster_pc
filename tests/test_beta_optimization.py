from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

import clustering_pipeline
from audio_cross_3band_utils import (
    AudioEnvelope,
    BetaProgressEpochs,
    extract_beta_progress_epochs,
    extract_beta_progress_features,
    extract_electrode_features,
    extract_three_band_power_epochs,
    load_speech_events,
)
from test_pipeline import FakeRaw
from visualization import _concat_beta_progress_display, plot_beta_cluster_grid


def raw_epochs(scale=1.0, flat=False):
    sfreq = 100.0
    times = np.arange(1800) / sfreq
    values = 1e-5 * (1.0 + 0.2 * np.sin(2 * np.pi * 2 * times))
    if flat:
        values[:] = 1e-5
    raw = FakeRaw((values * scale)[None, :], sfreq)
    events = load_speech_events([[2, 4], [6, 8], [10, 12], [14, 16]])
    audio = AudioEnvelope(np.sin(times) ** 2, times, sfreq)
    return extract_beta_progress_epochs(raw, events, audio, pre_post_window=0.5)


def curve_epochs(curves=None):
    progress = np.linspace(0, 100, 101)
    times = progress / 50.0
    if curves is None:
        clean = 2 * np.exp(-0.5 * ((times - 0.9) / 0.2) ** 2)
        curves = np.tile(clean, (6, 3, 1))
    curves = np.asarray(curves)
    n_trials, n_channels, _ = curves.shape
    context = np.linspace(0, 0.5, 51)
    return BetaProgressEpochs(
        speech=curves,
        pre=np.zeros((n_trials, n_channels, len(context))),
        post=np.zeros((n_trials, n_channels, len(context))),
        audio_speech=np.tile(np.exp(-0.5 * ((times - 0.9) / 0.2) ** 2), (n_trials, 1)),
        audio_pre=np.zeros((n_trials, len(context))),
        audio_post=np.zeros((n_trials, len(context))),
        progress_percent=progress,
        pre_times=context - 0.5,
        post_times=context,
        events=pd.DataFrame({"duration": [2.0] * n_trials, "original_trial_id": range(n_trials)}),
        dropped_events=pd.DataFrame(),
        channel_names=tuple(f"E{i}" for i in range(n_channels)),
        bad_channels=frozenset(),
        sfreq=100.0,
        beta_band=(13.0, 30.0),
        pre_post_window=0.5,
    )


def test_baseline_zscore_is_invariant_to_voltage_units():
    small, large = raw_epochs(), raw_epochs(scale=1e6)
    np.testing.assert_allclose(small.pre, large.pre, atol=1e-8)
    np.testing.assert_allclose(small.baseline_reference.normalized_std, 1.0, atol=1e-8)


def test_flat_baseline_is_flagged_and_not_imputed_as_a_response():
    epochs = raw_epochs(flat=True)
    assert np.isnan(epochs.speech).all()
    assert epochs.baseline_diagnostics["is_low_variance"].all()
    assert not epochs.baseline_diagnostics["valid"].any()
    features = extract_beta_progress_features(epochs)
    assert features.loc[0, "baseline_invalid_fraction"] == 1.0
    assert features.loc[0, "is_bad"]


def test_smoothed_slope_recovers_coupling_from_fast_noise():
    epochs = curve_epochs()
    noise = 0.8 * (-1.0) ** np.arange(101)
    noisy = replace(epochs, speech=epochs.speech + noise)
    features = extract_beta_progress_features(noisy, smoothing_sigma_seconds=0.06)
    raw_features = extract_beta_progress_features(noisy, smoothing_sigma_seconds=0, slope_method="gradient")
    key = "beta_progress__audio_slope_corr"
    assert features.loc[0, key] > 0.95
    assert features.loc[0, key] > raw_features.loc[0, key]
    np.testing.assert_array_equal(noisy.speech, epochs.speech + noise)


def test_morphology_features_measure_known_peak():
    features = extract_beta_progress_features(curve_epochs(), smoothing_sigma_seconds=0)
    assert features.loc[0, "beta_progress__middle_mean"] > features.loc[0, "beta_progress__early_mean"]
    assert features.loc[0, "beta_progress__positive_auc"] > 0
    assert features.loc[0, "beta_progress__negative_auc"] == 0
    assert features.loc[0, "beta_progress__positive_peak_percent"] == pytest.approx(45.0)
    assert 15 < features.loc[0, "beta_progress__dominant_peak_width_percent"] < 35


def test_feature_selection_excludes_quality_constants_and_duplicates():
    select = getattr(clustering_pipeline, "select_beta_progress_feature_columns", None)
    assert callable(select), "Missing redundancy-aware feature selection"
    table = pd.DataFrame({
        "channel": [f"E{i}" for i in range(8)], "is_bad": [False] * 7 + [True],
        "baseline_invalid_fraction": np.arange(8) / 8,
        "beta_progress__early_mean": [0, 1, 2, 3, 4, 5, 6, 100],
        "beta_progress__energy_mean": [0, 2, 4, 6, 8, 10, 12, -100],
        "beta_progress__constant": [0] * 8,
    })
    assert select(table) == ("beta_progress__early_mean",)


def test_display_grid_is_uniform_across_context_and_speech():
    epochs = curve_epochs()
    axis, beta, audio, duration = _concat_beta_progress_display(epochs)
    np.testing.assert_allclose(np.diff(axis), np.diff(axis)[0], atol=1e-12)
    assert beta.shape[-1] == audio.shape[-1] == len(axis)
    assert axis[0] == pytest.approx(-0.5)
    assert axis[-1] == pytest.approx(duration + 0.5)


def test_sem_is_calculated_from_smoothed_electrode_curves():
    oscillation = (-1.0) ** np.arange(101)
    epochs = curve_epochs(np.tile(np.stack([oscillation, -oscillation]), (4, 1, 1)))
    fig, axes = plot_beta_cluster_grid(epochs, [0, 0], sigma_seconds=0.10)
    bounds = axes[0, 0].collections[0].get_paths()[0].vertices
    interior = bounds[(bounds[:, 0] > 300) & (bounds[:, 0] < 1700), 1]
    assert np.max(np.abs(interior)) < 0.05
    plt.close(fig)


def test_trial_bootstrap_band_is_reproducible_and_reflects_trial_variation():
    epochs = curve_epochs()
    epochs = replace(epochs, speech=epochs.speech * np.arange(1, 7)[:, None, None])
    options = dict(sigma_seconds=0.1, uncertainty="trial_bootstrap", bootstrap_iterations=80, random_state=17)
    fig1, ax1 = plot_beta_cluster_grid(epochs, [0, 0, 0], **options)
    fig2, ax2 = plot_beta_cluster_grid(epochs, [0, 0, 0], **options)
    first = ax1[0, 0].collections[0].get_paths()[0].vertices
    second = ax2[0, 0].collections[0].get_paths()[0].vertices
    np.testing.assert_allclose(first, second)
    peak_band = first[(first[:, 0] > 850) & (first[:, 0] < 950), 1]
    assert np.ptp(peak_band) > 1
    plt.close(fig1)
    plt.close(fig2)


def test_feature_group_budget_is_independent_of_column_count():
    weights = clustering_pipeline.default_beta_progress_feature_weights(
        ["beta_progress__audio_slope_corr", "beta_progress__audio_slope_xcorr", "beta_progress__early_mean"],
        slope_audio_weight=5, morphology_weight=3, normalize_groups=True,
    )
    assert weights["beta_progress__audio_slope_corr"] == 2.5
    assert weights["beta_progress__audio_slope_xcorr"] == 2.5
    assert weights["beta_progress__early_mean"] == 3


def test_resampling_suppresses_out_of_band_signal():
    from signal_processing import resample_on_axis
    source = np.arange(801) / 400
    target = np.linspace(0, 2, 101)
    high_frequency = np.sin(2 * np.pi * 80 * source)
    result = resample_on_axis(high_frequency, source, target)
    assert np.max(np.abs(result[10:-10])) < 0.03


def test_gaussian_width_is_same_before_and_during_speech():
    from signal_processing import gaussian_smooth_nan
    axis, _, _, _ = _concat_beta_progress_display(curve_epochs())
    sfreq = 1 / np.diff(axis)[0]
    pulses = np.zeros((2, len(axis)))
    centers = [-0.25, 1.0]
    for pulse, center in zip(pulses, centers, strict=True):
        pulse[np.argmin(abs(axis - center))] = 1
    smooth = gaussian_smooth_nan(pulses, sfreq, sigma_seconds=0.06)
    widths = []
    for row, center in zip(smooth, centers, strict=True):
        row /= row.sum()
        widths.append(np.sqrt(np.sum(row * (axis - center) ** 2)))
    np.testing.assert_allclose(widths, 0.06, rtol=0.01)


def test_clustering_accepts_pandas_copy_on_write_arrays():
    table = pd.DataFrame({"channel": [f"E{i}" for i in range(8)], "feature": np.arange(8, dtype=float)})
    with pd.option_context("mode.copy_on_write", True):
        result = clustering_pipeline.cluster_electrodes(table, n_clusters=2)
    assert (result.labels >= 0).all()


def test_legacy_band_pipeline_flags_flat_baselines_as_bad():
    raw = FakeRaw(np.ones((1, 1800)) * 1e-5, sfreq=100)
    events = load_speech_events([[2, 4], [6, 8], [10, 12]])
    epochs = extract_three_band_power_epochs(raw, events, bands={"beta": (13, 30)})
    assert not epochs.baseline_diagnostics.valid.any()
    features = extract_electrode_features(epochs, np.zeros((3, len(epochs.onset_times))))
    assert features.loc[0, "is_bad"]


def test_audio_coupling_uses_electrode_matched_valid_trials():
    epochs = curve_epochs()
    speech = np.concatenate([epochs.speech, np.full_like(epochs.speech[:2], np.nan)], axis=0)
    audio = np.concatenate([epochs.audio_speech, -10 * epochs.audio_speech[:2]], axis=0)
    epochs = replace(epochs, speech=speech, audio_speech=audio,
        pre=np.concatenate([epochs.pre, np.full_like(epochs.pre[:2], np.nan)]),
        post=np.concatenate([epochs.post, np.full_like(epochs.post[:2], np.nan)]),
        audio_pre=np.concatenate([epochs.audio_pre, epochs.audio_pre[:2]]),
        audio_post=np.concatenate([epochs.audio_post, epochs.audio_post[:2]]),
        events=pd.DataFrame({"duration": [2.0] * 8, "original_trial_id": range(8)}))
    features = extract_beta_progress_features(epochs)
    assert not features.loc[0, "is_bad"]
    assert features.loc[0, "beta_progress__audio_slope_corr"] > 0.99


def test_resampling_brackets_non_aligned_endpoint():
    from signal_processing import resample_on_axis
    source = np.arange(10) / 10
    target = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(resample_on_axis(source, source, target), target, atol=0.005)


def test_resampling_preserves_missing_coverage_and_internal_gaps():
    from signal_processing import resample_on_axis
    source = np.arange(11, dtype=float)
    values = np.array([np.nan, np.nan, 2, 3, 4, np.nan, 6, 7, 8, np.nan, np.nan])
    target = np.arange(0, 10.5, 0.5)
    sampled = resample_on_axis(values, source, target)
    assert np.isnan(sampled[(target < 2) | (target > 8)]).all()
    assert np.isnan(sampled[(target > 4) & (target < 6)]).all()
    np.testing.assert_allclose(sampled[(target >= 2) & (target <= 4)], target[(target >= 2) & (target <= 4)])


def test_smoothing_does_not_create_observations_at_missing_samples():
    from signal_processing import gaussian_smooth_nan
    values = np.array([np.nan, np.nan, 1, 2, 3, np.nan, 4, 5, np.nan])
    result = gaussian_smooth_nan(values, 10, sigma_seconds=0.1)
    assert np.isnan(result[~np.isfinite(values)]).all()
    assert np.isfinite(result[np.isfinite(values)]).all()
