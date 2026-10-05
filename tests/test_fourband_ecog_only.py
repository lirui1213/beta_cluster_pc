"""Audio diagnostics cannot affect either four-band clustering round."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from test_fourband import feature_table, module


COUPLING_SUFFIXES = (
    "audio_corr", "audio_reg_slope", "audio_xcorr", "audio_xcorr_lag",
    "audio_slope_corr", "audio_slope_reg_slope", "audio_slope_xcorr",
    "audio_slope_xcorr_lag_percent", "slope_product_mean",
    "slope_product_abs_mean", "slope_same_sign_fraction",
    "power_slope_when_audio_rising", "power_slope_when_audio_falling",
    "slope_crossing_count", "first_slope_crossing_percent",
    "trialwise_slope_audio_corr_mean", "trialwise_slope_audio_corr_std",
    "duration_slope_audio_corr",
)


def legacy_feature_table():
    table = feature_table()
    rng = np.random.default_rng(813)
    for band in module().DEFAULT_BANDS:
        for suffix in COUPLING_SUFFIXES:
            table[f"{band}__{suffix}"] = rng.normal(size=len(table))
        table[f"{band}__power_slope_std"] = rng.normal(size=len(table))
        table.loc[3, f"{band}__power_slope_std"] = np.nan
    return table


def assert_equal_fits(original, changed):
    assert original.feature_columns == changed.feature_columns
    assert original.feature_weights == changed.feature_weights
    np.testing.assert_array_equal(original.clean_mask, changed.clean_mask)
    np.testing.assert_array_equal(original.labels, changed.labels)
    np.testing.assert_array_equal(original.standardized_clean_features,
                                  changed.standardized_clean_features)
    np.testing.assert_array_equal(original.weighted_clean_features,
                                  changed.weighted_clean_features)
    np.testing.assert_array_equal(original.imputer.statistics_, changed.imputer.statistics_)
    np.testing.assert_array_equal(original.scaler.mean_, changed.scaler.mean_)
    np.testing.assert_array_equal(original.pca_scores, changed.pca_scores)


@pytest.mark.parametrize("round_number", [1, 2])
def test_old_table_coupling_and_unrecognized_predictors_never_enter_either_round(round_number):
    table = legacy_feature_table()
    for band in module().DEFAULT_BANDS:
        # Prefixes alone must not admit metadata or unreviewed coupling aliases.
        for suffix in ("valid_trial_count", "future_predictor", "power_slope_voice_corr"):
            table[f"{band}__{suffix}"] = np.cos(np.arange(len(table)) * .7)
    expected = {f"{band}__{suffix}" for band in module().DEFAULT_BANDS
                for suffix in ("early_mean", "power_slope_std")}
    first = module().fit_joint(table, n_clusters=3)
    second, _ = module("review").refit_reviewed(
        table, first, n_clusters=2, manual_bad_channels=("E0", "E14"))
    result = first if round_number == 1 else second
    assert set(result.feature_columns) == expected
    for band in module().DEFAULT_BANDS:
        budget = sum(w for c, w in zip(result.feature_columns, result.feature_weights)
                     if c.startswith(band + "__"))
        assert budget == pytest.approx(1)


@pytest.mark.parametrize("audio_change", ["shuffle", "missing", "constant"])
@pytest.mark.parametrize("round_number", [1, 2])
def test_changing_only_audio_diagnostics_preserves_fit_and_pca_review(audio_change, round_number):
    table = legacy_feature_table()
    changed = table.copy(deep=True)
    rng = np.random.default_rng(271)
    for band in module().DEFAULT_BANDS:
        for suffix in COUPLING_SUFFIXES:
            column = f"{band}__{suffix}"
            if audio_change == "shuffle":
                changed[column] = rng.permutation(table[column].to_numpy()) * 1000
            else:
                changed[column] = np.nan if audio_change == "missing" else 7.0
    first = module().fit_joint(table, n_clusters=3)
    changed_first = module().fit_joint(changed, n_clusters=3)
    if round_number == 1:
        assert_equal_fits(first, changed_first)
        return

    distances = module("review").pca_review_table(first).pca_global_distance
    threshold = (distances.iloc[14] + distances.iloc[:14].max()) / 2
    review_settings = dict(n_clusters=2, manual_bad_channels=("E0",),
                           pca_distance_threshold=threshold)
    second, audit = module("review").refit_reviewed(table, first, **review_settings)
    changed_second, changed_audit = module("review").refit_reviewed(
        changed, changed_first, **review_settings)
    assert_equal_fits(second, changed_second)
    pd.testing.assert_frame_equal(audit, changed_audit)
    assert second.labels[0] == second.labels[14] == second.labels[15] == -1
    for estimator in ("imputer", "scaler", "pca", "kmeans"):
        assert getattr(second, estimator) is not getattr(first, estimator)


def test_coupling_is_filtered_before_it_can_prune_a_correlated_power_predictor():
    table = feature_table()
    # The shared selector prioritizes audio_slope_corr ahead of power_slope_std.
    table["theta__power_slope_std"] = table["theta__audio_slope_corr"]
    table["theta__audio_slope_corr"] = table["theta__audio_slope_corr"] * 2
    result = module().fit_joint(table, n_clusters=3)
    assert "theta__power_slope_std" in result.feature_columns
    assert "theta__audio_slope_corr" not in result.feature_columns


@pytest.mark.parametrize("power_state", ["absent", "constant", "missing"])
def test_audio_cannot_rescue_a_band_without_usable_ecog_predictors(power_state):
    table = legacy_feature_table()
    columns = ["theta__early_mean", "theta__power_slope_std"]
    if power_state == "absent":
        table = table.drop(columns=columns)
    else:
        table[columns] = 1.0 if power_state == "constant" else np.nan
    with pytest.raises(ValueError, match="theta"):
        module().fit_joint(table, n_clusters=3)


@pytest.mark.parametrize("suffix", [
    "early_mean", "middle_mean", "late_mean", "middle_minus_early", "late_minus_middle",
    "positive_auc", "negative_auc", "positive_peak_value", "negative_peak_value",
    "positive_peak_percent", "negative_peak_percent", "dominant_peak_width_percent",
    "offset_rebound", "late_to_post_change", "energy_mean", "energy_std", "energy_auc",
    "energy_peak_abs", "energy_peak_percent", "energy_range", "pre_mean", "post_mean",
    "post_minus_pre_mean", "power_slope_mean", "power_slope_std", "power_slope_peak_abs",
    "power_slope_peak_percent", "early_delta", "late_delta", "post_delta", "post_std",
    "post_auc", "onset_slope", "post_peak_abs", "post_peak_time", "articulation_mean",
    "offset_slope", "audio_dur_ecog_cov",
])
def test_power_morphology_and_legacy_timing_descriptors_remain_eligible(suffix):
    table = feature_table()
    for band in module().DEFAULT_BANDS:
        table[f"{band}__{suffix}"] = np.cos(np.arange(len(table)) * .7)
    result = module().fit_joint(table, n_clusters=3)
    assert all(f"{band}__{suffix}" in result.feature_columns for band in module().DEFAULT_BANDS)


def test_extracted_audio_diagnostics_remain_available_but_do_not_affect_fits(monkeypatch):
    from test_beta_optimization import curve_epochs

    analysis = module()
    progress = np.linspace(0, 100, 101)
    centers = np.linspace(15, 85, 16)
    curves = np.exp(-((progress[None, :] - centers[:, None]) / 12) ** 2)
    curves *= np.linspace(-2, 2, 16)[:, None]
    reference = curve_epochs(np.tile(curves, (6, 1, 1)))
    raw = SimpleNamespace(info={"sfreq": 400})
    changed_audio = np.full_like(reference.audio_speech, np.nan)

    def extract(raw, events, audio, **kwargs):
        return replace(reference, beta_band=kwargs["beta_band"], audio_speech=audio)

    monkeypatch.setattr(analysis, "extract_beta_progress_epochs", extract)
    original = analysis.extract_fourband(raw, reference.events, reference.audio_speech)
    changed = analysis.extract_fourband(raw, reference.events, changed_audio)
    for band in analysis.DEFAULT_BANDS:
        column = f"{band}__audio_slope_corr"
        assert original.features[column].notna().all()
        assert changed.features[column].isna().all()
        assert f"{band}__slope_same_sign_fraction" in original.features
        assert f"{band}__first_slope_crossing_percent" in original.features
    first = analysis.fit_joint(original.features, n_clusters=3)
    changed_first = analysis.fit_joint(changed.features, n_clusters=3)
    assert_equal_fits(first, changed_first)
    second, audit = module("review").refit_reviewed(
        original.features, first, n_clusters=2, manual_bad_channels=("E0",))
    changed_second, changed_audit = module("review").refit_reviewed(
        changed.features, changed_first, n_clusters=2, manual_bad_channels=("E0",))
    assert_equal_fits(second, changed_second)
    pd.testing.assert_frame_equal(audit, changed_audit)
