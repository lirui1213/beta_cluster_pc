"""Trial resampling must keep the reference electrode population and provenance."""
import importlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3-bands"))

import numpy as np
import pandas as pd
import pytest

from audio_cross_3band_utils import BetaProgressEpochs, extract_beta_progress_features
from fourband.analysis import DEFAULT_BANDS, FourBandExtraction, fit_joint


def stability():
    spec = importlib.util.find_spec("fourband.stability")
    assert spec is not None, "Task 2 stability implementation is missing"
    return importlib.import_module("fourband.stability")


def fixture(n_trials=8, amplitudes=None, settings=None, onset_order=None, fit_options=None):
    amplitudes = np.asarray(amplitudes if amplitudes is not None else [-3, -2.8, -2.5, 2.5, 2.8, 3, 90.])
    n_channels = len(amplitudes)
    channels = tuple(f"E{i}" for i in range(n_channels))
    progress = np.linspace(0, 100, 41)
    wave = np.exp(-((progress - 45) / 20) ** 2)
    speech = np.broadcast_to(amplitudes[None, :, None] * wave, (n_trials, n_channels, 41)).copy()
    context = np.linspace(0, .5, 11)
    ids = np.arange(n_trials) * 3 + 17
    events = pd.DataFrame({"duration": np.full(n_trials, 2.), "original_trial_id": ids,
                           "onset": np.arange(n_trials) * 4. if onset_order is None else onset_order},
                          index=np.arange(n_trials) + 100)
    events["offset"] = events.onset + events.duration
    diagnostics = pd.DataFrame({"original_trial_id": np.repeat(ids, n_channels),
                                "channel": np.tile(channels, n_trials), "valid": True})
    base = BetaProgressEpochs(speech, np.zeros((n_trials, n_channels, 11)),
        np.zeros((n_trials, n_channels, 11)), np.tile(wave, (n_trials, 1)),
        np.zeros((n_trials, 11)), np.zeros((n_trials, 11)), progress, context - .5,
        context, events, pd.DataFrame(), channels, frozenset({channels[-1]}), 100.,
        (13., 24.), .5, baseline_diagnostics=diagnostics)
    epochs = {band: replace(base, speech=speech.copy(), beta_band=bounds)
              for band, bounds in DEFAULT_BANDS.items()}
    columns = {"channel": channels, "is_bad": np.zeros(n_channels, dtype=bool)}
    for band, epoch in epochs.items():
        features = extract_beta_progress_features(epoch, **(settings or {}))
        columns["is_bad"] |= features.is_bad.to_numpy()
        for col in features:
            if col.startswith("beta_progress__"):
                suffix = col.removeprefix("beta_progress__").replace("beta_slope", "power_slope")
                columns[f"{band}__{suffix}"] = features[col].to_numpy()
        columns[f"qc_{band}_valid_trials"] = features.valid_trial_count.to_numpy()
        columns[f"qc_{band}_invalid_fraction"] = features.baseline_invalid_fraction.to_numpy()
    table = pd.DataFrame(columns)
    extraction = FourBandExtraction(table, epochs, dict(DEFAULT_BANDS))
    if settings is not None:
        extraction.feature_settings = dict(settings)
    options = dict(n_clusters=2, bad_mask=np.array([True] + [False] * (n_channels - 1)))
    options.update(fit_options or {})
    result = fit_joint(table, **options)
    return extraction, result


def test_identity_agreement_uses_requested_k_and_reviewed_channel_order():
    extraction, result = fixture()
    report = stability().evaluate_stability(extraction, result, n_resamples=2, random_state=12)
    assert report.summary["requested_k"] == 2
    assert report.summary["attempted"] == report.summary["supported"] == 8
    assert report.summary["failed"] == 0
    assert report.channels.channel.tolist() == ["E1", "E2", "E3", "E4", "E5"]
    assert report.replicates.ari_full.eq(1).all()
    assert report.replicates.loc[report.replicates.kind != "bootstrap", "ari_between_halves"].eq(1).all()
    assert report.channels.agreement.eq(1).all()
    assert report.channels.coverage.eq(1).all()


def test_permuted_reference_labels_are_matched_by_overlap():
    extraction, result = fixture()
    swapped = result.labels.copy()
    swapped[result.clean_mask] = 1 - swapped[result.clean_mask]
    report = stability().evaluate_stability(extraction, replace(result, labels=swapped), n_resamples=1)
    assert report.channels.agreement.eq(1).all()
    assert report.replicates.matched_agreement.eq(1).all()


def test_seeded_trial_draws_common_across_bands_keep_bootstrap_duplicates(monkeypatch):
    extraction, result = fixture()
    module = stability()
    calls = []
    real_extract = module.extract_beta_progress_features

    def capture(epoch, **kwargs):
        calls.append((epoch.beta_band, tuple(epoch.events.original_trial_id), epoch.speech.copy(),
                      epoch.baseline_diagnostics.copy()))
        return real_extract(epoch, **kwargs)

    monkeypatch.setattr(module, "extract_beta_progress_features", capture)
    first = module.evaluate_stability(extraction, result, n_resamples=3, random_state=12)
    second = module.evaluate_stability(extraction, result, n_resamples=3, random_state=12)
    pd.testing.assert_frame_equal(first.replicates, second.replicates)
    draws = first.replicates[first.replicates.kind == "bootstrap"]
    assert any(len(set(ids)) < len(ids) for ids in draws.original_trial_ids)
    for start in range(0, len(calls), 4):
        group = calls[start:start + 4]
        assert len({call[1] for call in group}) == 1
        assert {call[0] for call in group} == set(DEFAULT_BANDS.values())
        for _, ids, speech, quality in group:
            positions = [(id_ - 17) // 3 for id_ in ids]
            np.testing.assert_array_equal(speech, extraction.epochs["theta"].speech[positions])
            assert quality.original_trial_id.tolist() == np.repeat(ids, 7).tolist()
    other = module.evaluate_stability(extraction, result, n_resamples=3, random_state=13)
    assert first.replicates.trial_indices.tolist() != other.replicates.trial_indices.tolist()


def test_chronological_halves_follow_onsets_and_original_ids():
    extraction, result = fixture(onset_order=[12., 0., 28., 4., 24., 8., 20., 16.])
    report = stability().evaluate_stability(extraction, result, n_resamples=0)
    assert report.replicates.trial_indices.tolist() == [(1, 3, 5, 0), (7, 6, 4, 2)]
    assert report.replicates.original_trial_ids.tolist() == [(20, 26, 32, 17), (38, 35, 29, 23)]


def test_source_arrays_features_qc_and_reference_fit_are_immutable():
    extraction, result = fixture()
    table = extraction.features.copy(deep=True)
    arrays = {band: {key: getattr(epoch, key).copy() for key in
                    ("speech", "pre", "post", "audio_speech", "audio_pre", "audio_post")}
              for band, epoch in extraction.epochs.items()}
    labels, mask = result.labels.copy(), result.clean_mask.copy()
    quality = extraction.epochs["theta"].baseline_diagnostics.copy(deep=True)
    stability().evaluate_stability(extraction, result, n_resamples=1)
    pd.testing.assert_frame_equal(extraction.features, table)
    pd.testing.assert_frame_equal(extraction.epochs["theta"].baseline_diagnostics, quality)
    for band, values in arrays.items():
        for key, saved in values.items():
            np.testing.assert_array_equal(getattr(extraction.epochs[band], key), saved)
    np.testing.assert_array_equal(result.labels, labels)
    np.testing.assert_array_equal(result.clean_mask, mask)


def test_unsupported_halves_fail_without_changing_population():
    extraction, result = fixture()
    for epoch in extraction.epochs.values():
        epoch.speech[:2, 1, :] = np.nan
    # Full-data QC still supports E1 at the configured 25% threshold.
    # Rebuild the reference from these epochs to preserve its provenance.
    extraction, result = reaggregate(extraction)
    report = stability().evaluate_stability(extraction, result, n_resamples=0)
    assert report.summary["attempted"] == 2
    assert report.summary["failed"] == 1
    assert report.summary["supported"] == 1
    failed = report.replicates.loc[~report.replicates.supported].iloc[0]
    assert "E1" in failed.reason and "theta" in failed.reason
    assert np.isnan(failed.ari_full)
    assert report.channels.channel.tolist() == ["E1", "E2", "E3", "E4", "E5"]
    assert report.channels.coverage.eq(.5).all()


def reaggregate(extraction):
    table = extraction.features.copy()
    table["is_bad"] = False
    for band, epoch in extraction.epochs.items():
        local = extract_beta_progress_features(epoch, **(getattr(extraction, "feature_settings", {}) or {}))
        table["is_bad"] |= local.is_bad.to_numpy()
        for col in local:
            if col.startswith("beta_progress__"):
                suffix = col.removeprefix("beta_progress__").replace("beta_slope", "power_slope")
                table[f"{band}__{suffix}"] = local[col].to_numpy()
        table[f"qc_{band}_valid_trials"] = local.valid_trial_count.to_numpy()
        table[f"qc_{band}_invalid_fraction"] = local.baseline_invalid_fraction.to_numpy()
    extraction = replace(extraction, features=table)
    result = fit_joint(table, n_clusters=2, bad_mask=np.array([True] + [False] * (len(table) - 1)))
    return extraction, result


def test_no_supported_replicate_returns_unavailable_agreement_and_explanation():
    extraction, result = fixture(n_trials=2)
    report = stability().evaluate_stability(extraction, result, n_resamples=0)
    assert report.summary["supported"] == 0
    assert report.summary["failed"] == 2
    assert report.summary["ari_full_mean"] is None
    assert report.summary["reason"]
    assert report.channels.agreement.isna().all()
    assert report.channels.coverage.eq(0).all()


def test_tiny_clusters_are_supported_and_included_in_summary():
    extraction, result = fixture(amplitudes=[-3., -2.8, -2.5, -2., -1.5, 20., 90.])
    report = stability().evaluate_stability(extraction, result, n_resamples=1)
    assert report.summary["tiny"] == report.summary["supported"] == 5
    assert report.replicates.tiny_cluster.all()
    assert report.summary["ari_full_mean"] == 1
    assert report.channels.agreement.eq(1).all()


def test_collapsed_fits_keep_metrics_and_are_counted(monkeypatch):
    extraction, result = fixture()
    module = stability()
    real_fit = module.fit_joint

    def collapse(table, **kwargs):
        fitted = real_fit(table, **kwargs)
        if int(table["qc_theta_valid_trials"].max()) < 8:
            labels = fitted.labels.copy()
            labels[fitted.clean_mask] = 0
            return replace(fitted, labels=labels)
        return fitted

    monkeypatch.setattr(module, "fit_joint", collapse)
    report = module.evaluate_stability(extraction, result, n_resamples=0)
    assert report.summary["supported"] == report.summary["collapsed"] == 2
    assert report.replicates.ari_full.eq(0).all()
    assert report.replicates.ari_between_halves.eq(1).all()
    assert report.channels.agreement.mean() == pytest.approx(.6)


def test_fit_failures_are_recorded_and_do_not_abort_later_replicates(monkeypatch):
    extraction, result = fixture()
    module = stability()
    real_fit = module.fit_joint

    def fail_halves(table, **kwargs):
        if int(table["qc_theta_valid_trials"].max()) < 8:
            raise ValueError("No usable predictors for theta")
        return real_fit(table, **kwargs)

    monkeypatch.setattr(module, "fit_joint", fail_halves)
    report = module.evaluate_stability(extraction, result, n_resamples=1)
    assert report.summary["failed"] == 4
    assert report.summary["supported"] == 1
    assert report.replicates.iloc[-1].kind == "bootstrap"
    assert report.replicates.loc[~report.replicates.supported, "reason"].str.contains("theta").all()
    assert report.channels.coverage.eq(.2).all()


def test_progress_cancellation_returns_partial_report():
    extraction, result = fixture()
    updates = []
    report = stability().evaluate_stability(extraction, result, n_resamples=3,
        progress=lambda done, total: updates.append((done, total)),
        cancelled=lambda: bool(updates and updates[-1][0] >= 1))
    assert report.summary["cancelled"] is True
    assert report.summary["attempted"] == 1
    assert report.summary["planned"] == 11
    assert updates[-1] == (1, 11)
    assert len(report.replicates) == 1
    assert report.replicates.ari_between_halves.isna().all()
    assert report.channels.agreement.eq(1).all()
    empty = stability().evaluate_stability(extraction, result, cancelled=lambda: True)
    assert empty.summary["attempted"] == 0
    assert empty.channels.coverage.eq(0).all()


def test_actual_feature_settings_are_used_and_explicit_settings_reproduce_reference():
    settings = dict(smoothing_sigma_seconds=.15, slope_method="gradient", max_baseline_invalid_fraction=.5)
    extraction, result = fixture(settings=settings)
    module = stability()
    report = module.evaluate_stability(extraction, result, n_resamples=0)
    assert all(report.summary["feature_settings"][key] == value for key, value in settings.items())
    del extraction.feature_settings
    explicit = module.evaluate_stability(extraction, result, n_resamples=0, feature_kwargs=settings)
    pd.testing.assert_frame_equal(report.replicates, explicit.replicates)
    with pytest.raises(ValueError, match="settings|features|reference"):
        module.evaluate_stability(extraction, result, n_resamples=0)


@pytest.mark.parametrize("change", ["channels", "features", "audio", "mask", "weights"])
def test_mismatched_reference_is_rejected(change):
    extraction, result = fixture()
    if change == "channels":
        result = replace(result, channels=tuple(reversed(result.channels)))
    elif change == "features":
        extraction.features.loc[1, result.feature_columns[0]] += 20
    elif change == "audio":
        result = replace(result, feature_columns=("theta__audio_slope_corr", *result.feature_columns[1:]))
    elif change == "mask":
        mask = result.clean_mask.copy()
        mask[-1] = True
        result = replace(result, clean_mask=mask)
    else:
        result = replace(result, feature_weights=tuple(w * (2 if i == 0 else 1)
                         for i, w in enumerate(result.feature_weights)))
    with pytest.raises(ValueError):
        stability().evaluate_stability(extraction, result, n_resamples=0)


def test_mismatched_band_trials_are_rejected():
    extraction, result = fixture()
    extraction.epochs["beta1"] = replace(extraction.epochs["beta1"],
        events=extraction.epochs["beta1"].events.iloc[::-1])
    with pytest.raises(ValueError, match="align"):
        stability().evaluate_stability(extraction, result)


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_invalid_resample_counts_are_rejected(count):
    extraction, result = fixture()
    with pytest.raises(ValueError, match="n_resamples"):
        stability().evaluate_stability(extraction, result, n_resamples=count)


@pytest.mark.parametrize("n_trials,expected_ari", [(2, None), (8, 1.)])
def test_exports_write_strict_json_tables_and_nonempty_png(tmp_path, n_trials, expected_ari):
    extraction, result = fixture(n_trials=n_trials)
    module = stability()
    report = module.evaluate_stability(extraction, result, n_resamples=0)
    folder = module.save_stability(report, tmp_path / "stability")
    assert folder == tmp_path / "stability"
    summary = json.loads((folder / "stability_summary.json").read_text(encoding="utf-8"),
        parse_constant=lambda value: pytest.fail(f"Nonstandard JSON constant: {value}"))
    assert summary["ari_full_mean"] == expected_ari
    replicas = pd.read_csv(folder / "stability_replicates.csv")
    assert len(replicas) == 2
    assert json.loads(replicas.original_trial_ids.iloc[0]) == list(range(17, 17 + 3 * (n_trials // 2), 3))
    assert pd.read_csv(folder / "stability_channels.csv").channel.tolist() == ["E1", "E2", "E3", "E4", "E5"]
    from matplotlib import image
    pixels = image.imread(folder / "stability.png")
    assert pixels.shape[0] > 100 and np.std(pixels[..., :3]) > .01


@pytest.mark.parametrize("options", [
    {"weight_preset": "balanced"},
    {"weight_preset": "custom", "group_weights": {"morphology": 2., "amplitude": 1., "slope": 0.}},
    {"band_weights": {"theta": 2., "beta1": 1., "beta2": 3., "high_gamma": 4.},
     "correlation_threshold": 1., "random_state": 17},
])
def test_parent_fit_options_reproduce_actual_policy(options):
    extraction, result = fixture(fit_options=options)
    report = stability().evaluate_stability(extraction, result, n_resamples=0, fit_kwargs=options)
    assert report.summary["supported"] == 2
    assert report.channels.agreement.eq(1).all()
    for key, value in options.items():
        assert report.summary["fit_settings"][key] == value


def test_explicit_wrong_fit_policy_is_rejected():
    extraction, result = fixture()
    with pytest.raises(ValueError, match="policy|reference"):
        stability().evaluate_stability(extraction, result, n_resamples=0,
                                      fit_kwargs={"correlation_threshold": 1.})


def test_supported_resamples_refit_all_estimators_and_preserve_trial_exclusions(monkeypatch):
    extraction, result = fixture()
    extraction.trial_exclusions = (("E0", 17),)
    module = stability()
    fitted = []
    real_fit = module.fit_joint

    def capture(table, **kwargs):
        fit = real_fit(table, **kwargs)
        fitted.append(fit)
        return fit

    monkeypatch.setattr(module, "fit_joint", capture)
    report = module.evaluate_stability(extraction, result, n_resamples=1)
    assert report.summary["trial_exclusions"] == [("E0", 17)]
    assert len(fitted) == 6  # Reference policy check plus five resampled fits.
    for name in ("imputer", "scaler", "pca", "kmeans"):
        estimators = [getattr(item, name) for item in [result, *fitted]]
        assert len({id(item) for item in estimators}) == len(estimators)


def test_cancellation_during_band_reaggregation_keeps_completed_pair(monkeypatch):
    extraction, result = fixture()
    module = stability()
    calls = []
    real_extract = module.extract_beta_progress_features

    def stop_during_third_fit(epoch, **kwargs):
        calls.append(epoch)
        return real_extract(epoch, **kwargs)

    monkeypatch.setattr(module, "extract_beta_progress_features", stop_during_third_fit)
    report = module.evaluate_stability(extraction, result, n_resamples=2,
                                      cancelled=lambda: len(calls) >= 13)
    assert report.summary["cancelled"]
    assert report.summary["attempted"] == report.summary["supported"] == 2
    assert report.summary["failed"] == 0
    assert report.summary["ari_between_halves_mean"] == 1


def test_keyboard_interrupt_in_progress_returns_completed_metrics():
    extraction, result = fixture()

    def progress(done, total):
        if done == 2:
            raise KeyboardInterrupt()

    report = stability().evaluate_stability(extraction, result, n_resamples=2, progress=progress)
    assert report.summary["cancelled"]
    assert report.summary["attempted"] == 2
    assert report.summary["supported"] == 2
    assert report.summary["completed_half_pairs"] == 1


def test_empty_reference_population_returns_report_with_explicit_reason():
    extraction, result = fixture()
    empty = replace(result, labels=np.full(len(result.channels), -1),
                    clean_mask=np.zeros(len(result.channels), dtype=bool))
    report = stability().evaluate_stability(extraction, empty, n_resamples=0)
    assert report.summary["supported"] == report.summary["attempted"] == 0
    assert report.channels.empty
    assert "No included electrodes" in report.summary["reason"]


def test_real_collapsed_kmeans_fits_are_supported_without_leaking_warnings():
    with pytest.warns(UserWarning):
        extraction, result = fixture(amplitudes=[-3., -3., -3., 3., 3., 3., 90.],
                                     fit_options={"n_clusters": 3})
    # The warning is recorded on each supported replicate, including half fits.
    import warnings
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        report = stability().evaluate_stability(extraction, result, n_resamples=0)
    assert not emitted
    assert report.summary["requested_k"] == 3
    assert report.summary["supported"] == report.summary["collapsed"] == 2
    assert report.replicates.actual_k.eq(2).all()
    assert report.replicates.warnings.str.contains("distinct clusters").all()
    assert report.channels.agreement.eq(1).all()


@pytest.mark.parametrize("retain_source", [True, False])
def test_confirmed_half_trial_exclusions_are_not_baseline_failures(retain_source):
    from fourband.trial_qc import apply_trial_exclusions
    original, _ = fixture()
    pairs = [("E1", id_) for id_ in original.epochs["theta"].events.original_trial_id.iloc[:4]]
    reviewed = apply_trial_exclusions(original, pairs)
    assert not reviewed.features.loc[1, "is_bad"]
    assert reviewed.features.loc[1, "qc_theta_valid_trials"] == 4
    assert reviewed.features.loc[1, "qc_theta_invalid_fraction"] == 0
    result = fit_joint(reviewed.features, n_clusters=2,
                       bad_mask=np.array([True] + [False] * (len(reviewed.features) - 1)))
    if not retain_source:
        del reviewed._trial_qc_source
    reviewed_copy = reviewed.features.copy(deep=True)
    report = stability().evaluate_stability(reviewed, result, n_resamples=3, random_state=12)
    assert report.summary["supported"] >= 1
    assert report.summary["failed"] >= 1
    assert report.channels.channel.tolist() == ["E1", "E2", "E3", "E4", "E5"]
    assert not report.replicates.iloc[0].supported
    assert report.replicates.iloc[1].supported
    for row in report.replicates.itertuples():
        count = sum(id_ not in {17, 20, 23, 26} for id_ in row.original_trial_ids)
        assert row.min_valid_trial_count == count
        assert row.supported == (count >= 2)
    pd.testing.assert_frame_equal(reviewed.features, reviewed_copy)
    assert np.isfinite(original.epochs["theta"].speech[:, 1]).all()


def test_original_baseline_failures_and_manual_masks_use_separate_counts():
    from fourband.trial_qc import apply_trial_exclusions
    original, _ = fixture()
    for epoch in original.epochs.values():
        epoch.speech[:2, 1, :] = np.nan
        epoch.baseline_diagnostics.loc[
            (epoch.baseline_diagnostics.channel == "E1") &
            epoch.baseline_diagnostics.original_trial_id.isin([17, 20]), "valid"] = False
    original, _ = reaggregate(original)
    reviewed = apply_trial_exclusions(original, [("E1", id_) for id_ in [23, 26, 29, 32]])
    assert reviewed.features.loc[1, "qc_theta_invalid_fraction"] == .25
    assert reviewed.features.loc[1, "qc_theta_valid_trials"] == 2
    assert not reviewed.features.loc[1, "is_bad"]
    result = fit_joint(reviewed.features, n_clusters=2,
                       bad_mask=np.array([True] + [False] * (len(reviewed.features) - 1)))
    report = stability().evaluate_stability(reviewed, result, n_resamples=0)
    assert report.summary["supported"] == report.summary["failed"] == 1
    assert report.channels.coverage.eq(.5).all()
