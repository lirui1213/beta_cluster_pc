"""Manually confirmed trial QC, immutable features, and review provenance."""
from dataclasses import replace
import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3-bands"))

from audio_cross_3band_utils import extract_beta_progress_features
from fourband.analysis import DEFAULT_BANDS, FourBandExtraction
from test_beta_optimization import curve_epochs
from test_pipeline import FakeRaw


def qc_module(name="trial_qc"):
    assert importlib.util.find_spec(f"fourband.{name}") is not None, "Task 1 module is missing"
    return importlib.import_module(f"fourband.{name}")


def qc_fixture():
    values = np.broadcast_to(np.array([1., 10., 3., 5.])[:, None, None], (4, 2, 101)).copy()
    reference = replace(curve_epochs(values), channel_names=("E1", "E2"),
        events=pd.DataFrame({"original_trial_id": [5, 17, 31, 42],
            "onset": [2., 5., 8., 11.], "offset": [3., 6., 9., 12.], "duration": [1.] * 4}))
    epochs = {band: replace(reference, speech=values.copy(), beta_band=bounds)
              for band, bounds in DEFAULT_BANDS.items()}
    columns = {"channel": ["E1", "E2"], "is_bad": [False, False], "exclusion_reason": ["", ""]}
    for band, ep in epochs.items():
        table = extract_beta_progress_features(ep, smoothing_sigma_seconds=0, slope_method="gradient")
        for column in table:
            if column.startswith("beta_progress__"):
                columns[band + "__" + column.removeprefix("beta_progress__").replace("beta_slope", "power_slope")] = table[column]
        columns[f"qc_{band}_invalid_fraction"] = 0.
        columns[f"qc_{band}_valid_trials"] = 4
    features = pd.DataFrame(columns)
    extraction = FourBandExtraction(features, epochs, dict(DEFAULT_BANDS), {"n_jobs": 1})
    extraction.feature_settings = dict(smoothing_sigma_seconds=0, slope_method="gradient",
                                       max_baseline_invalid_fraction=.25)
    times = np.arange(1400) / 100.
    raw = FakeRaw(np.array([1e-5 * np.sin(2 * np.pi * 7 * times),
                           1e-5 * np.cos(2 * np.pi * 9 * times)]), 100., ["E1", "E2"])
    return raw, extraction


def test_detect_includes_unflagged_actual_pairs_and_never_excludes():
    raw, original = qc_fixture()
    before = original.epochs["theta"].speech.copy()
    report = qc_module().detect_trial_quality(raw, original)
    assert len(report) == 8
    assert set(zip(report.channel, report.original_trial_id)) == {
        (ch, trial) for ch in ("E1", "E2") for trial in (5, 17, 31, 42)}
    assert not report.candidate.any()
    assert not report.excluded.any()
    np.testing.assert_array_equal(original.epochs["theta"].speech, before)
    assert report.attrs["provenance"]["raw_fingerprint"]
    assert report.attrs["settings"]["reference"] == "full_record_channel_median_mad"
    assert report.attrs["units"]["flat_run_seconds"] == "seconds"


def test_spike_flat_clipping_nonfinite_and_adjustable_flags():
    raw, original = qc_fixture()
    raw._data[0, 550] = .01
    raw._data[1, 820:861] = 0
    raw._data[0, 1120:1151] = .01
    raw._data[1, 220] = np.nan
    report = qc_module().detect_trial_quality(raw, original).set_index(["channel", "original_trial_id"])
    assert report.loc[("E1", 17), "flag_spike"]
    assert report.loc[("E1", 17), "flag_jump"]
    assert report.loc[("E2", 31), "flag_flat"]
    assert report.loc[("E1", 42), "flag_clipping_candidate"]
    assert report.loc[("E2", 5), "flag_nonfinite"]
    adjusted = qc_module().detect_trial_quality(raw, original, dict(spike_z=1e9, jump_z=1e9,
        flat_run_seconds=2., clipping_run_seconds=2., clipping_fraction=1.))
    assert not adjusted.flag_spike.any()
    assert not adjusted.flag_jump.any()
    assert not adjusted.flag_flat.any()
    assert not adjusted.flag_clipping_candidate.any()
    assert adjusted.flag_nonfinite.any()


def test_baseline_flags_join_by_actual_pair_without_duplicating_rows():
    raw, original = qc_fixture()
    original.epochs["theta"] = replace(original.epochs["theta"], baseline_diagnostics=pd.DataFrame({
        "channel": ["E2"], "original_trial_id": [17], "valid": [False],
        "is_low_variance": [True], "invalid_reason": ["low_variance"]}))
    report = qc_module().detect_trial_quality(raw, original)
    row = report.set_index(["channel", "original_trial_id"]).loc[("E2", 17)]
    assert row.baseline_theta_invalid
    assert "theta:low_variance" in row.reasons
    assert len(report) == 8


def test_mask_all_bands_contexts_and_reaggregate_without_mutation():
    _, original = qc_fixture()
    before = {band: tuple(getattr(ep, part).copy() for part in ("speech", "pre", "post"))
              for band, ep in original.epochs.items()}
    features = original.features.copy(deep=True)
    reviewed = qc_module().apply_trial_exclusions(original, [("E1", 17)],
        smoothing_sigma_seconds=0, slope_method="gradient")
    assert reviewed.trial_exclusions == (("E1", 17),)
    assert reviewed.filter_kwargs == {"n_jobs": 1}
    assert reviewed.feature_settings["slope_method"] == "gradient"
    for band, ep in reviewed.epochs.items():
        for part, saved in zip(("speech", "pre", "post"), before[band]):
            actual = getattr(ep, part)
            assert np.isnan(actual[1, 0]).all()
            np.testing.assert_array_equal(actual[:, 1], saved[:, 1])
            np.testing.assert_array_equal(actual[[0, 2, 3], 0], saved[[0, 2, 3], 0])
            np.testing.assert_array_equal(getattr(original.epochs[band], part), saved)
        assert reviewed.features.loc[0, f"{band}__early_mean"] == pytest.approx(3.)
        assert reviewed.features.loc[0, f"qc_{band}_valid_trials"] == 3
        assert reviewed.features.loc[0, f"qc_{band}_invalid_fraction"] == 0
        assert not reviewed.features.is_bad.iloc[0]
        pd.testing.assert_frame_equal(ep.events, original.epochs[band].events)
    pd.testing.assert_frame_equal(original.features, features)


def test_remaining_support_recomputed_and_permanent_exclusions_not_restored():
    _, original = qc_fixture()
    original.features.loc[1, "is_bad"] = True
    original.features.loc[1, "exclusion_reason"] = "theta:invalid_trial_fraction"
    original.epochs["theta"] = replace(original.epochs["theta"], bad_channels=frozenset({"E2"}))
    reviewed = qc_module().apply_trial_exclusions(original, [("E1", 5), ("E1", 17), ("E1", 31)])
    assert reviewed.features.is_bad.tolist() == [True, True]
    assert reviewed.features.qc_theta_valid_trials.tolist() == [1, 4]
    assert "insufficient_valid_trials" in reviewed.features.exclusion_reason.iloc[0]
    assert "invalid_trial_fraction" in reviewed.features.exclusion_reason.iloc[1]
    assert reviewed.features.filter(regex="^theta__").isna().all().all()


def test_explicit_empty_review_restores_only_manual_masks():
    _, original = qc_fixture()
    original.epochs["theta"].speech[0, 1] = np.nan
    once = qc_module().apply_trial_exclusions(original, [("E1", 17)])
    restored = qc_module().apply_trial_exclusions(once, ())
    assert restored.trial_exclusions == ()
    np.testing.assert_array_equal(restored.epochs["theta"].speech, original.epochs["theta"].speech)
    assert np.isnan(restored.epochs["theta"].speech[0, 1]).all()


@pytest.mark.parametrize("pairs", [None, [("Unknown", 17)], [("E1", 999)], [("E1", True)], [("E1", 17.5)]])
def test_invalid_pairs_fail_before_masking(pairs):
    _, original = qc_fixture()
    with pytest.raises(ValueError):
        qc_module().apply_trial_exclusions(original, pairs)


@pytest.mark.parametrize("settings", [{"spike_z": -1}, {"flat_run_seconds": float("nan")},
    {"clipping_fraction": 1.1}, {"typo": 1}, {"interval": "bad"}])
def test_bad_detection_settings_fail(settings):
    raw, original = qc_fixture()
    with pytest.raises(ValueError):
        qc_module().detect_trial_quality(raw, original, settings)


@pytest.mark.parametrize("change", ["raw", "events", "settings", "filter", "channels", "epochs"])
def test_popup_rejects_report_reuse_after_provenance_change(change):
    raw, original = qc_fixture()
    report = qc_module().detect_trial_quality(raw, original)
    if change == "raw":
        raw._data[0, -1] += 1e-5
    elif change == "events":
        original.epochs["theta"].events.loc[0, "onset"] += .1
    elif change == "settings":
        report.attrs["settings"]["spike_z"] += 1
    elif change == "filter":
        original.filter_kwargs["n_jobs"] = 2
    elif change == "channels":
        raw.ch_names.reverse()
    else:
        original.epochs["theta"].speech[0, 0, 0] += .1
    with pytest.raises(ValueError, match="provenance|changed|alignment"):
        qc_module("trial_review_window").TrialQualityReviewWindow(raw, original, report)


def test_pending_selection_cancel_keep_all_and_provenance_checked_reuse():
    raw, original = qc_fixture()
    report = qc_module().detect_trial_quality(raw, original)
    cls = qc_module("trial_review_window").TrialQualityReviewWindow
    popup = cls(raw, original, report)
    assert not popup.selected_pairs
    popup.set_pair(("E1", 17), True)
    popup.cancel()
    assert popup.selection is None
    popup = cls(raw, original, report)
    popup.confirm()
    assert popup.selection == () and popup.selection is not None
    popup = cls(raw, original, report)
    popup.set_pair(("E1", 17), True)
    popup.confirm()
    selected = popup.selection
    assert selected == (("E1", 17),) and selected.provenance == report.attrs["provenance"]
    assert cls(raw, original, report, initial_pairs=selected).selected_pairs == {("E1", 17)}
    raw._data[0, -1] += 1e-5
    new_report = qc_module().detect_trial_quality(raw, original)
    with pytest.raises(ValueError, match="provenance|changed"):
        cls(raw, original, new_report, initial_pairs=selected)
    original.epochs["theta"].speech[0, 0, 0] += 1
    with pytest.raises(ValueError, match="provenance|changed"):
        qc_module().apply_trial_exclusions(original, selected)


def test_confirmed_provenance_keeps_exportable_thresholds_and_units():
    raw, original = qc_fixture()
    report = qc_module().detect_trial_quality(raw, original, {"spike_z": 12.})
    popup = qc_module("trial_review_window").TrialQualityReviewWindow(raw, original, report)
    popup.confirm()
    reviewed = qc_module().apply_trial_exclusions(original, popup.selection)
    assert reviewed.trial_qc_provenance["qc_settings"]["spike_z"] == 12.
    assert reviewed.trial_qc_provenance["qc_units"]["peak_abs_v"] == "volts"
    assert reviewed.trial_qc_provenance["qc_settings"]["pre_seconds"] == .5


def test_duplicate_pairs_and_all_removed_trials_have_explicit_support_counts():
    _, original = qc_fixture()
    pairs = [("E1", i) for i in (5, 17, 31, 42)] + [("E1", 17)]
    reviewed = qc_module().apply_trial_exclusions(original, pairs)
    assert len(reviewed.trial_exclusions) == 4
    assert reviewed.features.qc_theta_valid_trials.tolist() == [0, 4]
    assert reviewed.features.qc_theta_manual_fraction.tolist() == [1., 0.]
    assert reviewed.features.qc_theta_invalid_fraction.tolist() == [0., 0.]
    assert np.isnan(reviewed.features.qc_theta_remaining_invalid_fraction.iloc[0])
    assert reviewed.features.newly_ineligible.tolist() == [True, False]


def test_baseline_nan_is_not_restored_and_manual_fraction_is_independent():
    raw, original = qc_fixture()
    for band, ep in original.epochs.items():
        ep.speech[0, 1] = np.nan
        original.features.loc[1, f"qc_{band}_invalid_fraction"] = .25
        original.features.loc[1, f"qc_{band}_valid_trials"] = 3
    reviewed = qc_module().apply_trial_exclusions(original, [("E2", 5), ("E2", 17)])
    assert reviewed.features.qc_theta_invalid_fraction.iloc[1] == .25
    assert reviewed.features.qc_theta_manual_fraction.iloc[1] == .5
    assert reviewed.features.qc_theta_valid_trials.iloc[1] == 2
    assert not reviewed.features.is_bad.iloc[1]
    report = qc_module().detect_trial_quality(raw, reviewed)
    row = report.set_index(["channel", "original_trial_id"]).loc[("E2", 17)]
    assert row.excluded and not row.baseline_theta_invalid


def test_confirm_rejects_raw_change_while_popup_is_open():
    raw, original = qc_fixture()
    report = qc_module().detect_trial_quality(raw, original)
    popup = qc_module("trial_review_window").TrialQualityReviewWindow(raw, original, report)
    popup.set_pair(("E1", 17))
    raw._data[0, -1] += .001
    with pytest.raises(ValueError, match="provenance|changed"):
        popup.confirm()
    assert popup.selection is None


def test_reviewed_copy_changes_cannot_be_reused():
    raw, original = qc_fixture()
    reviewed = qc_module().apply_trial_exclusions(original, [("E1", 17)])
    report = qc_module().detect_trial_quality(raw, reviewed)
    reviewed.epochs["theta"].speech[0, 0, 0] += 1.
    with pytest.raises(ValueError, match="provenance|changed"):
        qc_module("trial_review_window").TrialQualityReviewWindow(raw, reviewed, report)


def test_lost_original_reference_cannot_silently_keep_manual_masks():
    _, original = qc_fixture()
    reviewed = qc_module().apply_trial_exclusions(original, [("E1", 17)])
    with pytest.raises(ValueError, match="source|original"):
        qc_module().apply_trial_exclusions(replace(reviewed), ())


def test_manual_baseline_only_invalidation_rejects_tampered_report():
    raw, original = qc_fixture()
    report = qc_module().detect_trial_quality(raw, original)
    report.loc[0, "original_trial_id"] = 999
    with pytest.raises(ValueError, match="provenance|changed"):
        qc_module("trial_review_window").TrialQualityReviewWindow(raw, original, report)


def test_flat_reference_and_infinities_remain_reviewable_without_warnings():
    raw, original = qc_fixture()
    raw._data[0] = 0
    raw._data[1, 550:560] = np.inf
    with np.errstate(all="raise"):
        report = qc_module().detect_trial_quality(raw, original)
    assert report.loc[report.channel == "E1", "flag_flat"].all()
    assert report.set_index(["channel", "original_trial_id"]).loc[("E2", 17), "flag_nonfinite"]


def test_real_trial_review_popup_in_isolated_process():
    worker = Path(__file__).with_name("fourband_trial_qc_gui_checks.py")
    result = subprocess.run([sys.executable, "-m", "pytest", str(worker), "-q", "--capture=sys"],
        cwd=worker.parents[1], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "5 passed" in result.stdout
