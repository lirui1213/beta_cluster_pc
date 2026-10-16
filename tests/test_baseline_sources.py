from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from audio_cross_3band_utils import AudioEnvelope, extract_beta_progress_epochs, load_speech_events
from test_pipeline import FakeRaw


def task(events=None, **kwargs):
    times = np.arange(2400) / 100.0
    raw = FakeRaw((1e-5 * (1 + 0.3 * np.sin(times * 3)))[None], 100)
    events = load_speech_events(events or [[2, 4], [6, 8], [10, 12], [14, 16]])
    audio = AudioEnvelope(np.sin(times) ** 2, times, 100)
    epochs = extract_beta_progress_epochs(raw, events, audio, pre_post_window=0.5, **kwargs)
    return epochs, raw, events


def test_local_baseline_is_independent_of_display_and_ends_before_onset():
    epochs, raw, events = task(baseline_mode="pre_onset", baseline_window=(-1.3, -0.3))
    times = np.linspace(-1.3, -0.3, 101)
    expected = np.stack([np.interp(start + times, raw.times, raw.get_data()[0] ** 2)
                         for start in events.onset])
    q = epochs.baseline_diagnostics
    np.testing.assert_allclose(q.baseline_mean, expected.mean(axis=-1))
    np.testing.assert_allclose(q.baseline_std, expected.std(axis=-1))
    assert epochs.pre_times[0] == -0.5
    assert np.all(q.baseline_mode == "pre_onset")
    assert np.all(q.baseline_stop_seconds == -0.3)
    np.testing.assert_allclose(epochs.baseline_reference.normalized_mean, 0, atol=1e-12)
    np.testing.assert_allclose(epochs.baseline_reference.normalized_std, 1, atol=1e-12)


def test_local_window_overlapping_speech_or_recording_boundary_is_invalid():
    epochs, _, _ = task(events=[[0.6, 2.6], [3, 5], [8, 10], [14, 16]],
                        baseline_mode="pre_onset", baseline_window=(-1.3, -0.3))
    q = epochs.baseline_diagnostics
    assert q.invalid_reason.iloc[0] == "outside_recording"
    assert q.invalid_reason.iloc[1] == "overlaps_speech"
    assert np.isnan(epochs.speech[:2]).all()
    assert q.valid.iloc[2:].all()


@pytest.mark.parametrize("window", [(-0.3, 0), (-0.3, -1.3), (0, 1), (np.nan, -0.3)])
def test_local_baseline_requires_a_finite_negative_window(window):
    with pytest.raises(ValueError, match="baseline_window"):
        task(baseline_mode="pre_onset", baseline_window=window)


def make_reference(tmp_path, *, reorder=False, flat=False):
    import mne
    sfreq = 100.0
    time = np.arange(1200) / sfreq
    values = 1e-5 * (1 + 0.3 * np.sin(2 * np.pi * 0.8 * time)) * np.sin(2 * np.pi * 20 * time)
    data = np.stack([values, 2 * values])
    if flat:
        data[:] = 0
    names = ["E0", "E1"]
    if reorder:
        names, data = names[::-1], data[::-1]
    raw = mne.io.RawArray(data, mne.create_info(names, sfreq, "ecog"), verbose="ERROR")
    path = tmp_path / "baseline.fif"
    raw.save(path, overwrite=True, verbose="ERROR")
    segments = tmp_path / "baseline__segments.csv"
    pd.DataFrame({"output_start_sec": [0, 6], "output_stop_sec": [6, 12]}).to_csv(segments, index=False)
    return path, segments, raw


def reference_result(path, segments, **kwargs):
    import baseline_sources
    return baseline_sources.load_file_baseline_statistics(
        path, ("E0", "E1"), beta_band=(13, 30), segments_csv=segments, **kwargs)


def test_file_baseline_matches_channels_and_pools_only_trimmed_segments(tmp_path):
    path, segments, _ = make_reference(tmp_path, reorder=True)
    result = reference_result(path, segments)
    assert result.mean.shape == (2,)
    assert result.mean[1] == pytest.approx(4 * result.mean[0])
    assert result.std[1] == pytest.approx(4 * result.std[0])
    assert not result.invalid.any()
    assert result.segments.used.all()
    assert (result.segments.edge_trim_seconds >= 0.3).all()
    assert result.reference.baseline_samples.min() < 1200
    np.testing.assert_allclose(result.reference.normalized_mean, 0, atol=1e-12)
    np.testing.assert_allclose(result.reference.normalized_std, 1, atol=1e-12)


def test_file_baseline_filters_independent_segments_without_join_leakage(tmp_path):
    import mne
    path, segments, raw = make_reference(tmp_path)
    original = reference_result(path, segments)
    data = raw.get_data()
    data[:, 600:] *= 100
    mne.io.RawArray(data, raw.info, verbose="ERROR").save(path, overwrite=True, verbose="ERROR")
    changed = reference_result(path, segments)
    # Each equally long segment contributes independent power samples.
    np.testing.assert_allclose(changed.mean, original.mean * (1 + 10000) / 2, rtol=0.015)


def test_file_baseline_rejects_missing_channels_and_malformed_segments(tmp_path):
    import baseline_sources
    path, segments, _ = make_reference(tmp_path)
    with pytest.raises(ValueError, match="channel"):
        baseline_sources.load_file_baseline_statistics(path, ("absent",))
    pd.DataFrame({"output_start_sec": [0, 5], "output_stop_sec": [6, 12]}).to_csv(segments, index=False)
    with pytest.raises(ValueError, match="overlap"):
        reference_result(path, segments)


def test_file_baseline_skips_short_segments_and_reports_flat_channels(tmp_path):
    path, segments, _ = make_reference(tmp_path, flat=True)
    pd.DataFrame({"output_start_sec": [0, 0.2], "output_stop_sec": [0.2, 12]}).to_csv(segments, index=False)
    result = reference_result(path, segments)
    assert not result.segments.used.iloc[0]
    assert result.segments.reason.iloc[0] == "too_short_after_edge_trim"
    assert result.invalid.all()
    assert result.reference.normalized_std.isna().all()


def test_file_baseline_refuses_insufficient_reference_and_metadata_mismatch(tmp_path):
    import mne
    path, segments, raw = make_reference(tmp_path)
    with pytest.raises(ValueError, match="usable baseline"):
        reference_result(path, segments, edge_seconds=6)
    info = mne.create_info(["E0", "E1"], 200, "ecog")
    with pytest.raises(ValueError, match="sampling"):
        reference_result(path, segments, task_info=info)
    info = raw.info.copy()
    with info._unlock():
        info["custom_ref_applied"] = 1
    with pytest.raises(ValueError, match="reference"):
        reference_result(path, segments, task_info=info)


@pytest.mark.parametrize("key,value", [("lowpass", 15.0), ("highpass", 18.0)])
def test_reference_rejects_different_stored_filter_preprocessing(tmp_path, key, value):
    path, segments, raw = make_reference(tmp_path)
    info = raw.info.copy()
    with info._unlock():
        info[key] = value
    with pytest.raises(ValueError, match="preprocessing"):
        reference_result(path, segments, task_info=info)


def test_file_mode_uses_one_reference_for_all_trials(tmp_path):
    import mne
    path, segments, _ = make_reference(tmp_path)
    sfreq = 100.0
    time = np.arange(2400) / sfreq
    data = np.tile((1e-5 * (1 + .4 * np.sin(time)) * np.sin(2 * np.pi * 20 * time))[None], (2, 1))
    raw = mne.io.RawArray(data, mne.create_info(["E0", "E1"], sfreq, "ecog"), verbose="ERROR")
    audio = AudioEnvelope(np.sin(time) ** 2, time, sfreq)
    epochs = extract_beta_progress_epochs(raw, load_speech_events([[2, 4], [6, 8], [10, 12]]), audio,
        pre_post_window=.5, baseline_mode="file", baseline_fif=path, baseline_segments_csv=segments)
    means = epochs.baseline_diagnostics.baseline_mean.to_numpy().reshape(3, 2)
    np.testing.assert_allclose(means, np.tile(means[0], (3, 1)))
    assert len(epochs.baseline_reference) == 2
    assert len(epochs.baseline_segments) == 2
    assert np.isfinite(epochs.speech).all()


def test_local_window_checks_speech_trials_rejected_by_duration_filter():
    epochs, _, _ = task(events=[[1, 8], [8.5, 10.5], [12, 14.1], [16, 18.2], [20, 22]],
                        baseline_mode="pre_onset")
    assert 0 in epochs.dropped_events.original_trial_id.to_list()
    assert epochs.baseline_diagnostics.invalid_reason.iloc[0] == "overlaps_speech"


def test_file_mode_bad_channel_is_not_misreported_as_low_variance(tmp_path):
    import mne
    path, segments, reference = make_reference(tmp_path)
    reference.info["bads"] = ["E1"]
    reference.save(path, overwrite=True, verbose="ERROR")
    time = np.arange(2400) / 100
    raw = mne.io.RawArray(np.tile((1e-5 * np.sin(2 * np.pi * 20 * time))[None], (2, 1)),
                         reference.info.copy(), verbose="ERROR")
    raw.info["bads"] = []
    epochs = extract_beta_progress_epochs(raw, load_speech_events([[2, 4], [6, 8], [10, 12]]),
        AudioEnvelope(np.sin(time) ** 2, time, 100), pre_post_window=.5,
        baseline_mode="file", baseline_fif=path, baseline_segments_csv=segments)
    bad = epochs.baseline_diagnostics.query("channel == 'E1'")
    assert not bad.valid.any()
    assert not bad.is_low_variance.any()
    assert (bad.invalid_reason == "baseline_file_bad_channel").all()


def test_reference_annotations_preserve_joins_and_exclude_bad_intervals(tmp_path):
    import mne
    path, segments, raw = make_reference(tmp_path)
    segments.unlink()
    raw.set_annotations(mne.Annotations([0, 6, 2], [6, 6, 1], ["quiet_first", "quiet_second", "BAD_artifact"]))
    raw.save(path, overwrite=True, verbose="ERROR")
    result = reference_result(path, None)
    assert len(result.segments) == 3
    assert (result.segments.boundary_source == "quiet_annotations").all()
    assert not ((result.segments.start_seconds < 3) & (result.segments.stop_seconds > 2)).any()


def test_reference_falls_back_to_continuous_recording_when_no_boundaries(tmp_path):
    path, segments, _ = make_reference(tmp_path)
    segments.unlink()
    result = reference_result(path, None)
    assert len(result.segments) == 1
    assert result.segments.boundary_source.iloc[0] == "whole_recording"


@pytest.mark.parametrize("quiet", [False, True])
def test_explicit_no_segment_selection_skips_companion_but_keeps_annotation_boundaries(tmp_path, quiet):
    import mne
    path, companion, raw = make_reference(tmp_path)
    onsets, durations, descriptions = [2., 8.], [1., 0.], ["BAD_artifact", "BAD boundary"]
    if quiet:
        onsets += [0., 6.]
        durations += [6., 6.]
        descriptions += ["quiet_first", "quiet_second"]
    raw.set_annotations(mne.Annotations(onsets, durations, descriptions))
    raw.save(path, overwrite=True, verbose="ERROR")
    explicit = reference_result(path, companion, auto_discover_segments=False)
    assert explicit.segments.boundary_source.eq("csv").all()
    legacy = reference_result(path, None)
    assert legacy.segments.boundary_source.eq("csv").all()
    result = reference_result(path, None, auto_discover_segments=False)
    assert result.segments.boundary_source.eq("quiet_annotations" if quiet else "whole_recording").all()
    assert not ((result.segments.start_seconds < 3) & (result.segments.stop_seconds > 2)).any()
    assert not ((result.segments.start_seconds < 8) & (result.segments.stop_seconds > 8)).any()
    companion.unlink()
    expected = reference_result(path, None)
    pd.testing.assert_frame_equal(result.segments, expected.segments)
    np.testing.assert_allclose(result.mean, expected.mean)
    np.testing.assert_allclose(result.std, expected.std)


def test_epoch_extraction_threads_segment_discovery_optout(tmp_path):
    path, _, raw = make_reference(tmp_path)
    time = np.arange(2400) / 100.
    task_raw = raw.copy()
    events = load_speech_events([[2, 3], [5, 6], [8, 9]])
    audio = AudioEnvelope(np.sin(time) ** 2, time, 100.)
    epochs = extract_beta_progress_epochs(task_raw, events, audio, pre_post_window=.5,
        baseline_mode="file", baseline_fif=path, baseline_auto_discover_segments=False)
    assert epochs.baseline_segments.boundary_source.tolist() == ["whole_recording"]


def test_file_baseline_is_invariant_to_voltage_scaling_and_rejects_unit_mismatch(tmp_path):
    import mne
    path, segments, raw = make_reference(tmp_path)
    before = reference_result(path, segments)
    scaled = mne.io.RawArray(raw.get_data() * 1e6, raw.info.copy(), verbose="ERROR")
    scaled.save(path, overwrite=True, verbose="ERROR")
    after = reference_result(path, segments)
    np.testing.assert_allclose(after.mean, before.mean * 1e12, rtol=1e-6)
    np.testing.assert_allclose(after.std, before.std * 1e12, rtol=1e-6)
    info = raw.info.copy()
    info["chs"][0]["unit"] = 0
    with pytest.raises(ValueError, match="units"):
        reference_result(path, segments, task_info=info)


def test_trial_preview_selects_valid_baselines_and_rejects_explicit_invalid_trial():
    import warnings
    from matplotlib import pyplot as plt
    from visualization import plot_two_nonoverlapping_beta_trials
    epochs, _, _ = task(events=[[0.6, 2.6], [3, 5], [8, 10], [14, 16]], baseline_mode="pre_onset")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figure, axes = plot_two_nonoverlapping_beta_trials(epochs)
    assert not any("empty slice" in str(w.message) for w in caught)
    assert all(np.isfinite(line.get_ydata()).all() for line in axes[0].lines)
    assert axes[0].lines[0].get_label().startswith("Trial 2:")
    plt.close(figure)
    with pytest.raises(ValueError, match="valid beta"):
        plot_two_nonoverlapping_beta_trials(epochs, trial_indices=(0, 2))
