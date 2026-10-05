"""Real two-trial time and baseline behavior, not concatenated epochs."""
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3-bands"))


def continuous():
    return importlib.import_module("fourband.continuous")


@pytest.fixture
def recording():
    import mne
    from fourband.analysis import DEFAULT_BANDS
    rate = 400.
    times = np.arange(4000) / rate
    # A genuine response in the inter-trial interval, outside either speech.
    amplitude = 1 + 2 * np.exp(-((times - 4) / .25) ** 2)
    wave = sum(np.sin(2 * np.pi * f * times) for f in (6, 18, 32, 95)) * amplitude
    raw = mne.io.RawArray(np.stack([wave, 2 * wave]),
        mne.create_info(["E1", "E2"], rate, ch_types="ecog"), verbose=False)
    events = pd.DataFrame(dict(original_trial_id=[10, 11, 12],
        onset=[2., 5., 7.], offset=[3., 6., 8.], duration=[1., 1., 1.]))
    diagnostics = pd.DataFrame(dict(original_trial_id=[10, 10, 11, 11],
        channel=["E1", "E2"] * 2, baseline_mean=[1., 2., 100., 200.],
        baseline_std=[2., 4., 10., 20.], valid=[True] * 4))
    epochs = {band: SimpleNamespace(channel_names=("E1", "E2"),
        events=events.iloc[:2].copy(), baseline_mode="pre_onset",
        baseline_diagnostics=diagnostics.copy(), beta_band=bounds, sfreq=rate,
        audio_speech=np.ones((2, 101))) for band, bounds in DEFAULT_BANDS.items()}
    extraction = SimpleNamespace(epochs=epochs, bands=dict(DEFAULT_BANDS))
    audio = SimpleNamespace(times=times, values=np.where((times >= 2) & (times <= 3), 1., .2), sfreq=rate)
    return raw, events, audio, extraction


def test_pairs_use_original_neighbors_not_neighbors_after_dropping():
    all_events = pd.DataFrame(dict(original_trial_id=[4, 5, 6, 7],
        onset=[1., 3., 5., 7.], offset=[2., 4., 6., 8.], duration=[1.] * 4))
    pairs = continuous().adjacent_trial_pairs(all_events, all_events.iloc[[0, 2, 3]])
    assert list(zip(pairs.first_trial_id, pairs.second_trial_id)) == [(6, 7)]


def test_unlabeled_original_events_get_same_sorted_ids_as_extraction():
    events = pd.DataFrame(dict(onset=[5., 1., 3.], offset=[6., 2., 4.], duration=[1.] * 3))
    retained = pd.DataFrame(dict(original_trial_id=[0, 1, 2]))
    pairs = continuous().adjacent_trial_pairs(events, retained)
    assert list(zip(pairs.first_trial_id, pairs.second_trial_id)) == [(0, 1), (1, 2)]
    assert pairs.first_onset.iloc[0] == 1.


def test_real_gap_is_sampled_and_first_baseline_applies_to_whole_interval(recording):
    raw, events, audio, extraction = recording
    data = continuous().prepare_continuous_trials(raw, events, audio, extraction,
        sigma_seconds=0, display_sfreq=400)
    assert data.trial_ids == (10, 11)
    np.testing.assert_allclose(np.diff(data.times), 1 / 400, atol=1e-12)
    assert data.times[0] == -1
    assert data.times[-1] == pytest.approx(5)
    assert data.metadata["gap_seconds"] == 2
    assert data.metadata["baseline_anchor_trial_id"] == 10
    filtered = raw.copy().filter(13, 24, phase="zero", verbose="ERROR")
    filtered.apply_hilbert(envelope=True, verbose="ERROR")
    start = 400
    stop = 2801
    expected = (filtered.get_data()[0, start:stop] ** 2 - 1) / 2
    np.testing.assert_allclose(data.power["beta1"][0], expected, atol=1e-10)
    gap = (data.times > 1) & (data.times < 3)
    assert data.power["beta1"][0, gap].max() > data.power["beta1"][0, (data.times >= 0) & (data.times < .5)].max() + 1
    np.testing.assert_allclose(data.audio[gap], .2, atol=1e-8)


@pytest.mark.parametrize("pair", [(10, 12), (11, 10), (10, 99), (10, 10), (10.5, 11), (True, 11)])
def test_invalid_pair_is_rejected_before_filtering(recording, pair):
    raw, events, audio, extraction = recording
    with pytest.raises(ValueError, match="trial|adjacent|retained"):
        continuous().prepare_continuous_trials(raw, events, audio, extraction, trial_ids=pair)


@pytest.mark.parametrize("kwargs", [dict(pre_seconds=-1), dict(post_seconds=np.inf),
    dict(display_sfreq=0), dict(sigma_seconds=-.1)])
def test_invalid_window_or_sampling_parameters(recording, kwargs):
    with pytest.raises(ValueError):
        continuous().prepare_continuous_trials(*recording, **kwargs)


def test_margins_clamp_to_recording_and_report_actual_bounds(recording):
    data = continuous().prepare_continuous_trials(*recording, pre_seconds=10, post_seconds=10)
    assert data.metadata["clipped_to_recording"]
    assert data.metadata["actual_bounds_seconds"][0] == 0
    assert data.metadata["actual_bounds_seconds"][1] <= 9.9975
    assert data.times[0] == -2


def test_baseline_and_channel_mismatch_fail_clearly(recording):
    raw, events, audio, extraction = recording
    extraction.epochs["theta"].channel_names = ("E2", "E1")
    with pytest.raises(ValueError, match="channel|align"):
        continuous().prepare_continuous_trials(raw, events, audio, extraction)
    extraction.epochs["theta"].channel_names = ("E1", "E2")
    extraction.epochs["theta"].baseline_diagnostics["valid"] = False
    with pytest.raises(ValueError, match="baseline"):
        continuous().prepare_continuous_trials(raw, events, audio, extraction)


def test_changed_original_event_times_require_reextraction(recording):
    raw, events, audio, extraction = recording
    changed = events.copy()
    changed.loc[1, ["onset", "duration"]] = [4.75, 1.25]
    with pytest.raises(ValueError, match="event|onset|extraction"):
        continuous().prepare_continuous_trials(raw, changed, audio, extraction)


def test_continuous_power_reuses_extraction_filter_configuration(recording):
    raw, events, audio, extraction = recording
    extraction.filter_kwargs = dict(method="iir", iir_params=dict(order=4, ftype="butter"))
    data = continuous().prepare_continuous_trials(raw, events, audio, extraction,
        sigma_seconds=0, display_sfreq=400)
    filtered = raw.copy().filter(13, 24, phase="zero", verbose="ERROR", **extraction.filter_kwargs)
    filtered.apply_hilbert(envelope=True, verbose="ERROR")
    expected = (filtered.get_data()[0, 400:2801] ** 2 - 1) / 2
    np.testing.assert_allclose(data.power["beta1"][0], expected, atol=1e-10)


def test_fourband_extraction_records_filter_configuration_for_display(recording):
    from fourband.analysis import extract_fourband
    raw, events, audio, _ = recording
    kwargs = dict(method="iir", iir_params=dict(order=4, ftype="butter"))
    extraction = extract_fourband(raw, events, audio, filter_kwargs=kwargs,
        pre_post_window=.5, min_trials_after_filter=3)
    assert extraction.filter_kwargs == kwargs


def test_invalid_channel_reference_stays_missing_and_audio_coverage_is_not_filled(recording):
    raw, events, audio, extraction = recording
    for epochs in extraction.epochs.values():
        epochs.baseline_diagnostics.loc[epochs.baseline_diagnostics.channel == "E2", "valid"] = False
    audio.times = audio.times[800:2001]
    audio.values = audio.values[800:2001]
    data = continuous().prepare_continuous_trials(raw, events, audio, extraction)
    assert np.isnan(data.power["theta"][1]).all()
    assert np.isnan(data.audio[data.times < 0]).all()
    assert np.isnan(data.audio[data.times > 3]).all()


def test_file_baseline_is_shared_and_does_not_claim_trial_anchoring(recording):
    raw, events, audio, extraction = recording
    for epochs in extraction.epochs.values():
        epochs.baseline_mode = "file"
        epochs.baseline_diagnostics.loc[epochs.baseline_diagnostics.original_trial_id == 11, "baseline_mean"] = [1., 2.]
        epochs.baseline_diagnostics.loc[epochs.baseline_diagnostics.original_trial_id == 11, "baseline_std"] = [2., 4.]
    data = continuous().prepare_continuous_trials(raw, events, audio, extraction)
    assert data.metadata["baseline_anchor_trial_id"] is None
    assert data.metadata["baseline_mode"] == "file"
    assert "Patient baseline file" in data.metadata["baseline_description"]


def test_selected_event_markers_keep_actual_durations_and_other_context_events(recording):
    raw, events, audio, extraction = recording
    # The second utterance lasts longer; real time cannot collapse it to 101 points.
    events.loc[1, ["offset", "duration"]] = [6.5, 1.5]
    for epochs in extraction.epochs.values():
        epochs.events = events.iloc[:2].copy()
    data = continuous().prepare_continuous_trials(raw, events, audio, extraction, post_seconds=2)
    assert data.times[-1] == pytest.approx(6.5)
    assert data.metadata["selected_events"][1]["offset"] == 6.5
    assert data.metadata["context_event_ids"] == [10, 11, 12]


def test_missing_real_gap_coverage_is_not_interpolated(recording):
    raw, events, audio, extraction = recording
    audio.values[(audio.times > 3.5) & (audio.times < 4.5)] = np.nan
    data = continuous().prepare_continuous_trials(raw, events, audio, extraction)
    assert np.isnan(data.audio[(data.times > 1.6) & (data.times < 2.4)]).all()


def test_plot_exports_actual_times_audio_and_visible_event_markers(recording, tmp_path):
    data = continuous().prepare_continuous_trials(*recording)
    result = SimpleNamespace(channels=("E1", "E2"), labels=np.array([0, 0]), clean_mask=np.array([True, True]))
    table = continuous().save_continuous_trial_plot(data, result, tmp_path / "display")
    assert set(table.band) == set(data.power)
    assert {"elapsed_seconds", "recording_seconds", "mean_audio", "mean_power", "sem_power", "n_electrodes"} <= set(table)
    assert table.elapsed_seconds.max() == pytest.approx(5)
    meta = json.loads((tmp_path / "display/continuous_trial_metadata.json").read_text())
    assert meta["trial_ids"] == [10, 11]
    assert meta["baseline_anchor_trial_id"] == 10
    assert (tmp_path / "display/fourband_continuous_trials.png").stat().st_size > 1000
    with pytest.raises(FileExistsError):
        continuous().save_continuous_trial_plot(data, result, tmp_path / "display")


def test_plot_rejects_stale_channel_order(recording, tmp_path):
    data = continuous().prepare_continuous_trials(*recording)
    result = SimpleNamespace(channels=("E2", "E1"), labels=np.array([0, 0]), clean_mask=np.array([True, True]))
    with pytest.raises(ValueError, match="channel|align"):
        continuous().save_continuous_trial_plot(data, result, tmp_path / "invalid")


def test_plot_rejects_no_valid_baseline_among_included_channels(recording, tmp_path):
    raw, events, audio, extraction = recording
    for epochs in extraction.epochs.values():
        epochs.baseline_diagnostics.loc[epochs.baseline_diagnostics.channel == "E2", "valid"] = False
    data = continuous().prepare_continuous_trials(raw, events, audio, extraction)
    result = SimpleNamespace(channels=("E1", "E2"), labels=np.array([-1, 0]), clean_mask=np.array([False, True]))
    with pytest.raises(ValueError, match="baseline|valid"):
        continuous().save_continuous_trial_plot(data, result, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
