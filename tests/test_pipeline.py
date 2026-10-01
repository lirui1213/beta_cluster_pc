from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from audio_cross_3band_utils import (
    AudioEnvelope,
    align_audio_to_epochs,
    extract_beta_progress_epochs,
    extract_beta_progress_features,
    extract_electrode_features,
    extract_three_band_power_epochs,
    filter_duration_outlier_trials,
    load_speech_events,
    minimum_intertrial_gap,
    resolve_frequency_bands,
)
from clustering_pipeline import (
    cluster_electrodes,
    combine_bad_channels,
    default_beta_progress_feature_weights,
    refit_after_bad_channel_review,
)


class FakeRaw:
    """Small MNE Raw-compatible object for deterministic unit tests."""

    def __init__(self, data, sfreq=400.0, names=None, bads=None):
        self._data = np.asarray(data, dtype=float)
        self.info = {"sfreq": sfreq, "bads": list(bads or [])}
        self.ch_names = list(names or [f"E{index}" for index in range(len(data))])
        self.times = np.arange(self._data.shape[1], dtype=float) / sfreq

    def copy(self):
        return FakeRaw(
            self._data.copy(), self.info["sfreq"], self.ch_names.copy(), self.info["bads"].copy()
        )

    def pick(self, _picks):
        return self

    def load_data(self):
        return self

    def filter(self, **_kwargs):
        return self

    def apply_hilbert(self, envelope=True, **_kwargs):
        assert envelope
        self._data = np.abs(self._data)
        return self

    def get_data(self):
        return self._data


def test_events_and_nyquist_resolution():
    events = load_speech_events(np.array([[1.0, 2.0], [3.0, 5.5]]))
    assert events["duration"].tolist() == [1.0, 2.5]
    bands = resolve_frequency_bands(250.0)
    assert bands["high_gamma"] == (70.0, 124.5)


def test_epochs_baseline_and_feature_dimensions():
    sfreq = 400.0
    times = np.arange(int(22 * sfreq)) / sfreq
    data = np.vstack(
        [
            1.0 + 0.2 * np.sin(2 * np.pi * 7 * times),
            1.0 + 0.3 * np.sin(2 * np.pi * 17 * times),
            1.0 + 0.2 * np.sin(2 * np.pi * 90 * times),
        ]
    )
    raw = FakeRaw(data, sfreq, bads=["E1"])
    events = load_speech_events(np.array([[2.0, 4.2], [7.0, 10.0], [13.0, 16.5]]))
    epochs = extract_three_band_power_epochs(raw, events)
    baseline = (epochs.onset_times >= -1.0) & (epochs.onset_times <= -0.7)
    assert epochs.onset["theta"].shape == (3, 3, 2801)
    assert epochs.offset["theta"].shape == (3, 3, 1201)
    assert np.max(np.abs(np.mean(epochs.onset["theta"][:, :, baseline], axis=-1))) < 1e-8

    audio = np.tile(np.linspace(0.0, 1.0, len(epochs.onset_times)), (3, 1))
    features = extract_electrode_features(epochs, audio)
    feature_columns = [column for column in features if "__" in column]
    assert features.shape == (3, 62)
    assert len(feature_columns) == 60
    assert features.loc[1, feature_columns].isna().all()


def test_audio_epoch_interpolation():
    from audio_cross_3band_utils import AudioEnvelope

    envelope = AudioEnvelope(
        values=np.array([0.0, 0.5, 1.0]),
        times=np.array([0.0, 1.0, 2.0]),
        sfreq=1.0,
    )
    events = load_speech_events(np.array([[1.0, 1.5]]))
    aligned = align_audio_to_epochs(envelope, events, [-0.5, 0.0, 0.5])
    np.testing.assert_allclose(aligned, [[0.25, 0.5, 0.75]])


def test_second_round_refits_and_keeps_bad_label_minus_one():
    rng = np.random.default_rng(12)
    count = 18
    table = pd.DataFrame(
        {
            "channel": [f"E{index}" for index in range(count)],
            "is_bad": [index in (1, 9) for index in range(count)],
            **{f"feature_{index}": rng.normal(size=count) for index in range(8)},
        }
    )
    table.loc[3, "feature_2"] = np.nan
    first = cluster_electrodes(table, n_clusters=5)
    combined = combine_bad_channels(
        first.channels,
        table["is_bad"].to_numpy(),
        first.labels,
        bad_clusters=[0],
        manual_bads=["E2"],
    )
    second = refit_after_bad_channel_review(table, combined, n_clusters=3)
    assert np.all(second.labels[combined] == -1)
    assert np.isnan(second.pca_scores[combined]).all()
    assert first.imputer is not second.imputer
    assert first.scaler is not second.scaler
    assert first.kmeans is not second.kmeans


def test_clustering_drops_all_missing_clean_feature():
    table = pd.DataFrame(
        {
            "channel": [f"E{index}" for index in range(8)],
            "is_bad": [False] * 8,
            "usable": np.arange(8, dtype=float),
            "insufficient_trials": [np.nan] * 8,
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = cluster_electrodes(table, n_clusters=2)
    assert result.feature_columns == ("usable",)


def test_beta_progress_duration_filter_and_features():
    sfreq = 100.0
    times = np.arange(int(36 * sfreq)) / sfreq
    data = np.vstack(
        [
            1.0 + 0.2 * np.sin(2 * np.pi * 18 * times),
            1.0 + 0.1 * np.sin(2 * np.pi * 22 * times),
            1.0 + 0.3 * np.sin(2 * np.pi * 15 * times),
        ]
    )
    raw = FakeRaw(data, sfreq, bads=["E1"])
    events = load_speech_events(
        np.array(
            [
                [2.0, 4.0],
                [6.0, 8.1],
                [10.0, 12.0],
                [14.0, 16.2],
                [18.0, 28.0],
            ]
        )
    )
    kept, dropped = filter_duration_outlier_trials(events, mad_threshold=2.5)
    assert len(kept) == 4
    assert len(dropped) == 1
    assert np.isclose(minimum_intertrial_gap(kept), 1.9)

    envelope = AudioEnvelope(
        values=np.linspace(0.0, 1.0, len(times)),
        times=times,
        sfreq=sfreq,
    )
    epochs = extract_beta_progress_epochs(
        raw,
        events,
        envelope,
        progress_points=51,
        duration_mad_threshold=2.5,
        min_trials_after_filter=3,
    )
    assert epochs.speech.shape == (4, 3, 51)
    assert epochs.audio_speech.shape == (4, 51)
    assert np.isclose(epochs.pre_post_window, 1.9)
    assert epochs.progress_percent[0] == 0.0
    assert epochs.progress_percent[-1] == 100.0

    features = extract_beta_progress_features(epochs)
    assert "beta_progress__audio_slope_corr" in features
    assert "beta_progress__energy_mean" in features
    feature_columns = [column for column in features if column.startswith("beta_progress__")]
    assert features.loc[1, feature_columns].isna().all()


def test_feature_weights_are_applied_after_scaling():
    rng = np.random.default_rng(20)
    table = pd.DataFrame(
        {
            "channel": [f"E{index}" for index in range(12)],
            "is_bad": [False] * 12,
            "beta_progress__audio_slope_corr": rng.normal(size=12),
            "beta_progress__energy_mean": rng.normal(size=12),
        }
    )
    weights = default_beta_progress_feature_weights(
        ["beta_progress__audio_slope_corr", "beta_progress__energy_mean"]
    )
    result = cluster_electrodes(
        table,
        n_clusters=3,
        feature_weights=weights,
    )
    assert result.feature_weights == (3.0, 0.35)
    np.testing.assert_allclose(
        result.weighted_clean_features[:, 0],
        result.standardized_clean_features[:, 0] * np.sqrt(3.0),
    )
