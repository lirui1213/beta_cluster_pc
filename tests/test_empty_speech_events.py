from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from audio_cross_3band_utils import (
    build_beta_progress_feature_table_from_files,
    filter_duration_outlier_trials,
    load_speech_events,
)


@pytest.mark.parametrize("source,offset", [
    (np.empty((0, 2)), None),
    (np.array([]), np.array([])),
    (pd.DataFrame(columns=["onset", "offset"]), None),
])
def test_event_loader_rejects_empty_intervals(source, offset):
    with pytest.raises(ValueError, match="No speech events"):
        load_speech_events(source, offset)


def test_empty_npy_error_identifies_both_input_paths(tmp_path):
    onset, offset = tmp_path / "onset.npy", tmp_path / "offset.npy"
    np.save(onset, np.array([]))
    np.save(offset, np.array([]))
    with pytest.raises(ValueError, match="No speech events") as caught:
        load_speech_events(onset, offset)
    assert str(onset) in str(caught.value)
    assert str(offset) in str(caught.value)


def test_empty_csv_error_identifies_input_path(tmp_path):
    path = tmp_path / "events.csv"
    pd.DataFrame(columns=["onset", "offset"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="No speech events") as caught:
        load_speech_events(path)
    assert str(path) in str(caught.value)


def test_duration_filter_rejects_empty_input_without_numpy_warnings():
    events = pd.DataFrame(columns=["onset", "offset", "duration"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="No speech events"):
            filter_duration_outlier_trials(events)


def test_file_pipeline_rejects_empty_events_before_loading_fif_or_audio(tmp_path):
    onset, offset = tmp_path / "onset.npy", tmp_path / "offset.npy"
    np.save(onset, np.array([]))
    np.save(offset, np.array([]))
    with pytest.raises(ValueError, match="No speech events"):
        build_beta_progress_feature_table_from_files(
            tmp_path / "not_loaded.fif", onset, tmp_path / "not_loaded.wav", offset_source=offset,
        )


def test_nonempty_equal_duration_trials_still_work():
    events = load_speech_events([[2, 4], [6, 8], [10, 12]])
    kept, dropped = filter_duration_outlier_trials(events)
    assert len(kept) == 3
    assert dropped.empty
    assert (kept.duration == 2).all()
