from __future__ import annotations

import numpy as np
import pandas as pd

from audio_cross_3band_utils import BetaProgressEpochs
from channelXtrial import (
    classify_trial_types,
    compute_channel_stability,
    extract_trial_type_features,
)


def _synthetic_beta_epochs() -> BetaProgressEpochs:
    progress = np.linspace(0.0, 100.0, 101)
    pre_times = np.linspace(-1.0, 0.0, 21)
    post_times = np.linspace(0.0, 1.0, 21)
    events = pd.DataFrame(
        {
            "trial_id": np.arange(6),
            "original_trial_id": np.arange(6),
            "onset": np.arange(6, dtype=float) * 3.0 + 1.0,
            "offset": np.arange(6, dtype=float) * 3.0 + 2.2,
            "duration": np.full(6, 1.2),
        }
    )
    type1 = np.interp(
        progress,
        [0, 12, 50, 78, 100],
        [0.0, -2.1, 0.9, 0.5, -1.2],
    )
    type2 = np.interp(
        progress,
        [0, 12, 55, 100],
        [0.0, -2.0, -1.7, -1.6],
    )
    pre = np.zeros((6, 2, len(pre_times)))
    post = np.zeros((6, 2, len(post_times)))
    speech = np.zeros((6, 2, len(progress)))
    audio_speech = np.tile(np.interp(progress, [0, 25, 100], [0.0, 1.0, 0.2]), (6, 1))
    audio_pre = np.zeros((6, len(pre_times)))
    audio_post = np.tile(np.interp(post_times, [0, 1], [0.2, 0.0]), (6, 1))

    for trial in range(6):
        speech[trial, 0] = type1
        speech[trial, 1] = type2
        post[trial, 0] = np.linspace(-1.0, 0.8, len(post_times))
        post[trial, 1] = np.linspace(-1.5, 0.8, len(post_times))

    return BetaProgressEpochs(
        speech=speech,
        pre=pre,
        post=post,
        audio_speech=audio_speech,
        audio_pre=audio_pre,
        audio_post=audio_post,
        progress_percent=progress,
        pre_times=pre_times,
        post_times=post_times,
        events=events,
        dropped_events=pd.DataFrame(),
        channel_names=("E0", "E1"),
        bad_channels=frozenset(),
        sfreq=100.0,
        beta_band=(13.0, 30.0),
        pre_post_window=1.0,
    )


def test_channelxtrial_type_features_and_classification():
    epochs = _synthetic_beta_epochs()
    features = extract_trial_type_features(epochs)
    assert features.shape[0] == 12
    assert features["mid_speech_rebound"].notna().all()

    classified = classify_trial_types(features)
    labels_by_channel = classified.groupby("channel")["trial_type"].agg(lambda values: values.mode()[0])
    assert labels_by_channel["E0"] == "Type1"
    assert labels_by_channel["E1"] == "Type2"


def test_channelxtrial_stability_metrics():
    epochs = _synthetic_beta_epochs()
    classified = classify_trial_types(extract_trial_type_features(epochs))
    stability = compute_channel_stability(classified)
    assert set(stability["channel"]) == {"E0", "E1"}
    assert (stability["stability_rate"] == 1.0).all()
    assert (stability["switch_count"] == 0).all()
