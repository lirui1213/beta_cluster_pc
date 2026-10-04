"""Independent second-round fitting and paired beta/audio display contracts."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "all-channel"))


def table():
    return pd.DataFrame({"subject_id": ["A"] * 9, "block_id": ["B01"] * 9,
                         "channel": [f"E{i}" for i in range(9)],
                         "sample_id": [f"A:B01:E{i}" for i in range(9)],
                         "is_bad": [False] * 8 + [True], "exclusion_reason": [""] * 8 + ["raw_bad"],
                         "beta_progress__early_mean": [-3, -2.8, -3.2, 3, 2.8, 3.2, 100, 110, np.nan]})


def test_second_round_removes_reviewed_samples_and_refits_every_estimator():
    import pipeline
    from review import refit_reviewed
    original = table()
    first, assignments = pipeline.fit_global(original, n_clusters=3)
    outlier_label = int(assignments.label.iloc[6])
    reviewed, second, labels, audit = refit_reviewed(
        original, first, second_n_clusters=2, bad_clusters=[outlier_label], manual_bad_samples=["A:B01:E0"])
    assert not original.is_bad.iloc[0]
    assert reviewed.is_bad.iloc[0]
    assert labels.label.iloc[6] == labels.label.iloc[7] == labels.label.iloc[8] == -1
    assert labels.label.iloc[0] == -1
    assert labels.pc1.iloc[0] != labels.pc1.iloc[0]
    assert second.scaler is not first.scaler
    assert second.imputer is not first.imputer
    assert second.kmeans is not first.kmeans
    assert second.scaler.mean_[0] == pytest.approx(.6)
    assert labels.label.iloc[1] != labels.label.iloc[3]
    assert "manual_sample" in labels.exclusion_reason.iloc[0]
    assert audit.sample_id.tolist() == original.sample_id.tolist()


@pytest.mark.parametrize("kwargs", [{"manual_bad_samples": ["E0"]}, {"bad_clusters": [999]}, {"pca_distance_threshold": -1}])
def test_unknown_review_selection_or_invalid_distance_fails(kwargs):
    import pipeline
    from review import refit_reviewed
    first, _ = pipeline.fit_global(table(), n_clusters=3)
    with pytest.raises(ValueError):
        refit_reviewed(table(), first, second_n_clusters=2, **kwargs)


def test_pca_distance_rule_uses_three_pc_distance_to_assigned_cluster_centroid():
    from review import pca_review_table
    first = SimpleNamespace(labels=np.array([0, 0, 0, 1, 1, -1]),
                            pca_scores=np.array([[0,0,0], [0,0,0], [0,0,9], [100,0,0], [100,0,0], [np.nan]*3]),
                            channels=tuple(f"S{i}" for i in range(6)))
    features = pd.DataFrame({"sample_id": list(first.channels)})
    audit = pca_review_table(features, first)
    np.testing.assert_allclose(audit.pca_cluster_distance.iloc[:5], [3,3,6,0,0])
    assert np.isnan(audit.pca_cluster_distance.iloc[5])


def test_relationship_curves_use_same_trials_and_average_within_trial_products():
    from dynamics import summarize_speech_relationship
    progress = np.array([0.,25.,50.,75.,100.])
    beta = np.array([[2 * progress], [np.full(5, np.nan)]])
    audio = np.array([progress / 100, 1 - progress / 100])
    epochs = SimpleNamespace(speech=beta, audio_speech=audio, progress_percent=progress,
                             events=pd.DataFrame({"duration": [1.,1.]}))
    values = summarize_speech_relationship(epochs, sigma_seconds=0, slope_method="gradient")
    np.testing.assert_allclose(values["beta"][0], 2 * progress)
    np.testing.assert_allclose(values["audio"][0], progress / 100)
    np.testing.assert_allclose(values["beta_slope"][0], 2)
    np.testing.assert_allclose(values["audio_slope"][0], .01)
    np.testing.assert_allclose(values["interaction"][0], .02)


def test_review_rejects_features_not_aligned_with_first_round():
    import pipeline
    from review import refit_reviewed
    first, _ = pipeline.fit_global(table(), n_clusters=3)
    with pytest.raises(ValueError, match="align"):
        refit_reviewed(table().iloc[::-1], first, second_n_clusters=2)


def test_distance_threshold_excludes_only_points_above_threshold():
    from review import refit_reviewed
    frame = table().iloc[:6].copy()
    frame.loc[5, "is_bad"] = True
    first = SimpleNamespace(labels=np.array([0,0,0,1,1,-1]),
                            pca_scores=np.array([[0,0,0],[0,0,0],[0,0,9],[100,0,0],[100,0,0],[np.nan]*3]),
                            channels=tuple(frame.sample_id))
    reviewed, _, labels, audit = refit_reviewed(frame, first, second_n_clusters=2, pca_distance_threshold=4)
    assert audit.excluded_distance.tolist() == [False,False,True,False,False,False]
    assert reviewed.is_bad.tolist() == [False,False,True,False,False,True]
    assert labels.label.iloc[2] == -1
    assert "pca_distance" in labels.exclusion_reason.iloc[2]


def test_slope_product_is_averaged_before_trial_slopes_cancel():
    from dynamics import summarize_speech_relationship
    p = np.array([0.,25.,50.,75.,100.])
    epochs = SimpleNamespace(speech=np.array([[2*p],[-2*p]]),
                             audio_speech=np.array([p/100,1-p/100]), progress_percent=p,
                             events=pd.DataFrame({"duration":[1.,1.]}))
    values = summarize_speech_relationship(epochs, sigma_seconds=0, slope_method="gradient")
    np.testing.assert_allclose(values["beta_slope"], 0)
    np.testing.assert_allclose(values["audio_slope"], 0)
    np.testing.assert_allclose(values["interaction"], .02)
