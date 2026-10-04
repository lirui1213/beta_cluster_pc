"""Small AND high-variance cluster pruning before the human review window."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "all-channel"))


def examples():
    values = [-1,0,1,-1,0,1, 49,50,51,49,50,51, -10,10, 3,3, 100, np.nan]
    labels = [0]*6 + [1]*6 + [2]*2 + [3]*2 + [4] + [-1]
    frame = pd.DataFrame({"subject_id": ["A"]*18, "block_id": ["B01"]*18,
                          "channel": [f"E{i}" for i in range(18)], "sample_id": [f"A:B01:E{i}" for i in range(18)],
                          "is_bad": [False]*17+[True], "exclusion_reason": [""]*17+["raw_bad"],
                          "beta_progress__early_mean": values})
    scores = np.c_[values, np.zeros(18), np.zeros(18)]
    scores[-1] = np.nan
    first = SimpleNamespace(labels=np.array(labels), pca_scores=scores, channels=tuple(frame.sample_id))
    return frame, first


def test_auto_pruning_requires_both_small_size_and_high_variance():
    from auto_prune import screen_small_high_variance_clusters
    frame, first = examples()
    result = screen_small_high_variance_clusters(frame, first, small_fraction=.15, variance_ratio=3)
    assert result.bad_clusters == (2,)
    assert result.bad_mask.tolist() == [False]*12+[True,True]+[False]*4
    report = result.cluster_report.set_index("label")
    assert report.loc[0,"pca_variance"] == pytest.approx(.8)
    assert report.loc[2,"pca_variance"] == pytest.approx(200)
    assert report.loc[2,"reference_variance"] == pytest.approx(.8)
    assert not report.loc[3,"auto_excluded"]  # Small but tight.
    assert not report.loc[4,"auto_excluded"]  # Singleton: undefined variance.
    assert -1 not in report.index
    assert result.sample_report.loc[12,"auto_excluded"]


def test_large_high_variance_cluster_is_retained():
    from auto_prune import screen_small_high_variance_clusters
    frame, first = examples()
    first.pca_scores[:6,0] = [-20,0,20,-20,0,20]
    result = screen_small_high_variance_clusters(frame, first, small_fraction=.15, variance_ratio=1.1)
    row = result.cluster_report.set_index("label").loc[0]
    assert row.pca_variance > 3
    assert not row.is_small
    assert not row.auto_excluded


def test_no_large_reference_cluster_does_not_guess_a_pruning_threshold():
    from auto_prune import screen_small_high_variance_clusters
    frame, first = examples()
    result = screen_small_high_variance_clusters(frame, first, small_fraction=1, variance_ratio=3)
    assert result.bad_clusters == ()
    assert result.cluster_report.reference_variance.isna().all()
    assert result.cluster_report.auto_reason.str.contains("no_reference").all()


@pytest.mark.parametrize("kwargs", [{"small_fraction":0},{"small_fraction":1.1},{"variance_ratio":0},{"variance_ratio":np.nan}])
def test_invalid_auto_thresholds_fail(kwargs):
    from auto_prune import screen_small_high_variance_clusters
    frame, first = examples()
    with pytest.raises(ValueError):
        screen_small_high_variance_clusters(frame, first, **kwargs)


def test_auto_exclusion_is_audited_and_cannot_be_restored_by_manual_review():
    from auto_prune import screen_small_high_variance_clusters
    from review import refit_reviewed
    frame, first = examples()
    automatic = screen_small_high_variance_clusters(frame, first, small_fraction=.15)
    reviewed, _, assignments, audit = refit_reviewed(frame, first, second_n_clusters=2,
                                                    auto_bad_clusters=automatic.bad_clusters,
                                                    manual_bad_samples=["A:B01:E0"])
    assert assignments.label.iloc[12] == assignments.label.iloc[13] == -1
    assert "auto_small_high_variance_cluster" in assignments.exclusion_reason.iloc[12]
    assert audit.excluded_auto_cluster.iloc[12]
    assert audit.excluded_manual.iloc[0]
    assert not frame.is_bad.iloc[12]


def test_popup_view_hides_auto_clusters_without_changing_first_round_model():
    import pipeline
    from auto_prune import make_popup_result
    frame, _ = examples()
    first, _ = pipeline.fit_global(frame, n_clusters=3)
    bad = np.zeros(len(frame), dtype=bool)
    bad[12:14] = True
    view = make_popup_result(first, bad)
    assert (view.labels[bad] == -1).all()
    assert (first.labels[bad] >= 0).all()
    assert np.isfinite(view.pca_scores[bad]).all()  # Fixed first-round coordinates for review.
    assert not view.clean_mask[bad].any()


def test_auto_policy_disabled_keeps_every_original_sample():
    from auto_prune import screen_small_high_variance_clusters
    frame, first = examples()
    result = screen_small_high_variance_clusters(frame, first, enabled=False, small_fraction=.15)
    assert not result.bad_mask.any()
    assert result.bad_clusters == ()


def test_unestimated_pc_dimensions_are_zero_only_in_review_coordinates():
    import pipeline
    from auto_prune import screen_small_high_variance_clusters, review_coordinates
    frame, _ = examples()
    first, _ = pipeline.fit_global(frame, n_clusters=3)
    assert np.isnan(first.pca_scores[:,1:]).all()
    coordinates = review_coordinates(first)
    np.testing.assert_allclose(coordinates[first.clean_mask,1:], 0)
    assert np.isnan(first.pca_scores[:,1:]).all()
    automatic = screen_small_high_variance_clusters(frame, first)
    assert np.isfinite(automatic.cluster_report.pca_variance.dropna()).all()


def test_stale_automatic_screening_cannot_be_applied_to_a_different_first_model(tmp_path):
    import pipeline
    from auto_prune import screen_small_high_variance_clusters
    from review import run_second_round
    frame, fake_first = examples()
    stale = screen_small_high_variance_clusters(frame, fake_first, small_fraction=.15)
    current, labels = pipeline.fit_global(frame,n_clusters=2)
    analysis = pipeline.GlobalAnalysisResult(frame,labels,current,tmp_path)
    with pytest.raises(ValueError,match="current first-round clustering"):
        run_second_round(analysis,tmp_path/'second',automatic_pruning=stale)
