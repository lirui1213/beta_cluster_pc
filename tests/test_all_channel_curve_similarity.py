"""Curve clustering removes level offsets and response amplitude differences."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'all-channel'))


def test_curve_vectors_ignore_vertical_offsets_and_amplitude_differences():
    from curve_similarity import build_curve_features
    pre=np.array([[5,5,5],[105,105,105],[5,5,5]],dtype=float)
    speech=np.array([[5,3,4,3,5],[105,103,104,103,105],[5,1,3,1,5]],dtype=float)
    post=np.array([[5,5,5],[105,105,105],[5,5,5]],dtype=float)
    cache={'pre_beta':pre,'beta':speech,'post_beta':post,'beta_slope':np.gradient(speech,axis=1),
           'audio_slope':np.tile([0,1,0,-1,0],(3,1)),'progress_percent':np.linspace(0,100,5)}
    values=build_curve_features(cache)
    columns=[c for c in values if c.startswith('curve__')]
    np.testing.assert_allclose(values[columns].iloc[0],values[columns].iloc[1])
    np.testing.assert_allclose(values[columns].iloc[2],values[columns].iloc[0])
    assert not any('energy' in c for c in values)


def test_curve_fit_groups_same_shapes_across_amplitudes_and_ignores_energy():
    import pipeline
    rows=8
    frame=pd.DataFrame({'subject_id':['A']*rows,'block_id':['B01']*rows,
                        'channel':[f'E{i}' for i in range(rows)],'sample_id':[f'A:B01:E{i}' for i in range(rows)],
                        'is_bad':[False]*rows, 'curve__pre_000':[0]*rows,
                        'curve__speech_000':[-1,-1.1,-2,-2.2,-5,-5.1,-10,-10.2],
                        'curve__speech_001':[-2,-2.2,-1,-1.1,-10,-10.2,-5,-5.1],
                        'curve__post_000':[0]*rows,'relation__slope_corr':[.5]*rows,
                        'beta_progress__energy_mean':np.arange(rows)*1e6})
    model,labels=pipeline.fit_global(frame,n_clusters=2)
    assert all(c.startswith(('curve__','relation__')) for c in model.feature_columns)
    assert labels.label.iloc[0] == labels.label.iloc[4]
    assert labels.label.iloc[0] != labels.label.iloc[2]
    np.testing.assert_allclose(model.standardized_clean_features[4],model.standardized_clean_features[0])
    changed=frame.copy()
    changed['beta_progress__energy_mean'] *= -1e10
    other,_=pipeline.fit_global(changed,n_clusters=2)
    np.testing.assert_allclose(other.weighted_clean_features,model.weighted_clean_features)


def test_complete_curves_keep_fixed_real_context_and_relative_power_amplitude():
    from full_curves import summarize_complete_curves
    p=np.linspace(0,100,5)
    pre_time=np.linspace(-.75,0,11)
    post_time=np.linspace(0,.75,11)
    pre=np.full((2,2,11),5.)
    speech=np.tile(np.array([[5,3,4,3,5],[5,1,3,1,5]])[None,:,:],(2,1,1))
    post=pre.copy()
    epochs=SimpleNamespace(pre=pre,speech=speech,post=post,pre_times=pre_time,post_times=post_time,
                           progress_percent=p,audio_pre=np.zeros((2,11)),audio_post=np.zeros((2,11)),
                           audio_speech=np.tile([0,1,.5,1,0],(2,1)),events=pd.DataFrame({'duration':[1.,1.]}))
    cache=summarize_complete_curves(epochs,context_seconds=.75,context_points=11,sigma_seconds=0,slope_method='gradient')
    assert cache['pre_beta'].shape==(2,11)
    assert cache['post_times'][-1]==pytest.approx(.75)
    np.testing.assert_allclose(cache['beta'],speech.mean(axis=0))
    np.testing.assert_allclose(cache['beta_sem'],0)
    with pytest.raises(ValueError,match='context'):
        summarize_complete_curves(epochs,context_seconds=1,context_points=11)


def test_curve_fit_accepts_read_only_pandas_arrays_without_mutating_input():
    from curve_similarity import fit_curve_clusters
    frame=pd.DataFrame({'sample_id':[f'S{i}' for i in range(6)],'is_bad':[False]*6,
                        'curve__speech_000':[-1.,-1.1,-.9,-5.,-5.1,-4.9],
                        'curve__speech_001':[-2.,-2.2,-1.8,-2.5,-2.55,-2.45],
                        'relation__slope_corr':[.1,.2,np.inf,.3,.4,.5]})
    original=frame.copy(deep=True)
    with pd.option_context('mode.copy_on_write',True):
        selected=frame.loc[np.ones(6,dtype=bool),['curve__speech_000','relation__slope_corr']].to_numpy(float)
        assert not selected.flags.writeable
        model=fit_curve_clusters(frame,n_clusters=2)
    pd.testing.assert_frame_equal(frame,original)
    assert np.isfinite(model.weighted_clean_features).all()
    assert model.labels[0]!=model.labels[3]


def test_flat_and_opposite_curves_remain_finite_and_distinct():
    from curve_similarity import normalize_curve_rows
    frame=pd.DataFrame({'curve__pre_000':[0.,0.,0.],
                        'curve__speech_000':[0.,-2.,4.],
                        'curve__post_000':[0.,0.,0.]})
    result=normalize_curve_rows(frame)
    assert np.isfinite(result.to_numpy()).all()
    np.testing.assert_allclose(result.iloc[0],0)
    np.testing.assert_allclose(result.iloc[1],-result.iloc[2])
