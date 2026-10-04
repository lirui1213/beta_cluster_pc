"""Continuous-source curves have no independently smoothed window seams."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'all-channel'))


def test_native_curves_sample_one_continuously_smoothed_power_signal():
    from native_curves import summarize_native_signal
    from signal_processing import gaussian_smooth_nan
    t=np.arange(0,10.001,.01)
    source=10+np.sin(t*2)+.2*t
    power=np.stack([source,100+source*5])
    events=pd.DataFrame({'onset':[2.,6.],'offset':[4.,8.],'duration':[2.,2.]})
    epochs=SimpleNamespace(events=events,channel_names=('E0','E1'),
                           speech=np.zeros((2,2,101)),progress_percent=np.linspace(0,100,101))
    cache=summarize_native_signal(power,t,1+.2*np.sin(t),t,epochs,
                                  context_seconds=.75,context_points=51,sigma_seconds=.1)
    np.testing.assert_allclose(cache['full_beta'][0],cache['full_beta'][1],atol=1e-12)
    query=np.r_[np.linspace(-.75,0,51)[:-1],np.linspace(0,2,101),2+np.linspace(0,.75,51)[1:]]
    smooth=gaussian_smooth_nan(source,100,sigma_seconds=.1)
    expected=np.mean([np.interp(start+query,t,smooth) for start in events.onset],axis=0)
    expected-=expected[:51].mean()
    expected/=np.sqrt(np.mean(expected**2))
    np.testing.assert_allclose(cache['full_beta'][0],expected,atol=1e-12)
    assert cache['full_beta'].shape==(2,201)
    assert np.isfinite(cache['full_beta_sem']).all()
    assert abs(cache['full_beta'][0,:51].mean())<1e-12
    assert abs(np.sqrt(np.mean(cache['full_beta'][0]**2))-1)<1e-12


def test_full_display_uses_native_whole_curve_instead_of_segment_arrays():
    from full_curves import display_arrays
    cache={'display_speech_seconds':np.asarray(2.),'pre_times':np.linspace(-.75,0,51),
           'post_times':np.linspace(0,.75,51),'progress_percent':np.linspace(0,100,101),
           'full_beta':np.ones((2,201)),'full_audio':np.zeros((2,201))}
    times,beta,audio,duration=display_arrays(cache)
    assert beta is cache['full_beta']
    assert audio is cache['full_audio']
    assert times[50]==0 and times[150]==2000


def test_constant_native_power_stays_zero_at_physiological_scale():
    from native_curves import summarize_native_signal
    t=np.arange(0,10.001,.01)
    epochs=SimpleNamespace(events=pd.DataFrame({'onset':[2.,6.],'offset':[4.,8.],'duration':[2.,2.]}),
                           channel_names=('E0',),speech=np.zeros((2,1,101)),
                           progress_percent=np.linspace(0,100,101))
    cache=summarize_native_signal(np.full((1,len(t)),1e-10),t,np.ones(len(t)),t,epochs)
    np.testing.assert_allclose(cache['full_beta'],0,atol=1e-12)
