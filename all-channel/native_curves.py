"""Sample each trial directly from continuous beta power, without window seams."""
import numpy as np
from audio_cross_3band_utils import load_ecog_fif,extract_audio_envelope,resolve_frequency_bands
from signal_processing import gaussian_smooth_nan,finite_mean
from full_curves import _sem


def summarize_native_signal(power,times,audio,audio_times,epochs,*,context_seconds=.75,
                            context_points=51,sigma_seconds=.10):
    if not np.isfinite(context_seconds) or context_seconds<=0 or context_points<3:
        raise ValueError('Invalid continuous curve context')
    times=np.asarray(times,dtype=float)
    audio_times=np.asarray(audio_times,dtype=float)
    power=gaussian_smooth_nan(power,1/np.median(np.diff(times)),sigma_seconds=sigma_seconds)
    audio=gaussian_smooth_nan(audio,1/np.median(np.diff(audio_times)),sigma_seconds=sigma_seconds)
    pre=np.linspace(-context_seconds,0,context_points)
    post=np.linspace(0,context_seconds,context_points)
    progress=np.asarray(epochs.progress_percent)
    n=context_points*2+len(progress)-2
    trial_beta=np.empty((len(epochs.events),len(epochs.channel_names),n))
    trial_audio=np.empty((len(epochs.events),n))
    for i,event in enumerate(epochs.events.itertuples()):
        # A single query on a continuously processed signal; no signal fragments join.
        query=event.onset+np.r_[pre[:-1],progress/100*event.duration,event.duration+post[1:]]
        for c in range(len(epochs.channel_names)):
            trial_beta[i,c]=np.interp(query,times,power[c],left=np.nan,right=np.nan)
        trial_audio[i]=np.interp(query,audio_times,audio,left=np.nan,right=np.nan)
    valid=np.isfinite(epochs.speech).all(axis=2)&np.isfinite(trial_beta).all(axis=2)
    valid &= np.isfinite(trial_audio).all(axis=1)[:,None]
    trial_beta=np.where(valid[:,:,None],trial_beta,np.nan)
    mean=finite_mean(trial_beta)
    reference=np.mean(mean[:,:context_points],axis=1)
    centered=mean-reference[:,None]
    rms=np.sqrt(np.mean(centered**2,axis=1))
    scale=np.where(rms>np.finfo(float).tiny,rms,1.)
    trial_beta=(trial_beta-reference[None,:,None])/scale[None,:,None]
    full_beta=finite_mean(trial_beta)
    full_sem=_sem(trial_beta)
    left=context_points-1
    right=left+len(progress)
    audio_peak=float(np.nanmax(trial_audio[:,left:right]))
    if not np.isfinite(audio_peak) or audio_peak<=0:
        raise ValueError('Continuous audio has no positive speech envelope')
    paired_audio=np.where(valid[:,:,None],trial_audio[:,None,:]/audio_peak,np.nan)
    speech_beta=trial_beta[:,:,left:right]
    speech_audio=paired_audio[:,:,left:right]
    b_slope=np.gradient(speech_beta,progress,axis=-1)
    a_slope=np.gradient(speech_audio,progress,axis=-1)
    full_audio=finite_mean(paired_audio)
    cache={'full_beta':full_beta,'full_beta_sem':full_sem,'full_audio':full_audio,
           'beta':full_beta[:,left:right],'beta_sem':full_sem[:,left:right],
           'pre_beta':full_beta[:,:context_points],'pre_beta_sem':full_sem[:,:context_points],
           'post_beta':full_beta[:,right-1:],'post_beta_sem':full_sem[:,right-1:],
           'audio':full_audio[:,left:right],'pre_audio':full_audio[:,:context_points],
           'post_audio':full_audio[:,right-1:], 'beta_slope':finite_mean(b_slope),
           'audio_slope':finite_mean(a_slope),'interaction':finite_mean(b_slope*a_slope),
           'pre_times':pre,'post_times':post,'progress_percent':progress,
           'curve_valid_trial_count':valid.sum(axis=0)}
    return cache


def summarize_native_files(fif_path,wav_path,epochs,**options):
    raw=load_ecog_fif(fif_path,preload=False).pick('ecog').load_data()
    if tuple(raw.ch_names)!=tuple(epochs.channel_names):
        raise ValueError('Native curve channel order differs from quality extraction')
    low,high=resolve_frequency_bands(float(raw.info['sfreq']),{'beta':epochs.beta_band})['beta']
    raw.filter(l_freq=low,h_freq=high,phase='zero',verbose='ERROR')
    raw.apply_hilbert(envelope=True,verbose='ERROR')
    power=np.square(raw.get_data())
    envelope=extract_audio_envelope(wav_path)
    return summarize_native_signal(power,raw.times,envelope.values,envelope.times,epochs,**options)
