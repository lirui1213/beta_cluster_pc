"""Complete beta/audio curves and example-style plots with real-time context."""
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from signal_processing import finite_mean,gaussian_smooth_nan
from dynamics import summarize_speech_relationship
from visualization import cluster_color


def _sem(data):
    count=np.isfinite(data).sum(axis=0)
    mean=finite_mean(data)
    residual=np.where(np.isfinite(data),data-mean,0)
    var=np.divide((residual**2).sum(axis=0),count-1,out=np.full(mean.shape,np.nan),where=count>1)
    return np.sqrt(var/np.maximum(count,1))


def summarize_complete_curves(epochs,*,context_seconds=.75,context_points=51,sigma_seconds=.10,slope_method='savgol'):
    if not np.isfinite(context_seconds) or context_seconds<=0 or context_points<3:
        raise ValueError('Invalid context window or point count')
    if epochs.pre_times[0]>-context_seconds+1e-8 or epochs.post_times[-1]<context_seconds-1e-8:
        raise ValueError('Extracted pre/post context is shorter than requested curve context')
    valid=np.isfinite(epochs.speech).all(axis=2)&np.isfinite(epochs.pre).all(axis=2)&np.isfinite(epochs.post).all(axis=2)
    audio_valid=np.isfinite(epochs.audio_speech).all(axis=1)&np.isfinite(epochs.audio_pre).all(axis=1)&np.isfinite(epochs.audio_post).all(axis=1)
    valid &= audio_valid[:,None]
    masked=np.where(valid[:,:,None],epochs.speech,np.nan)
    speech_epochs=SimpleNamespace(speech=masked,audio_speech=epochs.audio_speech,
                                  events=epochs.events,progress_percent=epochs.progress_percent)
    cache=summarize_speech_relationship(speech_epochs,sigma_seconds=sigma_seconds,slope_method=slope_method)
    finite_audio=epochs.audio_speech[np.isfinite(epochs.audio_speech)]
    audio_scale=float(np.max(np.abs(finite_audio))) if len(finite_audio) else 0
    if audio_scale<=0:raise ValueError('Invalid audio envelope')
    smooth_speech=[]
    for trial,duration in zip(masked,epochs.events.duration):
        rate=(len(epochs.progress_percent)-1)/float(duration)
        smooth_speech.append(gaussian_smooth_nan(trial,rate,sigma_seconds=sigma_seconds))
    cache['beta_sem']=_sem(np.asarray(smooth_speech))
    for phase,axis,beta,audio in [('pre',epochs.pre_times,epochs.pre,epochs.audio_pre),('post',epochs.post_times,epochs.post,epochs.audio_post)]:
        grid=np.linspace(-context_seconds,0,context_points) if phase=='pre' else np.linspace(0,context_seconds,context_points)
        beta_resampled=np.empty((*beta.shape[:2],context_points))
        audio_resampled=np.empty((len(audio),context_points))
        rate=1/float(np.median(np.diff(axis)))
        for i in range(len(beta)):
            smooth=gaussian_smooth_nan(beta[i],rate,sigma_seconds=sigma_seconds)
            for channel in range(beta.shape[1]):
                beta_resampled[i,channel]=np.interp(grid,axis,smooth[channel])
            smooth_audio=gaussian_smooth_nan(audio[i]/audio_scale,rate,sigma_seconds=sigma_seconds)
            audio_resampled[i]=np.interp(grid,axis,smooth_audio)
        beta_resampled=np.where(valid[:,:,None],beta_resampled,np.nan)
        paired_audio=np.where(valid[:,:,None],audio_resampled[:,None,:],np.nan)
        cache[phase+'_beta']=finite_mean(beta_resampled)
        cache[phase+'_beta_sem']=_sem(beta_resampled)
        cache[phase+'_audio']=finite_mean(paired_audio)
        cache[phase+'_times']=grid
    cache['progress_percent']=np.asarray(epochs.progress_percent)
    return cache


def display_arrays(cache):
    duration=float(np.asarray(cache['display_speech_seconds']).item())
    times=np.r_[cache['pre_times'][:-1],cache['progress_percent']/100*duration,duration+cache['post_times'][1:]]*1000
    if 'full_beta' in cache:
        return times,cache['full_beta'],cache['full_audio'],duration
    beta=np.concatenate([cache['pre_beta'][:,:-1],cache['beta'],cache['post_beta'][:,1:]],axis=1)
    audio=np.concatenate([cache['pre_audio'][:,:-1],cache['audio'],cache['post_audio'][:,1:]],axis=1)
    return times,beta,audio,duration


def _panel(ax,times,beta,sem,audio,duration,context,title,color):
    ax.plot(times,beta,color=color,lw=2)
    ax.fill_between(times,beta-sem,beta+sem,color=color,alpha=.20,lw=0)
    twin=ax.twinx()
    twin.plot(times,audio,color='orange',lw=1.5)
    twin.set_ylabel('Audio envelope / block peak',color='darkorange')
    twin.tick_params(axis='y',colors='darkorange')
    finite_audio=audio[np.isfinite(audio)]
    twin.set_ylim(0,max(1.05,float(finite_audio.max())*1.05 if len(finite_audio) else 1.05))
    ax.axvspan(times[0],0,color='gray',alpha=.07)
    ax.axvspan(duration*1000,times[-1],color='gray',alpha=.07)
    for position in [0,duration*1000]:ax.axvline(position,color='black',ls='--',lw=.9)
    ax.set(xlabel='Duration-normalized time (ms)',ylabel='Normalized beta change',title=title,xlim=(times[0],times[-1]))
    ax.text(.02,.94,f'pre/post {context:g}s',transform=ax.transAxes,va='top',fontsize=9)


def save_complete_plots(assignments,cache,output,*,save_channels=False):
    times,beta,audio,duration=display_arrays(cache)
    context=float(cache['post_times'][-1])
    labels=sorted(assignments.loc[assignments.label>=0,'label'].unique())
    columns=min(4,len(labels))
    rows=int(np.ceil(len(labels)/columns))
    fig,axes=plt.subplots(rows,columns,figsize=(5.3*columns,3.7*rows),squeeze=False)
    summaries=[]
    for ax,label in zip(axes.flat,labels):
        mask=assignments.label.to_numpy()==label
        mean=finite_mean(beta[mask])
        sem=_sem(beta[mask])
        envelope=finite_mean(audio[mask])
        _panel(ax,times,mean,sem,envelope,duration,context,f'Cluster {label}\n(n={int(mask.sum())})',cluster_color(int(label)))
        summaries.extend({'label':int(label),'display_time_ms':float(t),'mean_beta_shape':float(b),'sample_sem':float(s),'mean_audio_scaled':float(a)}
                         for t,b,s,a in zip(times,mean,sem,envelope))
    for ax in list(axes.flat)[len(labels):]:ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(output/'cluster_grid_beta_audio.png',dpi=180)
    plt.close(fig)
    pd.DataFrame(summaries).to_csv(output/'complete_curve_summary.csv',index=False,encoding='utf-8-sig')
    if not save_channels:return
    sem=cache['full_beta_sem'] if 'full_beta_sem' in cache else np.concatenate([cache['pre_beta_sem'][:,:-1],cache['beta_sem'],cache['post_beta_sem'][:,1:]],axis=1)
    records=[]
    for i,row in enumerate(assignments.itertuples()):
        if not np.isfinite(beta[i]).any():
            records.append({'sample_id':row.sample_id,'image_path':'','status':'no_finite_curve'})
            continue
        directory=output/'channel_figures'/row.subject_id/row.block_id
        directory.mkdir(parents=True,exist_ok=True)
        path=directory/f'{i:05d}_{quote(row.channel,safe="")}.png'
        fig,ax=plt.subplots(figsize=(6,4))
        _panel(ax,times,beta[i],sem[i],audio[i],duration,context,f'{row.subject_id}/{row.block_id}/{row.channel}','red')
        fig.tight_layout()
        fig.savefig(path,dpi=150)
        plt.close(fig)
        records.append({'sample_id':row.sample_id,'image_path':str(path.resolve()),'status':'saved'})
        if (i+1)%100==0:print(f'Channel figures: {i+1}/{len(assignments)}',flush=True)
    pd.DataFrame(records).to_csv(output/'channel_figure_index.csv',index=False,encoding='utf-8-sig')
