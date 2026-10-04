"""Weighted shape distances: remove offsets and per-sample response amplitude."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from clustering_pipeline import ClusteringResult


def _corr(x,y):
    valid=np.isfinite(x)&np.isfinite(y)
    if valid.sum()<3:
        return np.nan
    x=x[valid]-x[valid].mean()
    y=y[valid]-y[valid].mean()
    norm=np.linalg.norm(x)*np.linalg.norm(y)
    return float(np.dot(x,y)/norm) if norm>np.finfo(float).eps else np.nan


def normalize_curve_rows(features):
    """One RMS per full beta curve, preserving direction and segment proportions."""
    result=features.copy()
    columns=[c for c in result if c.startswith('curve__')]
    if not columns:
        return result
    values=result[columns].to_numpy(dtype=float,copy=True)
    finite=np.isfinite(values)
    count=finite.sum(axis=1)
    rms=np.sqrt(np.divide(np.where(finite,values**2,0).sum(axis=1),count,
                          out=np.zeros(len(values)),where=count>0))
    scale=np.where(rms>np.finfo(float).eps,rms,1.)
    result[columns]=values/scale[:,None]
    return result


def build_curve_features(cache):
    """Subtract the pre mean, then normalize the entire response curve's RMS."""
    pre=np.asarray(cache['pre_beta'])
    reference=np.mean(pre,axis=1)
    columns={}
    for phase,key in [('pre','pre_beta'),('speech','beta'),('post','post_beta')]:
        values=np.asarray(cache[key])-reference[:,None]
        columns.update({f'curve__{phase}_{i:03d}':values[:,i] for i in range(values.shape[1])})
    zero_corr,peak_corr,lag_fraction=[],[],[]
    n=np.asarray(cache['beta']).shape[1]
    limit=max(1,int(round((n-1)*.10)))
    for beta,audio in zip(cache['beta_slope'],cache['audio_slope']):
        zero_corr.append(_corr(beta,audio))
        best,best_lag=np.nan,0
        for lag in range(-limit,limit+1):
            if lag<0:
                x,y=beta[-lag:],audio[:lag]
            elif lag>0:
                x,y=beta[:-lag],audio[lag:]
            else:
                x,y=beta,audio
            value=_corr(x,y)
            if np.isfinite(value) and (not np.isfinite(best) or abs(value)>abs(best)):
                best,best_lag=value,lag
        peak_corr.append(best)
        lag_fraction.append(best_lag/(n-1) if np.isfinite(best) else np.nan)
    columns['relation__slope_corr']=zero_corr
    columns['relation__slope_xcorr']=peak_corr
    # Positive lag pairs beta(t) with audio(t+lag): audio occurs later.
    columns['relation__audio_delay_fraction']=lag_fraction
    return normalize_curve_rows(pd.DataFrame(columns))


class SharedBetaScaler(BaseEstimator,TransformerMixin):
    """One pooled beta scale for all rows and beta bins; relationships stay bounded."""
    def __init__(self,beta_indices):
        self.beta_indices=beta_indices

    def fit(self,values,y=None):
        values=np.asarray(values,dtype=float)
        rms=float(np.sqrt(np.mean(values[:,self.beta_indices]**2)))
        self.beta_scale_=rms if rms>np.finfo(float).eps else 1.
        self.scale_=np.ones(values.shape[1])
        self.scale_[self.beta_indices]=self.beta_scale_
        self.n_features_in_=values.shape[1]
        return self

    def transform(self,values):
        return np.asarray(values,dtype=float)/self.scale_

    def inverse_transform(self,values):
        return np.asarray(values,dtype=float)*self.scale_


def fit_curve_clusters(features,*,n_clusters=2,random_state=50):
    """K-means uses Euclidean distances on curve bins, not correlation distance."""
    # Also normalize existing curve tables; do this before selecting constant bins.
    features=normalize_curve_rows(features)
    clean=~features.is_bad.fillna(False).to_numpy(bool)
    count=int(clean.sum())
    if not isinstance(n_clusters,(int,np.integer)) or not 2<=n_clusters<=count:
        raise ValueError(f'n_clusters must be between 2 and {count}')
    columns=[c for c in features if c.startswith(('curve__','relation__'))]
    # Time bins are intentionally correlated: do not prune neighboring curve bins.
    selected=[]
    for c in columns:
        values=pd.to_numeric(features.loc[clean,c],errors='coerce').replace([np.inf,-np.inf],np.nan).dropna().to_numpy()
        if len(values)>=max(2,int(np.ceil(count/2))) and np.ptp(values)>max(float(np.max(np.abs(values))),1.)*1e-12:
            selected.append(c)
    if not any(c.startswith('curve__') for c in selected):
        raise ValueError('No varying response curves remain among eligible samples')
    # pandas Copy-on-Write may return a read-only view; sanitize an owned copy.
    raw=features.loc[clean,selected].to_numpy(dtype=float,copy=True)
    raw[~np.isfinite(raw)]=np.nan
    imputer=SimpleImputer(strategy='median')
    imputed=imputer.fit_transform(raw)
    beta_indices=np.array([i for i,c in enumerate(selected) if c.startswith('curve__')],dtype=int)
    scaler=SharedBetaScaler(beta_indices)
    standardized=scaler.fit_transform(imputed)
    budgets={'pre':.20,'speech':.60,'post':.20,'relation':.25}
    groups=[c.split('__')[1].split('_')[0] if c.startswith('curve__') else 'relation' for c in selected]
    weights=np.array([budgets[g]/groups.count(g) for g in groups])
    weighted=standardized*np.sqrt(weights)
    model=KMeans(n_clusters=n_clusters,n_init=50,random_state=random_state)
    labels=np.full(len(features),-1,dtype=int)
    labels[clean]=model.fit_predict(weighted)
    pca=PCA(n_components=min(3,count,weighted.shape[1]))
    scores=np.full((len(features),3),np.nan)
    transformed=pca.fit_transform(weighted)
    scores[clean,:transformed.shape[1]]=transformed
    return ClusteringResult(labels,scores,clean,tuple(features.sample_id),tuple(selected),imputer,scaler,model,pca,
                            standardized,tuple(weights),weighted)
