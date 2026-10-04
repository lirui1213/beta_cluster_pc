"""Descriptive beta cluster stability under independent trial resampling.

These diagnostics assess repeatability, not statistical significance. No
threading environment variables are modified by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import adjusted_rand_score, silhouette_score

from audio_cross_3band_utils import BetaProgressEpochs, extract_beta_progress_features
import clustering_pipeline
from clustering_pipeline import cluster_electrodes, infer_feature_columns
from signal_processing import gaussian_smooth_nan


@dataclass(frozen=True)
class ClusterStabilityEvaluation:
    """Per-K summaries, resamples, recommendation, and an epoch-aligned mask."""

    summary: pd.DataFrame
    resamples: pd.DataFrame
    recommended_k: int
    clean_mask: NDArray[np.bool_]


class NoEligibleClusterError(ValueError):
    """No recommendation, with diagnostics retained for inspection/export."""

    def __init__(self, summary, resamples, clean_mask):
        super().__init__(
            "No eligible K: full-data clusters are collapsed/tiny or fewer than "
            "80% supported complete stability replicates remain for either resampling type."
        )
        self.summary = summary
        self.resamples = resamples
        self.clean_mask = clean_mask.copy()


_METADATA = frozenset({
    "channel", "is_bad", "label", "pc1", "pc2", "pc3", "original_trial_id",
    "baseline_invalid_fraction", "valid_trial_count", "baseline_mean", "baseline_std",
    "denominator_floor", "is_low_variance", "valid",
})


def _slice_epochs(epochs: BetaProgressEpochs, indices: np.ndarray) -> BetaProgressEpochs:
    """Preserve bootstrap multiplicity in every trial array and QC table."""
    events = epochs.events.iloc[indices].copy().reset_index(drop=True)
    updates = {
        name: np.asarray(getattr(epochs, name))[indices].copy()
        for name in ("speech", "pre", "post", "audio_speech", "audio_pre", "audio_post")
    }
    diagnostics = getattr(epochs, "baseline_diagnostics", None)
    if diagnostics is not None:
        ids = epochs.events.get("original_trial_id", pd.Series(np.arange(len(epochs.events))))
        selected_ids = ids.to_numpy()[indices]
        if "original_trial_id" not in events:
            events["original_trial_id"] = selected_ids
        if "original_trial_id" in diagnostics:
            updates["baseline_diagnostics"] = pd.concat(
                [diagnostics.loc[diagnostics.original_trial_id == trial_id] for trial_id in selected_ids],
                ignore_index=True,
            ).copy()
        else:
            raise ValueError("baseline_diagnostics must contain original_trial_id.")
    return replace(epochs, events=events, **updates)


def _valid_trials(epochs: BetaProgressEpochs) -> np.ndarray:
    """Match feature QC: an observation needs a finite speech curve and valid QC."""
    valid = np.isfinite(epochs.speech).all(axis=2)
    diagnostics = getattr(epochs, "baseline_diagnostics", None)
    if diagnostics is not None:
        required = {"original_trial_id", "channel", "valid"}
        if not required.issubset(diagnostics.columns):
            raise ValueError(f"baseline_diagnostics must contain {sorted(required)}.")
        ids = epochs.events.get("original_trial_id", pd.Series(np.arange(len(epochs.events))))
        quality = diagnostics.copy()
        quality["valid"] = quality["valid"].fillna(False).astype(bool)
        by_trial = quality.groupby(["original_trial_id", "channel"])["valid"].all().unstack()
        aligned = by_trial.reindex(index=ids.to_numpy(), columns=epochs.channel_names)
        valid &= aligned.fillna(False).to_numpy(dtype=bool)
    return valid


def _aligned_features(epochs: BetaProgressEpochs, kwargs: Mapping[str, Any]) -> pd.DataFrame:
    table = extract_beta_progress_features(epochs, **kwargs).copy()
    if "channel" not in table or table.channel.duplicated().any():
        raise ValueError("Extracted features must identify each channel exactly once.")
    if set(table.channel) != set(epochs.channel_names):
        raise ValueError("Extracted feature channels must match epochs.channel_names.")
    table = table.set_index("channel").loc[list(epochs.channel_names)].reset_index()
    numeric = table.select_dtypes(include=np.number).columns
    table[numeric] = table[numeric].replace([np.inf, -np.inf], np.nan)
    return table


def _feature_bad_mask(table: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    numeric = table.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    bad = ~np.isfinite(numeric).any(axis=1)
    if "is_bad" in table:
        bad |= table.is_bad.fillna(False).to_numpy(bool)
    return bad


def _sample_support_reason(
    table: pd.DataFrame,
    valid_trials: np.ndarray,
    columns: Sequence[str],
    clean_mask: np.ndarray,
    max_invalid_fraction: float,
) -> str:
    """Reject unsupported samples rather than changing the original population."""
    counts = valid_trials.sum(axis=0)
    numeric = table.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    checks = {
        "insufficient_valid_observations": counts < 2,
        "invalid_fraction_exceeded": 1.0 - counts / len(valid_trials) > max_invalid_fraction,
        "sample_is_bad": table.get("is_bad", pd.Series(False, index=table.index)).fillna(False).to_numpy(bool),
        "wholly_missing_features": ~np.isfinite(numeric).any(axis=1),
    }
    reasons = []
    for reason, failed in checks.items():
        affected = table.loc[clean_mask & failed, "channel"].astype(str).tolist()
        if affected:
            reasons.append(f"{reason}:{','.join(affected)}")
    return "; ".join(reasons)


def _cluster_size(fit, k: int) -> tuple[int | float, bool]:
    if fit is None:
        return np.nan, False
    labels = fit.labels[fit.clean_mask]
    _, counts = np.unique(labels, return_counts=True)
    complete = len(counts) == k and np.all(labels >= 0)
    return (int(counts.min()) if complete else 0), bool(complete)


def _finite_mean(values: np.ndarray, axis: int) -> np.ndarray:
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    total = np.where(finite, values, 0.0).sum(axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)


def _trial_mean_waveforms(epochs: BetaProgressEpochs, *, sigma_seconds: float) -> np.ndarray:
    """Smooth trials in seconds before averaging on the speech-progress grid."""
    step_percent = float(np.median(np.diff(epochs.progress_percent)))
    durations = epochs.events["duration"].to_numpy(float)
    smoothed = np.stack([
        gaussian_smooth_nan(trial, 100.0 / (step_percent * duration), sigma_seconds=sigma_seconds)
        for trial, duration in zip(epochs.speech, durations, strict=True)
    ])
    return _finite_mean(smoothed, axis=0)


def _waveform_correlation(curves: np.ndarray, fit) -> tuple[float, int]:
    correlations = []
    for label in np.unique(fit.labels[fit.clean_mask]):
        members = np.flatnonzero(fit.clean_mask & (fit.labels == label))
        if len(members) < 2:
            continue
        for member in members:
            other_mean = _finite_mean(curves[members[members != member]], axis=0)
            curve = curves[member]
            finite = np.isfinite(curve) & np.isfinite(other_mean)
            if finite.sum() < 3:
                continue
            left, right = curve[finite], other_mean[finite]
            if np.ptp(left) == 0 or np.ptp(right) == 0:
                continue
            correlations.append(float(np.corrcoef(left, right)[0, 1]))
    values = np.asarray(correlations, dtype=float)
    values = values[np.isfinite(values)]
    return (float(values.mean()) if values.size else np.nan), int(values.size)


def _dispersion(values: pd.Series) -> tuple[float, float, int]:
    finite = values[np.isfinite(values)].to_numpy(float)
    if not finite.size:
        return np.nan, np.nan, 0
    return float(finite.mean()), float(finite.std(ddof=1) if finite.size > 1 else 0.0), len(finite)


def evaluate_beta_cluster_stability(
    epochs: BetaProgressEpochs,
    *,
    k_values: Iterable[int] = range(2, 7),
    n_resamples: int = 20,
    feature_kwargs: Mapping[str, Any] | None = None,
    bad_mask: np.ndarray | None = None,
    feature_columns: Sequence[str] | None = None,
    feature_weights: Mapping[str, float] | Sequence[float] | None = None,
    random_state: int = 50,
) -> ClusterStabilityEvaluation:
    """Compare independent split-half and trial-bootstrap electrode clusters.

    All fits use fresh imputation, scaling, and KMeans through
    ``cluster_electrodes``. Split-half ARI compares disjoint halves; bootstrap
    ARI compares each bootstrap fit with the full-data fit. Features are
    extracted once per trial sample and reused across K. Feature columns are
    fixed from the full-data table, using the parent selector when available.
    Numeric QC metadata are never predictors.

    The fixed population is determined only from original epochs and the
    full-data feature table: FIF/supplied bads, full-data feature-marked bads,
    wholly missing selected features, fewer than two valid original trials,
    or an original invalid fraction above the configured
    ``max_baseline_invalid_fraction`` (default 0.25) exclude an electrode.
    Resampling never changes this mask or the selected predictors. A half or
    bootstrap is unsupported if any required electrode fails these sample QC
    checks. Its fit is skipped (None), rather than imputing a wholly missing
    electrode or changing membership; other supported samples are still fitted.
    Each sample's ``*_supported`` and ``*_reason`` fields report availability.
    The returned ``clean_mask`` aligns with ``epochs.channel_names``; pass its
    complement as ``bad_mask`` to the final notebook clustering fit. Selected
    predictors are recorded in ``summary.attrs['feature_columns']``.

    A tiny cluster has fewer than ``max(2, ceil(0.05 * clean_count))`` members.
    Every supported complete fit contributes ARI regardless of cluster sizes;
    unsupported or collapsed comparisons yield NaN, with explicit reasons.
    Tiny replicate flags and minimum sizes are
    reported separately so unstable small groups remain in the ARI averages.
    Eligible K must have all K full-data clusters, no tiny full-data cluster,
    a finite silhouette, and at least ``ceil(0.8 * n_resamples)`` supported complete
    replicates of each type. Valid counts and fractions report this 80% rate.
    Among eligible K, descending average-tie ranks of silhouette, split-half
    ARI mean, and bootstrap ARI mean are averaged equally. The lowest average
    wins, with smaller K breaking ties; ``selection_rank`` is the resulting
    ordinal order. Waveform correlation is an electrode-weighted Pearson
    correlation of trial-mean speech curves with their leave-one-out cluster
    means. Each trial is Gaussian-smoothed before averaging, using its duration
    to convert progress spacing into physical sfreq and the configured
    ``feature_kwargs['smoothing_sigma_seconds']`` (default 0.06 seconds).
    This waveform calculation uses no derivatives. Smoothed trial means are
    cached across K. ARI std uses ddof=1, or zero for a single valid replicate.

    Raises ValueError for invalid inputs or when no K is eligible. Results are
    descriptive stability diagnostics, not tests of statistical significance.
    """
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, Integral) or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer.")
    candidates = tuple(k_values)
    if not candidates or any(isinstance(k, bool) or not isinstance(k, Integral) or k < 2 for k in candidates):
        raise ValueError("k_values must contain integers greater than or equal to 2.")
    candidates = tuple(dict.fromkeys(int(k) for k in candidates))
    speech = np.asarray(epochs.speech)
    if speech.ndim != 3:
        raise ValueError("speech must have shape (trials, channels, time).")
    n_trials, n_channels, _ = speech.shape
    if n_trials < 4:
        raise ValueError("Split-half evaluation requires at least 4 trials.")
    if len(epochs.events) != n_trials or len(epochs.channel_names) != n_channels:
        raise ValueError("events and channel_names must align with speech.")
    if len(set(epochs.channel_names)) != n_channels:
        raise ValueError("channel_names must be unique.")
    for name in ("pre", "post", "audio_speech", "audio_pre", "audio_post"):
        array = np.asarray(getattr(epochs, name))
        expected_ndim = 3 if name in {"pre", "post"} else 2
        if array.ndim != expected_ndim or array.shape[0] != n_trials:
            raise ValueError(f"{name} must align with the {n_trials} trials.")
        if expected_ndim == 3 and array.shape[1] != n_channels:
            raise ValueError(f"{name} must align with the {n_channels} channels.")
    supplied_bad = np.zeros(n_channels, dtype=bool) if bad_mask is None else np.asarray(bad_mask, dtype=bool)
    if supplied_bad.shape != (n_channels,):
        raise ValueError(f"bad_mask must have shape ({n_channels},).")
    excluded = supplied_bad | np.isin(epochs.channel_names, list(epochs.bad_channels))
    kwargs = dict(feature_kwargs or {})
    full_table = _aligned_features(epochs, kwargs)
    max_invalid_fraction = kwargs.get("max_baseline_invalid_fraction", 0.25)
    valid_trials = _valid_trials(epochs)
    valid_counts = valid_trials.sum(axis=0)
    excluded |= (valid_counts < 2) | (1.0 - valid_counts / n_trials > max_invalid_fraction)
    if feature_columns is not None:
        columns = tuple(feature_columns)
        if not columns:
            raise ValueError("feature_columns must not be empty.")
        if set(columns) & _METADATA:
            raise ValueError("Quality and channel metadata cannot be feature_columns.")
        missing = set(columns) - set(full_table.columns)
        if missing:
            raise ValueError(f"Unknown feature columns: {sorted(missing)}")
    else:
        columns = tuple(column for column in infer_feature_columns(full_table) if column not in _METADATA)
        if not columns:
            raise ValueError("No non-metadata feature_columns are available.")

    excluded |= _feature_bad_mask(full_table, columns)
    if feature_columns is None:
        selector = getattr(clustering_pipeline, "select_beta_progress_feature_columns", None)
        if selector is not None:
            columns = tuple(selector(full_table, bad_mask=excluded))
        else:
            columns = tuple(column for column in columns if full_table.loc[~excluded, column].nunique() > 1)
        if not columns or set(columns) & _METADATA:
            raise ValueError("No usable non-metadata feature_columns were selected.")
        excluded |= _feature_bad_mask(full_table, columns)
    clean_mask = ~excluded
    clean_count = int(clean_mask.sum())
    if any(k >= clean_count for k in candidates):
        raise ValueError(f"Every candidate K must be smaller than the {clean_count} common clean electrodes.")

    rng = np.random.default_rng(random_state)
    cached = []
    for replicate in range(n_resamples):
        shuffled = rng.permutation(n_trials)
        half_a, half_b = shuffled[:n_trials // 2], shuffled[n_trials // 2:]
        bootstrap = rng.integers(0, n_trials, size=n_trials)
        indices = (half_a, half_b, bootstrap)
        sampled_tables = []
        support_reasons = []
        for sample_indices in indices:
            table = _aligned_features(_slice_epochs(epochs, sample_indices), kwargs)
            sampled_tables.append(table)
            support_reasons.append(_sample_support_reason(
                table, valid_trials[sample_indices], columns, clean_mask, max_invalid_fraction,
            ))
        seeds = tuple(int(seed) for seed in rng.integers(0, np.iinfo(np.int32).max, size=3))
        cached.append((replicate, indices, tuple(sampled_tables), seeds, tuple(support_reasons)))
    minimum_size = max(2, int(np.ceil(0.05 * clean_count)))
    required_valid_count = int(np.ceil(0.8 * n_resamples))
    waveform_curves = _trial_mean_waveforms(epochs, sigma_seconds=kwargs.get("smoothing_sigma_seconds", 0.06))

    def fit(table, k, seed):
        result = cluster_electrodes(
            table, n_clusters=k, bad_mask=excluded, feature_columns=columns,
            feature_weights=feature_weights, random_state=seed,
        )
        if not np.array_equal(result.clean_mask, ~excluded):
            raise ValueError("Every refit must retain the same common clean electrode mask.")
        return result

    summary_rows, resample_rows = [], []
    for k in candidates:
        reference = fit(full_table, k, random_state)
        full_size, complete = _cluster_size(reference, k)
        silhouette = (float(silhouette_score(reference.weighted_clean_features, reference.labels[~excluded]))
                      if complete else np.nan)
        waveform, waveform_count = _waveform_correlation(waveform_curves, reference)
        rows = []
        for replicate, indices, sampled_tables, seeds, support_reasons in cached:
            half_a, half_b, bootstrap = [
                None if reason else fit(table, k, seed)
                for table, seed, reason in zip(sampled_tables, seeds, support_reasons, strict=True)
            ]
            a_size, a_complete = _cluster_size(half_a, k)
            b_size, b_complete = _cluster_size(half_b, k)
            boot_size, boot_complete = _cluster_size(bootstrap, k)
            split_valid = a_complete and b_complete
            boot_valid = complete and boot_complete
            a_reason, b_reason, boot_reason = [
                reason or ("collapsed_clusters" if fitted is not None and not is_complete else "")
                for reason, fitted, is_complete in zip(
                    support_reasons, (half_a, half_b, bootstrap),
                    (a_complete, b_complete, boot_complete), strict=True,
                )
            ]
            split_reason = "; ".join(
                f"{half}:{reason}" for half, reason in (("half_a", a_reason), ("half_b", b_reason)) if reason
            )
            if not complete:
                boot_reason = "; ".join(filter(None, (boot_reason, "reference_collapsed_clusters")))
            row = {
                "k": k, "resample": replicate,
                "split_half_ari": float(adjusted_rand_score(half_a.labels[~excluded], half_b.labels[~excluded]))
                if split_valid else np.nan,
                "bootstrap_ari": float(adjusted_rand_score(reference.labels[~excluded], bootstrap.labels[~excluded]))
                if boot_valid else np.nan,
                "split_half_valid": split_valid, "bootstrap_valid": boot_valid,
                "half_a_supported": half_a is not None, "half_b_supported": half_b is not None,
                "bootstrap_supported": bootstrap is not None,
                "half_a_reason": a_reason, "half_b_reason": b_reason,
                "split_half_reason": split_reason, "bootstrap_reason": boot_reason,
                "split_half_min_cluster_size": min(a_size, b_size)
                if half_a is not None and half_b is not None else np.nan,
                "bootstrap_min_cluster_size": boot_size,
                "split_half_has_tiny_cluster": bool(
                    (a_complete and a_size < minimum_size) or (b_complete and b_size < minimum_size)
                ),
                "bootstrap_has_tiny_cluster": bool(boot_complete and boot_size < minimum_size),
                "half_a_trials": tuple(int(i) for i in indices[0]),
                "half_b_trials": tuple(int(i) for i in indices[1]),
                "bootstrap_trials": tuple(int(i) for i in indices[2]),
            }
            rows.append(row)
            resample_rows.append(row)
        replicate_table = pd.DataFrame(rows)
        split_mean, split_std, split_count = _dispersion(replicate_table.split_half_ari)
        boot_mean, boot_std, boot_count = _dispersion(replicate_table.bootstrap_ari)
        tiny = full_size < minimum_size
        summary_rows.append({
            "k": k, "silhouette": silhouette,
            "split_half_ari_mean": split_mean, "split_half_ari_std": split_std,
            "bootstrap_ari_mean": boot_mean, "bootstrap_ari_std": boot_std,
            "min_cluster_size": full_size, "waveform_correlation_mean": waveform,
            "waveform_valid_count": waveform_count, "n_clean_electrodes": clean_count,
            "split_half_valid_count": split_count, "bootstrap_valid_count": boot_count,
            "split_half_valid_fraction": split_count / n_resamples,
            "bootstrap_valid_fraction": boot_count / n_resamples,
            "required_valid_count": required_valid_count,
            "has_tiny_cluster": tiny, "complete_clusters": complete,
            "eligible": bool(
                complete and not tiny and np.isfinite(silhouette)
                and split_count >= required_valid_count and boot_count >= required_valid_count
            ),
        })
    summary = pd.DataFrame(summary_rows)
    resamples = pd.DataFrame(resample_rows)
    summary["selection_score"] = np.nan
    summary["selection_rank"] = np.nan
    eligible = summary.loc[summary.eligible]
    if eligible.empty:
        raise NoEligibleClusterError(summary, resamples, clean_mask)
    metrics = ["silhouette", "split_half_ari_mean", "bootstrap_ari_mean"]
    scores = eligible[metrics].rank(ascending=False, method="average").mean(axis=1)
    summary.loc[scores.index, "selection_score"] = scores
    ranked = summary.loc[summary.eligible].sort_values(["selection_score", "k"])
    summary.loc[ranked.index, "selection_rank"] = np.arange(1, len(ranked) + 1)
    metadata = {"feature_columns": columns,
                "clean_channels": tuple(name for name, bad in zip(epochs.channel_names, excluded) if not bad),
                "excluded_channels": tuple(name for name, bad in zip(epochs.channel_names, excluded) if bad)}
    summary.attrs.update(metadata)
    resamples.attrs.update(metadata)
    return ClusterStabilityEvaluation(summary, resamples, int(ranked.iloc[0].k), (~excluded).copy())


__all__ = ["ClusterStabilityEvaluation", "NoEligibleClusterError", "evaluate_beta_cluster_stability"]
