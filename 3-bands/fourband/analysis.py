"""Actual per-band extraction and balanced, joint electrode clustering."""
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from audio_cross_3band_utils import (
    extract_beta_progress_epochs, extract_beta_progress_features,
    load_ecog_fif, load_speech_events, extract_audio_envelope,
)
from clustering_pipeline import (
    cluster_electrodes, select_beta_progress_feature_columns,
    default_beta_progress_feature_weights,
)

DEFAULT_BANDS = {"theta": (4.0, 8.0), "beta1": (13.0, 24.0),
                 "beta2": (24.0, 40.0), "high_gamma": (70.0, 150.0)}

# Exact suffixes keep unreviewed coupling descriptors out of both fitting rounds.
ECOG_FEATURE_SUFFIXES = frozenset({
    "early_mean", "middle_mean", "late_mean", "middle_minus_early", "late_minus_middle",
    "positive_auc", "negative_auc", "positive_peak_value", "negative_peak_value",
    "positive_peak_percent", "negative_peak_percent", "dominant_peak_width_percent",
    "offset_rebound", "late_to_post_change", "energy_mean", "energy_std", "energy_auc",
    "energy_peak_abs", "energy_peak_percent", "energy_range", "pre_mean", "post_mean",
    "post_minus_pre_mean", "power_slope_mean", "power_slope_std", "power_slope_peak_abs",
    "power_slope_peak_percent",
    # ECoG-only descriptors from older onset/offset feature tables.
    "early_delta", "late_delta", "post_delta", "post_std", "post_auc", "onset_slope",
    "post_peak_abs", "post_peak_time", "articulation_mean", "offset_slope",
    # Despite its legacy name, this correlates event duration with ECoG response.
    "audio_dur_ecog_cov",
})


@dataclass
class FourBandExtraction:
    features: pd.DataFrame
    epochs: dict
    bands: dict
    filter_kwargs: dict | None = None
    feature_settings: dict | None = None
    trial_exclusions: tuple = ()


def _validate_bands(bands, sfreq):
    if len(bands) != 4:
        raise ValueError("Specify exactly four bands.")
    for name, bounds in bands.items():
        values = np.asarray(bounds, dtype=float)
        if not isinstance(name, str) or not name.isidentifier() or "__" in name:
            raise ValueError("Band names must be distinct identifiers without double underscores.")
        if values.shape != (2,) or not np.isfinite(values).all() or not 0 < values[0] < values[1]:
            raise ValueError(f"Invalid passband for {name}.")
        if values[1] >= sfreq / 2:
            raise ValueError(f"{name} {tuple(values)} reaches/exceeds Nyquist ({sfreq / 2} Hz); no silent truncation.")


def extract_fourband(raw, events, audio, *, bands=None, smoothing_sigma_seconds=.06,
                     slope_method="savgol", max_baseline_invalid_fraction=.25, **epoch_kwargs):
    """Use the tested epoch machinery with each real passband, not four copies of beta."""
    bands = dict(DEFAULT_BANDS if bands is None else bands)
    _validate_bands(bands, float(raw.info["sfreq"]))
    if "beta_band" in epoch_kwargs:
        raise ValueError("Use bands rather than beta_band for four-band extraction.")
    epochs_by_band, feature_table, reasons = {}, None, None
    reference = None
    for band, bounds in bands.items():
        print(f"Extracting {band}: {bounds[0]:g}-{bounds[1]:g} Hz", flush=True)
        epochs = extract_beta_progress_epochs(raw, events, audio, beta_band=tuple(bounds), **epoch_kwargs)
        if reference is not None:
            aligned = (epochs.channel_names == reference.channel_names
                       and np.array_equal(epochs.progress_percent, reference.progress_percent)
                       and np.array_equal(epochs.pre_times, reference.pre_times)
                       and np.array_equal(epochs.post_times, reference.post_times)
                       and epochs.events.equals(reference.events))
            if not aligned:
                raise ValueError(f"Band {band} trial/channel/time alignment differs from the first band.")
        else:
            reference = epochs
        features = extract_beta_progress_features(epochs, smoothing_sigma_seconds=smoothing_sigma_seconds,
            slope_method=slope_method, max_baseline_invalid_fraction=max_baseline_invalid_fraction)
        if tuple(features.channel) != epochs.channel_names:
            raise ValueError(f"Band {band} features do not align with electrodes.")
        if feature_table is None:
            feature_table = {"channel": features.channel.to_numpy(), "is_bad": np.zeros(len(features), dtype=bool)}
            reasons = [[] for _ in range(len(features))]
        for column in features:
            if column.startswith("beta_progress__"):
                suffix = column.removeprefix("beta_progress__").replace("beta_slope", "power_slope")
                feature_table[f"{band}__{suffix}"] = features[column].to_numpy()
        feature_table[f"qc_{band}_invalid_fraction"] = features.baseline_invalid_fraction.to_numpy()
        feature_table[f"qc_{band}_valid_trials"] = features.valid_trial_count.to_numpy()
        feature_table["is_bad"] |= features.is_bad.to_numpy(bool)
        for index in np.flatnonzero(features.is_bad.to_numpy(bool)):
            parts = []
            if features.channel.iloc[index] in epochs.bad_channels:
                parts.append("raw_bad")
            if features.baseline_invalid_fraction.iloc[index] > max_baseline_invalid_fraction:
                parts.append("invalid_trial_fraction")
            if features.valid_trial_count.iloc[index] < 2:
                parts.append("insufficient_valid_trials")
            reasons[index].append(band + ":" + ",".join(parts or ["quality_excluded"]))
        epochs_by_band[band] = epochs
    feature_table = pd.DataFrame(feature_table)
    feature_table["exclusion_reason"] = [";".join(parts) for parts in reasons]
    return FourBandExtraction(feature_table, epochs_by_band, bands,
        filter_kwargs=dict(epoch_kwargs.get("filter_kwargs") or {}),
        feature_settings=dict(smoothing_sigma_seconds=smoothing_sigma_seconds,
            slope_method=slope_method, max_baseline_invalid_fraction=max_baseline_invalid_fraction))


def load_fourband(fif_path, onset_path, offset_path, wav_path, *, picks="ecog", **kwargs):
    paths = [Path(p) for p in (fif_path, onset_path, offset_path, wav_path)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing input files: " + "; ".join(missing))
    raw = load_ecog_fif(fif_path, preload=False)
    events = load_speech_events(onset_path, offset_path)
    audio = extract_audio_envelope(wav_path)
    return extract_fourband(raw, events, audio, picks=picks, **kwargs)


def fit_joint(features, *, n_clusters=4, bands=None, band_weights=None,
              bad_mask=None, correlation_threshold=.95, random_state=50,
              weight_preset="legacy", group_weights=None):
    """Prune ECoG-only predictors within bands and assign fixed band distance budgets.

    Audio diagnostics may remain in the input, including legacy feature tables,
    but are excluded before redundancy selection or any estimator is fitted.
    """
    from .weights import feature_group, resolve_group_budgets, group_feature_weights
    group_budgets = resolve_group_budgets(weight_preset, group_weights)
    bands = tuple(DEFAULT_BANDS if bands is None else bands)
    if len(bands) != 4 or len(set(bands)) != 4:
        raise ValueError("Four unique bands are required for joint fitting.")
    if "channel" not in features or features.channel.duplicated().any():
        raise ValueError("Joint feature rows must identify unique electrodes.")
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, Integral) or n_clusters < 2:
        raise ValueError("n_clusters must be an integer >= 2.")
    budgets = {band: 1.0 for band in bands}
    if band_weights is not None:
        if set(band_weights) - set(bands):
            raise ValueError("Unknown band in band_weights.")
        budgets.update(band_weights)
    if any(not np.isfinite(w) or w <= 0 for w in budgets.values()):
        raise ValueError("Every band weight must be finite and positive.")
    selected, weights = [], {}
    for band in bands:
        prefix = band + "__"
        columns = [c for c in features if c.startswith(prefix)
                   and c.removeprefix(prefix) in ECOG_FEATURE_SUFFIXES]
        if group_budgets is not None:
            columns = [c for c in columns if group_budgets[feature_group(c)] > 0]
        if not columns:
            raise ValueError(f"No ECoG predictors for {band}; cannot fit a four-band model.")
        table = features[["channel", "is_bad", *columns]].copy()
        try:
            keep = select_beta_progress_feature_columns(table, bad_mask=bad_mask,
                                                       correlation_threshold=correlation_threshold)
        except ValueError as error:
            raise ValueError(f"No usable predictors for {band}: {error}") from error
        if not keep:
            raise ValueError(f"No nonconstant predictors for {band} among included electrodes.")
        if group_budgets is None:
            canonical = [c.replace("power_slope", "beta_slope") for c in keep]
            local = default_beta_progress_feature_weights(canonical, normalize_groups=True)
            total = sum(local.values())
            weights.update({c: float(budgets[band] * local[k] / total) for c, k in zip(keep, canonical, strict=True)})
        else:
            weights.update(group_feature_weights(keep, group_budgets, budgets[band]))
        selected.extend(keep)
    return cluster_electrodes(features, n_clusters=n_clusters, bad_mask=bad_mask,
        feature_columns=selected, feature_weights=weights, random_state=random_state)
