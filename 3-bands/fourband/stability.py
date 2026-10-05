"""Descriptive trial-resampling agreement conditional on fixed electrodes and K."""
from dataclasses import dataclass, replace
import json
from numbers import Integral
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from audio_cross_3band_utils import extract_beta_progress_features
from .analysis import ECOG_FEATURE_SUFFIXES, fit_joint


@dataclass
class StabilityReport:
    summary: dict
    replicates: pd.DataFrame
    channels: pd.DataFrame


_FEATURE_DEFAULTS = dict(smoothing_sigma_seconds=.06, slope_method="savgol",
                         max_baseline_invalid_fraction=.25, max_lag_percent=20.,
                         mask_bad_features=True)
_REPLICATE_COLUMNS = ("kind", "split_id", "half", "trial_indices", "original_trial_ids",
    "n_trials", "supported", "status", "reason", "unsupported_channels", "actual_k",
    "cluster_sizes", "min_cluster_size", "tiny_cluster", "collapsed", "ari_full",
    "ari_between_halves", "matched_agreement", "min_valid_trial_count", "warnings")


class _Cancelled(Exception):
    pass


def _check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise _Cancelled()


def _trial_ids(events):
    return events.original_trial_id.to_numpy() if "original_trial_id" in events else events.index.to_numpy()


def _validate_alignment(extraction, result):
    channels = tuple(extraction.features.channel)
    if channels != tuple(result.channels) or len(set(channels)) != len(channels):
        raise ValueError("Reference channels must align with extraction in their original order.")
    mask = np.asarray(result.clean_mask)
    labels = np.asarray(result.labels)
    if mask.dtype != bool or mask.shape != (len(channels),) or labels.shape != mask.shape:
        raise ValueError("Reference labels and clean_mask must align with extraction.")
    if np.any(labels[mask] < 0) or np.any(labels[~mask] != -1):
        raise ValueError("Reference labels do not match its fixed clean_mask.")
    if np.any(extraction.features.is_bad.to_numpy(bool) & mask):
        raise ValueError("Reference population includes extraction QC exclusions.")
    bands = tuple(extraction.bands)
    if len(bands) != 4 or set(extraction.epochs) != set(bands):
        raise ValueError("Four aligned bands are required.")
    reference = extraction.epochs[bands[0]]
    ids = _trial_ids(reference.events)
    if len(set(ids)) != len(ids):
        raise ValueError("Original trial IDs must uniquely identify the retained source trials.")
    for band in bands:
        epoch = extraction.epochs[band]
        if (tuple(epoch.channel_names) != channels or not epoch.events.equals(reference.events)
                or not np.array_equal(epoch.beta_band, extraction.bands[band])
                or epoch.bad_channels != reference.bad_channels or epoch.sfreq != reference.sfreq):
            raise ValueError(f"Band {band} trial/channel/passband alignment differs from extraction.")
        for key in ("progress_percent", "pre_times", "post_times"):
            if not np.array_equal(getattr(epoch, key), getattr(reference, key)):
                raise ValueError(f"Band {band} time alignment differs from extraction.")
        for key in ("speech", "pre", "post", "audio_speech", "audio_pre", "audio_post"):
            values = getattr(epoch, key)
            expected = (len(ids), len(channels)) if not key.startswith("audio") else (len(ids),)
            if values.shape[:-1] != expected:
                raise ValueError(f"Band {band} {key} array does not align with trials/channels.")
    for column in result.feature_columns:
        band, separator, suffix = column.partition("__")
        if not separator or band not in bands or suffix not in ECOG_FEATURE_SUFFIXES:
            raise ValueError(f"Reference contains a predictor outside the ECoG allowlist: {column}")
        if column not in extraction.features:
            raise ValueError(f"Reference feature missing from extraction: {column}")
    return bands, mask.copy(), labels[mask].copy(), reference.events


def _sample_epochs(epoch, indices):
    events = epoch.events.iloc[indices].copy()
    diagnostics = epoch.baseline_diagnostics
    if diagnostics is not None:
        # Select each occurrence separately: isin() would lose bootstrap multiplicity.
        ids = _trial_ids(events)
        blocks = [diagnostics.loc[diagnostics.original_trial_id == trial_id].copy() for trial_id in ids]
        diagnostics = pd.concat(blocks, ignore_index=True) if blocks else diagnostics.iloc[:0].copy()
    values = {name: getattr(epoch, name)[indices].copy() for name in
              ("speech", "pre", "post", "audio_speech", "audio_pre", "audio_post")}
    return replace(epoch, **values, events=events, baseline_diagnostics=diagnostics)


def _baseline_validity(extraction, band):
    epoch = extraction.epochs[band]
    source = getattr(extraction, "_trial_qc_source", None)
    if source is not None:
        original = source.epochs[band]
        if (not original.events.equals(epoch.events) or original.channel_names != epoch.channel_names
                or original.speech.shape != epoch.speech.shape):
            raise ValueError(f"Band {band} original trial-QC source no longer aligns with extraction.")
        return np.isfinite(original.speech).all(axis=-1)
    if not getattr(extraction, "trial_exclusions", ()):
        return np.isfinite(epoch.speech).all(axis=-1)
    diagnostics = epoch.baseline_diagnostics
    keys = ["original_trial_id", "channel"]
    if (diagnostics is None or not {*keys, "valid"} <= set(diagnostics)
            or diagnostics.duplicated(keys).any()):
        raise ValueError(f"Band {band} manual masks need original baseline diagnostics or trial-QC source.")
    index = pd.MultiIndex.from_product([_trial_ids(epoch.events), epoch.channel_names], names=keys)
    valid = diagnostics.set_index(keys).valid.reindex(index)
    if valid.isna().any():
        raise ValueError(f"Band {band} original baseline diagnostics lack retained trial/channel keys.")
    return valid.to_numpy(bool).reshape(len(epoch.events), len(epoch.channel_names))


def _reaggregate(extraction, bands, indices, settings, cancelled=None):
    channels = tuple(extraction.features.channel)
    columns = {"channel": channels, "is_bad": np.zeros(len(channels), dtype=bool)}
    reasons = [[] for _ in channels]
    for band in bands:
        _check_cancelled(cancelled)
        epochs = _sample_epochs(extraction.epochs[band], indices)
        # Manual NaNs change support, not baseline validity. Preserve numerical
        # descriptors until QC has been recomputed from the original baseline.
        features = extract_beta_progress_features(epochs, **dict(settings, mask_bad_features=False))
        if tuple(features.channel) != channels:
            raise ValueError(f"Band {band} reaggregated channels do not align with extraction.")
        valid_count = np.isfinite(epochs.speech).all(axis=-1).sum(axis=0)
        baseline_fraction = 1 - _baseline_validity(extraction, band)[indices].mean(axis=0)
        bad = (np.isin(channels, list(epochs.bad_channels)) | (valid_count < 2)
               | (baseline_fraction > settings["max_baseline_invalid_fraction"]))
        columns["is_bad"] |= bad
        for column in features:
            if column.startswith("beta_progress__"):
                suffix = column.removeprefix("beta_progress__").replace("beta_slope", "power_slope")
                values = features[column].to_numpy(copy=True)
                if settings["mask_bad_features"]:
                    values[bad] = np.nan
                columns[f"{band}__{suffix}"] = values
        columns[f"qc_{band}_valid_trials"] = valid_count
        columns[f"qc_{band}_invalid_fraction"] = baseline_fraction
        for index in np.flatnonzero(bad):
            why = []
            if channels[index] in epochs.bad_channels:
                why.append("raw_bad")
            if valid_count[index] < 2:
                why.append("insufficient_valid_trials")
            if baseline_fraction[index] > settings["max_baseline_invalid_fraction"]:
                why.append("invalid_trial_fraction")
            reasons[index].append(f"{band}:" + ",".join(why or ["quality_excluded"]))
    columns["exclusion_reason"] = [";".join(reason) for reason in reasons]
    return pd.DataFrame(columns)


def _fit_settings(result, bands, mask, fit_kwargs):
    supplied = dict(fit_kwargs or {})
    allowed = {"n_clusters", "bands", "band_weights", "bad_mask", "correlation_threshold",
               "random_state", "weight_preset", "group_weights"}
    if set(supplied) - allowed:
        raise ValueError(f"Unknown fit_kwargs: {sorted(set(supplied) - allowed)}")
    k = result.kmeans.n_clusters
    if "n_clusters" in supplied and supplied["n_clusters"] != k:
        raise ValueError("Stability must retain the reference requested K.")
    if "bands" in supplied and tuple(supplied["bands"]) != bands:
        raise ValueError("Fit bands must match extraction order.")
    if "bad_mask" in supplied and not np.array_equal(supplied["bad_mask"], ~mask):
        raise ValueError("Fit bad_mask must retain the fixed reference population.")
    columns, weights = result.feature_columns, np.asarray(result.feature_weights, dtype=float)
    if weights.shape != (len(columns),) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Invalid reference feature weights.")
    budgets = {band: float(sum(w for c, w in zip(columns, weights, strict=True)
                              if c.startswith(band + "__"))) for band in bands}
    settings = dict(n_clusters=k, bands=bands, band_weights=budgets, bad_mask=~mask,
                    correlation_threshold=.95, random_state=result.kmeans.random_state)
    settings.update(supplied)
    settings["bad_mask"] = (~mask).copy()
    return settings


def _validate_reference(extraction, result, full_table, mask, settings):
    columns = list(result.feature_columns)
    original = extraction.features.loc[mask, columns].to_numpy(float)
    rebuilt = full_table.loc[mask, columns].to_numpy(float)
    if not np.allclose(original, rebuilt, rtol=1e-7, atol=1e-9, equal_nan=True):
        raise ValueError("Reference features differ from epochs at these feature settings; use actual extraction settings.")
    if full_table.is_bad.to_numpy(bool)[mask].any():
        raise ValueError("Reference population is unsupported at these feature settings.")
    for column in full_table:
        if column.startswith("qc_") and column in extraction.features:
            if not np.allclose(full_table.loc[mask, column], extraction.features.loc[mask, column], equal_nan=True):
                raise ValueError(f"Reference source QC differs from reaggregated epochs: {column}")
    original = np.where(np.isfinite(original), original, np.nan)
    transformed = result.scaler.transform(result.imputer.transform(original))
    if (transformed.shape != result.standardized_clean_features.shape
            or not np.allclose(transformed, result.standardized_clean_features, rtol=1e-7, atol=1e-9)):
        raise ValueError("Reference preprocessing was fitted to different extraction features.")
    weighted = transformed * np.sqrt(result.feature_weights)
    if (weighted.shape != result.weighted_clean_features.shape
            or not np.allclose(weighted, result.weighted_clean_features, rtol=1e-7, atol=1e-9)):
        raise ValueError("Reference feature weights differ from its fitted transform.")
    # A full-data refit checks the supplied selection/weighting policy before sampling.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = fit_joint(full_table, **settings)
    if (tuple(fitted.feature_columns) != tuple(result.feature_columns)
            or not np.allclose(fitted.feature_weights, result.feature_weights, rtol=1e-7, atol=1e-9)
            or not np.array_equal(fitted.clean_mask, mask)
            or adjusted_rand_score(result.labels[mask], fitted.labels[mask]) != 1):
        raise ValueError("Fit policy does not reproduce the reference; supply actual fit_kwargs (weights/selection/seed).")
    return [str(item.message) for item in caught]


def _matched(reference, labels):
    ref_values, ref_codes = np.unique(reference, return_inverse=True)
    values, codes = np.unique(labels, return_inverse=True)
    overlap = np.zeros((len(ref_values), len(values)), dtype=int)
    np.add.at(overlap, (ref_codes, codes), 1)
    rows, columns = linear_sum_assignment(-overlap)
    mapping = {values[col]: ref_values[row] for row, col in zip(rows, columns, strict=True)}
    return np.array([label in mapping and mapping[label] == expected
                     for label, expected in zip(labels, reference, strict=True)], dtype=bool)


def _draws(events, n_resamples, rng):
    count = len(events)
    middle = count // 2
    order = np.arange(count)
    if "onset" in events:
        onsets = events.onset.to_numpy(float)
        if not np.isfinite(onsets).all():
            raise ValueError("Chronological halves require finite trial onsets.")
        order = np.argsort(onsets, kind="stable")
    yield "chronological_half", 0, 0, order[:middle]
    yield "chronological_half", 0, 1, order[middle:]
    for split in range(n_resamples):
        order = rng.permutation(count)
        yield "random_half", split, 0, order[:middle]
        yield "random_half", split, 1, order[middle:]
    for repeat in range(n_resamples):
        yield "bootstrap", repeat, None, rng.integers(0, count, size=count)


def _finish(rows, channels, reference, matches, summary):
    replicates = pd.DataFrame(rows, columns=_REPLICATE_COLUMNS)
    supported = sum(row["supported"] for row in rows)
    attempted = len(rows)
    matched_counts = np.sum(matches, axis=0).astype(int) if matches else np.zeros(len(channels), dtype=int)
    table = pd.DataFrame(dict(channel=channels, reference_label=reference,
        attempted_replicates=attempted, supported_replicates=supported,
        matched_replicates=matched_counts,
        agreement=matched_counts / supported if supported else np.full(len(channels), np.nan),
        coverage=supported / attempted if attempted else 0.))
    full_ari = [row["ari_full"] for row in rows if row["supported"]]
    # Both half rows carry the same pair metric; aggregate each completed pair once.
    pair_ari = [row["ari_between_halves"] for row in rows
                if row["half"] == 0 and np.isfinite(row["ari_between_halves"])]
    failed_reasons = pd.Series([row["reason"] for row in rows if row["status"] == "failed"], dtype=object)
    summary.update(attempted=attempted, supported=supported,
        failed=sum(row["status"] == "failed" for row in rows),
        tiny=sum(row["tiny_cluster"] for row in rows),
        collapsed=sum(row["collapsed"] for row in rows),
        ari_full_mean=float(np.mean(full_ari)) if full_ari else None,
        ari_between_halves_mean=float(np.mean(pair_ari)) if pair_ari else None,
        completed_half_pairs=len(pair_ari),
        matched_agreement_mean=float(matched_counts.mean() / supported) if supported and len(channels) else None,
        failure_reasons={str(reason): int(count) for reason, count in failed_reasons.value_counts().items()},
        reason="" if supported else (summary.get("reason") or
            ("Cancelled before any supported replicate." if summary["cancelled"]
             else "No supported replicate on the fixed electrode population.")))
    return StabilityReport(summary, replicates, table)


def evaluate_stability(extraction, result, *, n_resamples=30, random_state=50,
                       fit_kwargs=None, feature_kwargs=None, progress=None, cancelled=None):
    """Return descriptive agreement for the supplied manual-K fitted round.

    Run one chronological pair, ``n_resamples`` random half pairs and that many
    trial bootstraps (2 + 3*n_resamples fits). Draw positions index retained original
    trials; original IDs and duplicate draws are retained in the report. All bands
    and electrodes share each draw; every transform is independently fitted.

    Feature settings come from ``extraction.feature_settings`` with legacy defaults
    when absent, then explicit ``feature_kwargs`` override them. Pass actual fit
    policy settings (especially nonlegacy weight_preset/group_weights and changed
    correlation_threshold) in fit_kwargs. Band totals and KMeans seed otherwise
    come from the reference. Validation rejects incompatible reference data/policy.

    ``progress(done, total)`` runs initially and after each attempted fit.
    ``cancelled()`` is checked between bands and before fits. Cancellation or
    KeyboardInterrupt returns a partial report without modifying the original fit.
    Coverage is supported fits / attempted fits; agreement is matched supported
    fits / supported fits, never a probability of physiological correctness.
    Baseline QC uses the original trial-QC source (or keyed baseline diagnostics
    for reviewed copies), while valid-trial counts use the current manual masks.
    Tiny means size < max(2, ceil(5% of included electrodes)); collapsed means
    fewer distinct clusters than requested. Both remain in agreement summaries.
    """
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, Integral) or n_resamples < 0:
        raise ValueError("n_resamples must be a nonnegative integer.")
    bands, mask, reference, events = _validate_alignment(extraction, result)
    feature_settings = dict(_FEATURE_DEFAULTS)
    feature_settings.update(getattr(extraction, "feature_settings", None) or {})
    feature_settings.update(feature_kwargs or {})
    if set(feature_settings) - set(_FEATURE_DEFAULTS):
        raise ValueError(f"Unknown feature settings: {sorted(set(feature_settings) - set(_FEATURE_DEFAULTS))}")
    fit_settings = _fit_settings(result, bands, mask, fit_kwargs)
    channels = np.asarray(result.channels)[mask].tolist()
    summary = dict(requested_k=int(result.kmeans.n_clusters), n_resamples=int(n_resamples),
        random_state=random_state, planned=2 + 3 * int(n_resamples), n_included=int(mask.sum()),
        included_channels=channels, bands=dict(extraction.bands), feature_settings=feature_settings,
        fit_settings={key: value for key, value in fit_settings.items() if key != "bad_mask"},
        trial_exclusions=list(getattr(extraction, "trial_exclusions", ()) or ()),
        cancelled=False, interpretation="Descriptive matched-label agreement conditional on fixed electrodes and K.")
    rows, matches, pairs = [], [], {}
    try:
        if progress is not None:
            progress(0, summary["planned"])
        _check_cancelled(cancelled)
        if not len(channels):
            summary["reason"] = "No included electrodes in the reference population."
            return _finish(rows, channels, reference, matches, summary)
        with threadpool_limits(limits=1):
            full_table = _reaggregate(extraction, bands, np.arange(len(events)), feature_settings, cancelled)
            _check_cancelled(cancelled)
            summary["reference_warnings"] = _validate_reference(extraction, result, full_table, mask, fit_settings)
            rng = np.random.default_rng(random_state)
            ids = _trial_ids(events)
            for kind, split, half, indices in _draws(events, n_resamples, rng):
                _check_cancelled(cancelled)
                row = dict(kind=kind, split_id=split, half=half,
                    trial_indices=tuple(int(i) for i in indices), original_trial_ids=tuple(ids[indices]),
                    n_trials=len(indices), supported=False, status="failed", reason="",
                    unsupported_channels=(), actual_k=None, cluster_sizes={}, min_cluster_size=None,
                    tiny_cluster=False, collapsed=False, ari_full=np.nan, ari_between_halves=np.nan,
                    matched_agreement=np.nan, min_valid_trial_count=None, warnings="")
                labels = None
                try:
                    if not len(indices):
                        raise ValueError("No trials in this half.")
                    table = _reaggregate(extraction, bands, indices, feature_settings, cancelled)
                    counts = table.loc[mask, [f"qc_{band}_valid_trials" for band in bands]]
                    row["min_valid_trial_count"] = int(counts.to_numpy().min())
                    unsupported = table.is_bad.to_numpy(bool) & mask
                    for band in bands:
                        columns = [c for c in table if c.startswith(band + "__")
                                   and c.partition("__")[2] in ECOG_FEATURE_SUFFIXES]
                        unsupported |= (~np.isfinite(table[columns].to_numpy(float)).any(axis=1)) & mask
                    row["unsupported_channels"] = tuple(table.channel[unsupported])
                    if unsupported.any():
                        details = [f"{table.channel.iloc[i]} ({table.exclusion_reason.iloc[i] or 'unsupported_features'})"
                                   for i in np.flatnonzero(unsupported)]
                        raise ValueError("Fixed population unsupported: " + "; ".join(details))
                    _check_cancelled(cancelled)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        fitted = fit_joint(table, **fit_settings)
                    row["warnings"] = ";".join(str(item.message) for item in caught)
                    if tuple(fitted.channels) != tuple(result.channels) or not np.array_equal(fitted.clean_mask, mask):
                        raise ValueError("Replicate fit changed the fixed electrode population.")
                    labels = fitted.labels[mask]
                    if np.any(labels < 0):
                        raise ValueError("Replicate fit omitted an included electrode.")
                    values, sizes = np.unique(labels, return_counts=True)
                    agreement = _matched(reference, labels)
                    row.update(supported=True, status="supported", actual_k=len(values),
                        cluster_sizes={str(int(value)): int(size) for value, size in zip(values, sizes, strict=True)},
                        min_cluster_size=int(sizes.min()), tiny_cluster=bool(sizes.min() < max(2, np.ceil(.05 * mask.sum()))),
                        collapsed=bool(len(values) < result.kmeans.n_clusters),
                        ari_full=float(adjusted_rand_score(reference, labels)), matched_agreement=float(agreement.mean()))
                    matches.append(agreement)
                except (_Cancelled, KeyboardInterrupt):
                    # An interrupted draw is not a fitted or failed replicate.
                    raise _Cancelled()
                except Exception as error:
                    row["reason"] = f"{type(error).__name__}: {error}"
                rows.append(row)
                if half is not None:
                    key = (kind, split)
                    if half == 0:
                        pairs[key] = (row, labels)
                    elif key in pairs:
                        first_row, first_labels = pairs.pop(key)
                        if first_row["supported"] and row["supported"]:
                            ari = float(adjusted_rand_score(first_labels, labels))
                            first_row["ari_between_halves"] = row["ari_between_halves"] = ari
                if progress is not None:
                    progress(len(rows), summary["planned"])
    except (_Cancelled, KeyboardInterrupt):
        summary["cancelled"] = True
    return _finish(rows, channels, reference, matches, summary)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_stability(report, output_dir):
    """Write stability_summary.json, stability_{replicates,channels}.csv and stability.png.

    Return the output Path. Object-valued CSV cells use JSON for trial draws,
    channel lists and cluster sizes. Uses the noninteractive Agg canvas.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "stability_summary.json").write_text(
        json.dumps(_json_safe(report.summary), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    replicates = report.replicates.copy(deep=True)
    for column in ("trial_indices", "original_trial_ids", "unsupported_channels", "cluster_sizes"):
        replicates[column] = replicates[column].map(lambda value: json.dumps(_json_safe(value), ensure_ascii=False))
    replicates.to_csv(output / "stability_replicates.csv", index=False, encoding="utf-8-sig")
    report.channels.to_csv(output / "stability_channels.csv", index=False, encoding="utf-8-sig")
    figure = Figure(figsize=(11, 7), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 1)
    rows = report.replicates
    if len(rows):
        positions = np.arange(len(rows))
        axes[0].scatter(positions, rows.ari_full, label="ARI vs full fit", color="#267761")
        pairs = rows.loc[rows.half == 0]
        axes[0].scatter(pairs.index, pairs.ari_between_halves, label="ARI between halves", color="#96507a", marker="x")
        failed = ~rows.supported
        axes[0].scatter(positions[failed], np.full(int(failed.sum()), -.1), label="Failed fit", color="#b44336", marker="x")
        axes[0].legend(loc="best")
    axes[0].set(xlabel="Attempted replicate", ylabel="Adjusted Rand index", ylim=(-1.05, 1.05))
    axes[0].set_title(f"K={report.summary['requested_k']}: {report.summary['supported']}/{report.summary['attempted']} supported")
    if not report.summary["supported"]:
        axes[0].text(.5, .5, report.summary["reason"], transform=axes[0].transAxes, ha="center", wrap=True)
    table = report.channels
    if len(table):
        positions = np.arange(len(table))
        axes[1].plot(positions, table.agreement, "o", color="#267761", label="Matched-label agreement")
        axes[1].plot(positions, table.coverage, ".", color="#96507a", label="Supported-fit coverage")
        stride = max(1, int(np.ceil(len(table) / 35)))
        axes[1].set_xticks(positions[::stride], table.channel.iloc[::stride], rotation=60, ha="right")
        axes[1].legend(loc="best")
    axes[1].set(xlabel="Included electrode (original order)", ylabel="Descriptive fraction", ylim=(-.05, 1.05))
    figure.savefig(output / "stability.png", dpi=150)
    return output
