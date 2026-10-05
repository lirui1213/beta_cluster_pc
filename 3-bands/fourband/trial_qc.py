"""Raw artifact candidates and explicitly confirmed, reversible trial masks.

Thresholds are review aids, not physiological rejection criteria. Raw voltage
is in volts (MNE units); detection never changes data or selects exclusions.
"""
from copy import deepcopy
from dataclasses import fields, replace
import hashlib
from numbers import Integral

import numpy as np
import pandas as pd

from audio_cross_3band_utils import extract_beta_progress_features
from .analysis import FourBandExtraction


DEFAULT_QC_SETTINGS = dict(
    interval="speech_context", pre_seconds=None, post_seconds=None,
    reference="full_record_channel_median_mad", spike_z=10., jump_z=10.,
    flat_run_seconds=.05, flat_tolerance_v=1e-12,
    clipping_fraction=.1, clipping_run_seconds=.05, clipping_tolerance_v=1e-12,
)
QC_UNITS = dict(raw_median_v="volts", raw_mad_scale_v="volts", raw_diff_median_v="volts",
    raw_diff_mad_scale_v="volts", peak_abs_v="volts", spike_z="scaled MAD",
    jump_z="scaled MAD of adjacent differences", flat_tolerance_v="volts", clipping_tolerance_v="volts",
    flat_run_seconds="seconds", clipping_run_seconds="seconds", pre_seconds="seconds", post_seconds="seconds",
    clipping_fraction="fraction of finite samples", nonfinite_count="samples", sample_count="samples",
    detection_start_seconds="recording seconds", detection_stop_seconds="recording seconds")


class TrialExclusions(tuple):
    """Tuple-compatible confirmed pairs carrying the report's provenance.

    Keep this object, rather than converting it to a plain tuple, when reusing a
    review. JSON exports should save ``provenance`` alongside the pair list.
    """

    def __new__(cls, pairs=(), *, provenance=None):
        instance = super().__new__(cls, pairs)
        instance.provenance = deepcopy(provenance)
        return instance

    def __getnewargs_ex__(self):
        return (tuple(self),), {"provenance": self.provenance}


def _source(extraction):
    return getattr(extraction, "_trial_qc_source", extraction)


def _hash_value(hasher, value):
    hasher.update(type(value).__name__.encode())
    if isinstance(value, np.ndarray):
        _hash_value(hasher, (value.shape, str(value.dtype)))
        if value.dtype.hasobject:
            _hash_value(hasher, value.tolist())
        else:
            hasher.update(memoryview(np.ascontiguousarray(value)).cast("B"))
    elif isinstance(value, pd.DataFrame):
        _hash_value(hasher, list(value.columns))
        _hash_value(hasher, [str(dtype) for dtype in value.dtypes])
        _hash_value(hasher, pd.util.hash_pandas_object(value, index=True).to_numpy())
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            _hash_value(hasher, key)
            _hash_value(hasher, value[key])
    elif isinstance(value, (set, frozenset)):
        _hash_value(hasher, sorted(value, key=str))
    elif isinstance(value, (list, tuple)):
        for item in value:
            _hash_value(hasher, item)
    else:
        hasher.update(repr(value).encode("utf-8"))
    hasher.update(b"\0")


def _fingerprint(value):
    hasher = hashlib.sha256()
    _hash_value(hasher, value)
    return hasher.hexdigest()


def _extraction_fingerprint(extraction):
    source = extraction
    content = dict(bands=source.bands, filter_kwargs=source.filter_kwargs,
                   feature_settings=getattr(source, "feature_settings", None), features=source.features,
                   trial_exclusions=tuple(getattr(source, "trial_exclusions", ()) or ()))
    content["epochs"] = {band: {field.name: getattr(ep, field.name) for field in fields(ep)}
                         for band, ep in source.epochs.items()}
    return _fingerprint(content)


def _aligned(extraction):
    source = _source(extraction)
    if not source.epochs or set(source.epochs) != set(source.bands):
        raise ValueError("Band/epoch alignment is invalid.")
    first = next(iter(source.epochs.values()))
    events = first.events
    required = {"original_trial_id", "onset", "offset", "duration"}
    if events.empty or not required <= set(events):
        raise ValueError("QC needs retained events with actual original trial IDs and intervals.")
    ids = events.original_trial_id.tolist()
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Original trial IDs must be unique integers.")
    intervals = events[["onset", "offset", "duration"]].to_numpy(float)
    if (not np.isfinite(intervals).all() or np.any(intervals[:, 1] <= intervals[:, 0])
            or np.any(intervals[:, 2] <= 0)):
        raise ValueError("Trial intervals must be finite and positive.")
    channels = first.channel_names
    if not channels or len(set(channels)) != len(channels) or tuple(source.features.channel) != tuple(channels):
        raise ValueError("Channel/feature alignment is invalid.")
    for ep in source.epochs.values():
        if (ep.channel_names != channels or not ep.events.equals(events) or ep.sfreq != first.sfreq
                or any(not np.array_equal(getattr(ep, name), getattr(first, name))
                       for name in ("progress_percent", "pre_times", "post_times"))):
            raise ValueError("Band trial/channel/time alignment differs.")
        for part, grid in (("speech", ep.progress_percent), ("pre", ep.pre_times), ("post", ep.post_times)):
            if getattr(ep, part).shape != (len(events), len(channels), len(grid)):
                raise ValueError("Epoch array alignment is invalid.")
    return source, first


def _settings(settings, reference):
    provided = dict(settings or {})
    if set(provided) - set(DEFAULT_QC_SETTINGS):
        raise ValueError("Unknown trial QC settings: " + ", ".join(sorted(set(provided) - set(DEFAULT_QC_SETTINGS))))
    result = dict(DEFAULT_QC_SETTINGS, **provided)
    if result["interval"] not in {"speech", "speech_context"}:
        raise ValueError("QC interval must be speech or speech_context.")
    if result["reference"] != DEFAULT_QC_SETTINGS["reference"]:
        raise ValueError("Unknown robust QC reference.")
    for name in ("pre_seconds", "post_seconds"):
        if result[name] is None:
            result[name] = float(reference.pre_post_window) if result["interval"] == "speech_context" else 0.
    for name in set(result) - {"interval", "reference"}:
        try:
            value = float(result[name])
        except (ValueError, TypeError) as error:
            raise ValueError(f"{name} must be finite and nonnegative.") from error
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
        result[name] = value
    if result["spike_z"] <= 0 or result["jump_z"] <= 0 or result["clipping_fraction"] > 1:
        raise ValueError("Z thresholds must be positive; clipping_fraction must lie in [0, 1].")
    if result["interval"] == "speech" and (result["pre_seconds"] or result["post_seconds"]):
        raise ValueError("Speech-only detection requires zero context margins.")
    return result


def _raw_inputs(raw, reference):
    data = np.asarray(raw.get_data())
    times = np.asarray(raw.times, dtype=float)
    names = tuple(raw.ch_names)
    if (data.ndim != 2 or data.shape != (len(names), len(times)) or len(set(names)) != len(names)
            or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("Raw data/channel/time alignment is invalid.")
    if not set(reference.channel_names) <= set(names) or float(raw.info["sfreq"]) != reference.sfreq:
        raise ValueError("Raw channel/sample-rate alignment changed since extraction.")
    fingerprint = _fingerprint(dict(data=data, times=times, channels=names,
        sfreq=float(raw.info["sfreq"]), bads=tuple(raw.info.get("bads", ())),
        first_samp=getattr(raw, "first_samp", 0)))
    return data, times, names, fingerprint


def _provenance(raw, extraction, settings, *, raw_fingerprint=None):
    source, reference = _aligned(extraction)
    if raw_fingerprint is None:
        _, _, _, raw_fingerprint = _raw_inputs(raw, reference)
    source_fingerprint = _extraction_fingerprint(source)
    return dict(version=1, raw_fingerprint=raw_fingerprint,
        extraction_fingerprint=source_fingerprint,
        reviewed_fingerprint=source_fingerprint if source is extraction else _extraction_fingerprint(extraction),
        settings_fingerprint=_fingerprint(settings), qc_settings=deepcopy(settings), qc_units=deepcopy(QC_UNITS))


def validate_trial_report(raw, extraction, report):
    """Reject stale or edited reports before opening or confirming a review."""
    _, reference = _aligned(extraction)
    settings = _settings(report.attrs.get("settings"), reference)
    expected = _provenance(raw, extraction, settings)
    if report.attrs.get("provenance") != expected or report.attrs.get("report_fingerprint") != _fingerprint(report):
        raise ValueError("Trial QC provenance changed; detect quality again before review.")
    return expected


def _median_scale(values):
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.nan, np.nan
    median = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - median)))
    # Sparse deviations from a constant reference still need a finite scale.
    return median, max(scale, np.finfo(float).eps * max(abs(median), 1e-12))


def _longest_run(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return int(np.max(np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1), initial=0))


def _baseline_rows(ep):
    table = ep.baseline_diagnostics
    if table is None or table.empty:
        return {}
    if not {"channel", "original_trial_id", "valid"} <= set(table) or table.duplicated(["channel", "original_trial_id"]).any():
        raise ValueError("Baseline diagnostics need unique channel/original trial keys and valid flags.")
    return {(str(row["channel"]), int(row["original_trial_id"])): row for row in table.to_dict("records")}


def detect_trial_quality(raw, extraction, settings=None) -> pd.DataFrame:
    """Return all actual retained channel-trial pairs, metrics and candidate flags.

    ``settings`` is a mapping overriding ``DEFAULT_QC_SETTINGS``. Robust spike
    and jump scores reference each channel's full-record median and scaled MAD;
    clipping candidates require repeated local extremes, without claiming an
    amplifier limit. Detection intervals and thresholds are retained in attrs.
    """
    source, reference = _aligned(extraction)
    settings = _settings(settings, reference)
    data, times, names, raw_fingerprint = _raw_inputs(raw, reference)
    sfreq = float(raw.info["sfreq"])
    baseline = {band: _baseline_rows(ep) for band, ep in source.epochs.items()}
    manual = set(getattr(extraction, "trial_exclusions", ()) or ())
    rows = []
    for column, channel in enumerate(reference.channel_names):
        signal = data[names.index(channel)]
        median, scale = _median_scale(signal)
        safe_signal = np.where(np.isfinite(signal), signal, np.nan)
        diff_median, diff_scale = _median_scale(np.diff(safe_signal))
        for trial, event in enumerate(reference.events.to_dict("records")):
            pair = (channel, int(event["original_trial_id"]))
            start = float(event["onset"] - settings["pre_seconds"])
            stop = float(event["offset"] + settings["post_seconds"])
            left = int(np.searchsorted(times, start, side="left"))
            right = int(np.searchsorted(times, stop, side="right"))
            values = signal[left:right]
            finite = np.isfinite(values)
            differences = np.diff(np.where(finite, values, np.nan))
            finite_differences = np.isfinite(differences)
            finite_values = values[finite]
            spike = float(np.max(np.abs(finite_values - median) / scale)) if len(finite_values) else np.nan
            jump = float(np.max(np.abs(differences[finite_differences] - diff_median) / diff_scale)) if finite_differences.any() else np.nan
            flat_count = _longest_run(finite_differences & (np.abs(differences) <= settings["flat_tolerance_v"]))
            flat_seconds = flat_count / sfreq
            extreme = np.zeros(len(values), dtype=bool)
            if len(finite_values) and np.ptp(finite_values) > settings["flat_tolerance_v"]:
                low, high = np.min(finite_values), np.max(finite_values)
                extreme = finite & ((np.abs(values - low) <= settings["clipping_tolerance_v"])
                                    | (np.abs(values - high) <= settings["clipping_tolerance_v"]))
            clipping_fraction = float(extreme.sum() / max(1, finite.sum()))
            clipping_seconds = _longest_run(extreme) / sfreq
            row = dict(channel=channel, original_trial_id=pair[1], onset=event["onset"], offset=event["offset"],
                detection_start_seconds=start, detection_stop_seconds=stop, sample_count=len(values),
                raw_median_v=median, raw_mad_scale_v=scale, raw_diff_median_v=diff_median,
                raw_diff_mad_scale_v=diff_scale, peak_abs_v=float(np.max(np.abs(finite_values))) if len(finite_values) else np.nan,
                spike_z=spike, jump_z=jump, flat_run_seconds=flat_seconds,
                clipping_fraction=clipping_fraction, clipping_run_seconds=clipping_seconds,
                nonfinite_count=int((~finite).sum()),
                flag_spike=bool(spike > settings["spike_z"]), flag_jump=bool(jump > settings["jump_z"]),
                flag_flat=bool(flat_count > 0 and flat_seconds >= settings["flat_run_seconds"]),
                flag_clipping_candidate=bool(extreme.any() and clipping_fraction >= settings["clipping_fraction"]
                    and clipping_seconds >= settings["clipping_run_seconds"]),
                flag_nonfinite=bool((~finite).any()), flag_outside_recording=bool(start < times[0] or stop > times[-1]),
                raw_bad=channel in raw.info.get("bads", ()) or channel in reference.bad_channels,
                excluded=pair in manual)
            reasons = [name.removeprefix("flag_") for name, value in row.items() if name.startswith("flag_") and value]
            for band, ep in source.epochs.items():
                diagnostic = baseline[band].get(pair)
                invalid = (not bool(diagnostic["valid"])) if diagnostic is not None else not np.isfinite(ep.speech[trial, column]).all()
                reason = str(diagnostic.get("invalid_reason", "invalid_baseline")) if diagnostic is not None else "missing_original_response"
                row[f"baseline_{band}_invalid"] = invalid
                row[f"baseline_{band}_reason"] = reason if invalid else ""
                if diagnostic is not None:
                    for name, value in diagnostic.items():
                        if name not in {"channel", "original_trial_id", "band", "valid", "invalid_reason"}:
                            row[f"baseline_{band}_{name}"] = value
                if invalid:
                    reasons.append(f"{band}:{reason or 'invalid_baseline'}")
            row["candidate"] = bool(reasons)
            row["reasons"] = ";".join(reasons)
            rows.append(row)
    report = pd.DataFrame(rows)
    report.attrs.update(settings=deepcopy(settings), units=deepcopy(QC_UNITS),
        provenance=_provenance(raw, extraction, settings, raw_fingerprint=raw_fingerprint),
        threshold_note="Candidate review aids; clipping is not a confirmed hardware limit.")
    report.attrs["report_fingerprint"] = _fingerprint(report)
    return report


def _validated_pairs(extraction, pairs):
    _, reference = _aligned(extraction)
    if pairs is None:
        raise ValueError("Cancelled review (None) cannot be applied; () explicitly keeps all trials.")
    allowed = {(channel, int(trial)) for channel in reference.channel_names for trial in reference.events.original_trial_id}
    result = set()
    try:
        for pair in pairs:
            if (not isinstance(pair, (tuple, list)) or len(pair) != 2 or not isinstance(pair[0], str)
                    or isinstance(pair[1], (bool, np.bool_)) or not isinstance(pair[1], Integral)):
                raise ValueError("Exclusions must contain (actual channel, integer original trial ID) pairs.")
            pair = (pair[0], int(pair[1]))
            if pair not in allowed:
                raise ValueError(f"Unknown channel/trial pair: {pair}.")
            result.add(pair)
    except TypeError as error:
        raise ValueError("Exclusions must be an iterable of channel/trial pairs.") from error
    return tuple(sorted(result, key=lambda p: (reference.channel_names.index(p[0]),
                                             reference.events.original_trial_id.tolist().index(p[1]))))


def apply_trial_exclusions(extraction, excluded_pairs, *, smoothing_sigma_seconds=.06,
                           slope_method="savgol", max_baseline_invalid_fraction=.25) -> FourBandExtraction:
    """Replace manual masks and reaggregate all bands without changing source arrays.

    Original raw/baseline exclusions remain permanent. Manual NaNs affect valid
    trial counts but never the reported baseline invalid fraction. Reapplying ()
    to a reviewed copy restores manually removed responses from its source.
    Pass the extraction's feature settings explicitly to preserve its policy.
    """
    source, reference = _aligned(extraction)
    if source is extraction and getattr(extraction, "trial_exclusions", ()):
        raise ValueError("Reviewed extraction lost its original source; use the original extraction to replace masks.")
    pairs = _validated_pairs(extraction, excluded_pairs)
    provenance = getattr(excluded_pairs, "provenance", None)
    fingerprint = _extraction_fingerprint(source)
    if provenance is not None and provenance.get("extraction_fingerprint") != fingerprint:
        raise ValueError("Trial exclusion provenance changed; review the current extraction again.")
    old_source_fingerprint = getattr(extraction, "_trial_qc_source_fingerprint", fingerprint)
    if old_source_fingerprint != fingerprint:
        raise ValueError("Original extraction changed since trial review; review again.")
    feature_settings = dict(smoothing_sigma_seconds=smoothing_sigma_seconds,
        slope_method=slope_method, max_baseline_invalid_fraction=max_baseline_invalid_fraction)
    mask = np.array([[(ch, int(trial)) in pairs for ch in reference.channel_names]
                     for trial in reference.events.original_trial_id], dtype=bool)
    table = {"channel": list(reference.channel_names)}
    permanent = source.features.is_bad.to_numpy(bool).copy()
    reasons = [[str(reason)] if pd.notna(reason) and str(reason) else [] for reason in
               source.features.get("exclusion_reason", pd.Series([""] * len(permanent)))]
    final_bad = permanent.copy()
    epochs_by_band = {}
    for band, ep in source.epochs.items():
        arrays = {part: np.array(getattr(ep, part), dtype=float, copy=True) for part in ("speech", "pre", "post")}
        for array in arrays.values():
            array[mask] = np.nan
        reviewed = replace(ep, **arrays)
        epochs_by_band[band] = reviewed
        local = extract_beta_progress_features(reviewed, smoothing_sigma_seconds=smoothing_sigma_seconds,
            slope_method=slope_method, max_baseline_invalid_fraction=max_baseline_invalid_fraction, mask_bad_features=False)
        original_valid = np.isfinite(ep.speech).all(axis=-1)
        baseline_fraction = source.features.get(f"qc_{band}_invalid_fraction", pd.Series(1 - original_valid.mean(axis=0))).to_numpy(float)
        valid_count = np.isfinite(reviewed.speech).all(axis=-1).sum(axis=0)
        remaining = (~mask).sum(axis=0)
        band_bad = (np.isin(reference.channel_names, list(ep.bad_channels))
                    | (baseline_fraction > max_baseline_invalid_fraction) | (valid_count < 2))
        final_bad |= band_bad
        for index in np.flatnonzero(band_bad):
            parts = []
            if reference.channel_names[index] in ep.bad_channels:
                parts.append("raw_bad")
            if baseline_fraction[index] > max_baseline_invalid_fraction:
                parts.append("invalid_trial_fraction")
            if valid_count[index] < 2:
                parts.append("insufficient_valid_trials")
            reasons[index].append(band + ":" + ",".join(parts))
        for column in local:
            if column.startswith("beta_progress__"):
                suffix = column.removeprefix("beta_progress__").replace("beta_slope", "power_slope")
                table[f"{band}__{suffix}"] = local[column].to_numpy().copy()
        table[f"qc_{band}_invalid_fraction"] = baseline_fraction
        table[f"qc_{band}_valid_trials"] = valid_count
        table[f"qc_{band}_remaining_trials"] = remaining
        table[f"qc_{band}_manual_excluded_trials"] = mask.sum(axis=0)
        table[f"qc_{band}_manual_fraction"] = mask.mean(axis=0)
        table[f"qc_{band}_remaining_invalid_fraction"] = np.divide(remaining - valid_count, remaining,
            out=np.full(len(remaining), np.nan), where=remaining > 0)
    for column in table:
        if "__" in column:
            table[column][final_bad] = np.nan
    table["is_bad"] = final_bad
    table["newly_ineligible"] = final_bad & ~permanent
    table["exclusion_reason"] = [";".join(dict.fromkeys(parts)) for parts in reasons]
    changes = dict(features=pd.DataFrame(table), epochs=epochs_by_band,
                   bands=deepcopy(source.bands), filter_kwargs=deepcopy(source.filter_kwargs))
    attributes = dict(feature_settings=feature_settings, trial_exclusions=TrialExclusions(pairs, provenance=provenance))
    available = {field.name for field in fields(extraction)}
    result = replace(extraction, **changes, **{k: v for k, v in attributes.items() if k in available})
    for name, value in attributes.items():
        setattr(result, name, value)
    result._trial_qc_source = source
    result._trial_qc_source_fingerprint = fingerprint
    result.trial_qc_provenance = deepcopy(provenance) if provenance is not None else dict(extraction_fingerprint=fingerprint)
    return result
