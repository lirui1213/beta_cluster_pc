"""Continuous real-time power across two neighboring speech events."""
from dataclasses import dataclass
from collections import OrderedDict
from collections.abc import Mapping
import hashlib
import json
from numbers import Integral
from pathlib import Path
import pickle
from functools import wraps
from threading import RLock

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from audio_cross_3band_utils import load_ecog_fif, load_speech_events, extract_audio_envelope
from signal_processing import finite_mean, gaussian_smooth_nan, normalize_power, resample_on_axis
from visualization import cluster_color
from .analysis import _validate_bands


_HASH_CHUNK_BYTES = 1024 ** 2


def _update_array_hash(digest, values):
    """Hash logical array contents with at most a 1 MiB temporary buffer."""
    values = np.asarray(values)
    digest.update(str((values.shape, values.dtype.str)).encode("ascii"))
    count = max(1, _HASH_CHUNK_BYTES // values.dtype.itemsize)
    iterator = np.nditer(values, flags=["external_loop", "buffered", "zerosize_ok"],
                         op_flags=["readonly"], order="C", buffersize=count)
    for block in iterator:
        for start in range(0, block.size, count):
            digest.update(block[start:start + count].tobytes())


def _freeze_config(value):
    if isinstance(value, Mapping):
        return tuple(sorted((key, _freeze_config(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_config(item) for item in value)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return (value.shape, tuple(_freeze_config(item) for item in value.flat))
        digest = hashlib.sha256()
        _update_array_hash(digest, value)
        return (value.shape, value.dtype.str, digest.digest())
    return value


def _power_cache_key(raw, channels, bands, filter_kwargs):
    digest = hashlib.sha256()
    # Hash all channels and the full recording, including preloaded mutable data.
    chunk_samples = max(1, _HASH_CHUNK_BYTES // (8 * len(raw.ch_names)))
    for start in range(0, raw.n_times, chunk_samples):
        stop = min(start + chunk_samples, raw.n_times)
        values = (raw._data[:, start:stop] if raw.preload
                  else raw.get_data(start=start, stop=stop))
        _update_array_hash(digest, values)
    annotations = raw.annotations
    provenance = (tuple(raw.ch_names), tuple(raw.get_channel_types()), channels,
                  float(raw.info["sfreq"]), raw.first_samp, raw.n_times,
                  annotations.orig_time, annotations.onset, annotations.duration,
                  tuple(annotations.description), getattr(annotations, "ch_names", ()),
                  bands, filter_kwargs, "zero", "hilbert-envelope-square")
    digest.update(pickle.dumps(_freeze_config(provenance), protocol=5))
    return digest.digest()


def _cache_locked(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return locked


class ContinuousPowerCache:
    """Bounded session LRU of raw band power, before any baseline or display.

    Every lookup validates the full current signal. ``info()`` reports resident
    array bytes, hits/misses/evictions/fallbacks and the last fallback reason.
    ``clear()`` releases arrays and resets counters. Oversized requests use the
    original band-at-a-time uncached path. Short metadata updates are locked so
    UI clearing and background numerical jobs cannot corrupt LRU accounting.
    """

    def __init__(self, max_bytes=512 * 1024 ** 2):
        if isinstance(max_bytes, (bool, np.bool_)) or not isinstance(max_bytes, Integral) or max_bytes < 0:
            raise ValueError("max_bytes must be a nonnegative integer.")
        self.max_bytes = int(max_bytes)
        self._lock = RLock()
        self.clear()

    @_cache_locked
    def clear(self):
        self._entries = OrderedDict()
        self._bytes = 0
        self._hits = self._misses = self._evictions = self._fallbacks = 0
        self._status, self._reason = "empty", None

    @_cache_locked
    def info(self):
        return dict(max_bytes=self.max_bytes, bytes=self._bytes, entries=len(self._entries),
                    hits=self._hits, misses=self._misses, evictions=self._evictions,
                    fallbacks=self._fallbacks, status=self._status, reason=self._reason)

    @_cache_locked
    def _lookup(self, key):
        self._reason = None
        if key in self._entries:
            self._hits += 1
            self._status = "hit"
            self._entries.move_to_end(key)
            return self._entries[key]
        self._misses += 1
        self._status = "miss"
        return None

    @_cache_locked
    def _fallback(self, required):
        self._misses += 1
        self._fallbacks += 1
        self._status = "fallback"
        self._reason = f"Raw power needs {required} bytes, exceeding cache limit {self.max_bytes}; computed uncached."

    @_cache_locked
    def _store(self, key, power):
        size = sum(values.nbytes for values in power.values())
        if size > self.max_bytes:
            self._fallback(size)
            return
        while self._bytes + size > self.max_bytes:
            _, removed = self._entries.popitem(last=False)
            self._bytes -= sum(values.nbytes for values in removed.values())
            self._evictions += 1
        for values in power.values():
            values.setflags(write=False)
        self._entries[key] = power
        self._bytes += size


@dataclass
class ContinuousTrials:
    times: np.ndarray
    power: dict
    audio: np.ndarray
    channels: tuple
    bands: dict
    trial_ids: tuple
    events: pd.DataFrame
    metadata: dict


def _ordered_events(events):
    if not {"onset", "offset"} <= set(events):
        raise ValueError("Original trial events need onset and offset columns.")
    ordered = events.sort_values("onset").reset_index(drop=True).copy()
    values = ordered[["onset", "offset"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values[:, 1] <= values[:, 0]):
        raise ValueError("Original trial intervals must be finite and positive.")
    if "original_trial_id" not in ordered:
        ordered["original_trial_id"] = np.arange(len(ordered))
    ids = pd.to_numeric(ordered.original_trial_id, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(ids).all() or np.any(ids != np.round(ids)) or len(np.unique(ids)) != len(ids):
        raise ValueError("Original trial IDs must be unique integers.")
    ordered["original_trial_id"] = ids.astype(int)
    return ordered


def adjacent_trial_pairs(events, retained_events):
    """Only pairs neighboring in the original recording, not after dropping."""
    ordered = _ordered_events(events)
    retained = set(retained_events.original_trial_id)
    rows = []
    for i in range(len(ordered) - 1):
        first, second = ordered.iloc[i], ordered.iloc[i + 1]
        if first.original_trial_id in retained and second.original_trial_id in retained and second.onset >= first.offset:
            rows.append(dict(first_trial_id=int(first.original_trial_id), second_trial_id=int(second.original_trial_id),
                first_onset=float(first.onset), first_offset=float(first.offset), second_onset=float(second.onset),
                second_offset=float(second.offset), gap_seconds=float(second.onset - first.offset)))
    return pd.DataFrame(rows, columns=["first_trial_id", "second_trial_id", "first_onset", "first_offset",
        "second_onset", "second_offset", "gap_seconds"])


def prepare_continuous_trials(raw, events, audio, extraction, *, trial_ids=None,
                             pre_seconds=1., post_seconds=1., sigma_seconds=.10, display_sfreq=100.,
                             cache=None):
    """Filter actual continuous data and keep one baseline across the interval.

    Full-record filtering/Hilbert matches extraction and avoids an artificial
    edge at either event or the gap. The returned axis is seconds from onset 1.
    """
    settings = [pre_seconds, post_seconds, sigma_seconds, display_sfreq]
    if not np.isfinite(settings).all() or min(settings[:3]) < 0 or display_sfreq <= 0:
        raise ValueError("Margins and smoothing must be nonnegative; display_sfreq must be positive.")
    _validate_bands(extraction.bands, float(raw.info["sfreq"]))
    reference = next(iter(extraction.epochs.values()))
    channels = tuple(reference.channel_names)
    pairs = adjacent_trial_pairs(events, reference.events)
    if trial_ids is None:
        if pairs.empty:
            raise ValueError("No adjacent retained trial pair; check original events and extraction exclusions.")
        trial_ids = (int(pairs.first_trial_id.iloc[0]), int(pairs.second_trial_id.iloc[0]))
    if (not isinstance(trial_ids, (tuple, list)) or len(trial_ids) != 2
            or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) for value in trial_ids)):
        raise ValueError("trial_ids must be two integer original trial IDs.")
    trial_ids = tuple(map(int, trial_ids))
    selected_pair = pairs.loc[(pairs.first_trial_id == trial_ids[0]) & (pairs.second_trial_id == trial_ids[1])]
    if len(selected_pair) != 1:
        raise ValueError("Choose two chronologically adjacent retained original trial IDs from the pair table.")
    ordered = _ordered_events(events)
    selected = ordered.loc[ordered.original_trial_id.isin(trial_ids)].copy()
    retained = reference.events.set_index("original_trial_id")
    if retained.index.has_duplicates or not np.allclose(
            retained.loc[list(trial_ids), ["onset", "offset"]].to_numpy(dtype=float),
            selected[["onset", "offset"]].to_numpy(dtype=float), rtol=0, atol=1e-9):
        raise ValueError("Original event onset/offset changed since extraction; rerun extraction and clustering.")
    first, second = selected.iloc[0], selected.iloc[1]
    recording_start, recording_stop = float(raw.times[0]), float(raw.times[-1])
    if first.onset < recording_start or second.offset > recording_stop:
        raise ValueError("Selected trials lie outside the continuous recording.")
    requested = [float(first.onset - pre_seconds), float(second.offset + post_seconds)]
    bounds = [max(requested[0], recording_start), min(requested[1], recording_stop)]
    count = int(np.floor((bounds[1] - bounds[0]) * display_sfreq + 1e-9)) + 1
    if count < 2:
        raise ValueError("Continuous display interval needs at least two samples.")
    absolute_times = bounds[0] + np.arange(count) / display_sfreq
    if not channels or any(name not in raw.ch_names for name in channels):
        raise ValueError("Continuous recording channels do not align with extraction.")
    if set(extraction.epochs) != set(extraction.bands):
        raise ValueError("Continuous bands must align with extracted bands.")

    # Validate every reference before any expensive signal processing.
    statistics, valid_counts = {}, {}
    for band, epochs in extraction.epochs.items():
        if (tuple(epochs.channel_names) != channels or not epochs.events.equals(reference.events)
                or tuple(epochs.beta_band) != tuple(extraction.bands[band])
                or not np.isclose(epochs.sfreq, raw.info["sfreq"])):
            raise ValueError(f"{band} trial/channel/band/sample-rate alignment differs from extraction.")
        if epochs.baseline_mode != reference.baseline_mode:
            raise ValueError("Baseline modes must align across bands.")
        diagnostics = epochs.baseline_diagnostics
        if diagnostics is None or not {"original_trial_id", "channel", "baseline_mean", "baseline_std", "valid"} <= set(diagnostics):
            raise ValueError(f"{band} baseline diagnostics are unavailable; rerun extraction.")
        anchored = diagnostics.loc[diagnostics.original_trial_id == trial_ids[0]]
        if tuple(anchored.channel) != channels:
            raise ValueError(f"{band} baseline channels do not align with continuous data.")
        mean = anchored.baseline_mean.to_numpy(dtype=float)
        std = anchored.baseline_std.to_numpy(dtype=float)
        invalid = ~anchored.valid.to_numpy(dtype=bool) | ~np.isfinite(mean) | ~np.isfinite(std) | (std <= 0)
        if invalid.all():
            raise ValueError(f"{band} has no valid baseline for the first selected trial.")
        statistics[band] = (mean, std, invalid)
        valid_counts[band] = int((~invalid).sum())

    if cache is not None and not isinstance(cache, ContinuousPowerCache):
        raise TypeError("cache must be a ContinuousPowerCache or None.")
    sample_times = np.asarray(raw.times)
    filter_kwargs = dict(getattr(extraction, "filter_kwargs", None) or {})
    filter_kwargs.pop("phase", None)
    filter_kwargs.pop("verbose", None)
    cached_power, key = None, None
    if cache is not None:
        required = len(channels) * raw.n_times * 8 * len(extraction.bands)
        if required > cache.max_bytes:
            cache._fallback(required)
        else:
            key = _power_cache_key(raw, channels, extraction.bands, filter_kwargs)
            cached_power = cache._lookup(key)
    picked = None if cached_power is not None else raw.copy().pick(list(channels)).load_data()
    pending_power = {} if key is not None and cached_power is None else None
    power = {}
    for band, (low, high) in extraction.bands.items():
        if cached_power is not None:
            values = cached_power[band]
        else:
            filtered = picked.copy().filter(low, high, phase="zero", verbose="ERROR", **filter_kwargs)
            filtered.apply_hilbert(envelope=True, verbose="ERROR")
            values = np.square(filtered.get_data().astype(float, copy=False))
            del filtered
            if pending_power is not None:
                pending_power[band] = values
        normalized = normalize_power(values, *statistics[band])
        smoothed = gaussian_smooth_nan(normalized, float(raw.info["sfreq"]), sigma_seconds=sigma_seconds)
        power[band] = resample_on_axis(smoothed, sample_times, absolute_times)
        del values, normalized, smoothed
    if pending_power is not None:
        cache._store(key, pending_power)
    audio_times = np.asarray(audio.times, dtype=float)
    audio_values = np.asarray(audio.values, dtype=float)
    if audio_values.ndim != 1 or len(audio_values) != len(audio_times):
        raise ValueError("Audio envelope values and times must align.")
    # One shared scale makes the two utterances directly comparable.
    peaks = [np.abs(e.audio_speech[np.isfinite(e.audio_speech)]) for e in extraction.epochs.values()]
    peak = max((float(values.max()) for values in peaks if len(values)), default=0.)
    scale = peak if peak > 0 else 1.
    envelope = gaussian_smooth_nan(audio_values / scale, audio.sfreq, sigma_seconds=sigma_seconds)
    sampled_audio = resample_on_axis(envelope, audio_times, absolute_times)
    context = ordered.loc[(ordered.offset >= absolute_times[0]) & (ordered.onset <= absolute_times[-1])].copy()
    mode = reference.baseline_mode
    metadata = dict(trial_ids=list(trial_ids), time_axis="Real seconds from first onset; no duration normalization",
        first_onset_seconds=float(first.onset), gap_seconds=float(second.onset - first.offset),
        requested_bounds_seconds=requested, actual_bounds_seconds=[float(absolute_times[0]), float(absolute_times[-1])],
        clipped_to_recording=bool(bounds != requested), display_sfreq=float(display_sfreq),
        sigma_seconds=float(sigma_seconds), baseline_mode=mode,
        baseline_anchor_trial_id=trial_ids[0] if mode == "pre_onset" else None,
        baseline_description="First selected trial baseline for entire display" if mode == "pre_onset" else "Patient baseline file; same reference as extraction",
        valid_baseline_channels=valid_counts, audio_scale=scale,
        selected_events=selected[["original_trial_id", "onset", "offset"]].to_dict("records"),
        context_event_ids=[int(value) for value in context.original_trial_id], bands=extraction.bands,
        trial_exclusions=[(str(channel), int(trial_id))
            for channel, trial_id in (getattr(extraction, "trial_exclusions", ()) or ())],
        power_cache=cache.info() if cache is not None else dict(status="disabled", reason=None))
    return ContinuousTrials(absolute_times - first.onset, power, sampled_audio, channels,
        dict(extraction.bands), trial_ids, context, metadata)


def load_continuous_trials(fif_path, onset_path, offset_path, wav_path, extraction, *, cache=None, **kwargs):
    """Load original continuous inputs; epoch arrays cannot supply the gap."""
    return prepare_continuous_trials(load_ecog_fif(fif_path, preload=False),
        load_speech_events(onset_path, offset_path), extract_audio_envelope(wav_path), extraction,
        cache=cache, **kwargs)


def save_continuous_trial_plot(data, result, output_dir):
    """Export cluster traces, electrode SEM, envelope and explicit time metadata."""
    if tuple(result.channels) != data.channels:
        raise ValueError("Continuous channels must align with fitted cluster channels.")
    labels = np.asarray(result.labels)
    included = np.asarray(result.clean_mask, dtype=bool)
    if labels.shape != (len(data.channels),) or included.shape != labels.shape:
        raise ValueError("Cluster labels must align with continuous channels.")
    unique = np.unique(labels[included & (labels >= 0)])
    if not len(unique):
        raise ValueError("No included clusters for continuous display.")
    for band, values in data.power.items():
        if not np.isfinite(values[included & (labels >= 0)]).any():
            raise ValueError(f"No valid {band} first-trial baseline among included electrodes; choose another pair.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(len(unique), 4, figsize=(24, 3.2 * len(unique)), squeeze=False,
        sharex=True, layout="constrained")
    origin = data.metadata["first_onset_seconds"]
    trial_exclusions = set(tuple(pair) for pair in data.metadata.get("trial_exclusions", ()))
    records = []
    try:
        for col, (band, bounds) in enumerate(data.bands.items()):
            for row, label in enumerate(unique):
                ax = axes[row, col]
                traces = data.power[band][included & (labels == label)]
                count = np.isfinite(traces).sum(axis=0)
                mean = finite_mean(traces)
                squared = np.where(np.isfinite(traces), (traces - mean) ** 2, 0).sum(axis=0)
                sem = np.full_like(mean, np.nan)
                np.divide(squared, count * (count - 1), out=sem, where=count > 1)
                sem = np.sqrt(sem)
                color = cluster_color(int(label))
                ax.plot(data.times, mean, color=color, linewidth=1.5)
                if not np.isfinite(mean).any():
                    ax.text(.5, .5, "No valid first-trial baseline in this cluster",
                        transform=ax.transAxes, ha="center", va="center", fontsize=8)
                ax.fill_between(data.times, mean - sem, mean + sem, color=color, alpha=.18)
                ax.axhline(0, color="gray", linewidth=.6)
                cluster_channels = [channel for channel, keep in zip(data.channels,
                    included & (labels == label), strict=True) if keep]
                for event in data.events.itertuples():
                    selected = event.original_trial_id in data.trial_ids
                    start, stop = event.onset - origin, event.offset - origin
                    qc_count = sum((channel, int(event.original_trial_id)) in trial_exclusions
                                   for channel in cluster_channels)
                    if qc_count:
                        ax.axvspan(start, stop, facecolor="orange", edgecolor="darkorange",
                                   hatch="//", alpha=.12)
                    else:
                        ax.axvspan(start, stop, color="green" if selected else "gray", alpha=.07)
                    ax.axvline(start, color="gray", linestyle="--", linewidth=.7)
                    ax.axvline(stop, color="gray", linestyle=":", linewidth=.7)
                    if row == 0 or qc_count:
                        left, right = max(start, data.times[0]), min(stop, data.times[-1])
                        event_label = f"Trial {event.original_trial_id}"
                        if qc_count:
                            event_label += f" | QC {qc_count}"
                        ax.text((left + right) / 2, .97, event_label,
                            transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
                right_ax = ax.twinx()
                right_ax.plot(data.times, data.audio, color="black", linestyle="--", linewidth=1., alpha=.7)
                right_ax.set_ylabel("Speech envelope / speech peak", fontsize=8)
                ax.set_ylabel("Power Z (mean +/- electrode SEM)", fontsize=8)
                ax.set_title(f"Cluster {int(label)} (n={len(traces)}) | {band} {bounds[0]:g}-{bounds[1]:g} Hz", fontsize=10)
                ax.set_xlim(data.times[0], data.times[-1])
                if row == len(unique) - 1:
                    ax.set_xlabel("Real time from first onset (s)", fontsize=9)
                ax.tick_params(labelsize=8)
                right_ax.tick_params(labelsize=8)
                for i, time in enumerate(data.times):
                    records.append(dict(band=band, label=int(label), elapsed_seconds=float(time),
                        recording_seconds=float(time + origin), mean_power=float(mean[i]), sem_power=float(sem[i]),
                        n_electrodes=int(count[i]), mean_audio=float(data.audio[i])))
        ids = " -> ".join(map(str, data.trial_ids))
        fig.suptitle(f"Continuous trials {ids} | real gap {data.metadata['gap_seconds']:.3f} s\n"
            + data.metadata["baseline_description"], fontsize=12)
        fig.savefig(output / "fourband_continuous_trials.png", dpi=130)
        table = pd.DataFrame(records)
        table.to_csv(output / "fourband_continuous_trials.csv", index=False, encoding="utf-8-sig")
        metadata = dict(data.metadata, cluster_labels=labels.tolist(), channel_names=list(data.channels),
            uncertainty="Electrode SEM conditional on the selected pair and fitted labels; not trial confidence interval")
        (output / "continuous_trial_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return table
    finally:
        plt.close(fig)
