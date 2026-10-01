"""Signal processing and feature engineering for three-band ECoG analysis.

The module keeps I/O, signal processing, epoch construction, audio alignment, and
electrode-level feature extraction independent from clustering and plotting. MNE
is imported lazily so event and audio utilities remain usable without MNE.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt


FloatArray = NDArray[np.float64]
DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "beta": (13.0, 30.0),
    "high_gamma": (70.0, 150.0),
}
BETA_ONLY_BANDS: dict[str, tuple[float, float]] = {"beta": (13.0, 30.0)}
ONSET_WINDOW = (-1.0, 6.0)
OFFSET_WINDOW = (-1.0, 2.0)
BASELINE_WINDOW = (-1.0, -0.7)


@dataclass(frozen=True)
class AudioEnvelope:
    """A normalized low-frequency audio envelope and its absolute time axis."""

    values: FloatArray
    times: FloatArray
    sfreq: float


@dataclass(frozen=True)
class BandPowerEpochs:
    """Trial-aligned, baseline-normalized power for all requested frequency bands.

    Arrays in ``onset`` and ``offset`` have shape ``(trials, channels, times)``.
    ``articulation_means`` has shape ``(trials, channels)`` and is calculated
    from the complete onset-to-offset interval in the continuous band-power
    signal. Offset epochs and articulation means use the corresponding trial's
    onset baseline parameters.
    """

    onset: Mapping[str, FloatArray]
    offset: Mapping[str, FloatArray]
    onset_times: FloatArray
    offset_times: FloatArray | None
    articulation_means: Mapping[str, FloatArray]
    events: pd.DataFrame
    channel_names: tuple[str, ...]
    bad_channels: frozenset[str]
    sfreq: float
    effective_bands: Mapping[str, tuple[float, float]]


@dataclass(frozen=True)
class BetaProgressEpochs:
    """Beta power normalized to each trial's speech progress.

    ``speech`` has shape ``(trials, channels, progress_points)`` and spans
    speech onset to speech offset as 0-100 percent progress. ``pre`` and
    ``post`` use fixed real-time windows whose size is derived from the minimum
    positive interval between each trial's offset and the next trial's onset.
    All ECoG arrays are baseline-Z scored per trial and channel using the
    complete pre-onset window.
    """

    speech: FloatArray
    pre: FloatArray
    post: FloatArray
    audio_speech: FloatArray
    audio_pre: FloatArray
    audio_post: FloatArray
    progress_percent: FloatArray
    pre_times: FloatArray
    post_times: FloatArray
    events: pd.DataFrame
    dropped_events: pd.DataFrame
    channel_names: tuple[str, ...]
    bad_channels: frozenset[str]
    sfreq: float
    beta_band: tuple[float, float]
    pre_post_window: float


def load_ecog_fif(path: str | Path, *, preload: bool = False) -> Any:
    """Load a continuous FIF recording while preserving ``raw.info['bads']``.

    Parameters
    ----------
    path:
        Path to a ``*_clean_car-raw.fif`` file.
    preload:
        Whether MNE should immediately load all samples into memory. Filtering
        later loads a band-specific copy regardless of this setting.
    """

    try:
        import mne
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("load_ecog_fif requires MNE-Python (`pip install mne`).") from exc

    fif_path = Path(path)
    if not fif_path.is_file():
        raise FileNotFoundError(f"ECoG FIF file does not exist: {fif_path}")
    return mne.io.read_raw_fif(fif_path, preload=preload, verbose="ERROR")


def _find_event_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str:
    normalized = {
        str(column).lower().strip().replace("_", " "): str(column)
        for column in frame.columns
    }
    for candidate in candidates:
        key = candidate.lower().strip().replace("_", " ")
        if key in normalized:
            return normalized[key]
    raise ValueError(
        f"Could not find any of {list(candidates)!r} in event columns "
        f"{list(frame.columns)!r}."
    )


def _array_to_event_frame(array: ArrayLike, offset: ArrayLike | None) -> pd.DataFrame:
    values = np.asarray(array)
    if values.dtype.names:
        frame = pd.DataFrame.from_records(values)
        return frame
    if values.ndim == 2 and values.shape[1] >= 2:
        return pd.DataFrame({"onset": values[:, 0], "offset": values[:, 1]})
    if values.ndim == 1 and offset is not None:
        offset_values = np.asarray(offset)
        if offset_values.ndim != 1 or len(offset_values) != len(values):
            raise ValueError("Separate onset and offset arrays must be one-dimensional and equal length.")
        return pd.DataFrame({"onset": values, "offset": offset_values})
    raise ValueError(
        "A NPY/array event source must contain two columns, structured onset/offset "
        "fields, or be accompanied by a separate offset source."
    )


def _load_array_source(source: str | Path | ArrayLike) -> NDArray[Any]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Event file does not exist: {path}")
        return np.load(path, allow_pickle=False)
    return np.asarray(source)


def load_speech_events(
    source: str | Path | pd.DataFrame | ArrayLike,
    offset_source: str | Path | ArrayLike | None = None,
    *,
    onset_column: str | None = None,
    offset_column: str | None = None,
) -> pd.DataFrame:
    """Load and validate speech onset/offset events.

    ``source`` may be a CSV path, a DataFrame, a structured NPY array, an
    ``(n_trials, >=2)`` array, or a one-dimensional onset array accompanied by
    ``offset_source``. The returned table always contains ``trial_id``,
    ``onset``, ``offset``, and ``duration``.
    """

    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    elif isinstance(source, (str, Path)) and Path(source).suffix.lower() in {".csv", ".tsv"}:
        separator = "\t" if Path(source).suffix.lower() == ".tsv" else ","
        frame = pd.read_csv(source, sep=separator)
    else:
        onset_values = _load_array_source(source)
        offset_values = None if offset_source is None else _load_array_source(offset_source)
        frame = _array_to_event_frame(onset_values, offset_values)

    onset_name = onset_column or _find_event_column(
        frame, ("onset", "speech onset", "speech_onset", "start", "start time")
    )
    offset_name = offset_column or _find_event_column(
        frame, ("offset", "speech offset", "speech_offset", "end", "end time")
    )
    onset = pd.to_numeric(frame[onset_name], errors="coerce").to_numpy(dtype=float)
    offset = pd.to_numeric(frame[offset_name], errors="coerce").to_numpy(dtype=float)

    if onset.shape != offset.shape:
        raise ValueError("Onset and offset columns must have equal length.")
    invalid = ~np.isfinite(onset) | ~np.isfinite(offset) | (offset <= onset)
    if invalid.any():
        indices = np.flatnonzero(invalid).tolist()
        raise ValueError(f"Invalid or non-positive speech intervals at rows {indices}.")

    result = pd.DataFrame(
        {
            "trial_id": np.arange(len(onset), dtype=int),
            "onset": onset,
            "offset": offset,
            "duration": offset - onset,
        }
    )
    return result


def resolve_frequency_bands(
    sfreq: float,
    bands: Mapping[str, tuple[float, float]] = DEFAULT_BANDS,
    *,
    nyquist_margin_hz: float | None = None,
) -> dict[str, tuple[float, float]]:
    """Validate bands and clip upper cutoffs that reach the Nyquist frequency."""

    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Sampling frequency must be a finite positive number.")
    nyquist = sfreq / 2.0
    margin = nyquist_margin_hz
    if margin is None:
        # A finite transition region is needed by practical FIR designs when a
        # requested cutoff lands exactly on Nyquist.
        margin = max(0.5, sfreq * 1e-6)
    resolved: dict[str, tuple[float, float]] = {}
    for name, (low, high) in bands.items():
        low_f, high_f = float(low), float(high)
        if low_f <= 0 or high_f <= low_f:
            raise ValueError(f"Invalid frequency band {name!r}: {(low, high)}")
        if low_f >= nyquist:
            raise ValueError(
                f"Band {name!r} starts at {low_f:g} Hz, not below the "
                f"{nyquist:g} Hz Nyquist frequency."
            )
        resolved[name] = (low_f, min(high_f, nyquist - margin))
    return resolved


def _time_grid(window: tuple[float, float], sfreq: float) -> FloatArray:
    start, stop = map(float, window)
    if stop <= start:
        raise ValueError(f"Invalid epoch window: {window}")
    count = int(round((stop - start) * sfreq)) + 1
    return np.arange(count, dtype=float) / sfreq + start


def _sample_continuous(
    data: FloatArray,
    sample_times: FloatArray,
    alignments: FloatArray,
    relative_times: FloatArray,
) -> FloatArray:
    """Linearly sample continuous channel-by-time data at trial-relative times."""

    output = np.empty((len(alignments), data.shape[0], len(relative_times)), dtype=float)
    for trial_index, alignment in enumerate(alignments):
        targets = alignment + relative_times
        for channel_index, channel_data in enumerate(data):
            output[trial_index, channel_index] = np.interp(targets, sample_times, channel_data)
    return output


def _valid_event_mask(
    events: pd.DataFrame,
    recording_start: float,
    recording_stop: float,
    onset_window: tuple[float, float],
    offset_window: tuple[float, float] | None,
) -> NDArray[np.bool_]:
    onset = events["onset"].to_numpy(float)
    offset = events["offset"].to_numpy(float)
    valid = (
        (onset + onset_window[0] >= recording_start)
        & (onset + onset_window[1] <= recording_stop)
        & (onset >= recording_start)
        & (offset <= recording_stop)
    )
    if offset_window is not None:
        valid &= (offset + offset_window[0] >= recording_start) & (
            offset + offset_window[1] <= recording_stop
        )
    return valid


def filter_duration_outlier_trials(
    events: pd.DataFrame,
    *,
    mad_threshold: float = 2.5,
    min_trials: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove trials whose speech duration is far from the robust center.

    Durations are compared to the median using the median absolute deviation
    (MAD). If MAD is effectively zero, the function falls back to an IQR fence.
    If filtering would leave fewer than ``min_trials`` trials, no duration
    trials are dropped and a warning is emitted.
    """

    if mad_threshold <= 0:
        raise ValueError("mad_threshold must be positive.")
    if min_trials < 1:
        raise ValueError("min_trials must be at least one.")
    required_columns = {"onset", "offset", "duration"}
    if not required_columns.issubset(events.columns):
        raise ValueError(f"Events must contain columns {sorted(required_columns)}.")

    frame = events.copy().reset_index(drop=True)
    if "original_trial_id" not in frame:
        frame["original_trial_id"] = np.arange(len(frame), dtype=int)
    duration = frame["duration"].to_numpy(float)
    median = float(np.median(duration))
    mad = float(np.median(np.abs(duration - median)))
    if mad > np.finfo(float).eps:
        robust_sigma = 1.4826 * mad
        robust_z = (duration - median) / robust_sigma
        keep = np.abs(robust_z) <= mad_threshold
    else:
        q1, q3 = np.percentile(duration, [25.0, 75.0])
        iqr = float(q3 - q1)
        if iqr <= np.finfo(float).eps:
            robust_z = np.zeros_like(duration, dtype=float)
            keep = np.ones_like(duration, dtype=bool)
        else:
            lower = q1 - mad_threshold * iqr
            upper = q3 + mad_threshold * iqr
            robust_z = (duration - median) / iqr
            keep = (duration >= lower) & (duration <= upper)

    frame["duration_robust_z"] = robust_z
    frame["duration_keep"] = keep
    if int(np.count_nonzero(keep)) < min_trials:
        warnings.warn(
            "Duration filtering would leave too few trials; keeping all trials.",
            RuntimeWarning,
            stacklevel=2,
        )
        frame["duration_keep"] = True
        keep = np.ones_like(keep, dtype=bool)

    kept = frame.loc[keep].drop(columns=["duration_keep"]).reset_index(drop=True)
    dropped = frame.loc[~keep].drop(columns=["duration_keep"]).reset_index(drop=True)
    return kept, dropped


def minimum_intertrial_gap(events: pd.DataFrame) -> float:
    """Return the minimum positive ``onset[n+1] - offset[n]`` gap in seconds."""

    if len(events) < 2:
        raise ValueError("At least two trials are required to estimate an intertrial gap.")
    ordered = events.sort_values("onset").reset_index(drop=True)
    onset = ordered["onset"].to_numpy(float)
    offset = ordered["offset"].to_numpy(float)
    gaps = onset[1:] - offset[:-1]
    positive = gaps[np.isfinite(gaps) & (gaps > 0)]
    if positive.size == 0:
        raise ValueError("No positive intertrial gaps were found.")
    return float(np.min(positive))


def _duration_valid_event_mask(
    events: pd.DataFrame,
    recording_start: float,
    recording_stop: float,
    pre_post_window: float,
) -> NDArray[np.bool_]:
    onset = events["onset"].to_numpy(float)
    offset = events["offset"].to_numpy(float)
    return (
        (onset - pre_post_window >= recording_start)
        & (offset + pre_post_window <= recording_stop)
        & (onset >= recording_start)
        & (offset <= recording_stop)
    )


def _sample_trial_progress(
    data: FloatArray,
    sample_times: FloatArray,
    onset: FloatArray,
    offset: FloatArray,
    progress_fraction: FloatArray,
) -> FloatArray:
    """Sample continuous data from onset to offset on a normalized progress grid."""

    output = np.empty((len(onset), data.shape[0], len(progress_fraction)), dtype=float)
    for trial_index, (start, stop) in enumerate(zip(onset, offset, strict=True)):
        targets = start + progress_fraction * (stop - start)
        for channel_index, channel_data in enumerate(data):
            output[trial_index, channel_index] = np.interp(targets, sample_times, channel_data)
    return output


def _sample_audio_trial_progress(
    envelope: AudioEnvelope,
    events: pd.DataFrame,
    progress_fraction: FloatArray,
    *,
    fill_value: float = np.nan,
) -> FloatArray:
    onset = events["onset"].to_numpy(float)
    offset = events["offset"].to_numpy(float)
    result = np.empty((len(events), len(progress_fraction)), dtype=float)
    for index, (start, stop) in enumerate(zip(onset, offset, strict=True)):
        targets = start + progress_fraction * (stop - start)
        result[index] = np.interp(
            targets,
            envelope.times,
            envelope.values,
            left=fill_value,
            right=fill_value,
        )
    return result


def _articulation_means(
    power: FloatArray,
    sample_times: FloatArray,
    onset: FloatArray,
    offset: FloatArray,
    baseline_mean: FloatArray,
    baseline_std: FloatArray,
    epsilon: float,
) -> FloatArray:
    result = np.empty((len(onset), power.shape[0]), dtype=float)
    for trial_index, (start, stop) in enumerate(zip(onset, offset, strict=True)):
        mask = (sample_times >= start) & (sample_times <= stop)
        if not np.any(mask):
            midpoint = np.asarray([(start + stop) / 2.0])
            values = np.vstack([np.interp(midpoint, sample_times, row) for row in power])[:, 0]
        else:
            values = np.mean(power[:, mask], axis=1)
        result[trial_index] = (values - baseline_mean[trial_index]) / (
            baseline_std[trial_index] + epsilon
        )
    return result


def extract_three_band_power_epochs(
    raw: Any,
    events: pd.DataFrame,
    *,
    bands: Mapping[str, tuple[float, float]] = DEFAULT_BANDS,
    onset_window: tuple[float, float] = ONSET_WINDOW,
    offset_window: tuple[float, float] | None = OFFSET_WINDOW,
    baseline_window: tuple[float, float] = BASELINE_WINDOW,
    picks: Any = "data",
    epsilon: float = 1e-6,
    filter_kwargs: Mapping[str, Any] | None = None,
) -> BandPowerEpochs:
    """Filter continuous ECoG, compute Hilbert power, epoch, and baseline-Z it.

    The filter is MNE's zero-phase FIR filter and power is the square of the
    Hilbert envelope. Trials that cannot provide every requested fixed epoch or
    their complete speech interval are removed before processing.
    """

    required_columns = {"onset", "offset", "duration"}
    if not required_columns.issubset(events.columns):
        raise ValueError(f"Events must contain columns {sorted(required_columns)}.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    sfreq = float(raw.info["sfreq"])
    effective_bands = resolve_frequency_bands(sfreq, bands)
    onset_times = _time_grid(onset_window, sfreq)
    offset_times = None if offset_window is None else _time_grid(offset_window, sfreq)
    baseline_mask = (onset_times >= baseline_window[0]) & (onset_times <= baseline_window[1])
    if np.count_nonzero(baseline_mask) < 2:
        raise ValueError("Baseline window must contain at least two epoch samples.")

    picked = raw.copy().pick(picks)
    channel_names = tuple(picked.ch_names)
    if not channel_names:
        raise ValueError("No channels matched the requested picks.")
    bad_channels = frozenset(name for name in raw.info.get("bads", []) if name in channel_names)
    recording_start = float(picked.times[0])
    recording_stop = float(picked.times[-1])
    valid = _valid_event_mask(events, recording_start, recording_stop, onset_window, offset_window)
    valid_events = events.loc[valid].reset_index(drop=True).copy()
    if valid_events.empty:
        raise ValueError("No trials fit completely within the requested recording windows.")

    onset = valid_events["onset"].to_numpy(float)
    offset = valid_events["offset"].to_numpy(float)
    onset_epochs: dict[str, FloatArray] = {}
    offset_epochs: dict[str, FloatArray] = {}
    articulation: dict[str, FloatArray] = {}
    kwargs = dict(filter_kwargs or {})
    kwargs.pop("phase", None)
    kwargs.pop("verbose", None)

    for name, (low, high) in effective_bands.items():
        filtered = picked.copy().load_data()
        filtered.filter(l_freq=low, h_freq=high, phase="zero", verbose="ERROR", **kwargs)
        filtered.apply_hilbert(envelope=True, verbose="ERROR")
        power = np.square(filtered.get_data().astype(float, copy=False))
        sample_times = np.asarray(filtered.times, dtype=float)

        onset_unscaled = _sample_continuous(power, sample_times, onset, onset_times)
        baseline = onset_unscaled[:, :, baseline_mask]
        baseline_mean = np.mean(baseline, axis=-1)
        baseline_std = np.std(baseline, axis=-1, ddof=0)
        denominator = baseline_std[:, :, None] + epsilon
        onset_epochs[name] = (onset_unscaled - baseline_mean[:, :, None]) / denominator

        if offset_times is not None:
            offset_unscaled = _sample_continuous(power, sample_times, offset, offset_times)
            offset_epochs[name] = (offset_unscaled - baseline_mean[:, :, None]) / denominator
        articulation[name] = _articulation_means(
            power,
            sample_times,
            onset,
            offset,
            baseline_mean,
            baseline_std,
            epsilon,
        )

    return BandPowerEpochs(
        onset=onset_epochs,
        offset=offset_epochs,
        onset_times=onset_times,
        offset_times=offset_times,
        articulation_means=articulation,
        events=valid_events,
        channel_names=channel_names,
        bad_channels=bad_channels,
        sfreq=sfreq,
        effective_bands=effective_bands,
    )


def extract_audio_envelope(
    wav_path: str | Path,
    *,
    lowpass_hz: float = 5.0,
    filter_order: int = 3,
    start_time: float = 0.0,
) -> AudioEnvelope:
    """Read a WAV file and produce a rectified, low-pass audio envelope."""

    path = Path(wav_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio WAV file does not exist: {path}")
    sfreq_int, audio_raw = wavfile.read(path)
    sfreq = float(sfreq_int)
    audio = np.asarray(audio_raw)
    if audio.ndim == 2:
        audio = np.mean(audio.astype(float), axis=1)
    elif audio.ndim != 1:
        raise ValueError(f"Expected mono or multichannel WAV data, got shape {audio.shape}.")
    else:
        audio = audio.astype(float)
    if audio.size < 2:
        raise ValueError("Audio file must contain at least two samples.")

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio /= peak
    rectified = np.abs(audio)
    if not 0 < lowpass_hz < sfreq / 2.0:
        raise ValueError("Audio low-pass cutoff must lie strictly between 0 and Nyquist.")
    sos = butter(filter_order, lowpass_hz, btype="lowpass", fs=sfreq, output="sos")
    try:
        envelope = sosfiltfilt(sos, rectified)
    except ValueError as exc:
        raise ValueError("Audio is too short for zero-phase envelope filtering.") from exc
    envelope = np.maximum(envelope, 0.0)
    envelope_peak = float(np.max(envelope))
    if envelope_peak > 0:
        envelope /= envelope_peak
    times = start_time + np.arange(envelope.size, dtype=float) / sfreq
    return AudioEnvelope(values=envelope.astype(float), times=times, sfreq=sfreq)


def align_audio_to_epochs(
    envelope: AudioEnvelope,
    events: pd.DataFrame,
    epoch_times: ArrayLike,
    *,
    alignment: str = "onset",
    fill_value: float = np.nan,
) -> FloatArray:
    """Linearly interpolate an audio envelope onto an ECoG epoch time grid."""

    if alignment not in events.columns:
        raise ValueError(f"Events do not contain alignment column {alignment!r}.")
    relative_times = np.asarray(epoch_times, dtype=float)
    alignments = events[alignment].to_numpy(dtype=float)
    result = np.empty((len(events), len(relative_times)), dtype=float)
    for index, event_time in enumerate(alignments):
        targets = event_time + relative_times
        result[index] = np.interp(
            targets,
            envelope.times,
            envelope.values,
            left=fill_value,
            right=fill_value,
        )
    return result


def extract_beta_progress_epochs(
    raw: Any,
    events: pd.DataFrame,
    audio_envelope: AudioEnvelope,
    *,
    beta_band: tuple[float, float] = (13.0, 30.0),
    progress_points: int = 101,
    duration_mad_threshold: float = 2.5,
    min_trials_after_filter: int = 3,
    pre_post_window: float | None = None,
    picks: Any = "data",
    epsilon: float = 1e-6,
    filter_kwargs: Mapping[str, Any] | None = None,
) -> BetaProgressEpochs:
    """Extract beta power on a normalized speech-progress time base.

    The speech interval is sampled at equally spaced progress points between
    onset and offset. The real-time pre-onset and post-offset windows are
    determined by the minimum positive interval between consecutive retained
    trials unless ``pre_post_window`` is supplied explicitly.
    """

    if progress_points < 3:
        raise ValueError("progress_points must be at least three.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    required_columns = {"onset", "offset", "duration"}
    if not required_columns.issubset(events.columns):
        raise ValueError(f"Events must contain columns {sorted(required_columns)}.")

    sfreq = float(raw.info["sfreq"])
    beta_low, beta_high = resolve_frequency_bands(sfreq, {"beta": beta_band})["beta"]
    picked = raw.copy().pick(picks)
    channel_names = tuple(picked.ch_names)
    if not channel_names:
        raise ValueError("No channels matched the requested picks.")
    bad_channels = frozenset(name for name in raw.info.get("bads", []) if name in channel_names)

    ordered = events.sort_values("onset").reset_index(drop=True).copy()
    if "original_trial_id" not in ordered:
        ordered["original_trial_id"] = np.arange(len(ordered), dtype=int)
    duration_kept, duration_dropped = filter_duration_outlier_trials(
        ordered,
        mad_threshold=duration_mad_threshold,
        min_trials=min_trials_after_filter,
    )
    if len(duration_kept) < 2:
        raise ValueError("At least two retained trials are required for beta progress analysis.")

    window = float(pre_post_window) if pre_post_window is not None else minimum_intertrial_gap(duration_kept)
    if not np.isfinite(window) or window <= 0:
        raise ValueError("pre_post_window must be a finite positive duration.")

    recording_start = float(picked.times[0])
    recording_stop = float(picked.times[-1])
    valid = _duration_valid_event_mask(duration_kept, recording_start, recording_stop, window)
    valid_events = duration_kept.loc[valid].reset_index(drop=True).copy()
    boundary_dropped = duration_kept.loc[~valid].reset_index(drop=True).copy()
    if len(valid_events) < min_trials_after_filter:
        raise ValueError(
            "Too few trials remain after applying duration and recording-boundary filters."
        )
    if pre_post_window is None and len(valid_events) >= 2:
        window = minimum_intertrial_gap(valid_events)

    dropped_events = pd.concat(
        [
            duration_dropped.assign(drop_reason="duration_outlier"),
            boundary_dropped.assign(drop_reason="recording_boundary"),
        ],
        ignore_index=True,
    )

    progress_fraction = np.linspace(0.0, 1.0, progress_points, dtype=float)
    progress_percent = progress_fraction * 100.0
    pre_count = int(round(window * sfreq)) + 1
    pre_times = np.linspace(-window, 0.0, pre_count, dtype=float)
    post_times = np.linspace(0.0, window, pre_count, dtype=float)

    kwargs = dict(filter_kwargs or {})
    kwargs.pop("phase", None)
    kwargs.pop("verbose", None)
    filtered = picked.copy().load_data()
    filtered.filter(l_freq=beta_low, h_freq=beta_high, phase="zero", verbose="ERROR", **kwargs)
    filtered.apply_hilbert(envelope=True, verbose="ERROR")
    power = np.square(filtered.get_data().astype(float, copy=False))
    sample_times = np.asarray(filtered.times, dtype=float)

    onset = valid_events["onset"].to_numpy(float)
    offset = valid_events["offset"].to_numpy(float)
    pre_unscaled = _sample_continuous(power, sample_times, onset, pre_times)
    speech_unscaled = _sample_trial_progress(power, sample_times, onset, offset, progress_fraction)
    post_unscaled = _sample_continuous(power, sample_times, offset, post_times)

    baseline_mean = np.mean(pre_unscaled, axis=-1)
    baseline_std = np.std(pre_unscaled, axis=-1, ddof=0)
    denominator = baseline_std[:, :, None] + epsilon
    pre = (pre_unscaled - baseline_mean[:, :, None]) / denominator
    speech = (speech_unscaled - baseline_mean[:, :, None]) / denominator
    post = (post_unscaled - baseline_mean[:, :, None]) / denominator

    audio_speech = _sample_audio_trial_progress(audio_envelope, valid_events, progress_fraction)
    audio_pre = align_audio_to_epochs(audio_envelope, valid_events, pre_times, alignment="onset")
    audio_post = align_audio_to_epochs(audio_envelope, valid_events, post_times, alignment="offset")

    return BetaProgressEpochs(
        speech=speech,
        pre=pre,
        post=post,
        audio_speech=audio_speech,
        audio_pre=audio_pre,
        audio_post=audio_post,
        progress_percent=progress_percent,
        pre_times=pre_times,
        post_times=post_times,
        events=valid_events,
        dropped_events=dropped_events,
        channel_names=channel_names,
        bad_channels=bad_channels,
        sfreq=sfreq,
        beta_band=(beta_low, beta_high),
        pre_post_window=window,
    )


def _window_mask(times: FloatArray, start: float, stop: float) -> NDArray[np.bool_]:
    mask = (times >= start) & (times <= stop)
    if not np.any(mask):
        raise ValueError(f"Time grid contains no samples in [{start}, {stop}] seconds.")
    return mask


def _safe_mean(values: FloatArray) -> float:
    finite = np.isfinite(values)
    return float(np.mean(values[finite])) if finite.any() else np.nan


def _safe_std(values: FloatArray) -> float:
    finite = np.isfinite(values)
    return float(np.std(values[finite], ddof=0)) if finite.any() else np.nan


def _safe_pearson(left: FloatArray, right: FloatArray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(valid) < 3:
        return np.nan
    x, y = left[valid], right[valid]
    if np.std(x) <= np.finfo(float).eps or np.std(y) <= np.finfo(float).eps:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _linear_slope(times: FloatArray, values: FloatArray) -> float:
    valid = np.isfinite(times) & np.isfinite(values)
    if np.count_nonzero(valid) < 2 or np.ptp(times[valid]) <= 0:
        return np.nan
    return float(np.polyfit(times[valid], values[valid], deg=1)[0])


def _trapezoidal_integral(values: FloatArray, times: FloatArray) -> float:
    """Integrate compatibly across NumPy 1.x and 2.x."""

    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(values, times))
    return float(np.trapz(values, times))


def _audio_regression_slope(ecog: FloatArray, audio: FloatArray) -> float:
    valid = np.isfinite(ecog) & np.isfinite(audio)
    if np.count_nonzero(valid) < 2:
        return np.nan
    predictor = audio[valid]
    variance = float(np.var(predictor))
    if variance <= np.finfo(float).eps:
        return np.nan
    return float(np.cov(predictor, ecog[valid], ddof=0)[0, 1] / variance)


def _max_normalized_xcorr(
    ecog: FloatArray,
    audio: FloatArray,
    sfreq: float,
    max_lag_seconds: float,
) -> tuple[float, float]:
    max_lag = max(0, int(round(max_lag_seconds * sfreq)))
    best_corr, best_lag = np.nan, np.nan
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x, y = ecog[-lag:], audio[:lag]
        elif lag > 0:
            x, y = ecog[:-lag], audio[lag:]
        else:
            x, y = ecog, audio
        correlation = _safe_pearson(x, y)
        if np.isfinite(correlation) and (
            not np.isfinite(best_corr) or abs(correlation) > abs(best_corr)
        ):
            best_corr = correlation
            best_lag = lag / sfreq
    return float(best_corr), float(best_lag)


def _max_normalized_xcorr_steps(
    left: FloatArray,
    right: FloatArray,
    step_size: float,
    max_lag_steps: int,
) -> tuple[float, float]:
    """Return max normalized cross-correlation and lag in axis units."""

    best_corr, best_lag = np.nan, np.nan
    max_lag = max(0, int(max_lag_steps))
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x, y = left[-lag:], right[:lag]
        elif lag > 0:
            x, y = left[:-lag], right[lag:]
        else:
            x, y = left, right
        correlation = _safe_pearson(x, y)
        if np.isfinite(correlation) and (
            not np.isfinite(best_corr) or abs(correlation) > abs(best_corr)
        ):
            best_corr = correlation
            best_lag = lag * step_size
    return float(best_corr), float(best_lag)


def _count_curve_crossings(left: FloatArray, right: FloatArray) -> tuple[float, float]:
    valid = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(valid) < 2:
        return np.nan, np.nan
    delta = left[valid] - right[valid]
    signs = np.sign(delta)
    nonzero = signs != 0
    if np.count_nonzero(nonzero) < 2:
        return 0.0, np.nan
    indices = np.flatnonzero(valid)[nonzero]
    signs = signs[nonzero]
    changes = np.flatnonzero(signs[1:] * signs[:-1] < 0)
    if changes.size == 0:
        return 0.0, np.nan
    first_index = indices[changes[0] + 1]
    return float(changes.size), float(first_index)


def _zscore_vector(values: FloatArray) -> FloatArray:
    result = np.full_like(values, np.nan, dtype=float)
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        return result
    std = float(np.std(values[finite], ddof=0))
    if std <= np.finfo(float).eps:
        return result
    result[finite] = (values[finite] - float(np.mean(values[finite]))) / std
    return result


def _beta_progress_channel_features(
    epochs: BetaProgressEpochs,
    channel_index: int,
    *,
    max_lag_percent: float = 20.0,
) -> dict[str, float]:
    speech_trials = np.asarray(epochs.speech[:, channel_index, :], dtype=float)
    mean_curve = np.nanmean(speech_trials, axis=0)
    progress = epochs.progress_percent
    audio_mean = np.nanmean(epochs.audio_speech, axis=0)
    beta_slope = np.gradient(mean_curve, progress)
    audio_slope = np.gradient(audio_mean, progress)
    step = float(np.nanmedian(np.diff(progress)))
    max_lag_steps = int(round(max_lag_percent / step)) if step > 0 else 0
    slope_xcorr, slope_xcorr_lag = _max_normalized_xcorr_steps(
        beta_slope, audio_slope, step, max_lag_steps
    )

    product = beta_slope * audio_slope
    finite_product = np.isfinite(product)
    same_sign = (
        np.sign(beta_slope) == np.sign(audio_slope)
    ) & np.isfinite(beta_slope) & np.isfinite(audio_slope)
    audio_rising = audio_slope >= np.nanpercentile(audio_slope, 75.0)
    audio_falling = audio_slope <= np.nanpercentile(audio_slope, 25.0)

    finite_peak = np.isfinite(mean_curve)
    if finite_peak.any():
        peak_index = int(np.nanargmax(np.abs(mean_curve)))
        peak_value = float(mean_curve[peak_index])
        peak_percent = float(progress[peak_index])
    else:
        peak_value, peak_percent = np.nan, np.nan

    finite_slope_peak = np.isfinite(beta_slope)
    if finite_slope_peak.any():
        slope_peak_index = int(np.nanargmax(np.abs(beta_slope)))
        slope_peak = float(beta_slope[slope_peak_index])
        slope_peak_percent = float(progress[slope_peak_index])
    else:
        slope_peak, slope_peak_percent = np.nan, np.nan

    beta_slope_z = _zscore_vector(beta_slope)
    audio_slope_z = _zscore_vector(audio_slope)
    crossing_count, first_crossing_index = _count_curve_crossings(beta_slope_z, audio_slope_z)
    first_crossing_percent = (
        float(progress[int(first_crossing_index)])
        if np.isfinite(first_crossing_index)
        else np.nan
    )

    trialwise_corr = np.asarray(
        [
            _safe_pearson(
                np.gradient(trial_curve, progress),
                np.gradient(audio_curve, progress),
            )
            for trial_curve, audio_curve in zip(speech_trials, epochs.audio_speech, strict=True)
        ],
        dtype=float,
    )

    finite_auc = np.isfinite(mean_curve) & np.isfinite(progress)
    energy_auc = (
        _trapezoidal_integral(mean_curve[finite_auc], progress[finite_auc])
        if np.count_nonzero(finite_auc) >= 2
        else np.nan
    )
    pre_mean = _safe_mean(np.nanmean(epochs.pre[:, channel_index, :], axis=0))
    post_mean = _safe_mean(np.nanmean(epochs.post[:, channel_index, :], axis=0))

    return {
        "energy_mean": _safe_mean(mean_curve),
        "energy_std": _safe_std(mean_curve),
        "energy_auc": energy_auc,
        "energy_peak_abs": peak_value,
        "energy_peak_percent": peak_percent,
        "energy_range": float(np.nanmax(mean_curve) - np.nanmin(mean_curve))
        if np.isfinite(mean_curve).any()
        else np.nan,
        "pre_mean": pre_mean,
        "post_mean": post_mean,
        "post_minus_pre_mean": post_mean - pre_mean,
        "beta_slope_mean": _safe_mean(beta_slope),
        "beta_slope_std": _safe_std(beta_slope),
        "beta_slope_peak_abs": slope_peak,
        "beta_slope_peak_percent": slope_peak_percent,
        "audio_slope_corr": _safe_pearson(beta_slope, audio_slope),
        "audio_slope_reg_slope": _audio_regression_slope(beta_slope, audio_slope),
        "audio_slope_xcorr": slope_xcorr,
        "audio_slope_xcorr_lag_percent": slope_xcorr_lag,
        "slope_product_mean": _safe_mean(product),
        "slope_product_abs_mean": _safe_mean(np.abs(product[finite_product])),
        "slope_same_sign_fraction": float(np.mean(same_sign))
        if np.isfinite(beta_slope).any() and np.isfinite(audio_slope).any()
        else np.nan,
        "beta_slope_when_audio_rising": _safe_mean(beta_slope[audio_rising]),
        "beta_slope_when_audio_falling": _safe_mean(beta_slope[audio_falling]),
        "slope_crossing_count": crossing_count,
        "first_slope_crossing_percent": first_crossing_percent,
        "trialwise_slope_audio_corr_mean": _safe_mean(trialwise_corr),
        "trialwise_slope_audio_corr_std": _safe_std(trialwise_corr),
        "duration_slope_audio_corr": _safe_pearson(
            epochs.events["duration"].to_numpy(float), trialwise_corr
        ),
    }


def extract_beta_progress_features(
    epochs: BetaProgressEpochs,
    *,
    max_lag_percent: float = 20.0,
    mask_bad_features: bool = True,
) -> pd.DataFrame:
    """Build electrode features from beta progress and audio-envelope slopes."""

    if max_lag_percent < 0:
        raise ValueError("max_lag_percent must be non-negative.")
    rows: list[dict[str, Any]] = []
    for channel_index, channel_name in enumerate(epochs.channel_names):
        is_bad = channel_name in epochs.bad_channels
        values = _beta_progress_channel_features(
            epochs,
            channel_index,
            max_lag_percent=max_lag_percent,
        )
        row: dict[str, Any] = {
            "channel": channel_name,
            "is_bad": is_bad,
            **{f"beta_progress__{key}": value for key, value in values.items()},
        }
        if is_bad and mask_bad_features:
            for key in tuple(row):
                if key not in {"channel", "is_bad"}:
                    row[key] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _band_channel_features(
    epochs: BandPowerEpochs,
    band: str,
    channel_index: int,
    audio_epochs: FloatArray,
    max_audio_lag: float,
) -> dict[str, float]:
    onset_trials = np.asarray(epochs.onset[band][:, channel_index], dtype=float)
    mean_curve = np.nanmean(onset_trials, axis=0)
    times = epochs.onset_times
    pre = _window_mask(times, -0.5, 0.0)
    early = _window_mask(times, 0.0, 0.5)
    late = _window_mask(times, 0.5, 1.5)
    post = _window_mask(times, 0.0, 2.0)
    pre_mean = _safe_mean(mean_curve[pre])
    early_mean = _safe_mean(mean_curve[early])
    late_mean = _safe_mean(mean_curve[late])
    post_mean = _safe_mean(mean_curve[post])
    post_curve, post_times = mean_curve[post], times[post]
    finite_peak = np.isfinite(post_curve)
    if finite_peak.any():
        peak_local = int(np.nanargmax(np.abs(post_curve)))
        peak_value = float(post_curve[peak_local])
        peak_time = float(post_times[peak_local])
    else:
        peak_value, peak_time = np.nan, np.nan

    mean_audio = np.nanmean(audio_epochs, axis=0)
    audio_post = mean_audio[post]
    xcorr, xcorr_lag = _max_normalized_xcorr(
        post_curve, audio_post, epochs.sfreq, max_audio_lag
    )
    finite_auc = np.isfinite(post_curve) & np.isfinite(post_times)
    post_auc = (
        _trapezoidal_integral(post_curve[finite_auc], post_times[finite_auc])
        if np.count_nonzero(finite_auc) >= 2
        else np.nan
    )

    features = {
        "pre_mean": pre_mean,
        "early_mean": early_mean,
        "late_mean": late_mean,
        "post_mean": post_mean,
        "early_delta": early_mean - pre_mean,
        "late_delta": late_mean - pre_mean,
        "post_delta": post_mean - pre_mean,
        "post_std": _safe_std(post_curve),
        "post_auc": post_auc,
        "onset_slope": _linear_slope(times[early], mean_curve[early]),
        "post_peak_abs": peak_value,
        "post_peak_time": peak_time,
        "audio_corr": _safe_pearson(post_curve, audio_post),
        "audio_reg_slope": _audio_regression_slope(post_curve, audio_post),
        "audio_xcorr": xcorr,
        "audio_xcorr_lag": xcorr_lag,
    }

    articulation = np.asarray(epochs.articulation_means[band][:, channel_index], dtype=float)
    features["articulation_mean"] = _safe_mean(articulation)
    features["audio_dur_ecog_cov"] = _safe_pearson(
        epochs.events["duration"].to_numpy(float), articulation
    )

    if band not in epochs.offset or epochs.offset_times is None:
        features["offset_rebound"] = np.nan
        features["offset_slope"] = np.nan
    else:
        offset_curve = np.nanmean(epochs.offset[band][:, channel_index], axis=0)
        offset_times = epochs.offset_times
        before = _window_mask(offset_times, -0.2, 0.0)
        after = _window_mask(offset_times, 0.0, 0.5)
        around = _window_mask(offset_times, -0.2, 0.2)
        features["offset_rebound"] = _safe_mean(offset_curve[after]) - _safe_mean(
            offset_curve[before]
        )
        features["offset_slope"] = _linear_slope(offset_times[around], offset_curve[around])
    return features


def extract_electrode_features(
    epochs: BandPowerEpochs,
    audio_epochs: ArrayLike,
    *,
    max_audio_lag: float = 0.5,
    mask_bad_features: bool = True,
) -> pd.DataFrame:
    """Build a 20-feature-per-band electrode table.

    The four offset/duration features are ``articulation_mean``,
    ``offset_rebound`` (post-offset mean minus pre-offset mean),
    ``offset_slope``, and ``audio_dur_ecog_cov`` (Pearson correlation between
    speech duration and per-trial articulation-period ECoG response).

    Initial bad channels remain as rows with ``is_bad=True``. By default their
    feature values are set to NaN so they cannot accidentally influence a later
    model even if a caller forgets to filter them.
    """

    audio = np.asarray(audio_epochs, dtype=float)
    expected = (len(epochs.events), len(epochs.onset_times))
    if audio.shape != expected:
        raise ValueError(f"audio_epochs has shape {audio.shape}; expected {expected}.")
    if max_audio_lag < 0:
        raise ValueError("max_audio_lag must be non-negative.")

    rows: list[dict[str, Any]] = []
    for channel_index, channel_name in enumerate(epochs.channel_names):
        is_bad = channel_name in epochs.bad_channels
        row: dict[str, Any] = {"channel": channel_name, "is_bad": is_bad}
        for band in epochs.onset:
            values = _band_channel_features(
                epochs, band, channel_index, audio, max_audio_lag
            )
            row.update({f"{band}__{name}": value for name, value in values.items()})
        if is_bad and mask_bad_features:
            for key in tuple(row):
                if key not in {"channel", "is_bad"}:
                    row[key] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_feature_table_from_files(
    fif_path: str | Path,
    event_source: str | Path | pd.DataFrame | ArrayLike,
    wav_path: str | Path,
    *,
    offset_source: str | Path | ArrayLike | None = None,
    picks: Any = "data",
    include_offset_epochs: bool = True,
) -> tuple[pd.DataFrame, BandPowerEpochs, FloatArray]:
    """Convenience end-to-end loader returning features, epochs, and audio epochs."""

    events = load_speech_events(event_source, offset_source)
    raw = load_ecog_fif(fif_path, preload=False)
    epochs = extract_three_band_power_epochs(
        raw,
        events,
        picks=picks,
        offset_window=OFFSET_WINDOW if include_offset_epochs else None,
    )
    audio_envelope = extract_audio_envelope(wav_path)
    audio_epochs = align_audio_to_epochs(
        audio_envelope, epochs.events, epochs.onset_times, alignment="onset"
    )
    features = extract_electrode_features(epochs, audio_epochs)
    return features, epochs, audio_epochs


def build_beta_progress_feature_table_from_files(
    fif_path: str | Path,
    event_source: str | Path | pd.DataFrame | ArrayLike,
    wav_path: str | Path,
    *,
    offset_source: str | Path | ArrayLike | None = None,
    picks: Any = "data",
    progress_points: int = 101,
    duration_mad_threshold: float = 2.5,
    min_trials_after_filter: int = 3,
    pre_post_window: float | None = None,
) -> tuple[pd.DataFrame, BetaProgressEpochs]:
    """Convenience loader for the beta-only normalized-progress pipeline."""

    events = load_speech_events(event_source, offset_source)
    raw = load_ecog_fif(fif_path, preload=False)
    audio_envelope = extract_audio_envelope(wav_path)
    epochs = extract_beta_progress_epochs(
        raw,
        events,
        audio_envelope,
        picks=picks,
        progress_points=progress_points,
        duration_mad_threshold=duration_mad_threshold,
        min_trials_after_filter=min_trials_after_filter,
        pre_post_window=pre_post_window,
    )
    features = extract_beta_progress_features(epochs)
    return features, epochs


__all__ = [
    "AudioEnvelope",
    "BASELINE_WINDOW",
    "BETA_ONLY_BANDS",
    "BetaProgressEpochs",
    "BandPowerEpochs",
    "DEFAULT_BANDS",
    "OFFSET_WINDOW",
    "ONSET_WINDOW",
    "align_audio_to_epochs",
    "build_beta_progress_feature_table_from_files",
    "build_feature_table_from_files",
    "extract_audio_envelope",
    "extract_beta_progress_epochs",
    "extract_beta_progress_features",
    "extract_electrode_features",
    "extract_three_band_power_epochs",
    "filter_duration_outlier_trials",
    "load_ecog_fif",
    "load_speech_events",
    "minimum_intertrial_gap",
    "resolve_frequency_bands",
]
