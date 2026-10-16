"""Patient-level baseline references with explicit segment and channel checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from signal_processing import baseline_statistics, finite_mean, normalize_power


@dataclass(frozen=True)
class FileBaselineStatistics:
    mean: np.ndarray
    std: np.ndarray
    floor: np.ndarray
    invalid: np.ndarray
    reference: pd.DataFrame
    segments: pd.DataFrame


def _file_segments(raw, segments_csv):
    sfreq = float(raw.info["sfreq"])
    duration = raw.n_times / sfreq
    if segments_csv is not None:
        table = pd.read_csv(segments_csv)
        columns = ["output_start_sec", "output_stop_sec"]
        if not set(columns).issubset(table):
            raise ValueError(f"Baseline segment CSV must contain {columns}.")
        intervals = table[columns].to_numpy(float)
        source = "csv"
    else:
        quiet = [(onset - raw.first_time, onset - raw.first_time + length)
                 for onset, length, desc in zip(raw.annotations.onset, raw.annotations.duration,
                                               raw.annotations.description, strict=True)
                 if str(desc).lower().startswith("quiet")]
        intervals = np.asarray(quiet or [(0.0, duration)], dtype=float)
        source = "quiet_annotations" if quiet else "whole_recording"
    if (intervals.ndim != 2 or intervals.shape[1] != 2 or not len(intervals)
            or not np.isfinite(intervals).all() or np.any(intervals[:, 1] <= intervals[:, 0])):
        raise ValueError("Baseline segments must be finite positive intervals.")
    if np.any(intervals[:, 0] < -0.5 / sfreq) or np.any(intervals[:, 1] > duration + 0.5 / sfreq):
        raise ValueError("Baseline segment lies outside the FIF recording.")
    intervals = intervals[np.argsort(intervals[:, 0])]
    if np.any(intervals[1:, 0] < intervals[:-1, 1] - 0.5 / sfreq):
        raise ValueError("Baseline segments overlap.")
    # Bad annotations remove observations; even zero-length joins split filters.
    for onset, length, desc in zip(raw.annotations.onset, raw.annotations.duration,
                                   raw.annotations.description, strict=True):
        if not str(desc).lower().startswith("bad") and "boundary" not in str(desc).lower():
            continue
        start, stop = onset - raw.first_time, onset - raw.first_time + length
        split = []
        for left, right in intervals:
            if stop <= left or start >= right:
                split.append((left, right))
            else:
                if left < start:
                    split.append((left, start))
                if stop < right:
                    split.append((stop, right))
        intervals = np.asarray(split, dtype=float).reshape(-1, 2)
    return intervals, source


def _check_task_metadata(raw, channels, task_info):
    if task_info is None:
        return
    if not np.isclose(raw.info["sfreq"], task_info["sfreq"]):
        raise ValueError("Baseline and task sampling rates differ; harmonize preprocessing first.")
    for key in ("highpass", "lowpass"):
        reference_value, task_value = raw.info[key], task_info[key]
        if (reference_value is None or task_value is None
                or not np.isfinite([reference_value, task_value]).all()
                or not np.isclose(reference_value, task_value, rtol=1e-7, atol=1e-9)):
            raise ValueError(f"Baseline and task {key} preprocessing differ or are unknown; harmonize first.")
    if raw.info["custom_ref_applied"] != task_info["custom_ref_applied"]:
        raise ValueError("Baseline and task reference flags differ; use the same reference preprocessing.")
    task_channels = {ch["ch_name"]: ch for ch in task_info["chs"]}
    reference_channels = {ch["ch_name"]: ch for ch in raw.info["chs"]}
    for name in channels:
        for key in ("kind", "unit", "unit_mul"):
            if reference_channels[name][key] != task_channels[name][key]:
                raise ValueError(f"Baseline/task channel type or units differ for {name} ({key}).")


def load_file_baseline_statistics(
    fif_path: str | Path,
    channels: Sequence[str],
    *,
    beta_band: tuple[float, float] = (13.0, 30.0),
    segments_csv: str | Path | None = None,
    auto_discover_segments: bool = True,
    edge_seconds: float = 0.3,
    relative_floor: float = 1e-3,
    absolute_floor: float | None = None,
    task_info: Any = None,
    filter_kwargs: Mapping[str, Any] | None = None,
) -> FileBaselineStatistics:
    """Pool independently filtered, trimmed power samples (not segment stds).

    FIR trimming includes half the filter support plus ``edge_seconds`` for
    Hilbert edges. This reduces, but cannot guarantee elimination of, analytic
    signal edge effects. At least one second of reference samples is required.
    Matching reference flags does not establish identical reference electrodes.
    Disabling companion CSV discovery retains quiet, BAD and boundary annotations.
    """
    import mne

    if not np.isfinite(edge_seconds) or edge_seconds < 0:
        raise ValueError("baseline_edge_seconds must be finite and non-negative.")
    path = Path(fif_path)
    raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
    if len(set(channels)) != len(channels) or not channels:
        raise ValueError("Baseline channel names must be nonempty and unique.")
    missing = set(channels) - set(raw.ch_names)
    if missing:
        raise ValueError(f"Baseline file missing task channels: {sorted(missing)}")
    _check_task_metadata(raw, channels, task_info)
    if segments_csv is None and auto_discover_segments:
        companion = path.with_name(path.stem + "__segments.csv")
        segments_csv = companion if companion.is_file() else None
    intervals, source = _file_segments(raw, segments_csv)
    raw.pick(list(channels))
    sfreq = float(raw.info["sfreq"])
    low, high = beta_band
    if not 0 < low < high < sfreq / 2:
        raise ValueError("Baseline beta band must lie below its Nyquist frequency.")
    kwargs = dict(filter_kwargs or {})
    kwargs.pop("phase", None)
    kwargs.pop("verbose", None)
    filter_keys = {"filter_length", "l_trans_bandwidth", "h_trans_bandwidth", "method",
                   "iir_params", "fir_window", "fir_design"}
    kernel = mne.filter.create_filter(None, sfreq, low, high, phase="zero", verbose="ERROR",
                                     **{k: v for k, v in kwargs.items() if k in filter_keys})
    half_support = (len(kernel) - 1) / (2 * sfreq) if isinstance(kernel, np.ndarray) else 0.0
    trim_count = int(np.ceil((half_support + edge_seconds) * sfreq))
    reports, powers = [], []
    for index, (start, stop) in enumerate(intervals):
        first, last = int(round(start * sfreq)), min(raw.n_times, int(round(stop * sfreq)))
        report = dict(segment_index=index + 1, start_seconds=start, stop_seconds=stop,
                      edge_trim_seconds=trim_count / sfreq, boundary_source=source,
                      used=False, usable_samples=0, usable_seconds=0.0,
                      reason="too_short_after_edge_trim")
        if last - first - 2 * trim_count >= 2:
            segment = mne.io.RawArray(raw.get_data(start=first, stop=last), raw.info.copy(), verbose="ERROR")
            segment.filter(low, high, phase="zero", verbose="ERROR", **kwargs)
            segment.apply_hilbert(envelope=True, verbose="ERROR")
            power = np.square(segment.get_data())[:, trim_count:(last - first - trim_count)]
            powers.append(power)
            report.update(used=True, usable_samples=power.shape[-1],
                          usable_seconds=power.shape[-1] / sfreq, reason="")
        reports.append(report)
    if not powers or sum(p.shape[-1] for p in powers) < sfreq:
        raise ValueError("Insufficient usable baseline: need >=1 second after segment edge trimming.")
    pooled = np.concatenate(powers, axis=-1)[None]
    mean, std, floor, invalid = baseline_statistics(pooled, relative_floor=relative_floor,
                                                   absolute_floor=absolute_floor)
    invalid |= np.isin(channels, raw.info["bads"])[None]
    normalized = normalize_power(pooled, mean, std, invalid)[0]
    reference = pd.DataFrame({
        "channel": channels, "baseline_mode": "file", "baseline_file": str(path),
        "baseline_mean": mean[0], "baseline_std": std[0], "denominator_floor": floor[0],
        "baseline_samples": pooled.shape[-1], "baseline_seconds": pooled.shape[-1] / sfreq,
        "normalized_mean": finite_mean(normalized, axis=-1),
        "normalized_std": [float(row.std()) if np.isfinite(row).all() else np.nan for row in normalized],
        "valid": ~invalid[0],
        "invalid_reason": np.where(np.isin(channels, raw.info["bads"]), "baseline_file_bad_channel",
                                   np.where(invalid[0], "low_variance_or_nonfinite", "")),
    })
    return FileBaselineStatistics(mean[0], std[0], floor[0], invalid[0], reference, pd.DataFrame(reports))
