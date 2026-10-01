"""Trial-level beta morphology features for channel x trial typing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

if TYPE_CHECKING:
    from audio_cross_3band_utils import BetaProgressEpochs


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TrialTypeFeatureConfig:
    """Window and smoothing settings for Type 1/Type 2 morphology features."""

    pre_tail_fraction: tuple[float, float] = (0.70, 1.00)
    early_speech_percent: tuple[float, float] = (0.0, 20.0)
    rebound_percent: tuple[float, float] = (30.0, 75.0)
    end_speech_percent: tuple[float, float] = (85.0, 100.0)
    post_early_fraction: tuple[float, float] = (0.0, 0.30)
    post_late_fraction: tuple[float, float] = (0.55, 1.00)
    low_power_threshold_z: float = -0.35
    smooth_sigma_points: float = 2.0


def _fraction_mask(length: int, fraction_window: tuple[float, float]) -> NDArray[np.bool_]:
    start, stop = fraction_window
    if not 0.0 <= start < stop <= 1.0:
        raise ValueError(f"Invalid fraction window: {fraction_window}")
    positions = np.linspace(0.0, 1.0, length, dtype=float)
    mask = (positions >= start) & (positions <= stop)
    if not np.any(mask):
        raise ValueError(f"Fraction window contains no samples: {fraction_window}")
    return mask


def _percent_mask(progress_percent: FloatArray, percent_window: tuple[float, float]) -> NDArray[np.bool_]:
    start, stop = percent_window
    if not 0.0 <= start < stop <= 100.0:
        raise ValueError(f"Invalid percent window: {percent_window}")
    mask = (progress_percent >= start) & (progress_percent <= stop)
    if not np.any(mask):
        raise ValueError(f"Percent window contains no samples: {percent_window}")
    return mask


def _smooth(values: FloatArray, sigma_points: float) -> FloatArray:
    data = np.asarray(values, dtype=float)
    if sigma_points <= 0:
        return data.copy()
    finite = np.isfinite(data)
    if not finite.any():
        return np.full_like(data, np.nan, dtype=float)
    filled = np.where(finite, data, 0.0)
    weights = finite.astype(float)
    numerator = gaussian_filter1d(filled, sigma_points, mode="nearest")
    denominator = gaussian_filter1d(weights, sigma_points, mode="nearest")
    smoothed = numerator / np.maximum(denominator, np.finfo(float).eps)
    smoothed[denominator <= np.finfo(float).eps] = np.nan
    return smoothed


def _safe_mean(values: FloatArray) -> float:
    finite = np.isfinite(values)
    return float(np.mean(values[finite])) if finite.any() else np.nan


def _safe_min(values: FloatArray) -> float:
    return float(np.nanmin(values)) if np.isfinite(values).any() else np.nan


def _safe_max(values: FloatArray) -> float:
    return float(np.nanmax(values)) if np.isfinite(values).any() else np.nan


def _safe_fraction(mask_values: NDArray[np.bool_]) -> float:
    return float(np.mean(mask_values)) if mask_values.size else np.nan


def _safe_pearson(left: FloatArray, right: FloatArray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(valid) < 3:
        return np.nan
    x = left[valid]
    y = right[valid]
    if np.std(x) <= np.finfo(float).eps or np.std(y) <= np.finfo(float).eps:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _linear_slope(axis: FloatArray, values: FloatArray) -> float:
    valid = np.isfinite(axis) & np.isfinite(values)
    if np.count_nonzero(valid) < 2 or np.ptp(axis[valid]) <= 0:
        return np.nan
    return float(np.polyfit(axis[valid], values[valid], deg=1)[0])


def _event_value(events: pd.DataFrame, row_index: int, column: str, default: Any = np.nan) -> Any:
    if column not in events:
        return default
    return events.iloc[row_index][column]


def extract_trial_type_features(
    epochs: "BetaProgressEpochs",
    *,
    include_bad: bool = False,
    config: TrialTypeFeatureConfig | None = None,
) -> pd.DataFrame:
    """Extract one Type 1/Type 2 morphology feature row per channel and trial.

    Parameters
    ----------
    epochs:
        Beta-only progress epochs from the main pipeline. Its events have
        already been filtered to valid trials.
    include_bad:
        Whether to keep channels marked in ``epochs.bad_channels``. By default
        bad channels are excluded from trial-wise type analysis.
    config:
        Feature extraction windows and smoothing settings.
    """

    cfg = config or TrialTypeFeatureConfig()
    progress = np.asarray(epochs.progress_percent, dtype=float)
    early = _percent_mask(progress, cfg.early_speech_percent)
    rebound = _percent_mask(progress, cfg.rebound_percent)
    end = _percent_mask(progress, cfg.end_speech_percent)
    pre_tail = _fraction_mask(epochs.pre.shape[-1], cfg.pre_tail_fraction)
    post_early = _fraction_mask(epochs.post.shape[-1], cfg.post_early_fraction)
    post_late = _fraction_mask(epochs.post.shape[-1], cfg.post_late_fraction)
    audio_speech = np.nanmean(np.asarray(epochs.audio_speech, dtype=float), axis=0)
    audio_speech = _smooth(audio_speech, cfg.smooth_sigma_points)
    audio_slope = np.gradient(audio_speech, progress)

    rows: list[dict[str, Any]] = []
    for channel_index, channel_name in enumerate(epochs.channel_names):
        is_bad = channel_name in epochs.bad_channels
        if is_bad and not include_bad:
            continue
        for trial_index in range(len(epochs.events)):
            pre_curve = _smooth(
                np.asarray(epochs.pre[trial_index, channel_index], dtype=float),
                cfg.smooth_sigma_points,
            )
            speech_curve = _smooth(
                np.asarray(epochs.speech[trial_index, channel_index], dtype=float),
                cfg.smooth_sigma_points,
            )
            post_curve = _smooth(
                np.asarray(epochs.post[trial_index, channel_index], dtype=float),
                cfg.smooth_sigma_points,
            )

            pre_tail_mean = _safe_mean(pre_curve[pre_tail])
            early_mean = _safe_mean(speech_curve[early])
            early_min = _safe_min(speech_curve[early])
            rebound_mean = _safe_mean(speech_curve[rebound])
            rebound_peak = _safe_max(speech_curve[rebound])
            end_mean = _safe_mean(speech_curve[end])
            post_early_mean = _safe_mean(post_curve[post_early])
            post_late_mean = _safe_mean(post_curve[post_late])
            speech_mean = _safe_mean(speech_curve)
            low_fraction = _safe_fraction(speech_curve < cfg.low_power_threshold_z)
            speech_slope = np.gradient(speech_curve, progress)
            rebound_slope = _linear_slope(progress[rebound], speech_curve[rebound])
            end_slope = _linear_slope(progress[end], speech_curve[end])

            finite_rebound = np.isfinite(speech_curve[rebound])
            if finite_rebound.any():
                rebound_indices = np.flatnonzero(rebound)
                peak_local = int(np.nanargmax(speech_curve[rebound]))
                rebound_peak_percent = float(progress[rebound_indices[peak_local]])
            else:
                rebound_peak_percent = np.nan

            onset_drop = pre_tail_mean - early_mean
            rebound_amplitude = rebound_peak - early_min
            sustained_low = -speech_mean
            offset_drop = rebound_peak - end_mean
            post_recovery = post_late_mean - end_mean
            post_recovery_slope = _linear_slope(
                epochs.post_times[post_late],
                post_curve[post_late],
            )

            row = {
                "channel": channel_name,
                "is_bad": is_bad,
                "trial_index": trial_index,
                "trial_id": _event_value(epochs.events, trial_index, "trial_id", trial_index),
                "original_trial_id": _event_value(
                    epochs.events, trial_index, "original_trial_id", trial_index
                ),
                "onset": float(_event_value(epochs.events, trial_index, "onset")),
                "offset": float(_event_value(epochs.events, trial_index, "offset")),
                "duration": float(_event_value(epochs.events, trial_index, "duration")),
                "block_fraction": trial_index / max(len(epochs.events) - 1, 1),
                "pre_tail_mean": pre_tail_mean,
                "early_speech_mean": early_mean,
                "early_speech_min": early_min,
                "rebound_mean": rebound_mean,
                "rebound_peak": rebound_peak,
                "rebound_peak_percent": rebound_peak_percent,
                "end_speech_mean": end_mean,
                "post_early_mean": post_early_mean,
                "post_late_mean": post_late_mean,
                "speech_mean": speech_mean,
                "low_power_fraction": low_fraction,
                "onset_drop": onset_drop,
                "mid_speech_rebound": rebound_amplitude,
                "speech_low_maintenance": sustained_low,
                "offset_drop": offset_drop,
                "post_offset_recovery": post_recovery,
                "rebound_slope": rebound_slope,
                "end_slope": end_slope,
                "post_recovery_slope": post_recovery_slope,
                "beta_slope_audio_slope_corr": _safe_pearson(speech_slope, audio_slope),
                "slope_product_mean": _safe_mean(speech_slope * audio_slope),
                "finite_fraction": float(
                    np.mean(
                        np.isfinite(pre_curve).tolist()
                        + np.isfinite(speech_curve).tolist()
                        + np.isfinite(post_curve).tolist()
                    )
                ),
            }
            rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["TrialTypeFeatureConfig", "extract_trial_type_features"]
