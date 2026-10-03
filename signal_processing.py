"""Scale-aware baseline statistics and physical-time signal operations."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample_poly, savgol_filter


def finite_mean(values: ArrayLike, axis: int = 0) -> NDArray[np.float64]:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    total = np.sum(np.where(valid, values, 0.0), axis=axis)
    count = valid.sum(axis=axis)
    result = np.full_like(total, np.nan, dtype=float)
    np.divide(total, count, out=result, where=count > 0)
    return result


def baseline_statistics(
    baseline: ArrayLike,
    *,
    relative_floor: float = 1e-3,
    absolute_floor: float | None = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Return mean, std, scale-aware floor and invalid trial/channel mask.

    A floor is a quality threshold, not an additive constant. Baselines below
    it are excluded instead of manufacturing a finite normalized response.
    """
    baseline = np.asarray(baseline, dtype=float)
    if baseline.ndim != 3 or baseline.shape[-1] < 2:
        raise ValueError("baseline must have shape (trials, channels, >=2 samples).")
    if not np.isfinite(relative_floor) or not 0 < relative_floor < 1:
        raise ValueError("relative_floor must lie strictly between zero and one.")
    if absolute_floor is not None and (not np.isfinite(absolute_floor) or absolute_floor <= 0):
        raise ValueError("absolute_floor must be finite and positive or None.")
    mean = np.mean(baseline, axis=-1)
    std = np.std(baseline, axis=-1)
    scale = np.ma.median(np.ma.masked_invalid(np.abs(mean)), axis=0).filled(0.0)
    numerical_floor = np.maximum(scale * 1e-12, np.finfo(float).tiny)
    usable_std = np.ma.masked_where(~np.isfinite(std) | (std <= numerical_floor), std)
    typical_std = np.ma.median(usable_std, axis=0).filled(0.0)
    floor = np.maximum(typical_std * relative_floor, numerical_floor)
    if absolute_floor is not None:
        floor = np.maximum(floor, absolute_floor)
    floor = np.broadcast_to(floor, std.shape).copy()
    invalid = ~np.isfinite(mean) | ~np.isfinite(std) | (std <= floor)
    return mean, std, floor, invalid


def normalize_power(values: ArrayLike, mean: ArrayLike, std: ArrayLike, invalid: ArrayLike) -> NDArray:
    values = np.asarray(values, dtype=float)
    denominator = np.asarray(std)[..., None]
    valid = ~np.asarray(invalid, dtype=bool)[..., None]
    result = np.full_like(values, np.nan)
    np.divide(values - np.asarray(mean)[..., None], denominator, out=result, where=valid)
    return result


def gaussian_smooth_nan(values: ArrayLike, sfreq: float, *, sigma_seconds: float = 0.06, axis: int = -1) -> NDArray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(sfreq) or sfreq <= 0 or not np.isfinite(sigma_seconds) or sigma_seconds < 0:
        raise ValueError("sfreq must be positive and sigma_seconds non-negative.")
    if sigma_seconds == 0:
        return values.copy()
    valid = np.isfinite(values)
    numerator = gaussian_filter1d(np.where(valid, values, 0), sigma_seconds * sfreq, axis=axis, mode="reflect")
    denominator = gaussian_filter1d(valid.astype(float), sigma_seconds * sfreq, axis=axis, mode="reflect")
    result = np.full_like(numerator, np.nan)
    np.divide(numerator, denominator, out=result, where=denominator > 1e-12)
    result[~valid] = np.nan
    return result


def resample_on_axis(values: ArrayLike, source_axis: ArrayLike, target_axis: ArrayLike) -> NDArray:
    """Interpolate uniform segments, using polyphase anti-aliasing on reduction."""
    values = np.asarray(values, dtype=float)
    source = np.asarray(source_axis, dtype=float)
    target = np.asarray(target_axis, dtype=float)
    if source.ndim != 1 or source.size < 2 or values.shape[-1] != source.size:
        raise ValueError("Source axis must match the last data dimension and contain >=2 samples.")
    if not np.isfinite(source).all() or np.any(np.diff(source) <= 0):
        raise ValueError("Source axis must be finite and strictly increasing.")
    if target.ndim != 1 or not np.isfinite(target).all() or np.any(np.diff(target) <= 0):
        raise ValueError("Target axis must be finite and strictly increasing.")
    if target.size == 0:
        return np.empty(values.shape[:-1] + (0,), dtype=float)
    target_step = float(np.median(np.diff(target))) if target.size > 1 else 0.0

    def sample_finite(data, axis, query):
        source_step = float(np.median(np.diff(axis)))
        filtered, interpolation_axis = data, axis
        if target_step > source_step * 1.01 and np.allclose(np.diff(axis), source_step, rtol=1e-5, atol=1e-12):
            ratio = Fraction(source_step / target_step).limit_denominator(1000)
            # A trailing extension gives the last supported target two bracketing
            # filtered samples even when downsampled grids have different phases.
            padding = int(np.ceil(target_step / source_step)) + 2
            tail = data[..., -1:] + (data[..., -1:] - data[..., -2:-1]) * np.arange(1, padding + 1)
            extended = np.concatenate([data, tail], axis=-1)
            filtered = resample_poly(extended, ratio.numerator, ratio.denominator, axis=-1, padtype="line")
            interpolation_axis = axis[0] + np.arange(filtered.shape[-1]) * source_step * ratio.denominator / ratio.numerator
        result = np.full(data.shape[:-1] + (len(query),), np.nan)
        supported = (query >= axis[0]) & (query <= axis[-1])
        for row, destination in zip(filtered.reshape(-1, filtered.shape[-1]), result.reshape(-1, len(query)), strict=True):
            destination[supported] = np.interp(query[supported], interpolation_axis, row, left=np.nan, right=np.nan)
        return result

    if np.isfinite(values).all():
        return sample_finite(values, source, target)
    output = np.full(values.shape[:-1] + (len(target),), np.nan)
    for row, destination in zip(values.reshape(-1, values.shape[-1]), output.reshape(-1, len(target)), strict=True):
        finite = np.isfinite(row)
        edges = np.flatnonzero(np.diff(np.r_[False, finite, False]))
        for start, stop in zip(edges[::2], edges[1::2], strict=True):
            if stop - start == 1:
                destination[np.isclose(target, source[start], rtol=0, atol=1e-12)] = row[start]
                continue
            supported = (target >= source[start]) & (target <= source[stop - 1])
            if supported.any():
                destination[supported] = sample_finite(row[start:stop], source[start:stop], target[supported])
    return output


def curve_slope(values: ArrayLike, axis_values: ArrayLike, *, sigma_seconds: float = 0.06, seconds_per_axis_unit: float = 1.0, method: str = "savgol") -> NDArray:
    """Estimate a derivative in axis units, with a physical-time local window."""
    values = np.asarray(values, dtype=float)
    axis_values = np.asarray(axis_values, dtype=float)
    if method not in {"gradient", "savgol"}:
        raise ValueError("slope method must be 'gradient' or 'savgol'.")
    step = float(np.median(np.diff(axis_values)))
    if step <= 0 or seconds_per_axis_unit <= 0:
        raise ValueError("Axis and physical-time scale must be positive.")
    if method == "gradient" or sigma_seconds == 0:
        return np.gradient(values, axis_values, axis=-1)
    window = max(5, int(round(4 * sigma_seconds / (step * seconds_per_axis_unit))) | 1)
    window = min(window, values.shape[-1] if values.shape[-1] % 2 else values.shape[-1] - 1)
    if window < 5:
        return np.gradient(values, axis_values, axis=-1)
    if not np.isfinite(values).all():
        result = np.full_like(values, np.nan)
        for row, destination in zip(values.reshape(-1, values.shape[-1]), result.reshape(-1, values.shape[-1]), strict=True):
            finite = np.isfinite(row)
            if finite.sum() >= window:
                filled = np.interp(axis_values, axis_values[finite], row[finite])
                derivative = savgol_filter(filled, window_length=window, polyorder=2, deriv=1, delta=step, mode="interp")
                destination[finite] = derivative[finite]
        return result
    return savgol_filter(values, window_length=window, polyorder=2, deriv=1, delta=step, axis=-1, mode="interp")
