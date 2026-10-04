"""Paired, per-band electrode means and descriptive power/audio diagnostics."""
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from signal_processing import finite_mean, gaussian_smooth_nan, curve_slope
from visualization import _concat_beta_progress_display, cluster_color, plot_pca_clusters


def summarize_relationship(epochs, *, sigma_seconds=.10, slope_method="savgol"):
    progress = np.asarray(epochs.progress_percent)
    step = float(np.median(np.diff(progress)))
    audio = np.asarray(epochs.audio_speech, dtype=float)
    scale = np.max(np.abs(audio[np.isfinite(audio)])) if np.isfinite(audio).any() else 0
    if scale <= 0:
        raise ValueError("Audio envelope needs nonzero finite speech samples.")
    audio = audio / scale
    arrays = {key: [] for key in ("power", "audio", "power_slope", "audio_slope", "interaction")}
    counts = np.zeros(epochs.speech.shape[1], dtype=int)
    for power, envelope, duration in zip(epochs.speech, audio, epochs.events.duration, strict=True):
        valid = np.isfinite(power).all(axis=-1) & np.isfinite(envelope).all()
        seconds_per_unit = float(duration) / 100
        rate = 1 / (step * seconds_per_unit)
        power = gaussian_smooth_nan(power, rate, sigma_seconds=sigma_seconds)
        envelope = gaussian_smooth_nan(envelope, rate, sigma_seconds=sigma_seconds)
        options = dict(sigma_seconds=sigma_seconds, seconds_per_axis_unit=seconds_per_unit, method=slope_method)
        p_slope = curve_slope(power, progress, **options)
        a_slope = curve_slope(envelope, progress, **options)
        counts += valid
        for key, values in (("power", power), ("audio", envelope[None]), ("power_slope", p_slope),
                            ("audio_slope", a_slope[None]), ("interaction", p_slope * a_slope[None])):
            arrays[key].append(np.where(valid[:, None], values, np.nan))
    result = {key: finite_mean(values) for key, values in arrays.items()}
    result["valid_trial_count"] = counts
    return result


def _mean_sem(curves):
    count = np.isfinite(curves).sum(axis=0)
    mean = finite_mean(curves)
    centered = np.where(np.isfinite(curves), curves - mean, 0)
    variance = np.divide((centered ** 2).sum(axis=0), count - 1,
                         out=np.full_like(mean, np.nan), where=count > 1)
    sem = np.sqrt(variance / np.maximum(count, 1))
    return mean, sem, count


def save_round_plots(epochs_by_band, result, output_dir, *, sigma_seconds=.10,
                     display_sfreq=100, slope_method="savgol"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels = np.unique(result.labels[result.clean_mask])
    bands = tuple(epochs_by_band)
    if len(bands) != 4 or not len(labels):
        raise ValueError("Four bands and at least one included cluster are required for plotting.")
    fig, axes = plt.subplots(len(labels), 4, figsize=(18, 2.9 * len(labels)), squeeze=False,
                             constrained_layout=True)
    records = []
    for column, (band, epochs) in enumerate(epochs_by_band.items()):
        if epochs.channel_names != result.channels:
            raise ValueError(f"{band} curve/electrode order does not align with labels.")
        times, power, audio, duration = _concat_beta_progress_display(epochs, display_sfreq=display_sfreq)
        valid = np.isfinite(epochs.speech).all(axis=-1) & np.isfinite(epochs.audio_speech).all(axis=-1)[:, None]
        power = finite_mean(np.where(valid[..., None], power, np.nan))
        audio = finite_mean(np.where(valid[..., None], audio[:, None, :], np.nan))
        rate = 1 / np.diff(times)[0]
        power = gaussian_smooth_nan(power, rate, sigma_seconds=sigma_seconds)
        audio = gaussian_smooth_nan(audio, rate, sigma_seconds=sigma_seconds)
        peak = float(np.nanmax(epochs.audio_speech))
        if not np.isfinite(peak) or peak <= 0:
            raise ValueError("Audio envelope needs a positive peak for display.")
        audio /= peak
        low, high = epochs.beta_band
        for row, label in enumerate(labels):
            mask = result.labels == label
            mean, sem, count = _mean_sem(power[mask])
            envelope = finite_mean(audio[mask])
            ax = axes[row, column]
            color = cluster_color(int(label))
            ax.plot(times * 1000, mean, color=color, linewidth=1.7)
            ax.fill_between(times * 1000, mean - sem, mean + sem, color=color, alpha=.18)
            right = ax.twinx()
            right.plot(times * 1000, envelope, color="black", linestyle="--", linewidth=1, alpha=.7)
            right.set_ylim(0, 1.05)
            if column == 3:
                right.set_ylabel("Audio / speech peak", fontsize=8)
            else:
                right.set_yticklabels([])
            ax.axhline(0, color="gray", linewidth=.6)
            ax.axvline(0, color="gray", linestyle="--", linewidth=.7)
            ax.axvline(duration * 1000, color="gray", linestyle="--", linewidth=.7)
            ax.set_title(f"Cluster {label} (n={mask.sum()}) | {band} {low:g}-{high:g} Hz", fontsize=10)
            ax.set_ylabel("Power Z", fontsize=9)
            if row == len(labels) - 1:
                ax.set_xlabel("Duration-normalized time (ms)", fontsize=9)
            for i, point in enumerate(times):
                records.append(dict(band=band, label=int(label), display_time_seconds=float(point),
                                    mean_power_z=float(mean[i]), electrode_sem=float(sem[i]),
                                    n_electrodes=int(count[i]), mean_audio=float(envelope[i])))
        _save_relationship_plot(epochs, band, result, output, sigma_seconds, slope_method)
    fig.savefig(output / "fourband_cluster_grid.png", dpi=150)
    plt.close(fig)
    summary = pd.DataFrame(records)
    summary.to_csv(output / "fourband_curve_summary.csv", index=False, encoding="utf-8-sig")
    fig, _ = plot_pca_clusters(result.pca_scores, result.labels, channel_names=result.channels, annotate=False)
    fig.savefig(output / "pca_clusters.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return summary


def _save_relationship_plot(epochs, band, result, output, sigma_seconds, slope_method):
    cache = summarize_relationship(epochs, sigma_seconds=sigma_seconds, slope_method=slope_method)
    labels = np.unique(result.labels[result.clean_mask])
    progress = epochs.progress_percent
    fig, axes = plt.subplots(len(labels), 3, figsize=(15, 3 * len(labels)), squeeze=False,
                             constrained_layout=True)
    records = []
    for row, label in enumerate(labels):
        mask = result.labels == label
        means = {key: finite_mean(cache[key][mask]) for key in ("power", "audio", "power_slope", "audio_slope", "interaction")}
        color = cluster_color(int(label))
        for column, (left_key, right_key) in enumerate((("power", "audio"), ("power_slope", "audio_slope"))):
            ax = axes[row, column]
            ax.plot(progress, means[left_key], color=color, label=left_key.replace("_", " "))
            right = ax.twinx()
            right.plot(progress, means[right_key], color="black", linestyle="--", label=right_key.replace("_", " "))
            ax.set_ylabel("Power Z" if column == 0 else "Power Z / percentage point", fontsize=8)
            right.set_ylabel("Audio / speech peak" if column == 0 else "Audio / percentage point", fontsize=8)
            ax.legend(loc="upper left", fontsize=7)
            right.legend(loc="upper right", fontsize=7)
        axes[row, 2].plot(progress, means["interaction"], color=color)
        axes[row, 2].set_ylabel("Mean within-trial slope product", fontsize=8)
        for column, title in enumerate(("Power / audio", "Power slope / audio slope", "Paired slope product")):
            ax = axes[row, column]
            ax.set_title(f"{band} | Cluster {label} (n={mask.sum()}) | {title}", fontsize=9)
            ax.set(xlabel="Speech progress (%)", xlim=(0, 100))
            ax.axhline(0, color="gray", linewidth=.6)
        for i, point in enumerate(progress):
            records.append(dict(label=int(label), progress_percent=float(point), n_electrodes=int(mask.sum()),
                                **{key: float(value[i]) for key, value in means.items()}))
    fig.savefig(output / f"{band}_audio_dynamics.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(records).to_csv(output / f"{band}_audio_dynamics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"channel": result.channels, "label": result.labels,
                  "paired_trial_count": cache["valid_trial_count"]}).to_csv(
                      output / f"{band}_paired_trial_counts.csv", index=False, encoding="utf-8-sig")
