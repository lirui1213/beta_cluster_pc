"""Paired beta/audio speech curves, averaged within each block/channel."""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from signal_processing import finite_mean, gaussian_smooth_nan, curve_slope
from visualization import cluster_color


def summarize_speech_relationship(epochs, *, sigma_seconds=.10, slope_method="savgol"):
    if not np.isfinite(sigma_seconds) or sigma_seconds < 0:
        raise ValueError("Display smoothing sigma must be finite and nonnegative")
    if slope_method not in {"savgol", "gradient"}:
        raise ValueError("Invalid slope_method")
    progress = np.asarray(epochs.progress_percent)
    step = float(np.median(np.diff(progress)))
    audio = np.asarray(epochs.audio_speech, dtype=float)
    finite = np.isfinite(audio)
    scale = float(np.max(np.abs(audio[finite]))) if finite.any() else 0
    if scale <= 0:
        raise ValueError("Audio envelope has no finite nonzero samples")
    audio = audio / scale
    beta_values, audio_values, beta_slopes, audio_slopes = [], [], [], []
    for beta, envelope, duration in zip(epochs.speech, audio, epochs.events.duration, strict=True):
        valid = np.isfinite(beta).all(axis=1) & np.isfinite(envelope).all()
        seconds_per_unit = float(duration) / 100
        rate = 1 / (step * seconds_per_unit)
        beta = gaussian_smooth_nan(beta, rate, sigma_seconds=sigma_seconds)
        envelope = gaussian_smooth_nan(envelope, rate, sigma_seconds=sigma_seconds)
        options = dict(sigma_seconds=sigma_seconds, seconds_per_axis_unit=seconds_per_unit, method=slope_method)
        b_slope = curve_slope(beta, progress, **options)
        a_slope = curve_slope(envelope, progress, **options)
        beta_values.append(np.where(valid[:, None], beta, np.nan))
        audio_values.append(np.where(valid[:, None], envelope[None, :], np.nan))
        beta_slopes.append(np.where(valid[:, None], b_slope, np.nan))
        audio_slopes.append(np.where(valid[:, None], a_slope[None, :], np.nan))
    beta_slopes = np.asarray(beta_slopes)
    audio_slopes = np.asarray(audio_slopes)
    return {"beta": finite_mean(beta_values), "audio": finite_mean(audio_values),
            "beta_slope": finite_mean(beta_slopes), "audio_slope": finite_mean(audio_slopes),
            "interaction": finite_mean(beta_slopes * audio_slopes)}


def save_audio_plots(assignments, cache, output):
    labels = sorted(assignments.loc[assignments.label >= 0, "label"].unique())
    progress = cache["progress_percent"]
    fig, axes = plt.subplots(len(labels), 3, figsize=(15, max(3.3, 3.3 * len(labels))), squeeze=False)
    records = []
    for row, label in enumerate(labels):
        mask = assignments.label.to_numpy() == label
        means = {key: finite_mean(cache[key][mask]) for key in ("beta", "audio", "beta_slope", "audio_slope", "interaction")}
        color = cluster_color(int(label))
        axes[row, 0].plot(progress, means["beta"], color=color, label="Normalized beta change")
        audio_axis = axes[row, 0].twinx()
        audio_axis.plot(progress, means["audio"], color="black", alpha=.8, linestyle="--", label="Audio envelope")
        audio_axis.set_ylabel("Audio / block peak")
        audio_axis.set_ylim(0, 1.05)
        axes[row, 0].set_ylabel("Normalized beta change")
        axes[row, 0].legend(loc="upper left", fontsize=8)
        audio_axis.legend(loc="upper right", fontsize=8)
        axes[row, 1].plot(progress, means["beta_slope"], color=color, label="Beta slope")
        slope_axis = axes[row, 1].twinx()
        slope_axis.plot(progress, means["audio_slope"], color="black", linestyle="--", label="Audio slope")
        axes[row, 1].set_ylabel("Beta shape / percentage point")
        slope_axis.set_ylabel("Normalized audio / percentage point")
        axes[row, 1].legend(loc="upper left", fontsize=8)
        slope_axis.legend(loc="upper right", fontsize=8)
        axes[row, 2].plot(progress, means["interaction"], color=color)
        axes[row, 2].set_ylabel("Mean within-trial slope product")
        for col, title in enumerate(["Beta and audio envelope", "Beta and audio slopes", "Beta slope x audio slope"]):
            axes[row, col].set_title(f"Cluster {label} (n={int(mask.sum())}): {title}", fontsize=10)
            axes[row, col].set(xlabel="Speech progress (%)", xlim=(0,100))
            axes[row, col].axhline(0, color="gray", linewidth=.7)
        for i, point in enumerate(progress):
            records.append({"label": int(label), "progress_percent": float(point), "n_samples": int(mask.sum()),
                            **{key: float(values[i]) for key, values in means.items()}})
    fig.tight_layout()
    fig.savefig(output / "beta_audio_dynamics.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(records).to_csv(output / "beta_audio_dynamics.csv", index=False, encoding="utf-8-sig")
